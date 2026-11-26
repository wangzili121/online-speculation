"""Carry the draft projection gradient through the frozen Attention input."""

from time import perf_counter

from .dsv41_packed_attention_tape import packed_attention_tape_shapes
from .dsv41_packed_final_vjp import PackedFinalGradient, make_packed_head_norm_vjp


def make_packed_attn_input_vjp(block, last_layers, model, dp, tp):
    import pypto.language as pl

    hidden, copies = model.hidden_size, model.hc_mult
    shapes = packed_attention_tape_shapes(block, last_layers, hidden, copies)
    residual_shape = shapes["attention_tape_residual"][0]
    mixes_shape = shapes["attention_tape_mixes"][0]
    backward = make_packed_head_norm_vjp(block, hidden, copies, model.rms_norm_eps, add_residual=True)

    @pl.jit(auto_scope=False)
    def rank_vjp(
        saved: pl.Tensor[residual_shape, pl.FP32],
        mixes: pl.Tensor[mixes_shape, pl.FP32],
        gamma: pl.Tensor[[hidden], pl.BF16],
        normed: pl.Tensor[[block, hidden], pl.FP32],
        direct: pl.Tensor[[block * copies, hidden], pl.FP32],
        streams: pl.Out[pl.Tensor[[block, copies, hidden], pl.FP32]],
        premix: pl.Out[pl.Tensor[[block, 16], pl.FP32]],
        slot: pl.Scalar[pl.INT32],
    ):
        offset = slot * block
        x = pl.reshape(pl.slice(saved, [block * copies, hidden], [offset * copies, 0]), [block, copies, hidden])
        # Keep the tape's row stride. The shared mHC code reads the leading coefficients.
        c = pl.slice(mixes, [block, 32], [offset, 0])
        backward(x, c, gamma, normed, direct, streams, premix)
        return streams, premix

    @pl.jit.host
    def dispatch(
        saved: pl.Tensor[[dp, *residual_shape], pl.FP32],
        mixes: pl.Tensor[[dp, *mixes_shape], pl.FP32],
        gamma: pl.Tensor[[dp, hidden], pl.BF16],
        normed: pl.Tensor[[dp, block, hidden], pl.FP32],
        direct: pl.Tensor[[dp, block * copies, hidden], pl.FP32],
        streams: pl.Out[pl.Tensor[[dp, block, copies, hidden], pl.FP32]],
        premix: pl.Out[pl.Tensor[[dp, block, 16], pl.FP32]],
        owner: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        replica = owner // tp
        rank_vjp(saved[replica], mixes[replica], gamma[replica], normed[replica], direct[replica],
                 streams[replica], premix[replica], slot, device=owner)

    return dispatch


class PackedAttentionInputVJP:
    def __init__(self, proposer, attn_post, qkv, graph, model):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.block, self.owners = proposer.block, qkv.owners
        self.normed, self.direct = qkv.outputs[0], attn_post.residual
        self.tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.first_layer = model.num_hidden_layers - proposer.last_layers
        self.hidden = model.hidden_size
        self.tape = {name: [proposer.ranks[r][name] for r in self.owners]
                     for name in ("attention_tape_residual", "attention_tape_mixes")}
        self.gamma = [proposer.ranks[r]["attn_norm_weight"] for r in self.owners]
        self.outputs = tuple(
            self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank) for rank in self.owners])
            for shape in ((self.block, model.hc_mult, self.hidden), (self.block, 16))
        )

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.owners)

    def __call__(self, student, seed):
        owner, layer = seed.owner, seed.layer
        replica = self.owners.index(owner)
        if (owner != student.owner or seed.normalized is not self.normed.shards[replica]
                or seed.residual is not self.direct.shards[replica]):
            raise ValueError("Attention input backward requires the live projection and residual gradients")
        tape = []
        for name, fallback in self.tape.items():
            handles = list(fallback)
            handles[replica] = student.ffn_tape[owner][name]
            tape.append(self._stack(handles))
        gamma = []
        for bank in self.gamma:
            offset = layer * self.hidden * bank.dtype.itemsize
            gamma.append(self.tensor(bank.data_ptr + offset, (self.hidden,), bank.dtype,
                                     buffer=bank.buffer, byte_offset=bank.byte_offset + offset))
        start = perf_counter()
        self.runtime.run(self.graph, *tape, self._stack(gamma), self.normed, self.direct, *self.outputs,
                         owner, layer - self.first_layer, config=self.config)
        return PackedFinalGradient(owner, seed.rows, *(output.shards[replica] for output in self.outputs),
                                   perf_counter() - start)
