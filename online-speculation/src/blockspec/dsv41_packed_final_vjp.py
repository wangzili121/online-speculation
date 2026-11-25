"""Reverse the packed draft's final RMSNorm and four-stream mHC collapse."""

from dataclasses import dataclass
from time import perf_counter


def make_packed_final_vjp(rows, hidden, copies, dp, tp, epsilon):
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.hc import hc_head

    @pl.jit(auto_scope=False)
    def rank_vjp(
        streams: pl.Tensor[[rows, copies, hidden], pl.FP32],
        premix: pl.Tensor[[rows, copies], pl.FP32],
        gamma: pl.Tensor[[hidden], pl.BF16],
        grad_normed: pl.Tensor[[rows, hidden], pl.FP32],
        grad_streams: pl.Out[pl.Tensor[[rows, copies, hidden], pl.FP32]],
        grad_premix: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
    ):
        collapsed = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        hc_head(streams, premix, collapsed)
        grad_collapsed = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        with pl.spmd(rows, name_hint="orthrus_packed_final_rms_vjp"):
            row = pl.tile.get_block_idx()
            x = pl.reshape(pl.cast(pl.slice(collapsed, [1, hidden], [row, 0]), pl.FP32), [8, hidden // 8])
            weight = pl.reshape(pl.cast(gamma, pl.FP32), [8, hidden // 8])
            dy = pl.reshape(pl.slice(grad_normed, [1, hidden], [row, 0]), [8, hidden // 8])
            scaled = pl.mul(weight, dy)
            ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
            squares = pl.row_expand_mul(ones, pl.row_sum(pl.mul(x, x)))
            mean = pl.mul(pl.row_sum(pl.transpose(squares, 0, 1)), 1. / hidden)
            inv = pl.rsqrt(pl.add(mean, epsilon), high_precision=True)
            dot_strips = pl.row_expand_mul(ones, pl.row_sum(pl.mul(x, scaled)))
            dot = pl.row_sum(pl.transpose(dot_strips, 0, 1))
            correction = pl.mul(pl.mul(pl.mul(inv, inv), inv), pl.mul(dot, 1. / hidden))
            value = pl.sub(pl.row_expand_mul(scaled, inv), pl.row_expand_mul(x, correction))
            grad_collapsed[row:row + 1, :] = pl.reshape(value, [1, hidden])
        flat = pl.reshape(streams, [rows * copies, hidden])
        gradients = pl.reshape(grad_streams, [rows * copies, hidden])
        with pl.spmd(rows, name_hint="orthrus_packed_final_hc_vjp"):
            row = pl.tile.get_block_idx()
            incoming = pl.slice(grad_collapsed, [1, hidden], [row, 0])
            coefficients = pl.tile.full([1, 16], dtype=pl.FP32, value=0.)
            ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
            for stream in pl.range(copies):
                index = row * copies + stream
                residual = pl.slice(flat, [1, hidden], [index, 0])
                gradients[index:index + 1, :] = pl.mul(incoming, pl.read(premix, [row, stream]))
                product = pl.reshape(pl.mul(incoming, residual), [8, hidden // 8])
                dot_strips = pl.row_expand_mul(ones, pl.row_sum(product))
                dot = pl.row_sum(pl.transpose(dot_strips, 0, 1))
                pl.tile.write(coefficients, [0, stream], pl.read(dot, [0, 0]))
            pl.store(coefficients, [row, 0], grad_premix)
        return grad_streams, grad_premix

    @pl.jit.host
    def dispatch(
        streams: pl.Tensor[[dp, rows, copies, hidden], pl.FP32],
        premix: pl.Tensor[[dp, rows, copies], pl.FP32],
        gamma: pl.Tensor[[dp, hidden], pl.BF16],
        grad_normed: pl.Tensor[[dp, rows, hidden], pl.FP32],
        grad_streams: pl.Out[pl.Tensor[[dp, rows, copies, hidden], pl.FP32]],
        grad_premix: pl.Out[pl.Tensor[[dp, rows, 16], pl.FP32]],
        owner: pl.Scalar[pl.INT32],
    ):
        replica = owner // tp
        rank_vjp(streams[replica], premix[replica], gamma[replica], grad_normed[replica],
                 grad_streams[replica], grad_premix[replica], device=owner)

    return dispatch


@dataclass(frozen=True)
class PackedFinalGradient:
    """Gradients at the last draft layer output, retained on the DP owner."""

    owner: int
    rows: int
    streams: object
    premix: object
    seconds: float


class PackedFinalVJP:
    def __init__(self, proposer, lm_head, graph):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.lm_head = lm_head
        self.owners = lm_head.loss.owners
        self.rows = lm_head.loss.rows
        self.tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.streams = [self._leading_rows(proposer.ranks[rank]["x_hc"]) for rank in self.owners]
        self.premix = [self._leading_rows(proposer.ranks[rank]["pre_mix"]) for rank in self.owners]
        self.gamma = self._stack([proposer.ranks[rank]["final_norm_weight"] for rank in self.owners])
        self.grad_normed = self._stack([lm_head.output.shards[rank] for rank in self.owners])

        def allocate(shape):
            return self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank)
                                for rank in self.owners])

        self.grad_streams = allocate(self.streams[0].shape)
        self.grad_premix = allocate((self.rows, 16))

    def _leading_rows(self, value):
        return self.tensor(value.data_ptr, (self.rows, *value.shape[1:]), value.dtype,
                           buffer=value.buffer, byte_offset=value.byte_offset)

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.owners)

    def __call__(self, student, head_gradient):
        owner = student.owner
        if student.final_state is None or head_gradient.owner != owner or head_gradient.rows > student.rows:
            raise ValueError("Packed final VJP requires captured activations and their LM head gradient")
        replica = self.owners.index(owner)
        if head_gradient.gradient is not self.lm_head.output.shards[owner]:
            raise ValueError("Packed final VJP requires the resident LM head input gradient")
        streams, premix = list(self.streams), list(self.premix)
        streams[replica] = self._leading_rows(student.final_state.streams)
        premix[replica] = self._leading_rows(student.final_state.premix)
        start = perf_counter()
        self.runtime.run(self.graph, self._stack(streams), self._stack(premix), self.gamma,
                         self.grad_normed, self.grad_streams, self.grad_premix, owner, config=self.config)
        return PackedFinalGradient(owner, head_gradient.rows, self.grad_streams.shards[replica],
                                   self.grad_premix.shards[replica], perf_counter() - start)
