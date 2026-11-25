"""Reverse the resident Attention mHC post and replicate the FP32 Wo gradient."""

from dataclasses import dataclass
from math import prod
from time import perf_counter

from .dsv41_packed_attention_tape import packed_attention_tape_shapes
from .dsv41_packed_mhc_vjp import make_packed_mhc_projection_vjp


def make_packed_attn_post_vjp(block, last_layers, model, dp, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld

    hidden, copies = model.hidden_size, model.hc_mult
    flat, channels, world = hidden * copies, (copies + 2) * copies, dp * tp
    shapes = packed_attention_tape_shapes(block, last_layers, hidden, copies)
    residual_shape = shapes["attention_tape_residual"][0]
    mixes_shape = shapes["attention_tape_mixes"][0]
    delta_shape = shapes["attention_tape_delta"][0]
    column_tile = 128
    columns = hidden // column_tile
    project, tile_vjp = make_packed_mhc_projection_vjp(block, model, column_tile)

    @pl.jit
    def finish(
        residual_tape: pl.Tensor[residual_shape, pl.FP32],
        mixes: pl.Tensor[mixes_shape, pl.FP32],
        delta_tape: pl.Tensor[delta_shape, pl.BF16],
        stream_seed: pl.Tensor[[block * copies, hidden], pl.FP32],
        premix_seed: pl.Tensor[[block, 16], pl.FP32],
        function: pl.Tensor[[channels, flat], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        bias: pl.Tensor[[channels], pl.FP32],
        delta_grad: pl.Out[pl.Tensor[[block, hidden], pl.FP32]],
        residual_grad: pl.Out[pl.Tensor[[block * copies, hidden], pl.FP32]],
        window: pld.DistributedTensor[[block, hidden], pl.FP32],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        offset = slot * block
        saved = pl.slice(residual_tape, [block * copies, hidden], [offset * copies, 0])
        with pl.spmd(block, name_hint="orthrus_attention_tp_output_gradient") as delta_ready:
            row = pl.tile.get_block_idx()
            value = pl.full([1, hidden], dtype=pl.FP32, value=0.)
            if row < active:
                for stream in pl.range(copies):
                    incoming = pl.slice(stream_seed, [1, hidden], [row * copies + stream, 0])
                    value = pl.add(value, pl.mul(incoming, pl.read(mixes, [offset + row, copies + stream])))
            delta_grad[row:row + 1, :] = value

        # The VJP of the forward sum is the same FP32 seed on every TP lane.
        with pl.spmd(active, name_hint="orthrus_attention_tp_gradient_publish", deps=[delta_ready]) as published:
            sending = pl.tile.get_block_idx()
            for lane in pl.range(1, tp):
                pld.tensor.put(dst=window, peer=owner + lane, src=delta_grad,
                               dst_offsets=[sending, 0], src_offsets=[sending, 0], shape=[1, hidden])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="orthrus_attention_tp_gradient_ready",
                   deps=[published], allow_early_resolve=False):
            for lane in pl.range(1, tp):
                pld.system.notify(target=ready, peer=owner + lane, offsets=[0, 0], value=epoch, op=pld.NotifyOp.Set)

        seeds = pl.create_tensor([block, 32], dtype=pl.FP32)
        with pl.spmd(block, name_hint="orthrus_attention_mhc_seeds"):
            token = pl.tile.get_block_idx()
            seed = pl.tile.full([1, 32], dtype=pl.FP32, value=0.)
            if token < active:
                partial = pl.cast(pl.slice(delta_tape, [1, hidden], [offset + token, 0]), pl.FP32)
                ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
                for target in pl.range(copies):
                    dy = pl.slice(stream_seed, [1, hidden], [token * copies + target, 0])
                    product = pl.reshape(pl.mul(dy, partial), [8, hidden // 8])
                    stripes = pl.row_expand_mul(ones, pl.row_sum(product))
                    dot = pl.row_sum(pl.transpose(stripes, 0, 1))
                    pl.tile.write(seed, [0, target], pl.read(premix_seed, [token, target]))
                    pl.tile.write(seed, [0, copies + target], pl.read(dot, [0, 0]))
                    for source in pl.range(copies):
                        x = pl.slice(saved, [1, hidden], [token * copies + source, 0])
                        product_x = pl.reshape(pl.mul(dy, x), [8, hidden // 8])
                        stripes_x = pl.row_expand_mul(ones, pl.row_sum(product_x))
                        dot_x = pl.row_sum(pl.transpose(stripes_x, 0, 1))
                        pl.tile.write(seed, [0, 2 * copies + source * copies + target], pl.read(dot_x, [0, 0]))
            pl.store(seed, [token, 0], seeds)

        projected, inverse, correction = project(saved, seeds, function, scale, bias, active)
        with pl.spmd(block * copies * columns, name_hint="orthrus_attention_mhc_residual"):
            panel = pl.tile.get_block_idx()
            subrow, col = panel // columns, panel % columns * column_tile
            position, stream = subrow // copies, subrow % copies
            dx = pl.full([1, column_tile], dtype=pl.FP32, value=0.)
            if position < active:
                dx = tile_vjp(saved, function, projected, inverse, correction, subrow, col)
                for target in pl.range(copies):
                    coefficient = pl.read(mixes, [offset + position, 2 * copies + stream * copies + target])
                    dy_tile = pl.slice(stream_seed, [1, column_tile], [position * copies + target, col])
                    dx = pl.add(dx, pl.mul(dy_tile, coefficient))
            residual_grad[subrow:subrow + 1, col:col + column_tile] = dx
        return delta_grad, residual_grad

    @pl.jit
    def receive(
        window: pld.DistributedTensor[[block, hidden], pl.FP32],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        delta_grad: pl.Out[pl.Tensor[[block, hidden], pl.FP32]],
        active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as available:
            pld.system.wait(signal=ready, offsets=[0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        with pl.spmd(block, name_hint="orthrus_attention_tp_gradient_receive", deps=[available]):
            row = pl.tile.get_block_idx()
            if row < active:
                delta_grad[row:row + 1, :] = window[row:row + 1, :]
            else:
                delta_grad[row:row + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
        return delta_grad

    @pl.jit.host
    def dispatch(
        residual: pl.Tensor[[dp, *residual_shape], pl.FP32],
        mixes: pl.Tensor[[dp, *mixes_shape], pl.FP32],
        reduced: pl.Tensor[[dp, *delta_shape], pl.BF16],
        stream_seed: pl.Tensor[[dp, block * copies, hidden], pl.FP32],
        premix_seed: pl.Tensor[[dp, block, 16], pl.FP32],
        function: pl.Tensor[[dp, channels, flat], pl.FP32],
        scale: pl.Tensor[[dp, 3], pl.FP32],
        bias: pl.Tensor[[dp, channels], pl.FP32],
        delta_grad: pl.Out[pl.Tensor[[world, block, hidden], pl.FP32]],
        residual_grad: pl.Out[pl.Tensor[[dp, block * copies, hidden], pl.FP32]],
        owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        gradient_window = pld.alloc_window_buffer([block, hidden], dtype=pl.FP32)
        gradient_ready = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        replica = owner // tp
        for rank in pl.range(pld.world_size()):
            window = pld.window(gradient_window, [block, hidden], dtype=pl.FP32)
            ready = pld.window(gradient_ready, [1, 1], dtype=pl.INT32)
            if rank == owner:
                finish(residual[replica], mixes[replica], reduced[replica], stream_seed[replica], premix_seed[replica],
                       function[replica], scale[replica], bias[replica], delta_grad[rank], residual_grad[replica],
                       window, ready, owner, active, slot, epoch, device=rank)
            elif rank > owner and rank < owner + tp:
                receive(window, ready, delta_grad[rank], active, epoch, device=rank)
    return dispatch


@dataclass(frozen=True)
class PackedAttentionPostGradient:
    owner: int
    layer: int
    rows: int
    delta: object
    residual: object
    seconds: float


class PackedAttentionPostVJP:
    def __init__(self, proposer, ffn_mhc, graph, banks, model):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.ffn_mhc, self.owners = ffn_mhc, ffn_mhc.owners
        self.ranks = tuple(range(proposer.world))
        self.device_tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.first_layer = model.num_hidden_layers - proposer.last_layers
        hidden, copies = model.hidden_size, model.hc_mult
        channels = copies * (copies + 2)
        geometry = (("hc_attn_fn", (channels, copies * hidden)), ("hc_attn_scale", (3,)),
                    ("hc_attn_base", (channels,)))
        self.banks = tuple((shape, tuple(banks[rank][key] for rank in self.owners)) for key, shape in geometry)
        names = ("attention_tape_residual", "attention_tape_mixes", "attention_tape_delta")
        self.tape = {key: tuple(proposer.ranks[rank][key] for rank in self.owners) for key in names}

        def allocate(shape, ranks):
            return self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank) for rank in ranks], ranks)

        self.delta = allocate((proposer.block, hidden), self.ranks)
        self.residual = allocate((proposer.block * copies, hidden), self.owners)
        self.epoch = 0

    def _stack(self, shards, ranks):
        return self.stacked(shards, (len(shards), *shards[0].shape), ranks)

    def _layer_weights(self, layer):
        for shape, shards in self.banks:
            views = []
            for bank in shards:
                offset = layer * prod(shape) * bank.dtype.itemsize
                views.append(self.device_tensor(bank.data_ptr + offset, shape, bank.dtype, buffer=bank.buffer,
                                                byte_offset=bank.byte_offset + offset))
            yield self._stack(views, self.owners)

    def __call__(self, student, seed):
        owner, layer, rows = seed.owner, seed.layer, seed.rows
        index = self.owners.index(owner)
        if (seed.streams is not self.ffn_mhc.outputs[0].shards[index]
                or seed.premix is not self.ffn_mhc.outputs[1].shards[index]):
            raise ValueError("Attention post backward requires both live FFN mHC gradient outputs")
        tape = []
        for key, fallback in self.tape.items():
            shards = list(fallback)
            shards[index] = student.ffn_tape[owner][key]
            tape.append(self._stack(shards, self.owners))
        self.epoch += 1
        start = perf_counter()
        self.runtime.run(self.graph, *tape, *self.ffn_mhc.outputs, *self._layer_weights(layer),
                         self.delta, self.residual, owner, rows, layer - self.first_layer,
                         self.epoch, config=self.config)
        return PackedAttentionPostGradient(owner, layer, rows, self.delta, self.residual.shards[index],
                                           perf_counter() - start)
