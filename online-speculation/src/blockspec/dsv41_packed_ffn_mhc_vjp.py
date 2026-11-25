"""Reverse the frozen mHC FFN boundary with FP32 residual streams."""

from dataclasses import dataclass
from math import prod
from time import perf_counter

from .dsv41_packed_ffn_tape import packed_ffn_tape_shapes
from .pypto_dsv41_tail_vjp import make_hc_pre_gate_vjp


def make_packed_ffn_mhc_vjp(block, last_layers, model, dp, tp):
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.hc import DECODE_LINEAR_OK, LINEAR_T_TILE, MIX_PAD, hc_project_decode

    hidden, copies = model.hidden_size, model.hc_mult
    flat, channels = hidden * copies, (copies + 2) * copies
    linear_rows = (block + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    residual_shape = shapes["ffn_tape_residual"][0]
    mix_shape = shapes["ffn_tape_mixes"][0]
    gate_vjp = make_hc_pre_gate_vjp(block, copies, model.hc_sinkhorn_iters, model.hc_eps, inline=True)
    column_tile = 128

    @pl.jit
    def finish(
        residual_tape: pl.Tensor[residual_shape, pl.FP32],
        mix_tape: pl.Tensor[mix_shape, pl.FP32],
        passthrough: pl.Tensor[[block * copies, hidden], pl.FP32],
        mix_grad: pl.Tensor[[block, 32], pl.FP32],
        collapsed_grad: pl.Tensor[[block, hidden], pl.FP32],
        function: pl.Tensor[[channels, flat], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        bias: pl.Tensor[[channels], pl.FP32],
        dx: pl.Out[pl.Tensor[[block * copies, hidden], pl.FP32]],
        dpremix: pl.Out[pl.Tensor[[block, 16], pl.FP32]],
        active: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        offset = slot * block
        saved = pl.slice(residual_tape, [block * copies, hidden], [offset * copies, 0])
        x = pl.reshape(saved, [block, copies, hidden])
        inverse = pl.create_tensor([linear_rows, 1], dtype=pl.FP32)
        partials = pl.create_tensor([DECODE_LINEAR_OK * linear_rows, MIX_PAD], dtype=pl.FP32)
        hc_project_decode(x, function, inverse, partials, short_attention=False, one_token=False)
        projection = pl.create_tensor([block, channels], dtype=pl.FP32)
        incoming_pre = pl.create_tensor([block, 8], dtype=pl.FP32)
        incoming_post = pl.create_tensor([block, 8], dtype=pl.FP32)
        incoming_mix = pl.create_tensor([block * copies, 8], dtype=pl.FP32)
        with pl.spmd(block, name_hint="orthrus_hc_projected_and_gate_seeds"):
            token = pl.tile.get_block_idx()
            combined = pl.tile.full([1, channels], dtype=pl.FP32, value=0.)
            for split in pl.range(DECODE_LINEAR_OK):
                combined = pl.add(combined, pl.load(partials, [split * linear_rows + token, 0], [1, channels]))
            pl.store(pl.mul(combined, pl.read(inverse, [token, 0])), [token, 0], projection)
            seed_pre = pl.tile.full([1, 8], dtype=pl.FP32, value=0.)
            seed_post = pl.tile.full([1, 8], dtype=pl.FP32, value=0.)
            for stream in pl.unroll(copies):
                if token < active:
                    pl.tile.write(seed_pre, [0, stream], pl.read(mix_grad, [token, stream]))
                    pl.tile.write(seed_post, [0, stream], pl.read(mix_grad, [token, copies + stream]))
                seed_mix = pl.tile.full([1, 8], dtype=pl.FP32, value=0.)
                if token < active:
                    for target in pl.unroll(copies):
                        pl.tile.write(seed_mix, [0, target],
                                      pl.read(mix_grad, [token, 2 * copies + stream * copies + target]))
                pl.store(seed_mix, [token * copies + stream, 0], incoming_mix)
            pl.store(seed_pre, [token, 0], incoming_pre)
            pl.store(seed_post, [token, 0], incoming_post)

        projected_grad = pl.create_tensor([block, channels], dtype=pl.FP32)
        gate_vjp(projection, scale, bias, incoming_pre, incoming_post, incoming_mix, projected_grad)
        correction = pl.create_tensor([block, 8], dtype=pl.FP32)
        with pl.spmd(block, name_hint="orthrus_hc_rms_vjp_coefficient"):
            tok = pl.tile.get_block_idx()
            dot = pl.cast(0., pl.FP32)
            for channel in pl.range(channels):
                dot = dot + pl.read(projection, [tok, channel]) * pl.read(projected_grad, [tok, channel])
            inv = pl.read(inverse, [tok, 0])
            value = dot * inv * inv * (1. / flat)
            pl.store(pl.add(pl.tile.full([1, 8], dtype=pl.FP32, value=0.), value), [tok, 0], correction)

        # One small projection transpose, followed by the direct residual,
        # collapse and normalized-projection contributions in FP32.
        columns = hidden // column_tile
        with pl.spmd(block * copies * columns, name_hint="orthrus_hc_vjp_add_residual"):
            idx = pl.tile.get_block_idx()
            row, col = idx // columns, idx % columns * column_tile
            token_idx, source = row // copies, row % copies
            total = pl.full([1, column_tile], dtype=pl.FP32, value=0.)
            if token_idx < active:
                for feature in pl.range(channels):
                    weight = pl.slice(function, [1, column_tile], [feature, source * hidden + col])
                    total = pl.add(total, pl.mul(weight, pl.read(projected_grad, [token_idx, feature])))
                orig = pl.slice(saved, [1, column_tile], [row, col])
                projected_dx = pl.mul(total, pl.read(inverse, [token_idx, 0]))
                projected_dx = pl.sub(projected_dx, pl.mul(orig, pl.read(correction, [token_idx, 0])))
                direct = pl.mul(pl.slice(collapsed_grad, [1, column_tile], [token_idx, col]),
                                pl.read(mix_tape, [offset + token_idx, source]))
                total = pl.add(pl.add(projected_dx, direct), pl.slice(passthrough, [1, column_tile], [row, col]))
            dx[row:row + 1, col:col + column_tile] = total

        with pl.spmd(block, name_hint="orthrus_hc_vjp_previous_premix"):
            row_token = pl.tile.get_block_idx()
            result = pl.tile.full([1, 16], dtype=pl.FP32, value=0.)
            if row_token < active:
                dy = pl.slice(collapsed_grad, [1, hidden], [row_token, 0])
                ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
                for mix in pl.range(copies):
                    part_value = pl.slice(saved, [1, hidden], [row_token * copies + mix, 0])
                    product = pl.reshape(pl.mul(part_value, dy), [8, hidden // 8])
                    summed = pl.row_expand_mul(ones, pl.row_sum(product))
                    total_dot = pl.row_sum(pl.transpose(summed, 0, 1))
                    pl.tile.write(result, [0, mix], pl.read(total_dot, [0, 0]))
            pl.store(result, [row_token, 0], dpremix)
        return dx, dpremix

    @pl.jit.host
    def dispatch(
        residual: pl.Tensor[[dp, *residual_shape], pl.FP32],
        mixes: pl.Tensor[[dp, *mix_shape], pl.FP32],
        passthrough: pl.Tensor[[dp, block * copies, hidden], pl.FP32],
        mix_grad: pl.Tensor[[dp, block, 32], pl.FP32],
        collapsed_grad: pl.Tensor[[dp, block, hidden], pl.FP32],
        function: pl.Tensor[[dp, channels, flat], pl.FP32],
        scale: pl.Tensor[[dp, 3], pl.FP32],
        bias: pl.Tensor[[dp, channels], pl.FP32],
        dx: pl.Out[pl.Tensor[[dp, block * copies, hidden], pl.FP32]],
        dpremix: pl.Out[pl.Tensor[[dp, block, 16], pl.FP32]],
        owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        replica = owner // tp
        finish(residual[replica], mixes[replica], passthrough[replica], mix_grad[replica], collapsed_grad[replica],
               function[replica], scale[replica], bias[replica], dx[replica], dpremix[replica], active, slot, device=owner)
    return dispatch


@dataclass(frozen=True)
class PackedFFNAttentionGradient:
    owner: int
    layer: int
    rows: int
    streams: object
    premix: object
    seconds: float


class PackedFFNMHCVJP:
    def __init__(self, proposer, post, ffn, graph, banks, model):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.post, self.ffn, self.owners = post, ffn, post.owners
        self.device_tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.first_layer = model.num_hidden_layers - proposer.last_layers
        hidden, copies = model.hidden_size, model.hc_mult
        widths = ((copies * (copies + 2), copies * hidden), (3,), (copies * (copies + 2),))
        names = ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base")
        self.banks = tuple((shape, tuple(banks[rank][key] for rank in self.owners))
                           for key, shape in zip(names, widths, strict=True))
        self.tape = {key: tuple(proposer.ranks[rank][key] for rank in self.owners)
                     for key in ("ffn_tape_residual", "ffn_tape_mixes")}
        self.outputs = tuple(self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank)
                                         for rank in self.owners])
                             for shape in ((proposer.block * copies, hidden), (proposer.block, 16)))

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.owners)

    def _weights(self, layer):
        for shape, banks in self.banks:
            views = []
            for bank in banks:
                offset = layer * prod(shape) * bank.dtype.itemsize
                views.append(self.device_tensor(bank.data_ptr + offset, shape, bank.dtype, buffer=bank.buffer,
                                                byte_offset=bank.byte_offset + offset))
            yield self._stack(views)

    def __call__(self, student, post_gradient, input_gradient):
        owner, layer, rows = input_gradient.owner, input_gradient.layer, input_gradient.rows
        index = self.owners.index(owner)
        if ((post_gradient.owner, post_gradient.layer, post_gradient.rows) != (owner, layer, rows)
                or post_gradient.residual is not self.post.outputs[1].shards[index]
                or post_gradient.mixes is not self.post.outputs[2].shards[index]
                or input_gradient.collapsed is not self.ffn.output.shards[owner]):
            raise ValueError("FFN mHC backward requires matching live residual and collapsed gradients")
        tape = []
        for key, fallback in self.tape.items():
            shards = list(fallback)
            shards[index] = student.ffn_tape[owner][key]
            tape.append(self._stack(shards))
        from_ffn = self._stack([self.ffn.output.shards[rank] for rank in self.owners])
        start = perf_counter()
        self.runtime.run(self.graph, *tape, self.post.outputs[1], self.post.outputs[2], from_ffn,
                         *self._weights(layer), *self.outputs, owner, rows, layer - self.first_layer,
                         config=self.config)
        return PackedFFNAttentionGradient(owner, layer, rows, *(item.shards[index] for item in self.outputs),
                                           perf_counter() - start)
