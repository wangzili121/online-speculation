# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident layer state for reversing the packed draft's native MoE suffix."""

from dataclasses import dataclass
from time import perf_counter


def packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies):
    """Give each compact route row a full 64-byte line."""
    local = block // tp
    return {
        "ffn_tape_residual": ((max(1, last_layers * block * copies), hidden), "float32"),
        "ffn_tape_mixes": ((max(1, last_layers * block), 32), "float32"),
        "ffn_tape_vectors": ((max(1, last_layers * (2 * block + local)), hidden), "bfloat16"),
        "ffn_tape_route_ids": ((max(1, last_layers * local), 16), "int32"),
        "ffn_tape_route_weights": ((max(1, last_layers * local), 16), "float32"),
    }


def make_packed_ffn_capture(block, last_layers, tp, hidden, copies, topk):
    import pypto.language as pl

    local = block // tp
    shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    residual_shape = shapes["ffn_tape_residual"][0]
    mix_shape = shapes["ffn_tape_mixes"][0]
    vector_shape = shapes["ffn_tape_vectors"][0]
    route_shape = shapes["ffn_tape_route_ids"][0]

    @pl.jit.inline(auto_scope=False)
    def save(
        residual: pl.Tensor[[block, copies, hidden], pl.FP32],
        pre: pl.Tensor[[block, copies], pl.FP32],
        post: pl.Tensor[[block, copies], pl.FP32],
        combination: pl.Tensor[[block, copies, copies], pl.FP32],
        collapsed: pl.Tensor[[block, hidden], pl.BF16],
        combined: pl.Tensor[[block, hidden], pl.BF16],
        normed: pl.Tensor,
        route_ids: pl.Tensor,
        route_weights: pl.Tensor,
        saved_residual: pl.Tensor[residual_shape, pl.FP32],
        saved_mixes: pl.Tensor[mix_shape, pl.FP32],
        saved_vectors: pl.Tensor[vector_shape, pl.BF16],
        saved_route_ids: pl.Tensor[route_shape, pl.INT32],
        saved_route_weights: pl.Tensor[route_shape, pl.FP32],
        slot: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(residual, [block * copies, hidden])
        mix_flat = pl.reshape(combination, [block, copies * copies])
        with pl.spmd(block, name_hint="orthrus_save_ffn_boundary"):
            row = pl.tile.get_block_idx()
            for stream in pl.range(copies):
                source = row * copies + stream
                dest = slot * block * copies + source
                saved_residual[dest:dest + 1, :] = pl.slice(flat, [1, hidden], [source, 0])
            coeff = pl.tile.full([1, 32], dtype=pl.FP32, value=0.)
            for stream in pl.range(copies):
                pl.tile.write(coeff, [0, stream], pl.read(pre, [row, stream]))
                pl.tile.write(coeff, [0, copies + stream], pl.read(post, [row, stream]))
            for index in pl.range(copies * copies):
                pl.tile.write(coeff, [0, 2 * copies + index], pl.read(mix_flat, [row, index]))
            pl.store(coeff, [slot * block + row, 0], saved_mixes)
            first = slot * (2 * block + local) + row
            saved_vectors[first:first + 1, :] = pl.slice(collapsed, [1, hidden], [row, 0])
            saved_vectors[first + block:first + block + 1, :] = pl.slice(combined, [1, hidden], [row, 0])
        with pl.spmd(local, name_hint="orthrus_save_ffn_routes"):
            row = pl.tile.get_block_idx()
            vector_row = slot * (2 * block + local) + 2 * block + row
            saved_vectors[vector_row:vector_row + 1, :] = pl.slice(normed, [1, hidden], [row, 0])
            ids = pl.load(route_ids, [row, 0], [1, 16], valid_shape=[1, topk])
            weights = pl.load(route_weights, [row, 0], [1, 16], valid_shape=[1, topk])
            dest = slot * local + row
            pl.store(pl.set_validshape(pl.fillpad(ids, pad_value=pl.PadValue.zero), 1, 16), [dest, 0], saved_route_ids)
            pl.store(pl.set_validshape(pl.fillpad(weights, pad_value=pl.PadValue.zero), 1, 16), [dest, 0], saved_route_weights)

    return save


def make_packed_ffn_post_vjp(block, last_layers, rows, hidden, copies, dp, tp):
    """Reverse native post-mix before the expert and mHC-pre input VJPs."""
    import pypto.language as pl

    local = block // tp
    shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    residual_shape = shapes["ffn_tape_residual"][0]
    mix_shape = shapes["ffn_tape_mixes"][0]
    vector_shape = shapes["ffn_tape_vectors"][0]

    @pl.jit
    def reverse(
        residual: pl.Tensor[residual_shape, pl.FP32],
        mixes: pl.Tensor[mix_shape, pl.FP32],
        vectors: pl.Tensor[vector_shape, pl.BF16],
        grad_hidden: pl.Tensor[[rows, copies, hidden], pl.FP32],
        grad_pre: pl.Tensor[[rows, 16], pl.FP32],
        grad_value: pl.Out[pl.Tensor[[block, hidden], pl.FP32]],
        grad_residual: pl.Out[pl.Tensor[[block * copies, hidden], pl.FP32]],
        grad_mixes: pl.Out[pl.Tensor[[block, 32], pl.FP32]],
        slot: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
    ):
        incoming = pl.reshape(grad_hidden, [rows * copies, hidden])
        with pl.spmd(block, name_hint="orthrus_packed_ffn_post_data_vjp"):
            row = pl.tile.get_block_idx()
            mix_row = slot * block + row
            dvalue = pl.full([1, hidden], dtype=pl.FP32, value=0.)
            if row < active:
                for target in pl.range(copies):
                    dy = pl.slice(incoming, [1, hidden], [row * copies + target, 0])
                    dvalue = pl.add(dvalue, pl.mul(dy, pl.read(mixes, [mix_row, copies + target])))
            grad_value[row:row + 1, :] = dvalue
            for source in pl.range(copies):
                dresidual = pl.full([1, hidden], dtype=pl.FP32, value=0.)
                if row < active:
                    for target in pl.range(copies):
                        dy = pl.slice(incoming, [1, hidden], [row * copies + target, 0])
                        column = 2 * copies + source * copies + target
                        dresidual = pl.add(dresidual, pl.mul(dy, pl.read(mixes, [mix_row, column])))
                dest = row * copies + source
                grad_residual[dest:dest + 1, :] = dresidual
        with pl.spmd(block, name_hint="orthrus_packed_ffn_post_mix_vjp"):
            row = pl.tile.get_block_idx()
            gradients = pl.tile.full([1, 32], dtype=pl.FP32, value=0.)
            if row < active:
                value_row = slot * (2 * block + local) + block + row
                value = pl.cast(pl.slice(vectors, [1, hidden], [value_row, 0]), pl.FP32)
                ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
                for target in pl.range(copies):
                    dy = pl.slice(incoming, [1, hidden], [row * copies + target, 0])
                    product = pl.reshape(pl.mul(dy, value), [8, hidden // 8])
                    partial = pl.row_expand_mul(ones, pl.row_sum(product))
                    dot = pl.row_sum(pl.transpose(partial, 0, 1))
                    pl.tile.write(gradients, [0, target], pl.read(grad_pre, [row, target]))
                    pl.tile.write(gradients, [0, copies + target], pl.read(dot, [0, 0]))
                    for source in pl.range(copies):
                        residual_row = (slot * block + row) * copies + source
                        x = pl.slice(residual, [1, hidden], [residual_row, 0])
                        product = pl.reshape(pl.mul(dy, x), [8, hidden // 8])
                        partial = pl.row_expand_mul(ones, pl.row_sum(product))
                        dot = pl.row_sum(pl.transpose(partial, 0, 1))
                        column = 2 * copies + source * copies + target
                        pl.tile.write(gradients, [0, column], pl.read(dot, [0, 0]))
            pl.store(gradients, [row, 0], grad_mixes)
        return grad_value, grad_residual, grad_mixes

    @pl.jit.host
    def dispatch(
        residual: pl.Tensor[[dp, *residual_shape], pl.FP32],
        mixes: pl.Tensor[[dp, *mix_shape], pl.FP32],
        vectors: pl.Tensor[[dp, *vector_shape], pl.BF16],
        grad_hidden: pl.Tensor[[dp, rows, copies, hidden], pl.FP32],
        grad_pre: pl.Tensor[[dp, rows, 16], pl.FP32],
        grad_value: pl.Out[pl.Tensor[[dp, block, hidden], pl.FP32]],
        grad_residual: pl.Out[pl.Tensor[[dp, block * copies, hidden], pl.FP32]],
        grad_mixes: pl.Out[pl.Tensor[[dp, block, 32], pl.FP32]],
        owner: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
    ):
        replica = owner // tp
        reverse(residual[replica], mixes[replica], vectors[replica], grad_hidden[replica], grad_pre[replica],
                grad_value[replica], grad_residual[replica], grad_mixes[replica], slot, active, device=owner)

    return dispatch


@dataclass(frozen=True)
class PackedFFNPostGradient:
    """FP32 gradients before MoE and the FFN mHC-pre operation."""

    owner: int
    layer: int
    rows: int
    value: object
    residual: object
    mixes: object
    seconds: float


class PackedFFNPostVJP:
    def __init__(self, proposer, final_vjp, graph, layer_count):
        import torch
        from pypto.runtime import StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.first_layer = layer_count - proposer.last_layers
        self.last_layer = layer_count - 1
        self.owners = final_vjp.owners
        self.stacked = StackedDeviceTensor
        self.tape_inputs = {
            name: [proposer.ranks[owner][name] for owner in self.owners]
            for name in ("ffn_tape_residual", "ffn_tape_mixes", "ffn_tape_vectors")
        }
        self.grad_streams, self.grad_pre = final_vjp.grad_streams, final_vjp.grad_premix
        _, _, copies, hidden = self.grad_streams.full_shape

        def allocate(shape):
            return self._stack([self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank)
                                for rank in self.owners])

        self.outputs = (allocate((proposer.block, hidden)), allocate((proposer.block * copies, hidden)),
                        allocate((proposer.block, 32)))

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.owners)

    def __call__(self, student, gradient, *, layer=None):
        owner = student.owner
        replica = self.owners.index(owner)
        layer = self.last_layer if layer is None else layer
        if (student.ffn_tape is None or gradient.owner != owner or not self.first_layer <= layer <= self.last_layer
                or not 1 <= gradient.rows <= self.grad_streams.shape[1]):
            raise ValueError("FFN post reverse requires the retained tape and live output gradients")
        args = []
        for name, bound in self.tape_inputs.items():
            shards = list(bound)
            shards[replica] = student.ffn_tape[owner][name]
            args.append(self._stack(shards))
        start = perf_counter()
        seeds = []
        for fallback, value in ((self.grad_streams, gradient.streams), (self.grad_pre, gradient.premix)):
            shards = list(fallback.shards)
            shards[replica] = value
            seeds.append(self._stack(shards))
        self.runtime.run(self.graph, *args, *seeds, *self.outputs,
                         owner, layer - self.first_layer, gradient.rows, config=self.config)
        return PackedFFNPostGradient(owner, layer, gradient.rows, *(output.shards[replica] for output in self.outputs),
                                     perf_counter() - start)
