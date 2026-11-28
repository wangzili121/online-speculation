# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO leaves for the frozen DSV4.1 suffix reverse pass."""

from .kernel_abi import with_resident_entry


def make_hc_pre_gate_vjp(rows=32, copies=4, iterations=20, eps=1e-6, *, inline=False):
    """Reverse mHC sigmoid gates and the native duplicated 8x8 Sinkhorn tile.

    The large frozen projection and RMS-normalization input VJPs use the
    generic linear/RMS kernels.  This leaf owns the nonstandard part that
    cannot be replaced by a GEMM: the exact native 4-stream gate transform.
    """
    if rows < 1 or copies != 4 or iterations < 1 or eps <= 0:
        raise ValueError("expected positive rows, hc_mult=4 and Sinkhorn iterations")
    import pypto.language as pl

    tile, channels = 8, (copies + 2) * copies
    stages = 2 * iterations
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def hc_pre_gate_vjp(
        projection: pl.Tensor,
        scale: pl.Tensor,
        base: pl.Tensor,
        pre_gradient: pl.Tensor,
        post_gradient: pl.Tensor,
        combination_gradient: pl.Tensor,
        projection_gradient: pl.Out[pl.Tensor],
    ):
        states = pl.create_tensor([rows * stages * tile, tile], dtype=pl.FP32)
        with pl.spmd(rows, name_hint="orthrus_dsv41_hc_pre_sinkhorn_backward"):
            token = pl.tile.get_block_idx()
            projected = pl.load(projection, [token, 0], [1, channels])
            projected_gradient = pl.create_tile([1, channels], dtype=pl.FP32)
            # PyPTO traces assignments, but deliberately does not lower a
            # Python generator expression inside a device program.
            scale0 = pl.read(scale, [0])
            scale1 = pl.read(scale, [1])
            scale2 = pl.read(scale, [2])

            # Sigmoid gate VJPs occupy channels [0:8].  Keep all eight in one
            # aligned FP32 vector: a 1x1 FP32 Vec tile is not representable on
            # Ascend because its contiguous extent is smaller than 32 bytes.
            gate_scales = pl.create_tile([1, 8], dtype=pl.FP32)
            gate_upstream = pl.create_tile([1, 8], dtype=pl.FP32)
            for channel in pl.unroll(copies):
                pl.tile.write(gate_scales, [0, channel], scale0)
                pl.tile.write(gate_scales, [0, copies + channel], scale1)
                pl.tile.write(
                    gate_upstream, [0, channel], pl.read(pre_gradient, [token, channel]),
                )
                pl.tile.write(
                    gate_upstream, [0, copies + channel],
                    pl.read(post_gradient, [token, channel]) * 2.,
                )
            gate_logits = pl.add(
                pl.mul(pl.slice(projected, [1, 8], [0, 0]), gate_scales),
                pl.reshape(pl.load(base, [0], [8]), [1, 8]),
            )
            gate_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_logits)), 1.))
            gate_gradient = pl.mul(
                pl.mul(pl.mul(gate_upstream, gate_scales), gate_sigmoid),
                pl.neg(pl.sub(gate_sigmoid, 1.)),
            )
            for channel in pl.unroll(2 * copies):
                pl.tile.write(
                    projected_gradient, [0, channel],
                    pl.tile.read(gate_gradient, [0, channel]),
                )

            # Native forward packs two independent 4x4 matrices into the
            # diagonal quadrants of one 8x8 vector tile.  Replaying one token
            # therefore duplicates its logits into the two diagonal blocks,
            # masks both off-diagonal blocks, and seeds only the first block's
            # output gradient.  This preserves the native reduction order
            # without introducing cross-token Sinkhorn coupling.
            logits = pl.create_tile([tile, tile], dtype=pl.FP32)
            diagonal = pl.tile.full([tile, tile], dtype=pl.FP32, value=0.)
            for row in pl.unroll(tile):
                for column in pl.unroll(tile):
                    logical = 2 * copies + (row % copies) * copies + column % copies
                    value = (pl.tile.read(projected, [0, logical]) * scale2
                             + pl.read(base, [logical]))
                    pl.tile.write(logits, [row, column], value)
            for block in pl.unroll(2):
                for row in pl.unroll(copies):
                    for column in pl.unroll(copies):
                        pl.tile.write(
                            diagonal,
                            [block * copies + row, block * copies + column],
                            1.,
                        )
            scratch = pl.create_tile([tile, tile], dtype=pl.FP32)
            maximum = pl.row_max(logits, scratch)
            exponential = pl.mul(
                pl.exp(pl.row_expand_sub(logits, maximum)), diagonal,
            )
            probability = pl.mul(
                pl.add(
                    pl.row_expand_div(exponential, pl.row_sum(exponential, scratch)), eps,
                ),
                diagonal,
            )
            offset = token * stages * tile
            pl.store(probability, [offset, 0], states)
            initial_column_denominator = pl.add(pl.col_sum(probability), eps)
            probability = pl.col_expand_div(probability, initial_column_denominator)
            pl.store(probability, [offset + tile, 0], states)
            stage = 2
            for _ in pl.unroll(iterations - 1):
                row_denominator = pl.add(pl.row_sum(probability, scratch), eps)
                probability = pl.row_expand_div(probability, row_denominator)
                pl.store(probability, [offset + stage * tile, 0], states)
                stage = stage + 1
                column_denominator = pl.add(pl.col_sum(probability), eps)
                probability = pl.col_expand_div(probability, column_denominator)
                pl.store(probability, [offset + stage * tile, 0], states)
                stage = stage + 1

            gradient = pl.create_tile([tile, tile], dtype=pl.FP32)
            for row in pl.unroll(tile):
                for column in pl.unroll(tile):
                    pl.tile.write(gradient, [row, column], 0.)
            for row in pl.unroll(copies):
                for column in pl.unroll(copies):
                    pl.tile.write(
                        gradient, [row, column],
                        pl.read(combination_gradient, [token * copies + row, column]),
                    )

            # Reverse column/row normalizations in exact opposite order.
            stage = stages - 1
            for _ in pl.unroll(iterations - 1):
                before = pl.load(states, [offset + (stage - 1) * tile, 0], [tile, tile])
                after = pl.load(states, [offset + stage * tile, 0], [tile, tile])
                reverse_column_denominator = pl.add(pl.col_sum(before), eps)
                reverse_column_correction = pl.col_sum(pl.mul(gradient, after))
                gradient = pl.col_expand_div(
                    pl.col_expand_sub(gradient, reverse_column_correction),
                    reverse_column_denominator,
                )
                stage = stage - 1
                before = pl.load(states, [offset + (stage - 1) * tile, 0], [tile, tile])
                after = pl.load(states, [offset + stage * tile, 0], [tile, tile])
                reverse_row_denominator = pl.add(pl.row_sum(before, scratch), eps)
                reverse_row_correction = pl.row_sum(pl.mul(gradient, after), scratch)
                gradient = pl.row_expand_div(
                    pl.row_expand_sub(gradient, reverse_row_correction),
                    reverse_row_denominator,
                )
                stage = stage - 1
            before = pl.load(states, [offset, 0], [tile, tile])
            after = pl.load(states, [offset + tile, 0], [tile, tile])
            first_column_denominator = pl.add(pl.col_sum(before), eps)
            first_column_correction = pl.col_sum(pl.mul(gradient, after))
            gradient = pl.col_expand_div(
                pl.col_expand_sub(gradient, first_column_correction), first_column_denominator,
            )

            softmax = pl.mul(pl.sub(before, eps), diagonal)
            softmax_correction = pl.row_sum(pl.mul(gradient, softmax), scratch)
            gradient = pl.mul(softmax, pl.row_expand_sub(gradient, softmax_correction))
            for row in pl.unroll(copies):
                for column in pl.unroll(copies):
                    total = 0.
                    for row_copy in pl.unroll(2):
                        for column_copy in pl.unroll(2):
                            total = total + pl.tile.read(
                                gradient, [row + row_copy * copies, column + column_copy * copies],
                            )
                    logical = 2 * copies + row * copies + column
                    pl.tile.write(projected_gradient, [0, logical], total * scale2)
            pl.store(projected_gradient, [token, 0], projection_gradient)
        return projection_gradient

    @pl.jit.host
    def entry(
        projection: pl.Tensor,
        scale: pl.Tensor,
        base: pl.Tensor,
        pre_gradient: pl.Tensor,
        post_gradient: pl.Tensor,
        combination_gradient: pl.Tensor,
        projection_gradient: pl.Out[pl.Tensor],
    ):
        hc_pre_gate_vjp(
            projection, scale, base, pre_gradient, post_gradient,
            combination_gradient, projection_gradient, device=0,
        )

    return hc_pre_gate_vjp if inline else with_resident_entry(entry, hc_pre_gate_vjp)
