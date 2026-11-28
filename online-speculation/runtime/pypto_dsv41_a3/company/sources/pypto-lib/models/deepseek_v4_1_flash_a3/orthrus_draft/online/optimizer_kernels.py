# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compute finite gradient statistics and reduction for the resident optimizer."""

from .kernel_abi import with_resident_entry


def make_stats(shape, *, input_bf16=False, inline=False):
    from math import prod
    import pypto.language as pl
    N = prod(shape)
    T = min(4096, N)
    if N % T or T % 8:
        raise ValueError("aligned gradient tensor required")
    from .loss_kernels import make_fp32_row
    load = make_fp32_row(T, input_fp32=not input_bf16)

    kernel = pl.jit.inline if inline else pl.jit
    @kernel
    def stats(value: pl.Tensor, sums: pl.Out[pl.Tensor], flags: pl.Out[pl.Tensor]):
        x = pl.reshape(value, [N // T, T])
        with pl.spmd((N // T + 15) // 16, name_hint="orthrus_gradient_statistics"):
            base = pl.tile.get_block_idx() * 16
            for part in pl.range(16):
                row = base + part
                if row < N // T:
                    values = load(x, row)
                    if T < 64:
                        # Keep a 32-byte row for short FP32 parameters such as the sink.
                        short_block = pl.reshape(values, [T // 8, 8])
                        sums[row:row + 1, 0:8] = pl.col_sum(pl.mul(short_block, short_block))
                        short_finite = pl.cmp(pl.abs(short_block), 3.4028234663852886e38, cmp_type=3)
                        flags[row:row + 1, 0:8] = pl.cmp(pl.col_sum(short_finite), T // 8, cmp_type=0)
                    else:
                        block = pl.reshape(values, [8, T // 8])
                        sums[row:row + 1, 0:8] = pl.reshape(pl.row_sum(pl.mul(block, block)), [1, 8])
                        finite = pl.cmp(pl.abs(block), 3.4028234663852886e38, cmp_type=3)
                        flags[row:row + 1, 0:8] = pl.reshape(pl.row_min(finite), [1, 8])
        return sums, flags

    @pl.jit.host
    def entry(value: pl.Tensor, sums: pl.Out[pl.Tensor], flags: pl.Out[pl.Tensor]):
        stats(value, sums, flags, device=0)
    return (stats if inline else with_resident_entry(entry, stats)), N // T


def make_reduce(rows, *, inline=False):
    import pypto.language as pl
    R = (rows + 63) // 64

    kernel = pl.jit.inline if inline else pl.jit
    @kernel
    def reduce(sums: pl.Tensor, flags: pl.Tensor, out_sums: pl.Out[pl.Tensor], out_flags: pl.Out[pl.Tensor]):
        padded = pl.create_tensor([R * 64, 8], dtype=pl.FP32)
        good = pl.create_tensor([R * 64, 8], dtype=pl.FP32)
        with pl.spmd(R, name_hint="orthrus_gradient_pad"):
            row = pl.tile.get_block_idx() * 64
            padded[row:row + 64, 0:8] = pl.full([64, 8], dtype=pl.FP32, value=0.)
            good[row:row + 64, 0:8] = pl.full([64, 8], dtype=pl.FP32, value=1.)
        with pl.spmd(R, name_hint="orthrus_gradient_actual"):
            base = pl.tile.get_block_idx() * 64
            for part in pl.range(64):
                row = base + part
                if row < rows:
                    padded[row:row + 1, 0:8] = pl.slice(sums, [1, 8], [row, 0])
                    good[row:row + 1, 0:8] = pl.slice(flags, [1, 8], [row, 0])
        with pl.spmd(R, name_hint="orthrus_gradient_reduce"):
            row = pl.tile.get_block_idx()
            out_sums[row:row + 1, 0:8] = pl.col_sum(pl.slice(padded, [64, 8], [row * 64, 0]))
            out_flags[row:row + 1, 0:8] = pl.reshape(pl.row_min(pl.transpose(pl.slice(good, [64, 8], [row * 64, 0]), 0, 1)), [1, 8])
        return out_sums, out_flags

    @pl.jit.host
    def entry(sums: pl.Tensor, flags: pl.Tensor, out_sums: pl.Out[pl.Tensor], out_flags: pl.Out[pl.Tensor]):
        reduce(sums, flags, out_sums, out_flags, device=0)
    return (reduce if inline else with_resident_entry(entry, reduce)), R
