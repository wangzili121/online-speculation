# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit loss and linear VJPs for the online draft head."""

from .kernel_abi import with_resident_entry


def loss_vocabulary_tile(vocabulary):
    """Use the widest aligned tile, including V4.1's 129280 vocabulary."""
    if type(vocabulary) is not int or vocabulary < 64 or vocabulary % 64:
        raise ValueError("a positive 64-aligned vocabulary required")
    return max(tile for tile in (64, 128, 256, 512, 1280, 2560)
               if tile <= vocabulary and vocabulary % tile == 0)


def make_vocab_gradient_shard(rows, vocabulary, *, source_rows=None):
    """Copy one contiguous TP lane from every full-vocabulary gradient row."""
    import pypto.language as pl

    width = vocabulary // 4
    source_rows = rows if source_rows is None else source_rows
    tile = next(size for size in (6464, 4096, 2048, 1024, 512, 256, 128, 64)
                if width % size == 0)

    @pl.jit
    def shard(gradient: pl.Tensor[[source_rows, vocabulary], pl.BF16],
              output: pl.Out[pl.Tensor[[rows, width], pl.BF16]],
              offset: pl.Scalar[pl.INT32]):
        for task in pl.spmd(rows * (width // tile), name_hint="orthrus_vocab_gradient_shard"):
            row = task // (width // tile)
            column = task % (width // tile) * tile
            if row < source_rows:
                values = pl.slice(gradient, [1, tile], [row, offset + column])
                output[row:row + 1, column:column + tile] = values
            else:
                output[row:row + 1, column:column + tile] = pl.full([1, tile], dtype=pl.BF16, value=0.)
        return output

    return shard


def _loss_entry(compute, *, inline, position_weighted):
    """Bind the loss and its optional per-row supervision weights."""
    import pypto.language as pl
    if position_weighted:
        if inline:
            return compute
        @pl.jit.host
        def entry(student: pl.Tensor, teacher: pl.Tensor, weights: pl.Tensor,
                  gradient: pl.Out[pl.Tensor], losses: pl.Out[pl.Tensor],
                  valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
            compute(student, teacher, weights, gradient, losses, valid, normalizer, device=0)
        return with_resident_entry(entry, compute)

    @pl.jit.inline
    def ordinary(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
                 losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        return compute(student, teacher, student, gradient, losses, valid, normalizer)
    if inline:
        return ordinary
    @pl.jit.host
    def entry(student: pl.Tensor, teacher: pl.Tensor, gradient: pl.Out[pl.Tensor],
              losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        compute(student, teacher, student, gradient, losses, valid, normalizer, device=0)
    return with_resident_entry(entry, compute)


def _make_loss_tile(width, input_fp32):
    import pypto.language as pl

    @pl.jit.inline
    def load(value: pl.Tensor, row: pl.Scalar[pl.INT32], column: pl.Scalar[pl.INT32]):
        tile = pl.slice(value, [1, width], [row, column])
        if input_fp32:
            result = pl.reshape(tile, [8, width // 8])
        else:
            result = pl.reshape(pl.cast(tile, pl.FP32), [8, width // 8])
        return result

    return load


def make_tv(rows, vocabulary, *, inline=False, focus_weight=1, position_weighted=False,
            log_overlap=False, input_fp32=False):
    """Compute full-vocabulary TV and its logit gradient on the verified rows."""
    if rows < 1 or vocabulary % 64:
        raise ValueError("positive rows and a 64-aligned vocabulary required")
    if type(focus_weight) is not int or focus_weight < 1:
        raise ValueError("TV candidate weights must be positive integers")
    if type(log_overlap) is not bool:
        raise ValueError("log_overlap must be a boolean")
    T = loss_vocabulary_tile(vocabulary)
    import pypto.language as pl

    load = _make_loss_tile(T, input_fp32)
    focus_reciprocal = 1.0 / focus_weight
    kernel = pl.jit.inline if inline else pl.jit
    @kernel
    def tv(student: pl.Tensor, teacher: pl.Tensor, position_weights: pl.Tensor,
           gradient: pl.Out[pl.Tensor],
           losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        with pl.spmd(rows, name_hint="orthrus_tv_backward"):
            row = pl.tile.get_block_idx()
            if row < valid:
                denominator = pl.cast(normalizer, pl.FP32)
                if position_weighted:
                    denominator = denominator / pl.read(position_weights, [row, 0])
                if focus_weight > 1:
                    if row >= 2 and row < 4:
                        denominator = denominator * focus_reciprocal
                sm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                tm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                for tile in pl.range(vocabulary // T):
                    s = load(student, row, tile * T)
                    t = load(teacher, row, tile * T)
                    sr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(s))
                    tr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(t))
                    sx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(tr, 0, 1)))
                    sm = pl.maximum(sm, sx)
                    tm = pl.maximum(tm, tx)
                sz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                tz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = load(student, row, tile * T)
                    t = load(teacher, row, tile * T)
                    se = pl.exp(pl.row_expand_sub(s, pl.row_max(sm)))
                    te = pl.exp(pl.row_expand_sub(t, pl.row_max(tm)))
                    sr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(se))
                    tr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(te))
                    sx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(tr, 0, 1)))
                    sz = pl.add(sz, sx)
                    tz = pl.add(tz, tx)
                dot = pl.full([8, 8], dtype=pl.FP32, value=0.)
                loss = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = load(student, row, tile * T)
                    t = load(teacher, row, tile * T)
                    p = pl.row_expand_div(pl.exp(pl.row_expand_sub(s, pl.row_max(sm))), pl.mul(pl.row_sum(sz), .125))
                    q = pl.row_expand_div(pl.exp(pl.row_expand_sub(t, pl.row_max(tm))), pl.mul(pl.row_sum(tz), .125))
                    diff = pl.sub(p, q)
                    sign = pl.mul(pl.sub(pl.cmp(diff, 0., cmp_type=4), pl.cmp(diff, 0., cmp_type=2)), .5)
                    if log_overlap:
                        # Split the gradient evenly at a tie, matching torch.minimum.
                        sign = pl.sub(sign, .5)
                    weighted = pl.mul(p, sign)
                    if log_overlap:
                        absolute = pl.minimum(p, q)
                    else:
                        absolute = pl.mul(pl.abs(diff), .5)
                    sr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(weighted))
                    tr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(absolute))
                    sx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(tr, 0, 1)))
                    dot = pl.add(dot, sx)
                    loss = pl.add(loss, tx)
                inverse_overlap = pl.full([8, 8], dtype=pl.FP32, value=1.)
                if log_overlap:
                    overlap = pl.maximum(loss, 1e-8)
                    loss_value = pl.mul(pl.log(overlap), -1.)
                    inverse_overlap = pl.div(pl.cmp(loss, 1e-8, cmp_type=5), overlap)
                else:
                    loss_value = loss
                losses[row:row + 1, 0:8] = pl.div(pl.slice(loss_value, [1, 8], [0, 0]), denominator)
                for tile in pl.range(vocabulary // T):
                    s = load(student, row, tile * T)
                    t = load(teacher, row, tile * T)
                    p = pl.row_expand_div(pl.exp(pl.row_expand_sub(s, pl.row_max(sm))), pl.mul(pl.row_sum(sz), .125))
                    q = pl.row_expand_div(pl.exp(pl.row_expand_sub(t, pl.row_max(tm))), pl.mul(pl.row_sum(tz), .125))
                    diff = pl.sub(p, q)
                    sign = pl.mul(pl.sub(pl.cmp(diff, 0., cmp_type=4), pl.cmp(diff, 0., cmp_type=2)), .5)
                    if log_overlap:
                        sign = pl.sub(sign, .5)
                    ds = pl.div(pl.mul(p, pl.row_expand_sub(sign, pl.mul(pl.row_sum(dot), .125))), denominator)
                    if log_overlap:
                        ds = pl.row_expand_mul(ds, pl.mul(pl.row_sum(inverse_overlap), .125))
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.cast(pl.reshape(ds, [1, T]), pl.BF16, mode="rint")
            else:
                losses[row:row + 1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.full([1, T], dtype=pl.BF16, value=0.)
        return gradient, losses

    return _loss_entry(tv, inline=inline, position_weighted=position_weighted)


def make_kl(rows, vocabulary, *, inline=False, position_weighted=False, input_fp32=False):
    """Compute full-vocabulary KL(teacher || draft) and its logit gradient."""
    if rows < 1 or vocabulary % 64:
        raise ValueError("positive rows and a 64-aligned vocabulary required")
    T = loss_vocabulary_tile(vocabulary)
    import pypto.language as pl

    load = _make_loss_tile(T, input_fp32)
    kernel = pl.jit.inline if inline else pl.jit
    @kernel
    def kl(student: pl.Tensor, teacher: pl.Tensor, position_weights: pl.Tensor,
           gradient: pl.Out[pl.Tensor],
           losses: pl.Out[pl.Tensor], valid: pl.Scalar[pl.INT32], normalizer: pl.Scalar[pl.INT32]):
        with pl.spmd(rows, name_hint="orthrus_kl_backward"):
            row = pl.tile.get_block_idx()
            if row < valid:
                denominator = pl.cast(normalizer, pl.FP32)
                if position_weighted:
                    denominator = denominator / pl.read(position_weights, [row, 0])
                sm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                tm = pl.full([8, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
                for tile in pl.range(vocabulary // T):
                    s = load(student, row, tile * T)
                    t = load(teacher, row, tile * T)
                    sr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(s))
                    tr = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(t))
                    sx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(pl.transpose(tr, 0, 1)))
                    sm = pl.maximum(sm, sx)
                    tm = pl.maximum(tm, tx)
                sz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                tz = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = load(student, row, tile * T)
                    t = load(teacher, row, tile * T)
                    se = pl.exp(pl.row_expand_sub(s, pl.row_max(sm)))
                    te = pl.exp(pl.row_expand_sub(t, pl.row_max(tm)))
                    sr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(se))
                    tr = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(te))
                    sx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(sr, 0, 1)))
                    tx = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(tr, 0, 1)))
                    sz = pl.add(sz, sx)
                    tz = pl.add(tz, tx)
                # Compute log-softmax directly from logits.
                lsz = pl.log(pl.mul(pl.row_sum(sz), .125))
                ltz = pl.log(pl.mul(pl.row_sum(tz), .125))
                loss = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    s = load(student, row, tile * T)
                    t = load(teacher, row, tile * T)
                    log_student = pl.row_expand_sub(pl.row_expand_sub(s, pl.row_max(sm)), lsz)
                    log_teacher = pl.row_expand_sub(pl.row_expand_sub(t, pl.row_max(tm)), ltz)
                    p = pl.exp(log_student)
                    q = pl.exp(log_teacher)
                    ds = pl.div(pl.sub(p, q), denominator)
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.cast(pl.reshape(ds, [1, T]), pl.BF16, mode="rint")
                    term = pl.mul(q, pl.sub(log_teacher, log_student))
                    partial = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(term))
                    total = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.transpose(partial, 0, 1)))
                    loss = pl.add(loss, total)
                losses[row:row + 1, 0:8] = pl.div(pl.slice(loss, [1, 8], [0, 0]), denominator)
            else:
                losses[row:row + 1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.)
                for tile in pl.range(vocabulary // T):
                    gradient[row:row + 1, tile * T:tile * T + T] = pl.full([1, T], dtype=pl.BF16, value=0.)
        return gradient, losses

    return _loss_entry(kl, inline=inline, position_weighted=position_weighted)


def reduction_tile(k, limit):
    """Select an aligned reduction tile that divides K."""
    if type(limit) is not int or limit not in (32, 64, 128, 256, 512):
        raise ValueError("reduction tile must be 32, 64, 128, 256 or 512")
    return max(t for t in (16, 32, 64, 128, 256, 512) if t <= limit and k % t == 0)


def make_linear(a_shape, b_shape, *, transpose_a=False, transpose_b=True, gradient_weight=False,
                batches=1, b_repeats=1, inline=False, accumulate=False, reduction_limit=None, task_group=16,
                output_fp32=False):
    """Compute the rank-local linear VJP with optional FP32 output for TP reduction."""
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    M, K = a_shape[::-1] if transpose_a else a_shape
    kb, N = b_shape[::-1] if transpose_b else b_shape
    if K != kb or M % 16 or N % 64 or K % 16:
        raise ValueError("aligned M/N/K required")
    if batches < 1 or b_repeats < 1 or batches % b_repeats:
        raise ValueError("whole GQA groups required")
    if type(task_group) is not int or not 1 <= task_group <= 16 or not gradient_weight and task_group != 16:
        raise ValueError("gradient task group must be an integer in 1..16")
    MT = 64 if M % 64 == 0 else 32 if M % 32 == 0 else 16
    NT = 64
    KT = reduction_tile(K, 64 if reduction_limit is None else reduction_limit)
    AR, AC = (KT, MT) if transpose_a else (MT, KT)
    BR, BC = (NT, KT) if transpose_b else (KT, NT)
    TA = int(transpose_a)
    TB = int(transpose_b)
    if gradient_weight:
        tasks = batches * (M // MT) * (N // NT)
        @kernel
        def linear(a: pl.Tensor, b: pl.Tensor, output: pl.InOut[pl.Tensor]):
            # Reuse Cube-to-Vector scratch across output tiles.
            with pl.spmd((tasks + task_group - 1) // task_group, name_hint="orthrus_training_linear"):
                base = pl.tile.get_block_idx() * task_group
                for part in pl.range(task_group):
                    task = base + part
                    if task < tasks:
                        head = task // ((M // MT) * (N // NT))
                        local = task % ((M // MT) * (N // NT))
                        row = local // (N // NT) * MT
                        col = local % (N // NT) * NT
                        if K == KT:
                            # Reduce the weight gradient across the physical token rows.
                            x = pl.slice(a, [AR, AC], [head * a_shape[0] + (1 - TA) * row, TA * row])
                            w = pl.slice(b, [BR, BC], [head // b_repeats * b_shape[0] + TB * col, (1 - TB) * col])
                            acc = pl.matmul(x, w, a_trans=transpose_a, b_trans=transpose_b, out_dtype=pl.FP32)
                        else:
                            acc = pl.create_tensor([MT, NT], dtype=pl.FP32)
                            for step in pl.pipeline(K // KT, stage=2):
                                begin = step * KT
                                x = pl.slice(a, [AR, AC], [head * a_shape[0] + TA * begin + (1 - TA) * row,
                                                           TA * row + (1 - TA) * begin])
                                w = pl.slice(b, [BR, BC], [head // b_repeats * b_shape[0] + TB * col + (1 - TB) * begin,
                                                           TB * begin + (1 - TB) * col])
                                acc = pl.matmul_acc(acc, x, w, a_trans=transpose_a, b_trans=transpose_b, init_cond=(step == 0))
                        rounded = pl.cast(acc, pl.BF16, mode="rint")
                        result = pl.cast(rounded, pl.FP32)
                        if accumulate:
                            result = pl.add(result, pl.slice(output, [MT, NT], [head * M + row, col]))
                        output[head * M + row:head * M + row + MT, col:col + NT] = result
            return output

    else:
        @kernel
        def linear(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
            with pl.spmd(batches * (M // MT) * (N // NT), name_hint="orthrus_training_linear"):
                task = pl.tile.get_block_idx()
                head = task // ((M // MT) * (N // NT))
                task = task % ((M // MT) * (N // NT))
                row = task // (N // NT) * MT
                col = task % (N // NT) * NT
                acc = pl.create_tensor([MT, NT], dtype=pl.FP32)
                for step in pl.pipeline(K // KT, stage=2):
                    begin = step * KT
                    x = pl.slice(a, [AR, AC], [head * a_shape[0] + TA * begin + (1 - TA) * row,
                                               TA * row + (1 - TA) * begin])
                    w = pl.slice(b, [BR, BC], [head // b_repeats * b_shape[0] + TB * col + (1 - TB) * begin,
                                               TB * begin + (1 - TB) * col])
                    acc = pl.matmul_acc(acc, x, w, a_trans=transpose_a, b_trans=transpose_b, init_cond=(step == 0))
                if output_fp32:
                    output[head * M + row:head * M + row + MT, col:col + NT] = acc
                else:
                    rounded = pl.cast(acc, pl.BF16, mode="rint")
                    output[head * M + row:head * M + row + MT, col:col + NT] = rounded
            return output

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        linear(a, b, output, device=0)

    if inline:
        return linear
    return with_resident_entry(entry, linear)


def make_fp32_row(tile, *, input_fp32):
    """Select the input conversion when constructing the kernel."""
    import pypto.language as pl
    if input_fp32:
        @pl.jit.inline
        def load(value: pl.Tensor, row: pl.Scalar[pl.INT32]):
            return pl.slice(value, [1, tile], [row, 0])
    else:
        @pl.jit.inline
        def load(value: pl.Tensor, row: pl.Scalar[pl.INT32]):
            return pl.cast(pl.slice(value, [1, tile], [row, 0]), pl.FP32)
    return load
