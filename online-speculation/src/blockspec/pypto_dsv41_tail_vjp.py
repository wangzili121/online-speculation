"""PyPTO leaves for the frozen DSV4.1 suffix reverse pass."""


def make_hc_post_data_vjp(rows=32, hidden=5120, copies=4, *, inline=False):
    """Backpropagate through mHC post mixing to value and residual streams.

    Gate gradients are intentionally a separate leaf because they feed the
    small mHC-pre reverse, whereas these two large tensors dominate bandwidth.
    Keeping this leaf fused avoids materializing four copies of ``value``.
    """
    if rows < 1 or hidden < 64 or hidden % 64 or copies != 4:
        raise ValueError("expected DSV4.1 rows, 64-aligned hidden and hc_mult=4")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    tile = 64
    tasks = rows * (hidden // tile)

    @kernel
    def hc_post_data_vjp(
        output_gradient: pl.Tensor,
        post: pl.Tensor,
        combination: pl.Tensor,
        value_gradient: pl.Out[pl.Tensor],
        residual_gradient: pl.Out[pl.Tensor],
    ):
        with pl.spmd(tasks, name_hint="orthrus_dsv41_hc_post_data_backward"):
            task = pl.tile.get_block_idx()
            token = task // (hidden // tile)
            column = task % (hidden // tile) * tile
            value_total = pl.full([1, tile], dtype=pl.FP32, value=0.)
            for target in pl.range(copies):
                dy = pl.cast(
                    pl.slice(output_gradient, [1, tile], [token * copies + target, column]),
                    pl.FP32,
                )
                value_total = pl.add(value_total, pl.mul(dy, pl.read(post, [token, target])))
            value_gradient[token:token + 1, column:column + tile] = pl.cast(
                value_total, pl.BF16, mode="rint",
            )
            for source in pl.range(copies):
                residual_total = pl.full([1, tile], dtype=pl.FP32, value=0.)
                for target in pl.range(copies):
                    dy = pl.cast(
                        pl.slice(output_gradient, [1, tile], [token * copies + target, column]),
                        pl.FP32,
                    )
                    residual_total = pl.add(
                        residual_total,
                        pl.mul(dy, pl.read(combination, [token * copies + source, target])),
                    )
                residual_gradient[
                    token * copies + source:token * copies + source + 1,
                    column:column + tile,
                ] = pl.cast(residual_total, pl.BF16, mode="rint")
        return value_gradient, residual_gradient

    @pl.jit.host
    def entry(
        output_gradient: pl.Tensor,
        post: pl.Tensor,
        combination: pl.Tensor,
        value_gradient: pl.Out[pl.Tensor],
        residual_gradient: pl.Out[pl.Tensor],
    ):
        hc_post_data_vjp(
            output_gradient, post, combination, value_gradient, residual_gradient, device=0,
        )

    return hc_post_data_vjp if inline else entry


def make_hc_post_gate_vjp(rows=32, hidden=5120, copies=4, *, inline=False):
    """Compute d(post) and d(combination) without retaining a Torch tape."""
    if rows < 1 or hidden < 64 or hidden % 64 or copies != 4:
        raise ValueError("expected DSV4.1 rows, 64-aligned hidden and hc_mult=4")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    tile = 64

    @kernel
    def hc_post_gate_vjp(
        value: pl.Tensor,
        residual: pl.Tensor,
        output_gradient: pl.Tensor,
        post_gradient: pl.Out[pl.Tensor],
        combination_gradient: pl.Out[pl.Tensor],
    ):
        with pl.spmd(rows, name_hint="orthrus_dsv41_hc_post_gate_backward"):
            token = pl.tile.get_block_idx()
            for target in pl.range(copies):
                post_total = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for part in pl.range(hidden // tile):
                    x = pl.reshape(
                        pl.cast(pl.slice(value, [1, tile], [token, part * tile]), pl.FP32),
                        [8, 8],
                    )
                    dy = pl.reshape(
                        pl.cast(
                            pl.slice(output_gradient, [1, tile],
                                     [token * copies + target, part * tile]),
                            pl.FP32,
                        ),
                        [8, 8],
                    )
                    partial = pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.),
                        pl.row_sum(pl.mul(x, dy)),
                    )
                    post_total = pl.add(
                        post_total,
                        pl.row_expand_mul(
                            pl.full([8, 8], dtype=pl.FP32, value=1.),
                            pl.row_sum(pl.transpose(partial, 0, 1)),
                        ),
                    )
                pl.write(post_gradient, [token, target], pl.read(post_total, [0, 0]))
                for source in pl.range(copies):
                    combination_total = pl.full([8, 8], dtype=pl.FP32, value=0.)
                    for part in pl.range(hidden // tile):
                        x = pl.reshape(
                            pl.cast(
                                pl.slice(residual, [1, tile],
                                         [token * copies + source, part * tile]),
                                pl.FP32,
                            ),
                            [8, 8],
                        )
                        dy = pl.reshape(
                            pl.cast(
                                pl.slice(output_gradient, [1, tile],
                                         [token * copies + target, part * tile]),
                                pl.FP32,
                            ),
                            [8, 8],
                        )
                        partial = pl.row_expand_mul(
                            pl.full([8, 8], dtype=pl.FP32, value=1.),
                            pl.row_sum(pl.mul(x, dy)),
                        )
                        combination_total = pl.add(
                            combination_total,
                            pl.row_expand_mul(
                                pl.full([8, 8], dtype=pl.FP32, value=1.),
                                pl.row_sum(pl.transpose(partial, 0, 1)),
                            ),
                        )
                    pl.write(
                        combination_gradient,
                        [token * copies + source, target],
                        pl.read(combination_total, [0, 0]),
                    )
        return post_gradient, combination_gradient

    @pl.jit.host
    def entry(
        value: pl.Tensor,
        residual: pl.Tensor,
        output_gradient: pl.Tensor,
        post_gradient: pl.Out[pl.Tensor],
        combination_gradient: pl.Out[pl.Tensor],
    ):
        hc_post_gate_vjp(
            value, residual, output_gradient, post_gradient, combination_gradient, device=0,
        )

    return hc_post_gate_vjp if inline else entry


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
            scale0, scale1, scale2 = (pl.read(scale, [i]) for i in range(3))

            # Sigmoid gate VJPs occupy channels [0:8].
            for channel in pl.unroll(copies):
                pre_logit = pl.tile.read(projected, [0, channel]) * scale0 + pl.read(base, [channel])
                pre_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(pre_logit)), 1.))
                pre_grad = (pl.read(pre_gradient, [token, channel]) * scale0
                            * pre_sigmoid * (1. - pre_sigmoid))
                pl.write(projection_gradient, [token, channel], pre_grad)
                post_logit = (pl.tile.read(projected, [0, copies + channel]) * scale1
                              + pl.read(base, [copies + channel]))
                post_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(post_logit)), 1.))
                post_grad = (pl.read(post_gradient, [token, channel]) * scale1
                             * 2. * post_sigmoid * (1. - post_sigmoid))
                pl.write(projection_gradient, [token, copies + channel], post_grad)

            # Native forward duplicates the logical 4x4 matrix into 8x8 so
            # Vec row/column reductions remain hardware aligned.
            logits = pl.full([tile, tile], dtype=pl.FP32, value=0.)
            for row in pl.unroll(tile):
                for column in pl.unroll(tile):
                    logical = 2 * copies + (row % copies) * copies + column % copies
                    value = (pl.tile.read(projected, [0, logical]) * scale2
                             + pl.read(base, [logical]))
                    pl.tile.write(logits, [row, column], value)
            scratch = pl.create_tile([tile, tile], dtype=pl.FP32)
            maximum = pl.row_max(logits, scratch)
            exponential = pl.exp(pl.row_expand_sub(logits, maximum))
            probability = pl.add(
                pl.row_expand_div(exponential, pl.mul(pl.row_sum(exponential, scratch), .5)), eps,
            )
            offset = token * stages * tile
            pl.store(probability, [offset, 0], states)
            denominator = pl.add(pl.mul(pl.col_sum(probability), .5), eps)
            probability = pl.col_expand_div(probability, denominator)
            pl.store(probability, [offset + tile, 0], states)
            stage = 2
            for _ in range(iterations - 1):
                denominator = pl.add(pl.mul(pl.row_sum(probability, scratch), .5), eps)
                probability = pl.row_expand_div(probability, denominator)
                pl.store(probability, [offset + stage * tile, 0], states)
                stage += 1
                denominator = pl.add(pl.mul(pl.col_sum(probability), .5), eps)
                probability = pl.col_expand_div(probability, denominator)
                pl.store(probability, [offset + stage * tile, 0], states)
                stage += 1

            gradient = pl.full([tile, tile], dtype=pl.FP32, value=0.)
            for row in pl.unroll(copies):
                for column in pl.unroll(copies):
                    pl.tile.write(
                        gradient, [row, column],
                        pl.read(combination_gradient, [token * copies + row, column]),
                    )

            # Reverse column/row normalizations in exact opposite order.
            stage = stages - 1
            for _ in range(iterations - 1):
                before = pl.load(states, [offset + (stage - 1) * tile, 0], [tile, tile])
                after = pl.load(states, [offset + stage * tile, 0], [tile, tile])
                denominator = pl.add(pl.mul(pl.col_sum(before), .5), eps)
                correction = pl.mul(pl.col_sum(pl.mul(gradient, after)), .5)
                gradient = pl.col_expand_div(pl.col_expand_sub(gradient, correction), denominator)
                stage -= 1
                before = pl.load(states, [offset + (stage - 1) * tile, 0], [tile, tile])
                after = pl.load(states, [offset + stage * tile, 0], [tile, tile])
                denominator = pl.add(pl.mul(pl.row_sum(before, scratch), .5), eps)
                correction = pl.mul(pl.row_sum(pl.mul(gradient, after), scratch), .5)
                gradient = pl.row_expand_div(pl.row_expand_sub(gradient, correction), denominator)
                stage -= 1
            before = pl.load(states, [offset, 0], [tile, tile])
            after = pl.load(states, [offset + tile, 0], [tile, tile])
            denominator = pl.add(pl.mul(pl.col_sum(before), .5), eps)
            correction = pl.mul(pl.col_sum(pl.mul(gradient, after)), .5)
            gradient = pl.col_expand_div(pl.col_expand_sub(gradient, correction), denominator)

            softmax = pl.sub(before, eps)
            correction = pl.row_sum(pl.mul(gradient, softmax), scratch)
            gradient = pl.mul(softmax, pl.row_expand_sub(gradient, correction))
            for row in pl.unroll(copies):
                for column in pl.unroll(copies):
                    total = 0.
                    for row_copy in pl.unroll(2):
                        for column_copy in pl.unroll(2):
                            total += pl.tile.read(
                                gradient, [row + row_copy * copies, column + column_copy * copies],
                            )
                    logical = 2 * copies + row * copies + column
                    pl.write(projection_gradient, [token, logical], total * scale2)
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

    return hc_pre_gate_vjp if inline else entry
