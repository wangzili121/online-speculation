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
