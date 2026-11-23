"""PyPTO leaves for the frozen DSV4.1 suffix reverse pass."""


def make_hc_collapse_forward(rows=32, hidden=5120, copies=4, *, inline=False):
    """Rebuild the final FP32-accumulated mHC collapse before RMSNorm."""
    if rows < 1 or hidden < 64 or hidden % 64 or copies != 4:
        raise ValueError("expected DSV4.1 rows, 64-aligned hidden and hc_mult=4")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    tile = 64

    @kernel
    def hc_collapse_forward(
        streams: pl.Tensor,
        premix: pl.Tensor,
        collapsed: pl.Out[pl.Tensor],
    ):
        with pl.spmd(rows, name_hint="orthrus_dsv41_final_hc_collapse_forward"):
            token = pl.tile.get_block_idx()
            for part in pl.range(hidden // tile):
                column = part * tile
                total = pl.full([1, tile], dtype=pl.FP32, value=0.)
                for stream in pl.range(copies):
                    value = pl.cast(
                        pl.slice(
                            streams, [1, tile],
                            [token * copies + stream, column],
                        ),
                        pl.FP32,
                    )
                    total = pl.add(
                        total,
                        pl.mul(value, pl.read(premix, [token, stream])),
                    )
                collapsed[token:token + 1, column:column + tile] = pl.cast(
                    total, pl.BF16, mode="rint",
                )
        return collapsed

    @pl.jit.host
    def entry(
        streams: pl.Tensor,
        premix: pl.Tensor,
        collapsed: pl.Out[pl.Tensor],
    ):
        hc_collapse_forward(streams, premix, collapsed, device=0)

    return hc_collapse_forward if inline else entry


def make_hc_collapse_vjp(rows=32, hidden=5120, copies=4, *, inline=False):
    """Reverse the final four-stream mHC collapse used before the LM head.

    The final model head computes ``sum_i streams[i] * premix[i]`` in FP32
    and rounds the result to BF16.  Online learning keeps both inputs frozen,
    but must propagate gradients through them to the selected draft suffix.
    ``premix_gradient`` is padded to sixteen FP32 columns so every token owns
    a complete cache line and no two device blocks race on scalar stores.
    """
    if rows < 1 or hidden < 64 or hidden % 64 or copies != 4:
        raise ValueError("expected DSV4.1 rows, 64-aligned hidden and hc_mult=4")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    tile = 64

    @kernel
    def hc_collapse_vjp(
        streams: pl.Tensor,
        premix: pl.Tensor,
        output_gradient: pl.Tensor,
        stream_gradient: pl.Out[pl.Tensor],
        premix_gradient: pl.Out[pl.Tensor],
    ):
        with pl.spmd(rows, name_hint="orthrus_dsv41_final_hc_collapse_backward"):
            token = pl.tile.get_block_idx()
            for stream in pl.range(copies):
                coefficient = pl.read(premix, [token, stream])
                total = pl.full([8, 8], dtype=pl.FP32, value=0.)
                for part in pl.range(hidden // tile):
                    column = part * tile
                    dy = pl.cast(
                        pl.slice(output_gradient, [1, tile], [token, column]),
                        pl.FP32,
                    )
                    value = pl.cast(
                        pl.slice(
                            streams, [1, tile],
                            [token * copies + stream, column],
                        ),
                        pl.FP32,
                    )
                    stream_gradient[
                        token * copies + stream:token * copies + stream + 1,
                        column:column + tile,
                    ] = pl.cast(pl.mul(dy, coefficient), pl.BF16, mode="rint")
                    product = pl.reshape(pl.mul(dy, value), [8, 8])
                    partial = pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.),
                        pl.row_sum(product),
                    )
                    total = pl.add(
                        total,
                        pl.row_expand_mul(
                            pl.full([8, 8], dtype=pl.FP32, value=1.),
                            pl.row_sum(pl.transpose(partial, 0, 1)),
                        ),
                    )
                pl.write(
                    premix_gradient, [token, stream], pl.read(total, [0, 0]),
                )
        return stream_gradient, premix_gradient

    @pl.jit.host
    def entry(
        streams: pl.Tensor,
        premix: pl.Tensor,
        output_gradient: pl.Tensor,
        stream_gradient: pl.Out[pl.Tensor],
        premix_gradient: pl.Out[pl.Tensor],
    ):
        hc_collapse_vjp(
            streams, premix, output_gradient,
            stream_gradient, premix_gradient, device=0,
        )

    return hc_collapse_vjp if inline else entry


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

    return hc_pre_gate_vjp if inline else entry


def make_hc_pre_projection_forward(
    rows=32, hidden=5120, copies=4, *, norm_eps=1e-20, inline=False,
):
    """Recompute the exact frozen mHC projection and saved inverse RMS.

    Native DSV4.1 projects the *FP32-normalized* concatenated four-stream
    residual directly to 24 FP32 gate channels.  A generic RMSNorm followed
    by a BF16 GEMM would insert a rounding boundary that is absent from the
    target forward, so online reverse uses this small dedicated replay leaf.
    """
    import math
    if (
        type(rows) is not int or rows < 1 or type(hidden) is not int
        or hidden < 64 or hidden % 64 or copies != 4
        or not isinstance(norm_eps, (int, float))
        or not math.isfinite(norm_eps) or norm_eps <= 0
    ):
        raise ValueError("expected aligned DSV4.1 mHC projection geometry")
    import pypto.language as pl

    flat, channels, tile = copies * hidden, (copies + 2) * copies, 64
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def projection_forward(
        residual: pl.Tensor,
        function: pl.Tensor,
        projection: pl.Out[pl.Tensor],
        inverse: pl.Out[pl.Tensor],
    ):
        flattened = pl.reshape(residual, [rows, flat])
        with pl.spmd(rows, name_hint="orthrus_dsv41_hc_pre_inverse"):
            token = pl.tile.get_block_idx()
            total = pl.full([8, 8], dtype=pl.FP32, value=0.)
            for part in pl.range(flat // tile):
                value = pl.reshape(
                    pl.cast(
                        pl.slice(flattened, [1, tile], [token, part * tile]),
                        pl.FP32,
                    ),
                    [8, 8],
                )
                square = pl.mul(value, value)
                partial = pl.row_expand_mul(
                    pl.full([8, 8], dtype=pl.FP32, value=1.),
                    pl.row_sum(square),
                )
                total = pl.add(
                    total,
                    pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.),
                        pl.row_sum(pl.transpose(partial, 0, 1)),
                    ),
                )
            inverse_tile = pl.rsqrt(
                pl.add(pl.mul(total, 1. / flat), norm_eps),
                high_precision=True,
            )
            inv = pl.tile.read(inverse_tile, [0, 0])
            for lane in pl.unroll(8):
                pl.write(inverse, [token, lane], inv)

        with pl.spmd(rows * channels, name_hint="orthrus_dsv41_hc_pre_projection_replay"):
            task = pl.tile.get_block_idx()
            token, channel = task // channels, task % channels
            total = pl.full([8, 8], dtype=pl.FP32, value=0.)
            for part in pl.range(flat // tile):
                value = pl.reshape(
                    pl.cast(
                        pl.slice(flattened, [1, tile], [token, part * tile]),
                        pl.FP32,
                    ),
                    [8, 8],
                )
                weight = pl.reshape(
                    pl.slice(function, [1, tile], [channel, part * tile]),
                    [8, 8],
                )
                product = pl.mul(value, weight)
                partial = pl.row_expand_mul(
                    pl.full([8, 8], dtype=pl.FP32, value=1.),
                    pl.row_sum(product),
                )
                total = pl.add(
                    total,
                    pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.),
                        pl.row_sum(pl.transpose(partial, 0, 1)),
                    ),
                )
            pl.write(
                projection, [token, channel],
                pl.read(total, [0, 0]) * pl.read(inverse, [token, 0]),
            )
        return projection, inverse

    @pl.jit.host
    def entry(
        residual: pl.Tensor,
        function: pl.Tensor,
        projection: pl.Out[pl.Tensor],
        inverse: pl.Out[pl.Tensor],
    ):
        projection_forward(residual, function, projection, inverse, device=0)

    return projection_forward if inline else entry


def make_hc_pre_data_vjp(
    rows=32, hidden=5120, copies=4, *, inline=False,
):
    """Reverse the frozen mHC projection/RMS and premix collapse.

    ``residual_gradient`` already contains the direct residual branch from
    the matching mHC-post VJP.  This leaf adds the projection and collapse
    contributions in FP32 and narrows once to the native BF16 boundary.
    ``incoming_gradient`` is padded to sixteen columns for race-free stores.
    """
    if (
        type(rows) is not int or rows < 1 or type(hidden) is not int
        or hidden < 64 or hidden % 64 or copies != 4
    ):
        raise ValueError("expected aligned DSV4.1 mHC data-VJP geometry")
    import pypto.language as pl

    flat, channels, tile = copies * hidden, (copies + 2) * copies, 64
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def data_vjp(
        residual: pl.Tensor,
        incoming: pl.Tensor,
        function: pl.Tensor,
        inverse: pl.Tensor,
        projection_gradient: pl.Tensor,
        collapsed_gradient: pl.Tensor,
        residual_passthrough: pl.Tensor,
        residual_gradient: pl.Out[pl.Tensor],
        incoming_gradient: pl.Out[pl.Tensor],
    ):
        flattened = pl.reshape(residual, [rows, flat])
        passthrough = pl.reshape(residual_passthrough, [rows, flat])
        output = pl.reshape(residual_gradient, [rows, flat])
        with pl.spmd(rows, name_hint="orthrus_dsv41_hc_pre_data_backward"):
            token = pl.tile.get_block_idx()
            inv = pl.read(inverse, [token, 0])
            dot = pl.full([8, 8], dtype=pl.FP32, value=0.)
            # First pass computes <d(normalized), residual>, required by the
            # exact RMS input VJP over all four streams.
            for part in pl.range(flat // tile):
                begin = part * tile
                normalized_gradient = pl.full([1, tile], dtype=pl.FP32, value=0.)
                for channel in pl.range(channels):
                    normalized_gradient = pl.add(
                        normalized_gradient,
                        pl.mul(
                            pl.slice(function, [1, tile], [channel, begin]),
                            pl.read(projection_gradient, [token, channel]),
                        ),
                    )
                value = pl.cast(
                    pl.slice(flattened, [1, tile], [token, begin]), pl.FP32,
                )
                product = pl.reshape(pl.mul(normalized_gradient, value), [8, 8])
                partial = pl.row_expand_mul(
                    pl.full([8, 8], dtype=pl.FP32, value=1.),
                    pl.row_sum(product),
                )
                dot = pl.add(
                    dot,
                    pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.),
                        pl.row_sum(pl.transpose(partial, 0, 1)),
                    ),
                )
            correction = pl.read(dot, [0, 0]) * inv * inv / flat

            for stream in pl.range(copies):
                incoming_total = pl.full([8, 8], dtype=pl.FP32, value=0.)
                coefficient = pl.read(incoming, [token, stream])
                for part in pl.range(hidden // tile):
                    column = part * tile
                    begin = stream * hidden + column
                    normalized_gradient = pl.full([1, tile], dtype=pl.FP32, value=0.)
                    for channel in pl.range(channels):
                        normalized_gradient = pl.add(
                            normalized_gradient,
                            pl.mul(
                                pl.slice(function, [1, tile], [channel, begin]),
                                pl.read(projection_gradient, [token, channel]),
                            ),
                        )
                    value = pl.cast(
                        pl.slice(flattened, [1, tile], [token, begin]), pl.FP32,
                    )
                    collapsed = pl.cast(
                        pl.slice(collapsed_gradient, [1, tile], [token, column]),
                        pl.FP32,
                    )
                    direct = pl.mul(collapsed, coefficient)
                    rms = pl.mul(
                        pl.sub(normalized_gradient, pl.mul(value, correction)), inv,
                    )
                    prior = pl.cast(
                        pl.slice(passthrough, [1, tile], [token, begin]), pl.FP32,
                    )
                    output[token:token + 1, begin:begin + tile] = pl.cast(
                        pl.add(pl.add(rms, direct), prior), pl.BF16, mode="rint",
                    )
                    product = pl.reshape(pl.mul(value, collapsed), [8, 8])
                    partial = pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.),
                        pl.row_sum(product),
                    )
                    incoming_total = pl.add(
                        incoming_total,
                        pl.row_expand_mul(
                            pl.full([8, 8], dtype=pl.FP32, value=1.),
                            pl.row_sum(pl.transpose(partial, 0, 1)),
                        ),
                    )
                pl.write(
                    incoming_gradient, [token, stream],
                    pl.read(incoming_total, [0, 0]),
                )
        return residual_gradient, incoming_gradient

    @pl.jit.host
    def entry(
        residual: pl.Tensor,
        incoming: pl.Tensor,
        function: pl.Tensor,
        inverse: pl.Tensor,
        projection_gradient: pl.Tensor,
        collapsed_gradient: pl.Tensor,
        residual_passthrough: pl.Tensor,
        residual_gradient: pl.Out[pl.Tensor],
        incoming_gradient: pl.Out[pl.Tensor],
    ):
        data_vjp(
            residual, incoming, function, inverse, projection_gradient,
            collapsed_gradient, residual_passthrough, residual_gradient,
            incoming_gradient, device=0,
        )

    return data_vjp if inline else entry
