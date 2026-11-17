"""Derivatives of the GLM mHC residual combination."""


def make_mhc_pre_backward(width=4096, tokens=6, *, device_entry=False):
    """Differentiate mHC's learned collapse of the four input streams."""
    import pypto.language as pl

    tile, streams, padded = 128, 4, 8
    if width < tile or width % tile or tokens < 1:
        raise ValueError("invalid GLM mHC collapse geometry")
    columns = width // tile

    @pl.jit
    def mhc_pre_backward(
        streams_in: pl.Tensor[[tokens * streams, width], pl.BF16],
        weights: pl.Tensor[[tokens, padded], pl.FP32],
        grad: pl.Tensor[[tokens, width], pl.FP32],
        d_streams: pl.Out[pl.Tensor[[tokens * streams, width], pl.FP32]],
        d_weights: pl.Out[pl.Tensor[[tokens, padded], pl.FP32]],
    ):
        for task in pl.spmd(tokens * columns, name_hint="glm_mhc_pre_streams_vjp"):
            row = task // columns
            col = task % columns * tile
            upstream = grad[row:row + 1, col:col + tile]
            for stream in pl.unroll(streams):
                destination = row * streams + stream
                gate = pl.read(weights, [row, stream])
                d_streams[destination:destination + 1, col:col + tile] = pl.mul(upstream, gate)

        for row in pl.spmd(tokens, name_hint="glm_mhc_pre_weights_vjp"):
            accumulated = pl.full([padded, tile], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, width, tile):
                x = pl.create_tensor([padded, tile], dtype=pl.FP32)
                x[:streams, :] = pl.cast(pl.slice(streams_in, [streams, tile], [row * streams, col]), pl.FP32)
                x[streams:, :] = pl.full([padded - streams, tile], dtype=pl.FP32, value=0.0)
                upstream = grad[row:row + 1, col:col + tile]
                accumulated = pl.add(accumulated, pl.col_expand_mul(x, upstream))
            d_weights[row:row + 1, :] = pl.reshape(pl.row_sum(accumulated), [1, padded])
        return d_streams, d_weights

    @pl.jit.host
    def entry(inputs: pl.Tensor, weights: pl.Tensor, grad: pl.Tensor,
              d_streams: pl.Out[pl.Tensor], d_weights: pl.Out[pl.Tensor]):
        mhc_pre_backward(inputs, weights, grad, d_streams, d_weights, device=0)

    return mhc_pre_backward if device_entry else entry


def make_mhc_post_backward(width=4096, tokens=6, *, device_entry=False):
    """Differentiate ``post * sublayer + residual_mix.T @ streams``."""
    import pypto.language as pl

    tile = 128
    if width < tile or width % tile or tokens < 1:
        raise ValueError("invalid GLM residual backward geometry")
    streams, padded = 4, 8
    shards = width // tile

    @pl.jit
    def backward(
        sublayer: pl.Tensor[[tokens, width], pl.BF16],
        residual: pl.Tensor[[tokens * streams, width], pl.BF16],
        post: pl.Tensor[[tokens, padded], pl.FP32],
        mix: pl.Tensor[[tokens * streams, padded], pl.FP32],
        upstream: pl.Tensor[[tokens * streams, width], pl.FP32],
        dy: pl.Out[pl.Tensor[[tokens, width], pl.FP32]],
        dx: pl.Out[pl.Tensor[[tokens * streams, width], pl.FP32]],
        dpost: pl.Out[pl.Tensor[[tokens, padded], pl.FP32]],
        dmix: pl.Out[pl.Tensor[[tokens * streams, padded], pl.FP32]],
    ):
        # Match the BF16 coefficients used by the forward mHC post-mix.
        # The coefficient gradients cross the casts with an identity VJP.
        rounded_post = pl.create_tensor([tokens, padded], dtype=pl.FP32)
        rounded_mix = pl.create_tensor([tokens * streams, padded], dtype=pl.FP32)
        for coefficient in pl.spmd(tokens, name_hint="glm_mhc_post_restore_precision"):
            row = coefficient * streams
            aligned = pl.create_tensor([streams + 1, 2 * padded], dtype=pl.FP32)
            aligned[:streams, :padded] = mix[row:row + streams, :]
            aligned[streams:streams + 1, :padded] = post[coefficient:coefficient + 1, :]
            aligned[:, padded:] = pl.full([streams + 1, padded], dtype=pl.FP32, value=0.0)
            values = pl.cast(pl.cast(aligned, pl.BF16, mode="rint"), pl.FP32)
            rounded_post[coefficient:coefficient + 1, :] = values[streams:streams + 1, :padded]
            rounded_mix[row:row + streams, :] = values[:streams, :padded]

        for task in pl.spmd(tokens * shards, name_hint="orthrus_glm53_mhc_dx"):
            token = task // shards
            begin = task % shards * tile
            sub_grad = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for destination in pl.unroll(streams):
                dest_row = token * streams + destination
                g = upstream[dest_row:dest_row + 1, begin:begin + tile]
                gate = pl.read(rounded_post, [token, destination])
                sub_grad = pl.add(sub_grad, pl.mul(g, gate))
            dy[token:token + 1, begin:begin + tile] = sub_grad
            for source in pl.unroll(streams):
                src_row = token * streams + source
                residual_grad = pl.full([1, tile], dtype=pl.FP32, value=0.0)
                for destination in pl.unroll(streams):
                    dest_row = token * streams + destination
                    g = upstream[dest_row:dest_row + 1, begin:begin + tile]
                    gate = pl.read(rounded_mix, [src_row, destination])
                    residual_grad = pl.add(residual_grad, pl.mul(g, gate))
                dx[src_row:src_row + 1, begin:begin + tile] = residual_grad

        for coefficient in pl.spmd(tokens * (streams + 1), name_hint="orthrus_glm53_mhc_dmix"):
            mix_token = coefficient // (streams + 1)
            mix_source = coefficient % (streams + 1)
            combined_dot = pl.full([padded, tile], dtype=pl.FP32, value=0.0)
            for segment in pl.range(0, width, tile):
                x = pl.full([1, tile], dtype=pl.FP32, value=0.0)
                if mix_source == streams:
                    x = pl.cast(sublayer[mix_token:mix_token + 1, segment:segment + tile], pl.FP32)
                else:
                    input_row = mix_token * streams + mix_source
                    x = pl.cast(residual[input_row:input_row + 1, segment:segment + tile], pl.FP32)
                target_start = mix_token * streams
                # Padding keeps the FP32 reduction aligned. The first four
                # slots hold the actual coefficients.
                first = pl.slice(upstream, [streams, tile], [target_start, segment])
                expanded = pl.create_tensor([padded, tile], dtype=pl.FP32)
                expanded[:streams, :] = first
                expanded[streams:padded, :] = pl.full([streams, tile], dtype=pl.FP32, value=0.0)
                combined_dot = pl.add(combined_dot, pl.col_expand_mul(expanded, x))
            scalar_gradient = pl.reshape(pl.row_sum(combined_dot), [1, padded])
            if mix_source == streams:
                dpost[mix_token:mix_token + 1, :] = scalar_gradient
            else:
                matrix_row = mix_token * streams + mix_source
                dmix[matrix_row:matrix_row + 1, :] = scalar_gradient
        return dy, dx, dpost, dmix

    @pl.jit.host
    def entry(sublayer: pl.Tensor, residual: pl.Tensor, post: pl.Tensor, mix: pl.Tensor, upstream: pl.Tensor,
              dy: pl.Out[pl.Tensor], dx: pl.Out[pl.Tensor], dpost: pl.Out[pl.Tensor], dmix: pl.Out[pl.Tensor]):
        backward(sublayer, residual, post, mix, upstream, dy, dx, dpost, dmix, device=0)

    return backward if device_entry else entry


def make_sinkhorn_backward(tokens, iterations, epsilon, *, device_entry=False):
    """Backpropagate exact GLM residual-mixing normalization on 4x4 matrices."""
    import pypto.language as pl

    if tokens < 1 or iterations < 1 or epsilon <= 0:
        raise ValueError("invalid GLM Sinkhorn replay geometry")
    rows = 8
    passes = 2 * iterations - 1

    @pl.jit.inline
    def normalize(x: pl.Tensor[[rows, rows], pl.FP32]):
        total = pl.add(pl.mul(pl.row_sum(x), 0.5), epsilon)
        return pl.row_expand_div(x, total)

    @pl.jit.inline
    def undo(x: pl.Tensor[[rows, rows], pl.FP32], gradient: pl.Tensor[[rows, rows], pl.FP32]):
        total = pl.add(pl.mul(pl.row_sum(x), 0.5), epsilon)
        y = pl.row_expand_div(x, total)
        coupling = pl.mul(pl.row_sum(pl.mul(y, gradient)), 0.5)
        return pl.row_expand_div(pl.row_expand_sub(gradient, coupling), total)

    @pl.jit
    def backward(
        logits: pl.Tensor[[tokens * rows, rows], pl.FP32],
        gradient: pl.Tensor[[tokens * rows, rows], pl.FP32],
        output: pl.Out[pl.Tensor[[tokens * rows, rows], pl.FP32]],
    ):
        tape = pl.create_tensor([tokens * passes * rows, rows], dtype=pl.FP32)
        for token in pl.spmd(tokens, name_hint="orthrus_glm53_sinkhorn_adjoint"):
            start = token * rows
            source = logits[start:start + rows, :]
            repr_exp = pl.exp(pl.row_expand_sub(source, pl.row_max(source)))
            softmax = pl.row_expand_div(repr_exp, pl.mul(pl.row_sum(repr_exp), 0.5))
            current = pl.create_tensor([rows, rows], dtype=pl.FP32)
            current[:, :] = pl.add(softmax, epsilon)
            for step in pl.range(passes):
                slot = (token * passes + step) * rows
                tape[slot:slot + rows, :] = current
                if step % 2 == 0:
                    transposed = pl.create_tensor([rows, rows], dtype=pl.FP32)
                    transposed[:, :] = pl.transpose(current, 0, 1)
                    oriented = normalize(transposed)
                    current[:, :] = pl.transpose(oriented, 0, 1)
                else:
                    current[:, :] = normalize(current)

            g = pl.create_tensor([rows, rows], dtype=pl.FP32)
            g[:, :] = gradient[start:start + rows, :]
            for reverse in pl.range(passes):
                step = passes - 1 - reverse
                slot = (token * passes + step) * rows
                recorded = tape[slot:slot + rows, :]
                if step % 2 == 0:
                    transposed_x = pl.create_tensor([rows, rows], dtype=pl.FP32)
                    transposed_g = pl.create_tensor([rows, rows], dtype=pl.FP32)
                    transposed_x[:, :] = pl.transpose(recorded, 0, 1)
                    transposed_g[:, :] = pl.transpose(g, 0, 1)
                    oriented_gradient = undo(transposed_x, transposed_g)
                    g[:, :] = pl.transpose(oriented_gradient, 0, 1)
                else:
                    g[:, :] = undo(recorded, g)
            bend = pl.mul(pl.row_sum(pl.mul(g, softmax)), 0.5)
            output[start:start + rows, :] = pl.mul(softmax, pl.row_expand_sub(g, bend))
        return output

    @pl.jit.host
    def entry(logits: pl.Tensor, gradient: pl.Tensor, output: pl.Out[pl.Tensor]):
        backward(logits, gradient, output, device=0)

    return backward if device_entry else entry


def make_mhc_mix_backward(tokens, width, norm_epsilon, mix_epsilon, *, device_entry=False):
    """Backpropagate the frozen coefficient projection and its input RMS."""
    import pypto.language as pl

    streams, padded, tile = 4, 8, 128
    flat_width = width * streams
    coefficients = 2 * streams + streams * streams
    if tokens < 1 or width < tile or width % tile or norm_epsilon <= 0 or mix_epsilon <= 0:
        raise ValueError("invalid GLM mHC coefficient geometry")

    saved_shape = [tokens * streams, width] if device_entry else [tokens, flat_width]

    @pl.jit
    def mhc_mix_backward(
        saved: pl.Tensor[saved_shape, pl.BF16],
        function: pl.Tensor[[coefficients, flat_width], pl.BF16],
        scale: pl.Tensor[[3], pl.FP32],
        pre: pl.Tensor[[tokens, padded], pl.FP32],
        post: pl.Tensor[[tokens, padded], pl.FP32],
        d_pre: pl.Tensor[[tokens, padded], pl.FP32],
        d_post: pl.Tensor[[tokens, padded], pl.FP32],
        d_residual_logits: pl.Tensor[[tokens * padded, padded], pl.FP32],
        output: pl.Out[pl.Tensor[[tokens, flat_width], pl.FP32]],
    ):
        activation = pl.reshape(saved, [tokens, flat_width])
        # A token's scalar writes need a dedicated 128-byte cache line.
        d_dot = pl.create_tensor([tokens, 32], dtype=pl.FP32)
        projected = pl.create_tensor([tokens, flat_width], dtype=pl.FP32)
        for row in pl.spmd(tokens, name_hint="glm_mhc_coefficients_vjp"):
            for stream in pl.unroll(streams):
                gate_pre = pl.read(pre, [row, stream]) - mix_epsilon
                gate_post = pl.read(post, [row, stream]) * 0.5
                a = pl.read(d_pre, [row, stream]) * gate_pre * (1.0 - gate_pre)
                b = pl.read(d_post, [row, stream]) * gate_post * (1.0 - gate_post) * 2.0
                pl.write(d_dot, [row, stream], a * pl.read(scale, [0]))
                pl.write(d_dot, [row, streams + stream], b * pl.read(scale, [1]))
                for dest in pl.unroll(streams):
                    dc = pl.read(d_residual_logits, [row * padded + stream, dest])
                    pl.write(d_dot, [row, 2 * streams + stream * streams + dest], dc * pl.read(scale, [2]))
        for task in pl.spmd(tokens * (flat_width // tile), name_hint="glm_mhc_projection_vjp"):
            row = task // (flat_width // tile)
            col = task % (flat_width // tile) * tile
            g = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for lane in pl.range(coefficients):
                w = pl.cast(function[lane:lane + 1, col:col + tile], pl.FP32)
                g = pl.add(g, pl.mul(w, pl.read(d_dot, [row, lane])))
            projected[row:row + 1, col:col + tile] = g
        for row in pl.spmd(tokens, name_hint="glm_mhc_input_rms_vjp"):
            squares = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            dots = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, flat_width, tile):
                x = pl.cast(activation[row:row + 1, col:col + tile], pl.FP32)
                g = projected[row:row + 1, col:col + tile]
                squares = pl.add(squares, pl.mul(x, x))
                dots = pl.add(dots, pl.mul(x, g))
            zero = pl.full([8, tile], dtype=pl.FP32, value=0.0)
            variance = pl.add(pl.mul(pl.row_sum(pl.col_expand(zero, squares)), 1.0 / flat_width), norm_epsilon)
            inverse = pl.recip(pl.sqrt(variance))
            inner = pl.row_sum(pl.col_expand(zero, dots))
            coupling = pl.mul(pl.mul(inner, pl.mul(inverse, inverse)), 1.0 / flat_width)
            inv = pl.read(inverse, [0, 0])
            correction = pl.read(coupling, [0, 0])
            for col in pl.range(0, flat_width, tile):
                x = pl.cast(activation[row:row + 1, col:col + tile], pl.FP32)
                g = projected[row:row + 1, col:col + tile]
                output[row:row + 1, col:col + tile] = pl.mul(pl.sub(g, pl.mul(x, correction)), inv)
        return output

    @pl.jit.host
    def entry(activation: pl.Tensor, function: pl.Tensor, scale: pl.Tensor, pre: pl.Tensor, post: pl.Tensor,
              d_pre: pl.Tensor, d_post: pl.Tensor, d_residual_logits: pl.Tensor, output: pl.Out[pl.Tensor]):
        mhc_mix_backward(activation, function, scale, pre, post, d_pre, d_post, d_residual_logits, output, device=0)

    return mhc_mix_backward if device_entry else entry


def make_mhc_sublayer_adjoint(ranks, block_rows=6):
    """Wrap an opaque sublayer VJP in the GLM four-stream residual VJP."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, FLASH, HC_MULT
    from models.glm5_3_flash.elementwise import accurate_rsqrt
    from models.glm5_3_flash.linear import bf16_linear_row
    from models.glm5_3_flash.mhc import mhc_mixes, mhc_pre

    rows, streams, pad, tile = 16, HC_MULT, 8, 128
    width, coeffs = streams * D, streams * (2 + streams)
    epsilon = FLASH.rms_norm_eps
    if not 0 < block_rows <= rows:
        raise ValueError("invalid GLM mHC tape rows")
    post_vjp = make_mhc_post_backward(D, rows, device_entry=True)
    pre_vjp = make_mhc_pre_backward(D, rows, device_entry=True)
    sinkhorn_vjp = make_sinkhorn_backward(rows, FLASH.hc_sinkhorn_iters, FLASH.hc_eps, device_entry=True)
    mix_vjp = make_mhc_mix_backward(rows, D, FLASH.rms_norm_eps, FLASH.hc_eps, device_entry=True)

    @pl.jit
    def replay(
        tape: pl.Tensor[[block_rows, streams + 1, D], pl.BF16],
        function: pl.Tensor[[coeffs, width], pl.BF16],
        scale: pl.Tensor[[3], pl.FP32], base: pl.Tensor[[coeffs], pl.FP32],
        residual: pl.Out[pl.Tensor[[rows * streams, D], pl.BF16]],
        sublayer: pl.Out[pl.Tensor[[rows, D], pl.BF16]],
        collapse: pl.Out[pl.Tensor[[rows, D], pl.BF16]],
        pre: pl.Out[pl.Tensor[[rows, pad], pl.FP32]],
        post: pl.Out[pl.Tensor[[rows, pad], pl.FP32]],
        mix: pl.Out[pl.Tensor[[rows * streams, pad], pl.FP32]],
        logits: pl.Out[pl.Tensor[[rows * pad, pad], pl.FP32]],
    ):
        input_rows = residual
        tape_rows = pl.reshape(tape, [block_rows * (streams + 1), D])
        for task in pl.spmd(rows * (D // tile), name_hint="glm_mhc_unpack_replay"):
            t = task // (D // tile)
            col = task % (D // tile) * tile
            if t < block_rows:
                source = t * (streams + 1)
                input_rows[t * streams:t * streams + streams, col:col + tile] = tape_rows[
                    source:source + streams, col:col + tile
                ]
                sublayer[t:t + 1, col:col + tile] = tape_rows[source + streams:source + streams + 1, col:col + tile]
            else:
                input_rows[t * streams:t * streams + streams, col:col + tile] = pl.full(
                    [streams, tile], dtype=pl.BF16, value=0.0
                )
                sublayer[t:t + 1, col:col + tile] = pl.full([1, tile], dtype=pl.BF16, value=0.0)
        x = pl.reshape(input_rows, [rows, streams, D])
        p = pl.create_tensor([rows, streams], dtype=pl.FP32)
        q = pl.create_tensor([rows, streams], dtype=pl.FP32)
        r = pl.create_tensor([rows, streams, streams], dtype=pl.FP32)
        p, q, r = mhc_mixes(x, function, scale, base, p, q, r)
        mhc_pre(x, p, collapse)
        r_flat = pl.reshape(r, [rows * streams, streams])
        flat = pl.reshape(x, [rows, width])
        for token in pl.spmd(rows, name_hint="glm_mhc_training_coefficients"):
            a = pl.full([1, pad], dtype=pl.FP32, value=0.0)
            b = pl.full([1, pad], dtype=pl.FP32, value=0.0)
            c = pl.full([streams, pad], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(streams):
                pl.write(a, [0, stream], pl.read(p, [token, stream]))
                pl.write(b, [0, stream], pl.read(q, [token, stream]))
                for dest in pl.unroll(streams):
                    pl.write(c, [stream, dest], pl.read(r_flat, [token * streams + stream, dest]))
            pre[token:token + 1, :] = a
            post[token:token + 1, :] = b
            mix[token * streams:token * streams + streams, :] = c
        # Sinkhorn's input logits are absent from the deployed forward outputs.
        # Replay the coefficient projection from its saved BF16 residual streams.
        for token_idx in pl.spmd(rows, name_hint="glm_mhc_training_sinkhorn_logits"):
            squares = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for offset in pl.range(0, width, tile):
                u = pl.cast(flat[token_idx:token_idx + 1, offset:offset + tile], pl.FP32)
                squares = pl.add(squares, pl.mul(u, u))
            square_rows = pl.col_expand(pl.full([pad, tile], dtype=pl.FP32, value=0.0), squares)
            variance = pl.add(pl.mul(pl.row_sum(square_rows), 1.0 / width), epsilon)
            inverse = pl.read(accurate_rsqrt(variance), [0, 0])
            first = bf16_linear_row(flat, function, token_idx, 2 * streams, pad, width, tile, 2)
            second = bf16_linear_row(flat, function, token_idx, 2 * streams + pad, pad, width, tile, 2)
            block = pl.create_tensor([pad, pad], dtype=pl.FP32)
            for i in pl.unroll(pad):
                for j in pl.unroll(pad):
                    lane = i % streams * streams + j % streams
                    value = pl.cast(0.0, pl.FP32)
                    if lane < pad:
                        value = pl.read(first, [0, lane])
                    else:
                        value = pl.read(second, [0, lane - pad])
                    pl.write(block, [i, j], value * inverse * pl.read(scale, [2]) + pl.read(base, [2 * streams + lane]))
            logits[token_idx * pad:token_idx * pad + pad, :] = block
        return residual, sublayer, collapse, pre, post, mix, logits

    @pl.jit
    def pad_mix_gradient(
        source: pl.Tensor[[rows * streams, pad], pl.FP32],
        output: pl.Out[pl.Tensor[[rows * pad, pad], pl.FP32]],
    ):
        for row in pl.spmd(rows * pad, name_hint="glm_mhc_sinkhorn_gradient_pad"):
            src = row // pad * streams + row % streams
            line = pl.full([1, pad], dtype=pl.FP32, value=0.0)
            for col in pl.unroll(pad):
                pl.write(line, [0, col], pl.read(source, [src, col % streams]))
            output[row:row + 1, :] = line
        return output

    @pl.jit
    def combine(
        direct: pl.Tensor[[rows * streams, D], pl.FP32],
        via_pre: pl.Tensor[[rows * streams, D], pl.FP32],
        via_mix: pl.Tensor[[rows, width], pl.FP32],
        output: pl.Out[pl.Tensor[[rows * streams, D], pl.FP32]],
    ):
        coeff = pl.reshape(via_mix, [rows * streams, D])
        for task in pl.spmd(rows * streams * (D // tile), name_hint="glm_mhc_residual_vjp"):
            row = task // (D // tile)
            col = task % (D // tile) * tile
            data = pl.add(direct[row:row + 1, col:col + tile], via_pre[row:row + 1, col:col + tile])
            output[row:row + 1, col:col + tile] = pl.add(data, coeff[row:row + 1, col:col + tile])
        return output

    @pl.jit.host
    def enter(
        tape: pl.Tensor[[ranks, block_rows, streams + 1, D], pl.BF16],
        function: pl.Tensor[[ranks, coeffs, width], pl.BF16],
        scale: pl.Tensor[[ranks, 3], pl.FP32],
        base: pl.Tensor[[ranks, coeffs], pl.FP32],
        upstream: pl.Tensor[[ranks, rows * streams, D], pl.FP32],
        residual: pl.Out[pl.Tensor[[ranks, rows * streams, D], pl.BF16]],
        collapse: pl.Out[pl.Tensor[[ranks, rows, D], pl.BF16]],
        pre: pl.Out[pl.Tensor[[ranks, rows, pad], pl.FP32]],
        post: pl.Out[pl.Tensor[[ranks, rows, pad], pl.FP32]],
        logits: pl.Out[pl.Tensor[[ranks, rows * pad, pad], pl.FP32]],
        dy: pl.Out[pl.Tensor[[ranks, rows, D], pl.FP32]],
        direct: pl.Out[pl.Tensor[[ranks, rows * streams, D], pl.FP32]],
        dpost: pl.Out[pl.Tensor[[ranks, rows, pad], pl.FP32]],
        dmix: pl.Out[pl.Tensor[[ranks, rows * streams, pad], pl.FP32]],
        sublayer_scratch: pl.InOut[pl.Tensor[[ranks, rows, D], pl.BF16]],
        mix_scratch: pl.InOut[pl.Tensor[[ranks, rows * streams, pad], pl.FP32]],
    ):
        for rank in pl.range(pld.world_size()):
            y = sublayer_scratch[rank]
            mix = mix_scratch[rank]
            x = residual[rank]
            p = pre[rank]
            q = post[rank]
            z = collapse[rank]
            logits_out = logits[rank]
            replayed_x, replayed_y, _, replayed_pre, replayed_post, replayed_mix, replayed_logits = replay(
                tape[rank], function[rank], scale[rank], base[rank], x, y, z, p, q, mix, logits_out, device=rank,
            )
            post_vjp(replayed_y, replayed_x, replayed_post, replayed_mix, upstream[rank],
                     dy[rank], direct[rank], dpost[rank], dmix[rank], device=rank)

    @pl.jit.host
    def leave(
        residual: pl.Tensor[[ranks, rows * streams, D], pl.BF16],
        function: pl.Tensor[[ranks, coeffs, width], pl.BF16],
        scale: pl.Tensor[[ranks, 3], pl.FP32],
        pre: pl.Tensor[[ranks, rows, pad], pl.FP32],
        post: pl.Tensor[[ranks, rows, pad], pl.FP32],
        logits: pl.Tensor[[ranks, rows * pad, pad], pl.FP32],
        direct: pl.Tensor[[ranks, rows * streams, D], pl.FP32],
        dpost: pl.Tensor[[ranks, rows, pad], pl.FP32],
        dmix: pl.Tensor[[ranks, rows * streams, pad], pl.FP32],
        dcollapse: pl.Tensor[[ranks, rows, D], pl.FP32],
        output: pl.Out[pl.Tensor[[ranks, rows * streams, D], pl.FP32]],
        via_pre_scratch: pl.InOut[pl.Tensor[[ranks, rows * streams, D], pl.FP32]],
        dpre_scratch: pl.InOut[pl.Tensor[[ranks, rows, pad], pl.FP32]],
        packed_grad_scratch: pl.InOut[pl.Tensor[[ranks, rows * pad, pad], pl.FP32]],
        dlogits_scratch: pl.InOut[pl.Tensor[[ranks, rows * pad, pad], pl.FP32]],
        via_mix_scratch: pl.InOut[pl.Tensor[[ranks, rows, width], pl.FP32]],
    ):
        for rank in pl.range(pld.world_size()):
            via_pre = via_pre_scratch[rank]
            dpre = dpre_scratch[rank]
            packed_grad = packed_grad_scratch[rank]
            dlogits = dlogits_scratch[rank]
            via_mix = via_mix_scratch[rank]
            x = residual[rank]
            dr, dp = pre_vjp(x, pre[rank], dcollapse[rank], via_pre, dpre, device=rank)
            ds = pad_mix_gradient(dmix[rank], packed_grad, device=rank)
            dl = sinkhorn_vjp(logits[rank], ds, dlogits, device=rank)
            dx = mix_vjp(x, function[rank], scale[rank], pre[rank], post[rank], dp, dpost[rank], dl, via_mix, device=rank)
            combine(direct[rank], dr, dx, output[rank], device=rank)

    return enter, leave
