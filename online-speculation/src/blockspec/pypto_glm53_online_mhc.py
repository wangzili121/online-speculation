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


def make_mhc_post_backward(width=4096, tokens=6):
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
        for task in pl.spmd(tokens * shards, name_hint="orthrus_glm53_mhc_dx"):
            token = task // shards
            begin = task % shards * tile
            sub_grad = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for destination in pl.unroll(streams):
                dest_row = token * streams + destination
                g = upstream[dest_row:dest_row + 1, begin:begin + tile]
                gate = pl.read(post, [token, destination])
                sub_grad = pl.add(sub_grad, pl.mul(g, gate))
            dy[token:token + 1, begin:begin + tile] = sub_grad
            for source in pl.unroll(streams):
                src_row = token * streams + source
                residual_grad = pl.full([1, tile], dtype=pl.FP32, value=0.0)
                for destination in pl.unroll(streams):
                    dest_row = token * streams + destination
                    g = upstream[dest_row:dest_row + 1, begin:begin + tile]
                    gate = pl.read(mix, [src_row, destination])
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

    return entry


def make_sinkhorn_backward(tokens, iterations, epsilon):
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

    return entry


def make_mhc_mix_backward(tokens, width, norm_epsilon, mix_epsilon):
    """Backpropagate the frozen coefficient projection and its input RMS."""
    import pypto.language as pl

    streams, padded, tile = 4, 8, 128
    flat_width = width * streams
    coefficients = 2 * streams + streams * streams
    if tokens < 1 or width < tile or width % tile or norm_epsilon <= 0 or mix_epsilon <= 0:
        raise ValueError("invalid GLM mHC coefficient geometry")

    @pl.jit
    def mhc_mix_backward(
        activation: pl.Tensor[[tokens, flat_width], pl.BF16],
        function: pl.Tensor[[coefficients, flat_width], pl.BF16],
        scale: pl.Tensor[[3], pl.FP32],
        pre: pl.Tensor[[tokens, padded], pl.FP32],
        post: pl.Tensor[[tokens, padded], pl.FP32],
        d_pre: pl.Tensor[[tokens, padded], pl.FP32],
        d_post: pl.Tensor[[tokens, padded], pl.FP32],
        d_residual_logits: pl.Tensor[[tokens * padded, padded], pl.FP32],
        output: pl.Out[pl.Tensor[[tokens, flat_width], pl.FP32]],
    ):
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

    return entry
