"""Derivatives of the GLM mHC residual combination."""


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
