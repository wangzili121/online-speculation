"""Activation-only adjoints for GLM's frozen causal KDA path."""


def make_kda_recurrence_backward(tokens=6, heads=8):
    """Use replayed states, normalized/scaled Q and unit-norm K from the forward."""
    import pypto.language as pl

    dim, v_tile = 128, 16
    parts = dim // v_tile
    rows = tokens * heads

    @pl.jit
    def backward(
        query: pl.Tensor[[rows, dim], pl.FP32], key: pl.Tensor[[rows, dim], pl.FP32],
        value: pl.Tensor[[rows, dim], pl.FP32], log_decay: pl.Tensor[[rows, dim], pl.FP32],
        beta: pl.Tensor[[rows, 8], pl.FP32], states: pl.Tensor[[(tokens + 1) * heads * dim, dim], pl.FP32],
        gradient: pl.Tensor[[rows, dim], pl.FP32],
        d_query: pl.Out[pl.Tensor[[rows, dim], pl.FP32]], d_key: pl.Out[pl.Tensor[[rows, dim], pl.FP32]],
        d_value: pl.Out[pl.Tensor[[rows, dim], pl.FP32]], d_decay: pl.Out[pl.Tensor[[rows, dim], pl.FP32]],
        d_beta: pl.Out[pl.Tensor[[rows, 8], pl.FP32]],
    ):
        q_parts = pl.create_tensor([rows * parts, dim], dtype=pl.FP32)
        k_parts = pl.create_tensor([rows * parts, dim], dtype=pl.FP32)
        decay_parts = pl.create_tensor([rows * parts, dim], dtype=pl.FP32)
        beta_parts = pl.create_tensor([rows * parts, 8], dtype=pl.FP32)
        for lane in pl.spmd(heads * parts, name_hint="orthrus_glm53_kda_adjoint"):
            head = lane // parts
            column = lane % parts * v_tile
            carry = pl.full([v_tile, dim], dtype=pl.FP32, value=0.0)
            for reverse in pl.range(tokens):
                step = tokens - 1 - reverse
                row = step * heads + head
                source = row * dim + column
                destination = source + heads * dim
                before = states[source:source + v_tile, :]
                after = states[destination:destination + v_tile, :]
                q = query[row:row + 1, :]
                k = key[row:row + 1, :]
                g_out = pl.reshape(gradient[row:row + 1, column:column + v_tile], [v_tile, 1])
                beta_scalar = pl.read(beta, [row, 0])
                decay = pl.exp(log_decay[row:row + 1, :])
                decayed = pl.col_expand_mul(before, decay)
                error = pl.sub(pl.reshape(value[row:row + 1, column:column + v_tile], [v_tile, 1]),
                               pl.row_sum(pl.col_expand_mul(decayed, k)))
                q_grid = pl.col_expand(pl.full([v_tile, dim], dtype=pl.FP32, value=0.0), q)
                d_state = pl.add(carry, pl.row_expand_mul(q_grid, g_out))
                d_update = pl.row_sum(pl.col_expand_mul(d_state, k))
                dv = pl.mul(d_update, beta_scalar)
                dk_update = pl.col_sum(pl.row_expand_mul(d_state, pl.mul(error, beta_scalar)))
                dk_memory = pl.col_sum(pl.row_expand_mul(decayed, dv))
                k_grid = pl.col_expand(pl.full([v_tile, dim], dtype=pl.FP32, value=0.0), k)
                d_decayed = pl.sub(d_state, pl.row_expand_mul(k_grid, dv))
                carry = pl.col_expand_mul(d_decayed, decay)

                piece = row * parts + lane % parts
                q_parts[piece:piece + 1, :] = pl.col_sum(pl.row_expand_mul(after, g_out))
                k_parts[piece:piece + 1, :] = pl.sub(dk_update, dk_memory)
                decay_parts[piece:piece + 1, :] = pl.col_sum(pl.mul(d_decayed, decayed))
                d_value[row:row + 1, column:column + v_tile] = pl.reshape(dv, [1, v_tile])
                db_cols = pl.row_expand_mul(pl.full([v_tile, 8], dtype=pl.FP32, value=1.0), pl.mul(d_update, error))
                beta_parts[piece:piece + 1, :] = pl.col_sum(db_cols)
        for row in pl.spmd(rows, name_hint="orthrus_glm53_kda_adjoint_reduce"):
            beginning = row * parts
            d_query[row:row + 1, :] = pl.col_sum(q_parts[beginning:beginning + parts, :])
            d_key[row:row + 1, :] = pl.col_sum(k_parts[beginning:beginning + parts, :])
            d_decay[row:row + 1, :] = pl.col_sum(decay_parts[beginning:beginning + parts, :])
            d_beta[row:row + 1, :] = pl.col_sum(beta_parts[beginning:beginning + parts, :])
        return d_query, d_key, d_value, d_decay, d_beta

    @pl.jit.host
    def entry(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
              states: pl.Tensor, gradient: pl.Tensor,
              d_query: pl.Out[pl.Tensor], d_key: pl.Out[pl.Tensor], d_value: pl.Out[pl.Tensor],
              d_decay: pl.Out[pl.Tensor], d_beta: pl.Out[pl.Tensor]):
        backward(query, key, value, log_decay, beta, states, gradient,
                 d_query, d_key, d_value, d_decay, d_beta, device=0)

    return entry


def make_kda_convolution_backward(tokens, channels, *, device_entry=False):
    """Backpropagate through four-tap causal convolution and SiLU."""
    import pypto.language as pl

    taps, tile, lanes = 4, 128, 48
    if tokens < 1 or channels < tile or channels % tile:
        raise ValueError("invalid GLM KDA convolution geometry")
    parts = channels // tile

    @pl.jit
    def backward(
        mixed: pl.Tensor[[tokens, channels], pl.BF16],
        history: pl.Tensor[[taps - 1, channels], pl.BF16],
        weight: pl.Tensor[[taps, channels], pl.BF16],
        gradient: pl.Tensor[[tokens, channels], pl.FP32],
        output: pl.Out[pl.Tensor[[tokens, channels], pl.FP32]],
    ):
        preactivation_gradient = pl.create_tensor([tokens, channels], dtype=pl.FP32)
        for lane in pl.spmd(lanes, name_hint="glm_kda_silu_vjp"):
            for task in pl.range(lane, tokens * parts, lanes):
                row, col = task // parts, (task % parts) * tile
                conv = pl.full([1, tile], dtype=pl.FP32, value=0.0)
                for tap in pl.unroll(taps):
                    source = row + tap - (taps - 1)
                    sample = pl.create_tensor([1, tile], dtype=pl.FP32)
                    if source < 0:
                        past = source + taps - 1
                        sample[:, :] = pl.cast(history[past:past + 1, col:col + tile], pl.FP32)
                    else:
                        sample[:, :] = pl.cast(mixed[source:source + 1, col:col + tile], pl.FP32)
                    coefficient = pl.cast(weight[tap:tap + 1, col:col + tile], pl.FP32)
                    conv = pl.add(conv, pl.mul(sample, coefficient))
                one = pl.full([1, tile], dtype=pl.FP32, value=1.0)
                # Clamp only the sigmoid's exponent to avoid overflow on saturated inputs.
                sig = pl.recip(pl.add(one, pl.exp(pl.minimum(pl.mul(conv, -1.0), 80.0))))
                slope = pl.mul(sig, pl.add(one, pl.mul(conv, pl.sub(one, sig))))
                preactivation_gradient[row:row + 1, col:col + tile] = pl.mul(
                    gradient[row:row + 1, col:col + tile], slope,
                )
        for lane in pl.spmd(lanes, name_hint="glm_kda_conv_vjp"):
            for task in pl.range(lane, tokens * parts, lanes):
                row, col = task // parts, (task % parts) * tile
                dx = pl.full([1, tile], dtype=pl.FP32, value=0.0)
                for tap in pl.unroll(taps):
                    downstream = row + taps - 1 - tap
                    if downstream < tokens:
                        dz = preactivation_gradient[downstream:downstream + 1, col:col + tile]
                        coefficient = pl.cast(weight[tap:tap + 1, col:col + tile], pl.FP32)
                        dx = pl.add(dx, pl.mul(dz, coefficient))
                output[row:row + 1, col:col + tile] = dx
        return output

    @pl.jit.host
    def entry(mixed: pl.Tensor, history: pl.Tensor, weight: pl.Tensor, gradient: pl.Tensor,
              output: pl.Out[pl.Tensor]):
        backward(mixed, history, weight, gradient, output, device=0)

    return backward if device_entry else entry


def make_kda_qk_backward(tokens, heads, *, device_entry=False):
    """Differentiate the Q and K L2 normalization that precedes the recurrence."""
    import pypto.language as pl

    dim, eps, lanes = 128, 1e-6, 48
    if tokens < 1 or heads < 1:
        raise ValueError("invalid GLM KDA Q/K geometry")
    rows = tokens * heads
    q_scale = dim ** -0.5

    @pl.jit.inline
    def adjoint(x: pl.Tensor[[1, dim], pl.FP32], g: pl.Tensor[[1, dim], pl.FP32]):
        zeros = pl.full([8, dim], dtype=pl.FP32, value=0.0)
        squared = pl.col_expand(zeros, pl.mul(x, x))
        product = pl.col_expand(zeros, pl.mul(x, g))
        norm_squared = pl.add(pl.row_sum(squared), eps)
        inverse = pl.recip(pl.sqrt(norm_squared))
        cross = pl.mul(pl.row_sum(product), pl.mul(inverse, inverse))
        inv = pl.read(inverse, [0, 0])
        correction = pl.read(cross, [0, 0])
        return pl.mul(pl.sub(g, pl.mul(x, correction)), inv)

    @pl.jit
    def kda_qk_backward(
        query: pl.Tensor[[rows, dim], pl.BF16], key: pl.Tensor[[rows, dim], pl.BF16],
        grad_query: pl.Tensor[[rows, dim], pl.FP32], grad_key: pl.Tensor[[rows, dim], pl.FP32],
        d_query: pl.Out[pl.Tensor[[rows, dim], pl.FP32]], d_key: pl.Out[pl.Tensor[[rows, dim], pl.FP32]],
    ):
        for lane in pl.spmd(lanes, name_hint="glm_kda_qk_l2_vjp"):
            for row in pl.range(lane, rows, lanes):
                q = pl.create_tensor([1, dim], dtype=pl.FP32)
                k = pl.create_tensor([1, dim], dtype=pl.FP32)
                dq = pl.create_tensor([1, dim], dtype=pl.FP32)
                dk = pl.create_tensor([1, dim], dtype=pl.FP32)
                q[:, :] = pl.cast(query[row:row + 1, :], pl.FP32)
                k[:, :] = pl.cast(key[row:row + 1, :], pl.FP32)
                dq[:, :] = grad_query[row:row + 1, :]
                dk[:, :] = grad_key[row:row + 1, :]
                d_query[row:row + 1, :] = pl.mul(adjoint(q, dq), q_scale)
                d_key[row:row + 1, :] = adjoint(k, dk)
        return d_query, d_key

    @pl.jit.host
    def entry(q: pl.Tensor, k: pl.Tensor, dq: pl.Tensor, dk: pl.Tensor,
              out_q: pl.Out[pl.Tensor], out_k: pl.Out[pl.Tensor]):
        kda_qk_backward(q, k, dq, dk, out_q, out_k, device=0)

    return kda_qk_backward if device_entry else entry
