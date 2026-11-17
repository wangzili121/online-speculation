"""Activation-only adjoints for GLM's frozen causal KDA path."""


def make_kda_recurrence_backward(tokens=6, heads=8, *, device_entry=False):
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

    return backward if device_entry else entry


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


def make_kda_output_backward(tokens, heads, width, epsilon, *, device_stages=False):
    """Propagate a frozen KDA output projection through gated RMSNorm."""
    import pypto.language as pl

    from .pypto_glm53_online_head import make_frozen_bf16_input_backward

    dim = 128
    if tokens < 1 or tokens % 16 or heads < 1 or width < 64 or width % 64 or epsilon <= 0:
        raise ValueError("invalid GLM KDA output gradient geometry")
    rows = tokens * heads
    projection = make_frozen_bf16_input_backward(width, heads * dim, tokens, device_entry=True, refine=True)

    @pl.jit
    def gated_norm_backward(
        core: pl.Tensor[[rows, dim], pl.BF16], gate: pl.Tensor[[rows, dim], pl.BF16],
        weight: pl.Tensor[[dim], pl.BF16], dy: pl.Tensor[[tokens, heads * dim], pl.FP32],
        d_core: pl.Out[pl.Tensor[[rows, dim], pl.FP32]], d_gate: pl.Out[pl.Tensor[[rows, dim], pl.FP32]],
    ):
        flat = pl.reshape(dy, [rows, dim])
        for row in pl.spmd(rows, name_hint="glm_kda_gated_norm_vjp"):
            x = pl.cast(core[row:row + 1, :], pl.FP32)
            logit = pl.cast(gate[row:row + 1, :], pl.FP32)
            gamma = pl.cast(pl.reshape(weight, [1, dim]), pl.FP32)
            g = pl.mul(flat[row:row + 1, :], gamma)
            e = pl.exp(pl.neg(pl.abs(logit)))
            positive = pl.cast(pl.cast(pl.cmp(logit, pl.abs(logit), cmp_type=0), pl.INT32), pl.FP32)
            sigmoid = pl.div(pl.add(e, pl.mul(positive, pl.add(pl.neg(e), 1.0))), pl.add(e, 1.0))
            upstream = pl.mul(g, sigmoid)
            zero = pl.full([8, dim], dtype=pl.FP32, value=0.0)
            second_moment = pl.mul(pl.row_sum(pl.col_expand(zero, pl.mul(x, x))), 1.0 / dim)
            inverse = pl.recip(pl.sqrt(pl.add(second_moment, epsilon)))
            dots = pl.row_sum(pl.col_expand(zero, pl.mul(upstream, x)))
            coupling = pl.mul(pl.mul(dots, pl.mul(inverse, inverse)), 1.0 / dim)
            inv = pl.read(inverse, [0, 0])
            adjustment = pl.read(coupling, [0, 0])
            d_core[row:row + 1, :] = pl.mul(pl.sub(upstream, pl.mul(x, adjustment)), inv)
            d_gate[row:row + 1, :] = pl.mul(pl.mul(x, upstream), pl.mul(pl.add(pl.neg(sigmoid), 1.0), inv))
        return d_core, d_gate

    @pl.jit.host
    def entry(upstream: pl.Tensor, core: pl.Tensor, gate: pl.Tensor, projection_weight: pl.Tensor,
              norm_weight: pl.Tensor, d_core: pl.Out[pl.Tensor], d_gate: pl.Out[pl.Tensor]):
        partial = pl.create_tensor([tokens, heads * dim], dtype=pl.FP32)
        dy = projection(upstream, projection_weight, partial, device=0)
        gated_norm_backward(core, gate, norm_weight, dy, d_core, d_gate, device=0)

    return (projection, gated_norm_backward) if device_stages else entry


def make_kda_projection_backward(tokens, heads, width, lower_bound, *, beta_width=None, device_stages=False):
    """Propagate the frozen QKV, forget, beta and output-gate branches to KDA input."""
    import pypto.language as pl

    from .pypto_glm53_online_head import make_frozen_bf16_input_backward

    dim, tile = 128, 128
    local = heads * dim
    beta_width = heads if beta_width is None else beta_width
    if (tokens < 1 or tokens % 16 or heads != 8 or width < tile or width % tile
            or lower_bound >= 0.0 or beta_width < heads):
        raise ValueError("KDA projection backward expects the GLM TP8 geometry")
    to_input = make_frozen_bf16_input_backward(local, width, tokens, device_entry=True)
    to_inner = make_frozen_bf16_input_backward(local, dim, tokens, device_entry=True)
    from_inner = make_frozen_bf16_input_backward(dim, width, tokens, device_entry=True)

    @pl.jit
    def prepare(
        d_qkv: pl.Tensor[[tokens, 3 * local], pl.FP32],
        decay: pl.Tensor[[tokens, local], pl.FP32],
        decay_rate: pl.Tensor[[heads], pl.FP32],
        d_decay: pl.Tensor[[tokens, local], pl.FP32],
        dq: pl.Out[pl.Tensor[[tokens, local], pl.FP32]],
        dk: pl.Out[pl.Tensor[[tokens, local], pl.FP32]],
        dv: pl.Out[pl.Tensor[[tokens, local], pl.FP32]],
        d_forget: pl.Out[pl.Tensor[[tokens, local], pl.FP32]],
    ):
        for task in pl.spmd(tokens * heads, name_hint="glm_kda_prepare_projection_vjp"):
            row = task // heads
            head = task % heads
            col = head * dim
            dq[row:row + 1, col:col + dim] = pl.slice(d_qkv, [1, dim], [row, col])
            dk[row:row + 1, col:col + dim] = pl.slice(d_qkv, [1, dim], [row, col + local])
            dv[row:row + 1, col:col + dim] = pl.slice(d_qkv, [1, dim], [row, col + 2 * local])
            sigmoid = pl.mul(decay[row:row + 1, col:col + dim], 1.0 / lower_bound)
            a_log = pl.read(decay_rate, [head])
            aligned = pl.add(pl.full([1, dim], dtype=pl.FP32, value=0.0), a_log)
            slope = pl.mul(pl.mul(sigmoid, pl.add(pl.neg(sigmoid), 1.0)), pl.exp(aligned))
            d_forget[row:row + 1, col:col + dim] = pl.mul(
                pl.mul(slope, lower_bound), d_decay[row:row + 1, col:col + dim],
            )
        return dq, dk, dv, d_forget

    @pl.jit
    def combine(
        q: pl.Tensor[[tokens, width], pl.FP32], k: pl.Tensor[[tokens, width], pl.FP32],
        v: pl.Tensor[[tokens, width], pl.FP32], forget: pl.Tensor[[tokens, width], pl.FP32],
        gate: pl.Tensor[[tokens, width], pl.FP32], beta: pl.Tensor[[tokens, beta_width], pl.FP32],
        d_beta: pl.Tensor[[tokens, heads], pl.FP32], beta_weight: pl.Tensor[[heads, width], pl.BF16],
        output: pl.Out[pl.Tensor[[tokens, width], pl.FP32]],
    ):
        for task in pl.spmd(tokens * (width // tile), name_hint="glm_kda_projection_vjp_sum"):
            row = task // (width // tile)
            col = task % (width // tile) * tile
            dx = pl.add(q[row:row + 1, col:col + tile], k[row:row + 1, col:col + tile])
            dx = pl.add(dx, v[row:row + 1, col:col + tile])
            dx = pl.add(dx, forget[row:row + 1, col:col + tile])
            dx = pl.add(dx, gate[row:row + 1, col:col + tile])
            for head in pl.unroll(heads):
                b = pl.read(beta, [row, head])
                scale = pl.read(d_beta, [row, head]) * b * (1.0 - b)
                weight = pl.cast(beta_weight[head:head + 1, col:col + tile], pl.FP32)
                dx = pl.add(dx, pl.mul(weight, scale))
            output[row:row + 1, col:col + tile] = dx
        return output

    @pl.jit.host
    def entry(
        d_qkv: pl.Tensor, decay: pl.Tensor, a_log: pl.Tensor, d_decay: pl.Tensor,
        beta: pl.Tensor, d_beta: pl.Tensor, d_output_gate: pl.Tensor,
        w_q: pl.Tensor, w_k: pl.Tensor, w_v: pl.Tensor, w_f_a: pl.Tensor, w_f_b: pl.Tensor,
        w_b: pl.Tensor, w_g_a: pl.Tensor, w_g_b: pl.Tensor, out: pl.Out[pl.Tensor],
    ):
        q = pl.create_tensor([tokens, local], dtype=pl.FP32)
        k = pl.create_tensor([tokens, local], dtype=pl.FP32)
        v = pl.create_tensor([tokens, local], dtype=pl.FP32)
        forget = pl.create_tensor([tokens, local], dtype=pl.FP32)
        forget_inner = pl.create_tensor([tokens, dim], dtype=pl.FP32)
        gate_inner = pl.create_tensor([tokens, dim], dtype=pl.FP32)
        dx_q = pl.create_tensor([tokens, width], dtype=pl.FP32)
        dx_k = pl.create_tensor([tokens, width], dtype=pl.FP32)
        dx_v = pl.create_tensor([tokens, width], dtype=pl.FP32)
        dx_forget = pl.create_tensor([tokens, width], dtype=pl.FP32)
        dx_gate = pl.create_tensor([tokens, width], dtype=pl.FP32)
        d_q, d_k, d_v, d_f = prepare(d_qkv, decay, a_log, d_decay, q, k, v, forget, device=0)
        q_input = to_input(d_q, w_q, dx_q, device=0)
        k_input = to_input(d_k, w_k, dx_k, device=0)
        v_input = to_input(d_v, w_v, dx_v, device=0)
        f_inner = to_inner(d_f, w_f_b, forget_inner, device=0)
        g_inner = to_inner(d_output_gate, w_g_b, gate_inner, device=0)
        f_input = from_inner(f_inner, w_f_a, dx_forget, device=0)
        g_input = from_inner(g_inner, w_g_a, dx_gate, device=0)
        combine(q_input, k_input, v_input, f_input, g_input, beta, d_beta, w_b, out, device=0)

    return (prepare, to_input, to_inner, from_inner, combine) if device_stages else entry


def make_kda_training_replay(tokens, *, ranks=None):
    """Run the native frozen KDA forward with one snapshot per replayed token."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, KDA_DIM, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM
    from models.glm5_3_flash.kda import BETA_PAD, kda_conv_recurrent, kda_projection, recurrent_kda

    heads, dim, local = LOCAL_KDA_H, KDA_DIM, LOCAL_KDA_QKV_DIM
    if tokens != 16 or heads != 8:
        raise ValueError("KDA training replay expects a padded 16-token TP8 shard")

    @pl.jit
    def replay(
        x: pl.Tensor[[tokens, D], pl.BF16],
        starts: pl.Tensor[[2], pl.INT32], slots: pl.Tensor[[1, tokens + 2], pl.INT32],
        w_q: pl.Tensor[[local, D], pl.BF16], w_k: pl.Tensor[[local, D], pl.BF16], w_v: pl.Tensor[[local, D], pl.BF16],
        w_f_a: pl.Tensor[[dim, D], pl.BF16], w_f_b: pl.Tensor[[local, dim], pl.BF16],
        dt_bias: pl.Tensor[[local], pl.FP32], a_log: pl.Tensor[[heads], pl.FP32],
        w_b: pl.Tensor[[heads, D], pl.BF16], w_g_a: pl.Tensor[[dim, D], pl.BF16], w_g_b: pl.Tensor[[local, dim], pl.BF16],
        weight: pl.Tensor[[4, 3 * local], pl.BF16],
        conv_states: pl.InOut[pl.Tensor[[tokens + 1, 3, 3 * local], pl.BF16]],
        states: pl.InOut[pl.Tensor[[tokens + 1, heads, dim, dim], pl.FP32]],
        mixed: pl.Out[pl.Tensor[[tokens, 3 * local], pl.BF16]],
        decay: pl.Out[pl.Tensor[[tokens, heads, dim], pl.FP32]],
        beta: pl.Out[pl.Tensor[[tokens, BETA_PAD], pl.FP32]],
        gate: pl.Out[pl.Tensor[[tokens, heads, dim], pl.BF16]],
        q: pl.Out[pl.Tensor[[tokens, heads, dim], pl.BF16]],
        k: pl.Out[pl.Tensor[[tokens, heads, dim], pl.BF16]],
        v: pl.Out[pl.Tensor[[tokens, heads, dim], pl.BF16]],
        core: pl.Out[pl.Tensor[[tokens, heads, dim], pl.BF16]],
    ):
        kda_projection(x, starts, w_q, w_k, w_v, w_f_a, w_f_b, dt_bias, a_log, w_b, w_g_a, w_g_b,
                       mixed, decay, beta, gate)
        kda_conv_recurrent(mixed, weight, conv_states, starts, slots, q, k, v)
        recurrent_kda(q, k, v, decay, beta, states, starts, slots, core)
        return mixed, decay, beta, gate, q, k, v, core

    participants = ranks or 1

    @pl.jit.host
    def distributed(
        x: pl.Tensor[[participants, tokens, D], pl.BF16],
        starts: pl.Tensor[[participants, 2], pl.INT32], slots: pl.Tensor[[participants, 1, tokens + 2], pl.INT32],
        w_q: pl.Tensor[[participants, local, D], pl.BF16],
        w_k: pl.Tensor[[participants, local, D], pl.BF16],
        w_v: pl.Tensor[[participants, local, D], pl.BF16],
        w_f_a: pl.Tensor[[participants, dim, D], pl.BF16], w_f_b: pl.Tensor[[participants, local, dim], pl.BF16],
        dt_bias: pl.Tensor[[participants, local], pl.FP32], a_log: pl.Tensor[[participants, heads], pl.FP32],
        w_b: pl.Tensor[[participants, heads, D], pl.BF16],
        w_g_a: pl.Tensor[[participants, dim, D], pl.BF16], w_g_b: pl.Tensor[[participants, local, dim], pl.BF16],
        weight: pl.Tensor[[participants, 4, 3 * local], pl.BF16],
        conv_states: pl.InOut[pl.Tensor[[participants, tokens + 1, 3, 3 * local], pl.BF16]],
        states: pl.InOut[pl.Tensor[[participants, tokens + 1, heads, dim, dim], pl.FP32]],
        mixed: pl.Out[pl.Tensor[[participants, tokens, 3 * local], pl.BF16]],
        decay: pl.Out[pl.Tensor[[participants, tokens, heads, dim], pl.FP32]],
        beta: pl.Out[pl.Tensor[[participants, tokens, BETA_PAD], pl.FP32]],
        gate: pl.Out[pl.Tensor[[participants, tokens, heads, dim], pl.BF16]],
        q: pl.Out[pl.Tensor[[participants, tokens, heads, dim], pl.BF16]],
        k: pl.Out[pl.Tensor[[participants, tokens, heads, dim], pl.BF16]],
        v: pl.Out[pl.Tensor[[participants, tokens, heads, dim], pl.BF16]],
        core: pl.Out[pl.Tensor[[participants, tokens, heads, dim], pl.BF16]],
    ):
        for rank in pl.range(pld.world_size()):
            replay(x[rank], starts[rank], slots[rank], w_q[rank], w_k[rank], w_v[rank], w_f_a[rank],
                   w_f_b[rank], dt_bias[rank], a_log[rank], w_b[rank], w_g_a[rank], w_g_b[rank], weight[rank],
                   conv_states[rank], states[rank], mixed[rank], decay[rank], beta[rank], gate[rank],
                   q[rank], k[rank], v[rank], core[rank], device=rank)

    if ranks is not None:
        return distributed

    @pl.jit.host
    def entry(
        x: pl.Tensor, starts: pl.Tensor, slots: pl.Tensor,
        w_q: pl.Tensor, w_k: pl.Tensor, w_v: pl.Tensor, w_f_a: pl.Tensor, w_f_b: pl.Tensor,
        dt_bias: pl.Tensor, a_log: pl.Tensor, w_b: pl.Tensor, w_g_a: pl.Tensor, w_g_b: pl.Tensor, weight: pl.Tensor,
        conv_states: pl.InOut[pl.Tensor], states: pl.InOut[pl.Tensor],
        mixed: pl.Out[pl.Tensor], decay: pl.Out[pl.Tensor], beta: pl.Out[pl.Tensor], gate: pl.Out[pl.Tensor],
        q: pl.Out[pl.Tensor], k: pl.Out[pl.Tensor], v: pl.Out[pl.Tensor], core: pl.Out[pl.Tensor],
    ):
        replay(x, starts, slots, w_q, w_k, w_v, w_f_a, w_f_b, dt_bias, a_log, w_b, w_g_a, w_g_b, weight,
               conv_states, states, mixed, decay, beta, gate, q, k, v, core, device=0)

    return entry


def make_kda_recurrence_inputs(tokens, heads, beta_width, *, device_entry=False):
    """Prepare the same normalized vectors used by the native KDA recurrence."""
    import pypto.language as pl
    from models.glm5_3_flash.elementwise import accurate_divide, accurate_row_sum, accurate_sqrt

    dim, chunk = 128, 16
    rows = tokens * heads
    if rows % chunk:
        raise ValueError("KDA recurrence tape requires whole normalization tiles")

    @pl.jit
    def prepare(
        q: pl.Tensor[[rows, dim], pl.BF16], k: pl.Tensor[[rows, dim], pl.BF16],
        v: pl.Tensor[[rows, dim], pl.BF16], beta: pl.Tensor[[tokens, beta_width], pl.FP32],
        q_norm: pl.Out[pl.Tensor[[rows, dim], pl.FP32]], k_norm: pl.Out[pl.Tensor[[rows, dim], pl.FP32]],
        values: pl.Out[pl.Tensor[[rows, dim], pl.FP32]], beta_per_head: pl.Out[pl.Tensor[[rows, 8], pl.FP32]],
    ):
        for tile in pl.spmd(rows // chunk, name_hint="glm_kda_replay_normalize"):
            row = tile * chunk
            for kind in pl.unroll(2):
                activation = pl.create_tensor([chunk, dim], dtype=pl.FP32)
                if kind == 0:
                    activation[:, :] = pl.cast(q[row:row + chunk, :], pl.FP32)
                else:
                    activation[:, :] = pl.cast(k[row:row + chunk, :], pl.FP32)
                squares = pl.create_tensor([chunk, dim], dtype=pl.FP32)
                squares[:, :] = pl.mul(activation, activation)
                variance = pl.create_tensor([chunk, 1], dtype=pl.FP32)
                variance[:, :] = pl.add(accurate_row_sum(squares), 1e-6)
                divisor = pl.create_tensor([chunk, dim], dtype=pl.FP32)
                divisor[:, :] = pl.row_expand_mul(
                    pl.full([chunk, dim], dtype=pl.FP32, value=1.0), accurate_sqrt(variance),
                )
                unit = accurate_divide(activation, divisor)
                if kind == 0:
                    q_norm[row:row + chunk, :] = pl.mul(unit, dim ** -0.5)
                else:
                    k_norm[row:row + chunk, :] = unit
            values[row:row + chunk, :] = pl.cast(v[row:row + chunk, :], pl.FP32)
        for row in pl.spmd(rows, name_hint="glm_kda_replay_beta"):
            b = pl.read(beta, [row // heads, row % heads])
            beta_per_head[row:row + 1, :] = pl.add(pl.full([1, 8], dtype=pl.FP32, value=0.0), b)
        return q_norm, k_norm, values, beta_per_head

    @pl.jit.host
    def entry(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, beta: pl.Tensor, q_norm: pl.Out[pl.Tensor],
              k_norm: pl.Out[pl.Tensor], values: pl.Out[pl.Tensor], beta_per_head: pl.Out[pl.Tensor]):
        prepare(q, k, v, beta, q_norm, k_norm, values, beta_per_head, device=0)

    return prepare if device_entry else entry


def make_kda_projection_inputs(tokens, heads, *, device_entry=False):
    """Pack convolution gradients and the per-head beta gradient for the projections."""
    import pypto.language as pl

    dim = 128
    local = heads * dim
    rows = tokens * heads

    @pl.jit
    def pack(
        dq: pl.Tensor[[rows, dim], pl.FP32], dk: pl.Tensor[[rows, dim], pl.FP32],
        dv: pl.Tensor[[rows, dim], pl.FP32], d_beta: pl.Tensor[[rows, 8], pl.FP32],
        convolution: pl.Out[pl.Tensor[[tokens, 3 * local], pl.FP32]],
        beta: pl.Out[pl.Tensor[[tokens, heads], pl.FP32]],
    ):
        for row in pl.spmd(rows, name_hint="glm_kda_pack_convolution_vjp"):
            token = row // heads
            col = row % heads * dim
            convolution[token:token + 1, col:col + dim] = dq[row:row + 1, :]
            second = col + local
            third = col + 2 * local
            convolution[token:token + 1, second:second + dim] = dk[row:row + 1, :]
            convolution[token:token + 1, third:third + dim] = dv[row:row + 1, :]
        for token in pl.spmd(1, name_hint="glm_kda_pack_beta_vjp"):
            for step in pl.range(token, tokens, 1):
                for head in pl.unroll(heads):
                    pl.write(beta, [step, head], pl.read(d_beta, [step * heads + head, 0]))
        return convolution, beta

    @pl.jit.host
    def entry(dq: pl.Tensor, dk: pl.Tensor, dv: pl.Tensor, d_beta: pl.Tensor,
              convolution: pl.Out[pl.Tensor], beta: pl.Out[pl.Tensor]):
        pack(dq, dk, dv, d_beta, convolution, beta, device=0)

    return pack if device_entry else entry


def make_kda_replay_seed(ranks, tokens=16):
    """Prepare the frozen KDA replay using only the committed state, not verifier scratch."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, KDA_DIM, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM
    from models.glm5_3_flash.kda import CONV_DIM, KDA_CONV_K, STATE_LEN, stack_conv_weight
    from models.glm5_3_flash.rmsnorm import rmsnorm

    if tokens != 16:
        raise ValueError("GLM KDA replay uses sixteen padded rows")
    state_rows = LOCAL_KDA_H * KDA_DIM
    tile = 128

    @pl.jit
    def seed(
        collapsed: pl.Tensor[[tokens, D], pl.BF16],
        norm: pl.Tensor[[D], pl.BF16],
        conv_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
        conv_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
        conv_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
        conv_state: pl.Tensor[[STATE_LEN, CONV_DIM], pl.BF16],
        recurrent_state: pl.Tensor[[LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32],
        normalized: pl.Out[pl.Tensor[[tokens, D], pl.BF16]],
        weight: pl.Out[pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16]],
        conv_replay: pl.InOut[pl.Tensor[[tokens + 1, STATE_LEN, CONV_DIM], pl.BF16]],
        recurrent_replay: pl.InOut[pl.Tensor[[tokens + 1, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    ):
        rmsnorm(collapsed, norm, normalized)
        stack_conv_weight(conv_q, conv_k, conv_v, weight)
        destination = pl.reshape(conv_replay, [(tokens + 1) * STATE_LEN, CONV_DIM])
        recurrent_in = pl.reshape(recurrent_state, [state_rows, KDA_DIM])
        recurrent_out = pl.reshape(recurrent_replay, [(tokens + 1) * state_rows, KDA_DIM])
        for task in pl.spmd(STATE_LEN * (CONV_DIM // tile), name_hint="glm_kda_seed_convolution"):
            row = task // (CONV_DIM // tile)
            col = task % (CONV_DIM // tile) * tile
            destination[row:row + 1, col:col + tile] = conv_state[row:row + 1, col:col + tile]
        for part in pl.spmd(state_rows // 8, name_hint="glm_kda_seed_recurrence"):
            row = part * 8
            recurrent_out[row:row + 8, :] = recurrent_in[row:row + 8, :]

    @pl.jit.host
    def entry(
        collapsed: pl.Tensor[[ranks, tokens, D], pl.BF16],
        norm: pl.Tensor[[ranks, D], pl.BF16],
        conv_q: pl.Tensor[[ranks, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
        conv_k: pl.Tensor[[ranks, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
        conv_v: pl.Tensor[[ranks, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
        conv_state: pl.Tensor[[ranks, STATE_LEN, CONV_DIM], pl.BF16],
        recurrent_state: pl.Tensor[[ranks, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32],
        normalized: pl.Out[pl.Tensor[[ranks, tokens, D], pl.BF16]],
        weight: pl.Out[pl.Tensor[[ranks, KDA_CONV_K, CONV_DIM], pl.BF16]],
        conv_replay: pl.InOut[pl.Tensor[[ranks, tokens + 1, STATE_LEN, CONV_DIM], pl.BF16]],
        recurrent_replay: pl.InOut[pl.Tensor[[ranks, tokens + 1, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    ):
        for rank in pl.range(pld.world_size()):
            seed(collapsed[rank], norm[rank], conv_q[rank], conv_k[rank], conv_v[rank],
                 conv_state[rank], recurrent_state[rank], normalized[rank], weight[rank],
                 conv_replay[rank], recurrent_replay[rank], device=rank)

    return entry


KDA_PROGRAMS = ("orthrus_kda_seed", "orthrus_kda_replay", "orthrus_kda_temporal",
                "orthrus_kda_projection", "orthrus_kda_reduce")


def prepare_kda_programs(api, compiler, rows):
    ranks = api.RANKS
    programs = (make_kda_replay_seed(ranks, rows), make_kda_training_replay(rows, ranks=ranks),
                *make_kda_resident_backward(ranks, rows))
    return {name: compiler.compile(name, entry) for name, entry in zip(KDA_PROGRAMS, programs, strict=True)}


class ResidentKDAReplay:
    """Replay and differentiate a frozen KDA layer using the committed request state."""

    def __init__(self, session, rows):
        import torch
        from models.glm5_3_flash.kda import BETA_PAD, CONV_DIM, KDA_CONV_K, KDA_DIM, LOCAL_KDA_H, STATE_LEN

        def alloc(shape, dtype):
            return session.allocate(shape, dtype)

        def upload_metadata(values):
            values = values.contiguous().share_memory_()
            tensor = alloc(values.shape, values.dtype)
            for rank, shard in zip(tensor.worker_ids, tensor.shards, strict=True):
                session.runtime.copy_to(shard.data_ptr, values.data_ptr(), values.nbytes, worker_id=rank)
            return tensor

        bf16, fp32 = torch.bfloat16, torch.float32
        self.starts = upload_metadata(torch.tensor([0, rows], dtype=torch.int32))
        self.slots = upload_metadata(torch.tensor([[0, rows, *range(1, rows + 1)]], dtype=torch.int32))
        self.normalized = alloc((rows, session.runner._api.D), bf16)
        self.conv_weight = alloc((KDA_CONV_K, CONV_DIM), bf16)
        self.conv = alloc((rows + 1, STATE_LEN, CONV_DIM), bf16)
        self.recurrent = alloc((rows + 1, LOCAL_KDA_H, KDA_DIM, KDA_DIM), fp32)
        self.mixed = alloc((rows, CONV_DIM), bf16)
        self.decay = alloc((rows, LOCAL_KDA_H, KDA_DIM), fp32)
        self.beta = alloc((rows, BETA_PAD), fp32)
        self.gate = alloc((rows, LOCAL_KDA_H, KDA_DIM), bf16)
        self.query = alloc((rows, LOCAL_KDA_H, KDA_DIM), bf16)
        self.key = alloc((rows, LOCAL_KDA_H, KDA_DIM), bf16)
        self.value = alloc((rows, LOCAL_KDA_H, KDA_DIM), bf16)
        self.core = alloc((rows, LOCAL_KDA_H, KDA_DIM), bf16)
        self.dmixed = alloc((rows, CONV_DIM), fp32)
        self.dgate = alloc((rows * LOCAL_KDA_H, KDA_DIM), fp32)
        self.ddecay = alloc((rows * LOCAL_KDA_H, KDA_DIM), fp32)
        self.dbeta = alloc((rows, LOCAL_KDA_H), fp32)
        self.partial = alloc((rows, session.runner._api.D), fp32)
        self.reduced = alloc((rows, session.runner._api.D), fp32)
        self.input_gradient = alloc((rows, session.runner._api.D), fp32)

    def step(self, launch, collapse, upstream, norm, weights, initial_conv, initial_recurrence, epoch):
        (conv_q, conv_k, conv_v, wq, wk, wv, wfa, wfb, dt_bias, a_log, wb, wga, wgb, wo, out_norm) = weights
        launch(KDA_PROGRAMS[0], collapse, norm, conv_q, conv_k, conv_v, initial_conv,
               initial_recurrence, self.normalized, self.conv_weight, self.conv, self.recurrent)
        launch(KDA_PROGRAMS[1], self.normalized, self.starts, self.slots, wq, wk, wv, wfa, wfb,
               dt_bias, a_log, wb, wga, wgb, self.conv_weight, self.conv, self.recurrent,
               self.mixed, self.decay, self.beta, self.gate, self.query, self.key, self.value, self.core)
        launch(KDA_PROGRAMS[2], upstream, self.core, self.gate, self.query, self.key, self.value,
               self.beta, self.decay, self.recurrent, self.mixed, initial_conv, self.conv_weight,
               wo, out_norm, self.dmixed, self.ddecay, self.dbeta, self.dgate)
        launch(KDA_PROGRAMS[3], self.dmixed, self.decay, a_log, self.ddecay, self.beta, self.dbeta,
               self.dgate, wq, wk, wv, wfa, wfb, wb, wga, wgb, self.partial)
        launch(KDA_PROGRAMS[4], collapse, norm, self.partial, self.reduced, self.input_gradient, epoch)
        return self.input_gradient


def make_kda_resident_backward(ranks, tokens=16):
    """Differentiate the replayed KDA shard and reduce the gradient to its mHC input."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, FLASH, KDA_DIM, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM
    from models.glm5_3_flash.kda import BETA_PAD, CONV_DIM, KDA_CONV_K, STATE_LEN

    from .pypto_glm53_online_head import make_final_norm_backward
    from .pypto_glm53_online_suffix import make_fp32_tp_sum

    dim, heads, local = KDA_DIM, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM
    rt = tokens * heads
    project_out, gated_norm = make_kda_output_backward(tokens, heads, D, FLASH.rms_norm_eps, device_stages=True)
    split, project_wide, project_inner, project_outer, combine = make_kda_projection_backward(
        tokens, heads, D, FLASH.linear_lower_bound, beta_width=BETA_PAD, device_stages=True,
    )
    vjp_out = pl.jit.inline(project_out._func)
    vjp_gate = pl.jit.inline(gated_norm._func)
    project_inputs = pl.jit.inline(split._func)
    project_to_input = pl.jit.inline(project_wide._func)
    project_to_inner = pl.jit.inline(project_inner._func)
    project_from_inner = pl.jit.inline(project_outer._func)
    sum_projections = pl.jit.inline(combine._func)
    prepare = pl.jit.inline(make_kda_recurrence_inputs(tokens, heads, BETA_PAD, device_entry=True)._func)
    recurrence = pl.jit.inline(make_kda_recurrence_backward(tokens, heads, device_entry=True)._func)
    qk_vjp = pl.jit.inline(make_kda_qk_backward(tokens, heads, device_entry=True)._func)
    pack = pl.jit.inline(make_kda_projection_inputs(tokens, heads, device_entry=True)._func)
    conv_vjp = pl.jit.inline(make_kda_convolution_backward(tokens, CONV_DIM, device_entry=True)._func)
    tp_sum = make_fp32_tp_sum(ranks, tokens, D)
    rms_vjp = make_final_norm_backward(D, FLASH.rms_norm_eps, tokens, streams=1, device_entry=True)

    @pl.jit
    def temporal_backward(
        upstream: pl.Tensor[[tokens, D], pl.FP32],
        core: pl.Tensor[[tokens, heads, dim], pl.BF16], gate: pl.Tensor[[tokens, heads, dim], pl.BF16],
        query: pl.Tensor[[tokens, heads, dim], pl.BF16], key: pl.Tensor[[tokens, heads, dim], pl.BF16],
        value: pl.Tensor[[tokens, heads, dim], pl.BF16], beta: pl.Tensor[[tokens, BETA_PAD], pl.FP32],
        decay: pl.Tensor[[tokens, heads, dim], pl.FP32],
        states: pl.Tensor[[tokens + 1, heads, dim, dim], pl.FP32],
        mixed: pl.Tensor[[tokens, CONV_DIM], pl.BF16], initial: pl.Tensor[[STATE_LEN, CONV_DIM], pl.BF16],
        conv_weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
        output_weight: pl.Tensor[[D, local], pl.BF16], output_norm: pl.Tensor[[dim], pl.BF16],
        mixed_grad: pl.Out[pl.Tensor[[tokens, CONV_DIM], pl.FP32]],
        decay_grad: pl.Out[pl.Tensor[[rt, dim], pl.FP32]],
        beta_grad: pl.Out[pl.Tensor[[tokens, heads], pl.FP32]],
        gate_grad: pl.Out[pl.Tensor[[rt, dim], pl.FP32]],
    ):
        projected = pl.create_tensor([tokens, local], dtype=pl.FP32)
        core_grad = pl.create_tensor([rt, dim], dtype=pl.FP32)
        qn = pl.create_tensor([rt, dim], dtype=pl.FP32)
        kn = pl.create_tensor([rt, dim], dtype=pl.FP32)
        vf = pl.create_tensor([rt, dim], dtype=pl.FP32)
        step_beta = pl.create_tensor([rt, 8], dtype=pl.FP32)
        dq = pl.create_tensor([rt, dim], dtype=pl.FP32)
        dk = pl.create_tensor([rt, dim], dtype=pl.FP32)
        dv = pl.create_tensor([rt, dim], dtype=pl.FP32)
        step_beta_grad = pl.create_tensor([rt, 8], dtype=pl.FP32)
        dqn = pl.create_tensor([rt, dim], dtype=pl.FP32)
        dkn = pl.create_tensor([rt, dim], dtype=pl.FP32)
        conv_grad = pl.create_tensor([tokens, CONV_DIM], dtype=pl.FP32)
        q_flat = pl.reshape(query, [rt, dim])
        k_flat = pl.reshape(key, [rt, dim])
        v_flat = pl.reshape(value, [rt, dim])
        core_flat = pl.reshape(core, [rt, dim])
        gate_flat = pl.reshape(gate, [rt, dim])
        decay_flat = pl.reshape(decay, [rt, dim])
        states_flat = pl.reshape(states, [(tokens + 1) * heads * dim, dim])
        dy = vjp_out(upstream, output_weight, projected)
        dcore, ignored_gate = vjp_gate(core_flat, gate_flat, output_norm, dy, core_grad, gate_grad)
        q_unit, k_unit, val, betas = prepare(q_flat, k_flat, v_flat, beta, qn, kn, vf, step_beta)
        dquery, dkey, dvalue, ignored_decay, dbeta = recurrence(
            q_unit, k_unit, val, decay_flat, betas, states_flat, dcore, dq, dk, dv, decay_grad, step_beta_grad,
        )
        a, b = qk_vjp(q_flat, k_flat, dquery, dkey, dqn, dkn)
        conv, _ = pack(a, b, dvalue, dbeta, conv_grad, beta_grad)
        conv_vjp(mixed, initial, conv_weight, conv, mixed_grad)

    @pl.jit
    def projection_backward(
        d_qkv: pl.Tensor[[tokens, CONV_DIM], pl.FP32],
        decay: pl.Tensor[[tokens, heads, dim], pl.FP32], a_log: pl.Tensor[[heads], pl.FP32],
        d_decay: pl.Tensor[[rt, dim], pl.FP32], beta: pl.Tensor[[tokens, BETA_PAD], pl.FP32],
        d_beta: pl.Tensor[[tokens, heads], pl.FP32], d_gate: pl.Tensor[[rt, dim], pl.FP32],
        w_q: pl.Tensor[[local, D], pl.BF16], w_k: pl.Tensor[[local, D], pl.BF16],
        w_v: pl.Tensor[[local, D], pl.BF16],
        w_f_a: pl.Tensor[[dim, D], pl.BF16], w_f_b: pl.Tensor[[local, dim], pl.BF16],
        w_b: pl.Tensor[[heads, D], pl.BF16],
        w_g_a: pl.Tensor[[dim, D], pl.BF16], w_g_b: pl.Tensor[[local, dim], pl.BF16],
        output: pl.Out[pl.Tensor[[tokens, D], pl.FP32]],
    ):
        d_q = pl.create_tensor([tokens, local], dtype=pl.FP32)
        d_k = pl.create_tensor([tokens, local], dtype=pl.FP32)
        d_v = pl.create_tensor([tokens, local], dtype=pl.FP32)
        d_f = pl.create_tensor([tokens, local], dtype=pl.FP32)
        fg = pl.create_tensor([tokens, dim], dtype=pl.FP32)
        gg = pl.create_tensor([tokens, dim], dtype=pl.FP32)
        dx_q = pl.create_tensor([tokens, D], dtype=pl.FP32)
        dx_k = pl.create_tensor([tokens, D], dtype=pl.FP32)
        dx_v = pl.create_tensor([tokens, D], dtype=pl.FP32)
        dx_f = pl.create_tensor([tokens, D], dtype=pl.FP32)
        dx_g = pl.create_tensor([tokens, D], dtype=pl.FP32)
        joined_decay = pl.reshape(decay, [tokens, local])
        joined_decay_grad = pl.reshape(d_decay, [tokens, local])
        joined_gate_grad = pl.reshape(d_gate, [tokens, local])
        a, b, c, f = project_inputs(d_qkv, joined_decay, a_log, joined_decay_grad, d_q, d_k, d_v, d_f)
        dq = project_to_input(a, w_q, dx_q)
        dk = project_to_input(b, w_k, dx_k)
        dv = project_to_input(c, w_v, dx_v)
        inner_f = project_to_inner(f, w_f_b, fg)
        inner_g = project_to_inner(joined_gate_grad, w_g_b, gg)
        df = project_from_inner(inner_f, w_f_a, dx_f)
        dg = project_from_inner(inner_g, w_g_a, dx_g)
        sum_projections(dq, dk, dv, df, dg, beta, d_beta, w_b, output)

    @pl.jit.host
    def temporal(
        upstream: pl.Tensor[[ranks, tokens, D], pl.FP32],
        core: pl.Tensor[[ranks, tokens, heads, dim], pl.BF16],
        gate: pl.Tensor[[ranks, tokens, heads, dim], pl.BF16],
        query: pl.Tensor[[ranks, tokens, heads, dim], pl.BF16],
        key: pl.Tensor[[ranks, tokens, heads, dim], pl.BF16],
        value: pl.Tensor[[ranks, tokens, heads, dim], pl.BF16],
        beta: pl.Tensor[[ranks, tokens, BETA_PAD], pl.FP32],
        decay: pl.Tensor[[ranks, tokens, heads, dim], pl.FP32],
        states: pl.Tensor[[ranks, tokens + 1, heads, dim, dim], pl.FP32],
        mixed: pl.Tensor[[ranks, tokens, CONV_DIM], pl.BF16],
        initial: pl.Tensor[[ranks, STATE_LEN, CONV_DIM], pl.BF16],
        conv_weight: pl.Tensor[[ranks, KDA_CONV_K, CONV_DIM], pl.BF16],
        output_weight: pl.Tensor[[ranks, D, local], pl.BF16], output_norm: pl.Tensor[[ranks, dim], pl.BF16],
        mixed_grad: pl.Out[pl.Tensor[[ranks, tokens, CONV_DIM], pl.FP32]],
        decay_grad: pl.Out[pl.Tensor[[ranks, rt, dim], pl.FP32]],
        beta_grad: pl.Out[pl.Tensor[[ranks, tokens, heads], pl.FP32]],
        gate_grad: pl.Out[pl.Tensor[[ranks, rt, dim], pl.FP32]],
    ):
        for rank in pl.range(pld.world_size()):
            temporal_backward(upstream[rank], core[rank], gate[rank], query[rank], key[rank], value[rank], beta[rank],
                              decay[rank], states[rank], mixed[rank], initial[rank], conv_weight[rank],
                              output_weight[rank], output_norm[rank], mixed_grad[rank], decay_grad[rank],
                              beta_grad[rank], gate_grad[rank], device=rank)

    @pl.jit.host
    def projection(
        d_qkv: pl.Tensor[[ranks, tokens, CONV_DIM], pl.FP32],
        decay: pl.Tensor[[ranks, tokens, heads, dim], pl.FP32], a_log: pl.Tensor[[ranks, heads], pl.FP32],
        d_decay: pl.Tensor[[ranks, rt, dim], pl.FP32], beta: pl.Tensor[[ranks, tokens, BETA_PAD], pl.FP32],
        d_beta: pl.Tensor[[ranks, tokens, heads], pl.FP32], d_gate: pl.Tensor[[ranks, rt, dim], pl.FP32],
        w_q: pl.Tensor[[ranks, local, D], pl.BF16], w_k: pl.Tensor[[ranks, local, D], pl.BF16],
        w_v: pl.Tensor[[ranks, local, D], pl.BF16],
        w_f_a: pl.Tensor[[ranks, dim, D], pl.BF16], w_f_b: pl.Tensor[[ranks, local, dim], pl.BF16],
        w_b: pl.Tensor[[ranks, heads, D], pl.BF16],
        w_g_a: pl.Tensor[[ranks, dim, D], pl.BF16], w_g_b: pl.Tensor[[ranks, local, dim], pl.BF16],
        partial: pl.Out[pl.Tensor[[ranks, tokens, D], pl.FP32]],
    ):
        for rank in pl.range(pld.world_size()):
            projection_backward(d_qkv[rank], decay[rank], a_log[rank], d_decay[rank], beta[rank], d_beta[rank],
                                d_gate[rank], w_q[rank], w_k[rank], w_v[rank], w_f_a[rank], w_f_b[rank],
                                w_b[rank], w_g_a[rank], w_g_b[rank], partial[rank], device=rank)

    @pl.jit.host
    def finish(
        collapsed: pl.Tensor[[ranks, tokens, D], pl.BF16], norm: pl.Tensor[[ranks, D], pl.BF16],
        partial: pl.Tensor[[ranks, tokens, D], pl.FP32], reduced: pl.InOut[pl.Tensor[[ranks, tokens, D], pl.FP32]],
        output: pl.Out[pl.Tensor[[ranks, tokens, D], pl.FP32]],
        epoch: pl.Scalar[pl.INT32],
    ):
        exchange_buffer = pld.alloc_window_buffer([tokens, D], dtype=pl.FP32)
        arrivals_buffer = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            exchange = pld.window(exchange_buffer, [tokens, D], dtype=pl.FP32)
            arrivals = pld.window(arrivals_buffer, [ranks, 1], dtype=pl.INT32)
            local = partial[rank]
            total = reduced[rank]
            gradient = tp_sum(local, total, exchange, arrivals, rank, epoch, device=rank)
            rms_vjp(collapsed[rank], norm[rank], gradient, output[rank], device=rank)

    return temporal, projection, finish
