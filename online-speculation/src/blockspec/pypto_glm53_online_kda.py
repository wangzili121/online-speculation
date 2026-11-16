"""Activation-only adjoint of GLM's frozen causal KDA recurrence."""


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
