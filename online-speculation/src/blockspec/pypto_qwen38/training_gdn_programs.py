"""GDN 的 PyPTO 显式反向；AR 初态只读，所有临时状态均独立保存。"""


def make_recurrence(shape, *, value_tile=16, inline=False):
    import pypto.language as pl
    H, Q, D, T = shape.heads, shape.query_heads, shape.dimension, shape.rows
    G, NT = H // Q, D // value_tile
    V = value_tile
    if D % V or V not in (16, 32) or H % 8:
        raise ValueError("complete 16/32 value tiles required")

    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def forward(packed: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor, initial: pl.Tensor,
                output: pl.Out[pl.Tensor], saved: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        old = pl.reshape(initial, [H * D, D])
        states = pl.reshape(saved, [(T + 1) * H * D, D])
        result = pl.reshape(output, [T, H * D])
        with pl.spmd(H * NT, name_hint="orthrus_gdn_saved_recurrence"):
            task = pl.tile.get_block_idx()
            h = task // NT
            first = task % NT * V
            state = pl.slice(old, [V, D], [h * D + first, 0])
            states[h * D + first:h * D + first + V, 0:D] = state
            qr = pl.cast(pl.slice(packed, [T, D], [0, h // G * D]), pl.FP32)
            kr = pl.cast(pl.slice(packed, [T, D], [0, Q * D + h // G * D]), pl.FP32)
            qi = pl.rsqrt(pl.add(pl.row_sum(pl.mul(qr, qr)), 1e-6), high_precision=True)
            ki = pl.rsqrt(pl.add(pl.row_sum(pl.mul(kr, kr)), 1e-6), high_precision=True)
            q_all = pl.mul(pl.row_expand_mul(qr, qi), D ** -.5)
            k_all = pl.row_expand_mul(kr, ki)
            for t in pl.range(width):
                q = pl.slice(q_all, [1, D], [t, 0])
                k = pl.slice(k_all, [1, D], [t, 0])
                g = pl.read(decay, [t, h])
                b = pl.read(beta, [t, h])
                factor = pl.exp(pl.mul(pl.full([1, D], dtype=pl.FP32, value=1.), g))
                before = pl.col_expand_mul(state, factor)
                remembered = pl.row_sum(pl.col_expand_mul(before, k))
                value = pl.reshape(pl.cast(pl.slice(packed, [1, V], [t, 2 * Q * D + h * D + first]), pl.FP32), [V, 1])
                delta = pl.mul(pl.sub(value, remembered), b)
                expanded = pl.row_expand_mul(pl.full([V, D], dtype=pl.FP32, value=1.), delta)
                state = pl.add(before, pl.col_expand_mul(expanded, k))
                result[t:t + 1, h * D + first:h * D + first + V] = pl.cast(
                    pl.reshape(pl.row_sum(pl.col_expand_mul(state, q)), [1, V]), pl.BF16, mode="rint")
                row = (t + 1) * H * D + h * D + first
                states[row:row + V, 0:D] = state
            for t in pl.range(width, T):
                result[t:t + 1, h * D + first:h * D + first + V] = pl.full([1, V], dtype=pl.BF16, value=0.)
                row = (t + 1) * H * D + h * D + first
                states[row:row + V, 0:D] = state
        return output, saved

    @pl.jit.host
    def entry(packed: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor, initial: pl.Tensor,
              output: pl.Out[pl.Tensor], saved: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        forward(packed, decay, beta, initial, output, saved, width, device=0)
    return forward if inline else entry


def make_recurrence_vjp(shape, *, value_tile=16, inline=False):
    import pypto.language as pl
    H, Q, D, T = shape.heads, shape.query_heads, shape.dimension, shape.rows
    G, NT, V = H // Q, D // value_tile, value_tile
    if H % 8 or D % V or V not in (16, 32):
        raise ValueError("eight-aligned heads and complete 16/32 value tiles required")

    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def backward(packed: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor, saved: pl.Tensor, upstream: pl.Tensor,
                 dq: pl.Out[pl.Tensor], dk: pl.Out[pl.Tensor], dv: pl.Out[pl.Tensor],
                 dg: pl.Out[pl.Tensor], db: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        states = pl.reshape(saved, [(T + 1) * H * D, D])
        dy = pl.reshape(upstream, [T, H * D])
        with pl.spmd(H * NT, name_hint="orthrus_gdn_reverse_recurrence"):
            task = pl.tile.get_block_idx()
            h = task // NT
            first = task % NT * V
            carry = pl.full([V, D], dtype=pl.FP32, value=0.)
            qr = pl.cast(pl.slice(packed, [T, D], [0, h // G * D]), pl.FP32)
            kr = pl.cast(pl.slice(packed, [T, D], [0, Q * D + h // G * D]), pl.FP32)
            qi = pl.rsqrt(pl.add(pl.row_sum(pl.mul(qr, qr)), 1e-6), high_precision=True)
            ki = pl.rsqrt(pl.add(pl.row_sum(pl.mul(kr, kr)), 1e-6), high_precision=True)
            q_all = pl.mul(pl.row_expand_mul(qr, qi), D ** -.5)
            k_all = pl.row_expand_mul(kr, ki)
            for reverse in pl.range(width):
                t = width - 1 - reverse
                q = pl.slice(q_all, [1, D], [t, 0])
                k = pl.slice(k_all, [1, D], [t, 0])
                gradient = pl.reshape(pl.cast(pl.slice(dy, [1, V], [t, h * D + first]), pl.FP32), [V, 1])
                repeated = pl.row_expand_mul(pl.full([V, D], dtype=pl.FP32, value=1.), gradient)
                ds = pl.add(carry, pl.col_expand_mul(repeated, q))
                now = pl.slice(states, [V, D], [(t + 1) * H * D + h * D + first, 0])
                row = t * H * NT + task
                dq[row:row + 1, 0:D] = pl.col_sum(pl.row_expand_mul(now, gradient))
                g = pl.read(decay, [t, h])
                factor = pl.exp(pl.mul(pl.full([1, D], dtype=pl.FP32, value=1.), g))
                old = pl.col_expand_mul(pl.slice(states, [V, D], [t * H * D + h * D + first, 0]), factor)
                value = pl.reshape(pl.cast(pl.slice(packed, [1, V], [t, 2 * Q * D + h * D + first]), pl.FP32), [V, 1])
                residual = pl.sub(value, pl.row_sum(pl.col_expand_mul(old, k)))
                b = pl.read(beta, [t, h])
                delta = pl.mul(residual, b)
                dd = pl.row_sum(pl.col_expand_mul(ds, k))
                dr = pl.mul(dd, b)
                dk[row:row + 1, 0:D] = pl.col_sum(pl.sub(pl.row_expand_mul(ds, delta), pl.row_expand_mul(old, dr)))
                dv[t:t + 1, h * D + first:h * D + first + V] = pl.cast(pl.reshape(dr, [1, V]), pl.BF16, mode="rint")
                expanded = pl.row_expand_mul(pl.full([V, D], dtype=pl.FP32, value=1.), dr)
                dold = pl.sub(ds, pl.col_expand_mul(expanded, k))
                sum_g = pl.row_sum(pl.reshape(pl.col_sum(pl.mul(dold, old)), [8, D // 8]))
                beta_terms = pl.reshape(pl.mul(dd, residual), [1, V])
                sum_b = pl.full([1, 8], dtype=pl.FP32, value=0.)
                for fragment in pl.range(V // 8):
                    sum_b = pl.add(sum_b, pl.slice(beta_terms, [1, 8], [0, fragment * 8]))
                dg[row:row + 1, 0:8] = pl.reshape(sum_g, [1, 8])
                db[row:row + 1, 0:8] = pl.reshape(sum_b, [1, 8])
                carry = pl.col_expand_mul(dold, factor)
            for t in pl.range(width, T):
                row = t * H * NT + task
                dq[row:row + 1, 0:D] = pl.full([1, D], dtype=pl.FP32, value=0.)
                dk[row:row + 1, 0:D] = pl.full([1, D], dtype=pl.FP32, value=0.)
                dg[row:row + 1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.)
                db[row:row + 1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.)
                dv[t:t + 1, h * D + first:h * D + first + V] = pl.full([1, V], dtype=pl.BF16, value=0.)
        return dq, dk, dv, dg, db

    @pl.jit.host
    def entry(packed: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor, saved: pl.Tensor, upstream: pl.Tensor,
              dq: pl.Out[pl.Tensor], dk: pl.Out[pl.Tensor], dv: pl.Out[pl.Tensor],
              dg: pl.Out[pl.Tensor], db: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        backward(packed, decay, beta, saved, upstream, dq, dk, dv, dg, db, width, device=0)
    return backward if inline else entry


def make_recurrence_reduce(shape, *, value_tile=16, inline=False):
    import pypto.language as pl
    H, Q, D, T = shape.heads, shape.query_heads, shape.dimension, shape.rows
    G, NT = H // Q, D // value_tile
    if H % 8 or D % value_tile or value_tile not in (16, 32):
        raise ValueError("eight-aligned heads and complete 16/32 value tiles required")

    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def reduce(packed: pl.Tensor, dq: pl.Tensor, dk: pl.Tensor, dv: pl.Tensor, dg: pl.Tensor, db: pl.Tensor,
               dx: pl.Out[pl.Tensor], out_g: pl.Out[pl.Tensor], out_b: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        with pl.spmd(T * Q, name_hint="orthrus_gdn_normalization_vjp"):
            task = pl.tile.get_block_idx()
            t = task // Q
            head = task % Q
            q = pl.full([1, D], dtype=pl.FP32, value=0.)
            k = pl.full([1, D], dtype=pl.FP32, value=0.)
            if t < width:
                q = pl.cast(pl.slice(packed, [1, D], [t, head * D]), pl.FP32)
                k = pl.cast(pl.slice(packed, [1, D], [t, Q * D + head * D]), pl.FP32)
            aq = pl.full([1, D], dtype=pl.FP32, value=0.)
            ak = pl.full([1, D], dtype=pl.FP32, value=0.)
            for part in pl.range(G * NT):
                row = t * H * NT + head * G * NT + part
                aq = pl.add(aq, pl.slice(dq, [1, D], [row, 0]))
                ak = pl.add(ak, pl.slice(dk, [1, D], [row, 0]))
            aq = pl.mul(aq, D ** -.5)
            q = pl.reshape(q, [8, D // 8])
            k = pl.reshape(k, [8, D // 8])
            aq = pl.reshape(aq, [8, D // 8])
            ak = pl.reshape(ak, [8, D // 8])
            qs = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(q, q)))
            ks = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(k, k)))
            qi = pl.rsqrt(pl.add(pl.row_sum(pl.transpose(qs, 0, 1)), 1e-6), high_precision=True)
            ki = pl.rsqrt(pl.add(pl.row_sum(pl.transpose(ks, 0, 1)), 1e-6), high_precision=True)
            qd = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(aq, q)))
            kd = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(ak, k)))
            qdot = pl.row_sum(pl.transpose(qd, 0, 1))
            kdot = pl.row_sum(pl.transpose(kd, 0, 1))
            qgrad = pl.sub(pl.row_expand_mul(aq, qi), pl.row_expand_mul(q, pl.mul(qdot, pl.mul(qi, pl.mul(qi, qi)))))
            kgrad = pl.sub(pl.row_expand_mul(ak, ki), pl.row_expand_mul(k, pl.mul(kdot, pl.mul(ki, pl.mul(ki, ki)))))
            dx[t:t + 1, head * D:head * D + D] = pl.cast(pl.reshape(qgrad, [1, D]), pl.BF16, mode="rint")
            dx[t:t + 1, Q * D + head * D:Q * D + head * D + D] = pl.cast(pl.reshape(kgrad, [1, D]), pl.BF16, mode="rint")
        with pl.spmd(T, name_hint="orthrus_gdn_zero_gate_padding"):
            t = pl.tile.get_block_idx()
            out_g[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.FP32, value=0.)
            out_b[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.FP32, value=0.)
        with pl.spmd(T * (H // 8), name_hint="orthrus_gdn_partial_vjp_reduce"):
            task = pl.tile.get_block_idx()
            t = task // (H // 8)
            group = task % (H // 8)
            row = t * H * NT + group * 8 * NT
            vg = pl.reshape(pl.slice(dg, [8 * NT, 8], [row, 0]), [8, NT * 8])
            vb = pl.reshape(pl.slice(db, [8 * NT, 8], [row, 0]), [8, NT * 8])
            out_g[t:t + 1, group * 8:group * 8 + 8] = pl.reshape(pl.row_sum(vg), [1, 8])
            out_b[t:t + 1, group * 8:group * 8 + 8] = pl.reshape(pl.row_sum(vb), [1, 8])
        with pl.spmd(T * H, name_hint="orthrus_gdn_value_vjp_copy"):
            task = pl.tile.get_block_idx()
            t = task // H
            head = task % H
            value_gradient = pl.slice(dv, [1, D], [t, head * D])
            dx[t:t + 1, 2 * Q * D + head * D:2 * Q * D + head * D + D] = value_gradient
        return dx, out_g, out_b

    @pl.jit.host
    def entry(packed: pl.Tensor, dq: pl.Tensor, dk: pl.Tensor, dv: pl.Tensor, dg: pl.Tensor, db: pl.Tensor,
              dx: pl.Out[pl.Tensor], out_g: pl.Out[pl.Tensor], out_b: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        reduce(packed, dq, dk, dv, dg, db, dx, out_g, out_b, width, device=0)
    return reduce if inline else entry
