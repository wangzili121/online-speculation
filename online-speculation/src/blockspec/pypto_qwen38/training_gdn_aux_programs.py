"""GDN 的卷积、门控及 gated RMSNorm 前向/显式反向。"""


def make_convolution(shape, *, backward=False, inline=False):
    import pypto.language as pl
    C, T = shape.channels, shape.rows
    if C % 128:
        raise ValueError("128-aligned convolution channels required")
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def forward(x: pl.Tensor, weight: pl.Tensor, history: pl.Tensor,
                y: pl.Out[pl.Tensor], pre: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        with pl.spmd(C // 128, name_hint="orthrus_gdn_conv_saved_forward"):
            col = pl.tile.get_block_idx() * 128
            p1 = pl.cast(pl.slice(history, [1, 128], [1, col]), pl.FP32)
            p2 = pl.cast(pl.slice(history, [1, 128], [2, col]), pl.FP32)
            p3 = pl.cast(pl.slice(history, [1, 128], [3, col]), pl.FP32)
            w0 = pl.cast(pl.slice(weight, [1, 128], [0, col]), pl.FP32)
            w1 = pl.cast(pl.slice(weight, [1, 128], [1, col]), pl.FP32)
            w2 = pl.cast(pl.slice(weight, [1, 128], [2, col]), pl.FP32)
            w3 = pl.cast(pl.slice(weight, [1, 128], [3, col]), pl.FP32)
            for t in pl.range(width):
                current = pl.cast(pl.slice(x, [1, 128], [t, col]), pl.FP32)
                z16 = pl.cast(pl.add(pl.add(pl.mul(p1, w0), pl.mul(p2, w1)),
                                    pl.add(pl.mul(p3, w2), pl.mul(current, w3))), pl.BF16, mode="rint")
                pre[t:t + 1, col:col + 128] = z16
                z = pl.cast(z16, pl.FP32)
                y[t:t + 1, col:col + 128] = pl.cast(pl.div(z, pl.add(pl.exp(pl.neg(z)), 1.)), pl.BF16, mode="rint")
                p1 = p2
                p2 = p3
                p3 = current
            for t in pl.range(width, T):
                pre[t:t + 1, col:col + 128] = pl.full([1, 128], dtype=pl.BF16, value=0.)
                y[t:t + 1, col:col + 128] = pl.full([1, 128], dtype=pl.BF16, value=0.)
        return y, pre

    @kernel
    def vjp(x: pl.Tensor, weight: pl.Tensor, history: pl.Tensor, pre: pl.Tensor, dy: pl.Tensor,
            dx: pl.Out[pl.Tensor], dw: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        with pl.spmd(C // 128, name_hint="orthrus_gdn_causal_conv_vjp"):
            col = pl.tile.get_block_idx() * 128
            p1 = pl.cast(pl.slice(history, [1, 128], [1, col]), pl.FP32)
            p2 = pl.cast(pl.slice(history, [1, 128], [2, col]), pl.FP32)
            p3 = pl.cast(pl.slice(history, [1, 128], [3, col]), pl.FP32)
            a0 = pl.full([1, 128], dtype=pl.FP32, value=0.)
            a1 = pl.full([1, 128], dtype=pl.FP32, value=0.)
            a2 = pl.full([1, 128], dtype=pl.FP32, value=0.)
            a3 = pl.full([1, 128], dtype=pl.FP32, value=0.)
            for t in pl.range(width):
                total = pl.full([1, 128], dtype=pl.FP32, value=0.)
                for future in pl.range(4):
                    if t + future < width:
                        z = pl.cast(pl.slice(pre, [1, 128], [t + future, col]), pl.FP32)
                        upstream = pl.cast(pl.slice(dy, [1, 128], [t + future, col]), pl.FP32)
                        sig = pl.recip(pl.add(pl.exp(pl.neg(z)), 1.))
                        dz = pl.cast(pl.cast(pl.mul(pl.mul(upstream, sig),
                            pl.add(pl.mul(z, pl.add(pl.neg(sig), 1.)), 1.)), pl.BF16, mode="rint"), pl.FP32)
                        w = pl.cast(pl.slice(weight, [1, 128], [3 - future, col]), pl.FP32)
                        total = pl.add(total, pl.mul(dz, w))
                dx[t:t + 1, col:col + 128] = pl.cast(total, pl.BF16, mode="rint")
                z = pl.cast(pl.slice(pre, [1, 128], [t, col]), pl.FP32)
                upstream = pl.cast(pl.slice(dy, [1, 128], [t, col]), pl.FP32)
                sig = pl.recip(pl.add(pl.exp(pl.neg(z)), 1.))
                dz = pl.cast(pl.cast(pl.mul(pl.mul(upstream, sig),
                    pl.add(pl.mul(z, pl.add(pl.neg(sig), 1.)), 1.)), pl.BF16, mode="rint"), pl.FP32)
                current = pl.cast(pl.slice(x, [1, 128], [t, col]), pl.FP32)
                a0 = pl.add(a0, pl.mul(dz, p1))
                a1 = pl.add(a1, pl.mul(dz, p2))
                a2 = pl.add(a2, pl.mul(dz, p3))
                a3 = pl.add(a3, pl.mul(dz, current))
                p1 = p2
                p2 = p3
                p3 = current
            for t in pl.range(width, T):
                dx[t:t + 1, col:col + 128] = pl.full([1, 128], dtype=pl.BF16, value=0.)
            dw[0:1, col:col + 128] = pl.cast(pl.cast(a0, pl.BF16, mode="rint"), pl.FP32)
            dw[1:2, col:col + 128] = pl.cast(pl.cast(a1, pl.BF16, mode="rint"), pl.FP32)
            dw[2:3, col:col + 128] = pl.cast(pl.cast(a2, pl.BF16, mode="rint"), pl.FP32)
            dw[3:4, col:col + 128] = pl.cast(pl.cast(a3, pl.BF16, mode="rint"), pl.FP32)
        return dx, dw

    if backward:
        @pl.jit.host
        def entry(x: pl.Tensor, weight: pl.Tensor, history: pl.Tensor, pre: pl.Tensor, dy: pl.Tensor,
                  dx: pl.Out[pl.Tensor], dw: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
            vjp(x, weight, history, pre, dy, dx, dw, width, device=0)
    else:
        @pl.jit.host
        def entry(x: pl.Tensor, weight: pl.Tensor, history: pl.Tensor,
                  y: pl.Out[pl.Tensor], pre: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
            forward(x, weight, history, y, pre, width, device=0)
    return (vjp if backward else forward) if inline else entry


def make_gates(shape, *, backward=False, inline=False):
    import pypto.language as pl
    H, T = shape.heads, shape.rows
    HF = float(H)
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def forward(a: pl.Tensor, b: pl.Tensor, al: pl.Tensor, bias: pl.Tensor,
                g: pl.Out[pl.Tensor], beta: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        with pl.spmd(T, name_hint="orthrus_gdn_saved_gates"):
            t = pl.tile.get_block_idx()
            if t < width:
                av = pl.cast(pl.slice(a, [1, 64], [t, 0]), pl.FP32)
                bv = pl.cast(pl.slice(b, [1, 64], [t, 0]), pl.FP32)
                z = pl.add(av, pl.cast(pl.slice(bias, [1, 64], [0, 0]), pl.FP32))
                active = pl.cmp(pl.cast(pl.arange(0, [1, 64], dtype=pl.INT32), pl.FP32), HF, cmp_type=2)
                value = pl.neg(pl.mul(pl.exp(pl.slice(al, [1, 64], [0, 0])), pl.log(pl.add(pl.exp(z), 1.))))
                g[t:t + 1, 0:64] = pl.mul(value, active)
                probability = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(bv)), 1.)), pl.BF16, mode="rint")
                beta[t:t + 1, 0:64] = pl.mul(pl.cast(probability, pl.FP32), active)
            else:
                g[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.FP32, value=0.)
                beta[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.FP32, value=0.)
        return g, beta

    @kernel
    def vjp(a: pl.Tensor, b: pl.Tensor, al: pl.Tensor, bias: pl.Tensor, dg: pl.Tensor, db: pl.Tensor,
            da: pl.Out[pl.Tensor], d_b: pl.Out[pl.Tensor], partial_al: pl.Out[pl.Tensor],
            partial_bias: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        with pl.spmd(T, name_hint="orthrus_gdn_gate_vjp"):
            t = pl.tile.get_block_idx()
            if t < width:
                av = pl.cast(pl.slice(a, [1, 64], [t, 0]), pl.FP32)
                bv = pl.cast(pl.slice(b, [1, 64], [t, 0]), pl.FP32)
                z = pl.add(av, pl.cast(pl.slice(bias, [1, 64], [0, 0]), pl.FP32))
                factor = pl.neg(pl.mul(pl.exp(pl.slice(al, [1, 64], [0, 0])), pl.slice(dg, [1, 64], [t, 0])))
                active = pl.cmp(pl.cast(pl.arange(0, [1, 64], dtype=pl.INT32), pl.FP32), HF, cmp_type=2)
                factor = pl.mul(factor, active)
                dz = pl.div(factor, pl.add(pl.exp(pl.neg(z)), 1.))
                sig = pl.recip(pl.add(pl.exp(pl.neg(bv)), 1.))
                delta = pl.cast(pl.cast(pl.slice(db, [1, 64], [t, 0]), pl.BF16, mode="rint"), pl.FP32)
                delta = pl.mul(pl.mul(pl.mul(delta, sig), pl.add(pl.neg(sig), 1.)), active)
                da[t:t + 1, 0:64] = pl.cast(dz, pl.BF16, mode="rint")
                d_b[t:t + 1, 0:64] = pl.cast(delta, pl.BF16, mode="rint")
                partial_al[t:t + 1, 0:64] = pl.mul(factor, pl.log(pl.add(pl.exp(z), 1.)))
                partial_bias[t:t + 1, 0:64] = dz
            else:
                da[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.BF16, value=0.)
                d_b[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.BF16, value=0.)
                partial_al[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.FP32, value=0.)
                partial_bias[t:t + 1, 0:64] = pl.full([1, 64], dtype=pl.FP32, value=0.)
            da[t:t + 1, 64:128] = pl.full([1, 64], dtype=pl.BF16, value=0.)
            d_b[t:t + 1, 64:128] = pl.full([1, 64], dtype=pl.BF16, value=0.)
        return da, d_b, partial_al, partial_bias

    if backward:
        @pl.jit.host
        def entry(a: pl.Tensor, b: pl.Tensor, al: pl.Tensor, bias: pl.Tensor, dg: pl.Tensor, db: pl.Tensor,
                  da: pl.Out[pl.Tensor], d_b: pl.Out[pl.Tensor], partial_al: pl.Out[pl.Tensor],
                  partial_bias: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
            vjp(a, b, al, bias, dg, db, da, d_b, partial_al, partial_bias, width, device=0)
    else:
        @pl.jit.host
        def entry(a: pl.Tensor, b: pl.Tensor, al: pl.Tensor, bias: pl.Tensor,
                  g: pl.Out[pl.Tensor], beta: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
            forward(a, b, al, bias, g, beta, width, device=0)
    return (vjp if backward else forward) if inline else entry


def make_gate_reduce(shape, *, inline=False):
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def reduce(pa: pl.Tensor, pb: pl.Tensor, da: pl.Out[pl.Tensor], db: pl.Out[pl.Tensor]):
        with pl.spmd(1, name_hint="orthrus_gdn_gate_parameter_reduce"):
            row = pl.tile.get_block_idx()
            da[row:row + 1, 0:64] = pl.col_sum(pa)
            db[row:row + 1, 0:64] = pl.cast(pl.cast(pl.col_sum(pb), pl.BF16, mode="rint"), pl.FP32)
        return da, db

    @pl.jit.host
    def entry(pa: pl.Tensor, pb: pl.Tensor, da: pl.Out[pl.Tensor], db: pl.Out[pl.Tensor]):
        reduce(pa, pb, da, db, device=0)
    return reduce if inline else entry


def make_tail(shape, *, backward=False, inline=False):
    import pypto.language as pl
    R, D = shape.rows * shape.heads, shape.dimension
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def forward(core: pl.Tensor, z: pl.Tensor, weight: pl.Tensor, output: pl.Out[pl.Tensor],
                normed: pl.Out[pl.Tensor], weighted: pl.Out[pl.Tensor], inverse: pl.Out[pl.Tensor]):
        inputs = pl.reshape(core, [R, D])
        gates = pl.reshape(z, [R, D])
        result = pl.reshape(output, [R, D])
        with pl.spmd(R, name_hint="orthrus_gdn_saved_gated_rms"):
            row = pl.tile.get_block_idx()
            x = pl.reshape(pl.cast(pl.slice(inputs, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            gate = pl.reshape(pl.cast(pl.slice(gates, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            gamma = pl.reshape(pl.cast(pl.slice(weight, [1, D], [0, 0]), pl.FP32), [8, D // 8])
            sums = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(x, x)))
            inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.transpose(sums, 0, 1)), 1. / D), 1e-6), high_precision=True)
            normalized = pl.cast(pl.reshape(pl.row_expand_mul(x, inv), [1, D]), pl.BF16, mode="rint")
            value = pl.cast(pl.mul(pl.cast(normalized, pl.FP32), pl.reshape(gamma, [1, D])), pl.BF16, mode="rint")
            silu = pl.div(gate, pl.add(pl.exp(pl.neg(gate)), 1.))
            result[row:row + 1, 0:D] = pl.cast(pl.mul(pl.cast(value, pl.FP32), pl.reshape(silu, [1, D])), pl.BF16, mode="rint")
            normed[row:row + 1, 0:D] = pl.reshape(normalized, [1, D])
            weighted[row:row + 1, 0:D] = pl.reshape(value, [1, D])
            inverse[row:row + 1, 0:8] = pl.reshape(inv, [1, 8])
        return output, normed, weighted, inverse

    @kernel
    def vjp(core: pl.Tensor, z: pl.Tensor, weight: pl.Tensor, weighted: pl.Tensor, inverse: pl.Tensor,
            upstream: pl.Tensor, dx: pl.Out[pl.Tensor], dz: pl.Out[pl.Tensor], dweighted: pl.Out[pl.Tensor]):
        inputs = pl.reshape(core, [R, D])
        gates = pl.reshape(z, [R, D])
        dy = pl.reshape(upstream, [R, D])
        dinputs = pl.reshape(dx, [R, D])
        dgates = pl.reshape(dz, [R, D])
        with pl.spmd(R, name_hint="orthrus_gdn_gated_rms_vjp"):
            row = pl.tile.get_block_idx()
            x = pl.reshape(pl.cast(pl.slice(inputs, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            gate = pl.reshape(pl.cast(pl.slice(gates, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            grad = pl.reshape(pl.cast(pl.slice(dy, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            gamma = pl.reshape(pl.cast(pl.slice(weight, [1, D], [0, 0]), pl.FP32), [8, D // 8])
            value = pl.reshape(pl.cast(pl.slice(weighted, [1, D], [row, 0]), pl.FP32), [8, D // 8])
            inv = pl.reshape(pl.slice(inverse, [1, 8], [row, 0]), [8, 1])
            sig = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.))
            dw = pl.cast(pl.reshape(pl.mul(pl.mul(grad, gate), sig), [1, D]), pl.BF16, mode="rint")
            dweighted[row:row + 1, 0:D] = pl.reshape(dw, [1, D])
            delta = pl.cast(pl.cast(pl.mul(pl.cast(dw, pl.FP32), pl.reshape(gamma, [1, D])), pl.BF16, mode="rint"), pl.FP32)
            delta = pl.reshape(delta, [8, D // 8])
            sums = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(delta, x)))
            dot = pl.row_sum(pl.transpose(sums, 0, 1))
            correction = pl.mul(pl.mul(dot, pl.mul(inv, pl.mul(inv, inv))), 1. / D)
            dinputs[row:row + 1, 0:D] = pl.cast(pl.reshape(pl.sub(pl.row_expand_mul(delta, inv),
                pl.row_expand_mul(x, correction)), [1, D]), pl.BF16, mode="rint")
            derivative = pl.mul(sig, pl.add(pl.mul(gate, pl.add(pl.neg(sig), 1.)), 1.))
            dgates[row:row + 1, 0:D] = pl.cast(pl.reshape(pl.mul(pl.mul(grad, value), derivative), [1, D]), pl.BF16, mode="rint")
        return dx, dz, dweighted

    if backward:
        @pl.jit.host
        def entry(core: pl.Tensor, z: pl.Tensor, weight: pl.Tensor, weighted: pl.Tensor, inverse: pl.Tensor,
                  upstream: pl.Tensor, dx: pl.Out[pl.Tensor], dz: pl.Out[pl.Tensor], dweighted: pl.Out[pl.Tensor]):
            vjp(core, z, weight, weighted, inverse, upstream, dx, dz, dweighted, device=0)
    else:
        @pl.jit.host
        def entry(core: pl.Tensor, z: pl.Tensor, weight: pl.Tensor, output: pl.Out[pl.Tensor],
                  normed: pl.Out[pl.Tensor], weighted: pl.Out[pl.Tensor], inverse: pl.Out[pl.Tensor]):
            forward(core, z, weight, output, normed, weighted, inverse, device=0)
    return (vjp if backward else forward) if inline else entry


def make_tail_weight(shape, *, inline=False):
    import pypto.language as pl
    R, D = shape.rows * shape.heads, shape.dimension
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def weight_grad(normed: pl.Tensor, delta: pl.Tensor, dw: pl.Out[pl.Tensor]):
        with pl.spmd(D // 64, name_hint="orthrus_gdn_norm_parameter_vjp"):
            col = pl.tile.get_block_idx() * 64
            total = pl.full([1, 64], dtype=pl.FP32, value=0.)
            for step in pl.range(R // 32):
                row = step * 32
                a = pl.cast(pl.slice(normed, [32, 64], [row, col]), pl.FP32)
                b = pl.cast(pl.slice(delta, [32, 64], [row, col]), pl.FP32)
                total = pl.add(total, pl.col_sum(pl.mul(a, b)))
            dw[0:1, col:col + 64] = pl.cast(pl.cast(total, pl.BF16, mode="rint"), pl.FP32)
        return dw

    @pl.jit.host
    def entry(normed: pl.Tensor, delta: pl.Tensor, dw: pl.Out[pl.Tensor]):
        weight_grad(normed, delta, dw, device=0)
    return weight_grad if inline else entry
