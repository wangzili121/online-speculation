"""GDN 前后处理：合并 gate 去 padding，直接读取连续 Z 投影做 gated norm。"""

from .baseline import native_module


def make_auxiliary(source, *, backend="fused", rows=16):
    if backend not in {"native", "fused"}:
        raise ValueError("auxiliary backend must be native or fused")
    if type(rows) is not int or rows not in {16, 32} or rows == 32 and backend == "native":
        raise ValueError("rows must be 16 or 32; native reference uses 16 rows")
    import pypto.language as pl
    model = native_module(source, "model")
    linear = model.linear_hidden_projection_inline
    T, FLAT = rows, rows * 48
    if rows == 32:
        from .projection_programs import make_projection
        linear, _ = make_projection(source, 6144, 5120)

    @pl.jit.inline
    def padded_gates(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
                     decay: pl.Tensor, beta: pl.Tensor):
        with pl.spmd(1, name_hint="orthrus_decay_beta"):
            row = pl.tile.get_block_idx() * 16
            av = pl.cast(pl.slice(a, [16, 64], [row, 0]), pl.FP32)
            bv = pl.cast(pl.slice(b, [16, 64], [row, 0]), pl.FP32)
            bias = pl.cast(pl.slice(dt_bias, [1, 64], [0, 0]), pl.FP32)
            parameter = pl.slice(a_log, [1, 64], [0, 0])
            softplus = pl.log(pl.add(pl.exp(pl.col_expand_add(av, bias)), 1.))
            decay[0:16, 0:64] = pl.mul(pl.col_expand_mul(softplus, pl.exp(parameter)), -1.)
            beta[0:16, 0:64] = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(bv)), 1.)), pl.BF16, mode="rint")
        return decay, beta

    @pl.jit.inline
    def native_gates(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
                     decay: pl.Tensor, beta: pl.Tensor):
        gp = pl.create_tensor([16, 64], dtype=pl.FP32)
        bp = pl.create_tensor([16, 64], dtype=pl.BF16)
        gp, bp = padded_gates(a, b, a_log, dt_bias, gp, bp)
        with pl.spmd(16, name_hint="orthrus_unpad_gates"):
            row = pl.tile.get_block_idx()
            decay[row:row + 1, 0:48] = pl.slice(gp, [1, 48], [row, 0])
            beta[row:row + 1, 0:48] = pl.cast(pl.slice(bp, [1, 48], [row, 0]), pl.FP32)
        return decay, beta

    @pl.jit.inline
    def fused_gates(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
                    decay: pl.Tensor, beta: pl.Tensor):
        with pl.spmd(1, name_hint="orthrus_gdn_compact_gates"):
            row = pl.tile.get_block_idx() * T
            av = pl.cast(pl.slice(a, [T, 64], [row, 0]), pl.FP32)
            bv = pl.cast(pl.slice(b, [T, 64], [row, 0]), pl.FP32)
            bias = pl.cast(pl.slice(dt_bias, [1, 64], [0, 0]), pl.FP32)
            parameter = pl.slice(a_log, [1, 64], [0, 0])
            softplus = pl.log(pl.add(pl.exp(pl.col_expand_add(av, bias)), 1.))
            g = pl.mul(pl.col_expand_mul(softplus, pl.exp(parameter)), -1.)
            # beta 必须保留原生 sigmoid 的 BF16 舍入，再转 FP32 给递推。
            b16 = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(bv)), 1.)), pl.BF16, mode="rint")
            decay[0:T, 0:48] = pl.slice(g, [T, 48], [0, 0])
            beta[0:T, 0:48] = pl.cast(pl.slice(b16, [T, 48], [0, 0]), pl.FP32)
        return decay, beta

    @pl.jit.inline
    def fused_tail(core: pl.Tensor, z: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor,
                   gated: pl.Tensor, attention: pl.Tensor):
        # Z 本来就是连续的 [T,48*128]，reshape 即可，不需要逐 head 复制。
        zr = pl.reshape(z, [FLAT, 128])
        cr = pl.reshape(core, [FLAT, 128])
        with pl.spmd(FLAT // 16, name_hint="orthrus_gdn_gated_norm"):
            row = pl.tile.get_block_idx() * 16
            x = pl.cast(pl.slice(cr, [16, 128], [row, 0]), pl.FP32)
            gate = pl.cast(pl.slice(zr, [16, 128], [row, 0]), pl.FP32)
            weight = pl.cast(pl.slice(norm_weight, [1, 128], [0, 0]), pl.FP32)
            inverse = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(x, x)), 1. / 128), 1e-6), high_precision=True)
            # 保留归一化、乘权重两处原生 BF16 舍入。
            normalized = pl.cast(pl.row_expand_mul(x, inverse), pl.BF16, mode="rint")
            weighted = pl.cast(pl.col_expand_mul(pl.cast(normalized, pl.FP32), weight), pl.BF16, mode="rint")
            silu = pl.div(gate, pl.add(pl.exp(pl.neg(gate)), 1.))
            gated[row:row + 16, 0:128] = pl.cast(pl.mul(pl.cast(weighted, pl.FP32), silu), pl.BF16, mode="rint")
        projection = pl.reshape(gated, [T, 6144])
        attention = linear(projection, out_weight, attention)
        return gated, attention

    gates = fused_gates if backend == "fused" else native_gates
    tail = fused_tail if backend == "fused" else model.delta_window16_attention_tail_window16_inline
    return gates, tail


def make_auxiliary_entries(source, *, backend="fused", rows=16):
    gates, tail = make_auxiliary(source, backend=backend, rows=rows)
    from .gdn_convolution_programs import make_convolution
    convolution = make_convolution(source, backend=backend)
    import pypto.language as pl

    @pl.jit
    def gate_program(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
                     decay: pl.Out[pl.Tensor], beta: pl.Out[pl.Tensor]):
        return gates(a, b, a_log, dt_bias, decay, beta)

    @pl.jit.host
    def gate_entry(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
                   decay: pl.Out[pl.Tensor], beta: pl.Out[pl.Tensor]):
        gate_program(a, b, a_log, dt_bias, decay, beta, device=0)

    @pl.jit
    def tail_program(core: pl.Tensor, z: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor,
                     gated: pl.Out[pl.Tensor], attention: pl.Out[pl.Tensor]):
        return tail(core, z, norm_weight, out_weight, gated, attention)

    @pl.jit.host
    def tail_entry(core: pl.Tensor, z: pl.Tensor, norm_weight: pl.Tensor, out_weight: pl.Tensor,
                   gated: pl.Out[pl.Tensor], attention: pl.Out[pl.Tensor]):
        tail_program(core, z, norm_weight, out_weight, gated, attention, device=0)

    @pl.jit
    def conv_program(projected: pl.Tensor, weight: pl.Tensor, history: pl.Tensor, output: pl.Out[pl.Tensor]):
        return convolution(projected, weight, history, output, 0)

    @pl.jit.host
    def conv_entry(projected: pl.Tensor, weight: pl.Tensor, history: pl.Tensor, output: pl.Out[pl.Tensor]):
        conv_program(projected, weight, history, output, device=0)

    return dict(gates=gate_entry, tail=tail_entry, conv=conv_entry)
