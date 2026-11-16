"""Activation derivatives for the frozen GLM expert layers."""


def make_swiglu_backward(rows, width, limit):
    """Differentiate the clamped SwiGLU used in GLM shared and routed experts."""
    import pypto.language as pl

    tile = 64
    if rows < 1 or width < tile or width % tile or limit < 0:
        raise ValueError("invalid GLM SwiGLU backward geometry")
    parts = width // tile
    clamped = limit > 0

    @pl.jit
    def backward(
        gate: pl.Tensor[[rows, width], pl.BF16], up: pl.Tensor[[rows, width], pl.BF16],
        upstream: pl.Tensor[[rows, width], pl.FP32],
        d_gate: pl.Out[pl.Tensor[[rows, width], pl.FP32]], d_up: pl.Out[pl.Tensor[[rows, width], pl.FP32]],
    ):
        for task in pl.spmd(rows * parts, name_hint="orthrus_glm53_swiglu_backward"):
            row = task // parts
            column = task % parts * tile
            gate_raw = pl.cast(gate[row:row + 1, column:column + tile], pl.FP32)
            up_raw = pl.cast(up[row:row + 1, column:column + tile], pl.FP32)
            if clamped:
                g = pl.minimum(gate_raw, limit)
                u = pl.minimum(pl.maximum(up_raw, -limit), limit)
            else:
                g, u = gate_raw, up_raw
            # exp(-abs(g)) stays finite at both tails of SiLU.
            e = pl.exp(pl.neg(pl.abs(g)))
            nonnegative = pl.cast(pl.cast(pl.cmp(g, pl.abs(g), cmp_type=0), pl.INT32), pl.FP32)
            numerator = pl.add(e, pl.mul(nonnegative, pl.add(pl.neg(e), 1.0)))
            sigmoid = pl.div(numerator, pl.add(e, 1.0))
            d_silu = pl.mul(sigmoid, pl.add(pl.mul(g, pl.add(pl.neg(sigmoid), 1.0)), 1.0))
            dy = upstream[row:row + 1, column:column + tile]
            dg = pl.mul(pl.mul(dy, u), d_silu)
            du = pl.mul(pl.mul(dy, g), sigmoid)
            if clamped:
                gate_pass = pl.cast(pl.cast(pl.cmp(gate_raw, g, cmp_type=0), pl.INT32), pl.FP32)
                up_pass = pl.cast(pl.cast(pl.cmp(up_raw, u, cmp_type=0), pl.INT32), pl.FP32)
                dg = pl.mul(dg, gate_pass)
                du = pl.mul(du, up_pass)
            d_gate[row:row + 1, column:column + tile] = dg
            d_up[row:row + 1, column:column + tile] = du
        return d_gate, d_up

    @pl.jit.host
    def entry(gate: pl.Tensor, up: pl.Tensor, upstream: pl.Tensor, d_gate: pl.Out[pl.Tensor], d_up: pl.Out[pl.Tensor]):
        backward(gate, up, upstream, d_gate, d_up, device=0)

    return entry


def make_frozen_int8_linear_backward(rows, in_features, out_features):
    """Compute dX with frozen row-scaled weights and an activation-quantization STE."""
    import pypto.language as pl

    columns, reduction = 64, 128
    if rows < 16 or rows % 16 or in_features < columns or in_features % columns:
        raise ValueError("invalid GLM expert input gradient geometry")
    if out_features < reduction or out_features % reduction:
        raise ValueError("expert reduction width must be 128-aligned")

    @pl.jit
    def backward(
        upstream: pl.Tensor[[rows, out_features], pl.FP32],
        weight: pl.Tensor[[out_features, in_features], pl.INT8], scale: pl.Tensor[[out_features], pl.FP32],
        gradient: pl.Out[pl.Tensor[[rows, in_features], pl.FP32]],
    ):
        for piece in pl.spmd(in_features // columns, name_hint="orthrus_glm53_frozen_linear_backward"):
            col = piece * columns
            accum = pl.create_tensor([rows, columns], dtype=pl.FP32)
            for start in pl.pipeline(0, out_features, reduction, stage=2):
                dy = pl.slice(upstream, [rows, reduction], [0, start])
                channel_scale = pl.reshape(scale[start:start + reduction], [1, reduction])
                scaled = pl.col_expand_mul(dy, channel_scale)
                high = pl.cast(scaled, pl.BF16, mode="rint")
                low = pl.cast(pl.sub(scaled, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
                packed = pl.slice(weight, [reduction, columns], [start, col])
                unpacked = pl.cast(pl.cast(pl.cast(packed, pl.FP16), pl.FP32), pl.BF16, mode="rint")
                accum = pl.matmul_acc(accum, high, unpacked, init_cond=start == 0)
                accum = pl.matmul_acc(accum, low, unpacked)
            gradient[:, col:col + columns] = accum
        return gradient

    @pl.jit.host
    def entry(upstream: pl.Tensor, weight: pl.Tensor, scale: pl.Tensor, gradient: pl.Out[pl.Tensor]):
        backward(upstream, weight, scale, gradient, device=0)

    return entry
