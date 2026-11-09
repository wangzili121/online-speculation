"""训练重放的 head 布局、部分 RoPE、历史读取和 softmax VJP。"""


def make_layout(s, kind, *, inline=False, **options):
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    D = s.dimension
    K = s.block
    Q = s.q_heads
    V = s.kv_heads
    C = s.context
    heads = options.get("heads", Q)
    pack = options.get("mode", "pack") == "pack"
    interleaved = options.get("interleaved", False)
    gate = options.get("gate", False)
    stride = D * (2 if interleaved else 1)
    gate_offset = D if gate else 0

    @kernel
    def layout(x: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.spmd(heads * K, name_hint="orthrus_training_head_layout"):
            task = pl.tile.get_block_idx()
            head = task // K
            row = task % K
            if pack:
                col = head * stride + gate_offset
                out[task:task + 1, 0:D] = pl.slice(x, [1, D], [row, col])
            else:
                out[row:row + 1, head * D:head * D + D] = pl.slice(x, [1, D], [task, 0])
        return out

    @kernel
    def q_join(q: pl.Tensor, g: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.spmd(Q * K, name_hint="orthrus_query_gate_backward_join"):
            task = pl.tile.get_block_idx()
            head = task // K
            row = task % K
            out[row:row + 1, head * 2 * D:head * 2 * D + D] = pl.slice(q, [1, D], [task, 0])
            out[row:row + 1, head * 2 * D + D:(head + 1) * 2 * D] = pl.slice(g, [1, D], [task, 0])
        return out

    @kernel
    def join(block: pl.Tensor, history: pl.Tensor, out: pl.Out[pl.Tensor],
             past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        with pl.spmd(V * C, name_hint="orthrus_training_kv_join"):
            task = pl.tile.get_block_idx()
            head = task // C
            row = task % C
            if row < past:
                value = pl.slice(history, [1, D], [task, 0])
            elif row < past + width:
                value = pl.slice(block, [1, D], [head * K + row - past, 0])
            else:
                value = pl.full([1, D], dtype=pl.BF16, value=0.)
            out[task:task + 1, 0:D] = value
        return out

    @kernel
    def reduce_block(x: pl.Tensor, out: pl.Out[pl.Tensor],
                     past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        with pl.spmd(V * K, name_hint="orthrus_gqa_block_backward"):
            task = pl.tile.get_block_idx()
            head = task // K
            row = task % K
            value = pl.full([1, D], dtype=pl.FP32, value=0.)
            if row < width:
                for group in pl.range(Q // V):
                    source = (head * (Q // V) + group) * C + past + row
                    value = pl.add(value, pl.cast(pl.slice(x, [1, D], [source, 0]), pl.FP32))
            out[task:task + 1, 0:D] = pl.cast(value, pl.BF16, mode="rint")
        return out

    if kind == "layout":
        @pl.jit.host
        def entry(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            layout(x, out, device=0)
    elif kind == "q_join":
        @pl.jit.host
        def entry(q: pl.Tensor, g: pl.Tensor, out: pl.Out[pl.Tensor]):
            q_join(q, g, out, device=0)
    elif kind == "join":
        @pl.jit.host
        def entry(block: pl.Tensor, history: pl.Tensor, out: pl.Out[pl.Tensor],
                  past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
            join(block, history, out, past, width, device=0)
    elif kind == "reduce_block":
        @pl.jit.host
        def entry(x: pl.Tensor, out: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
            reduce_block(x, out, past, width, device=0)
    else:
        raise ValueError(kind)
    if inline:
        return {"layout": layout, "q_join": q_join, "join": join, "reduce_block": reduce_block}[kind]
    return entry


def make_rope(s, rows, *, backward=False, inline=False):
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    K = s.block
    D = s.dimension
    R = s.rotary

    @kernel
    def rope(x: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.spmd(rows, name_hint="orthrus_training_partial_rope"):
            row = pl.tile.get_block_idx()
            a = pl.cast(pl.slice(x, [1, R // 2], [row, 0]), pl.FP32)
            b = pl.cast(pl.slice(x, [1, R // 2], [row, R // 2]), pl.FP32)
            c0 = pl.cast(pl.slice(cos, [1, R // 2], [row % K, 0]), pl.FP32)
            c1 = pl.cast(pl.slice(cos, [1, R // 2], [row % K, R // 2]), pl.FP32)
            s0 = pl.cast(pl.slice(sin, [1, R // 2], [row % K, 0]), pl.FP32)
            s1 = pl.cast(pl.slice(sin, [1, R // 2], [row % K, R // 2]), pl.FP32)
            if backward:
                first = pl.add(pl.mul(a, c0), pl.mul(b, s1))
                second = pl.sub(pl.mul(b, c1), pl.mul(a, s0))
            else:
                first = pl.sub(pl.mul(a, c0), pl.mul(b, s0))
                second = pl.add(pl.mul(b, c1), pl.mul(a, s1))
            out[row:row + 1, 0:R // 2] = pl.cast(first, pl.BF16, mode="rint")
            out[row:row + 1, R // 2:R] = pl.cast(second, pl.BF16, mode="rint")
            out[row:row + 1, R:D] = pl.slice(x, [1, D - R], [row, R])
        return out

    @pl.jit.host
    def entry(x: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, out: pl.Out[pl.Tensor]):
        rope(x, cos, sin, out, device=0)
    if inline:
        return rope
    return entry


def make_softmax(rows, columns, *, backward=False, inline=False):
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    C = columns

    @kernel
    def forward(x: pl.Tensor, probability: pl.Out[pl.Tensor], rounded: pl.Out[pl.Tensor],
                past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        with pl.spmd(rows, name_hint="orthrus_training_attention_softmax"):
            row = pl.tile.get_block_idx()
            raw = pl.cast(pl.slice(x, [1, C], [row, 0]), pl.FP32)
            position = pl.arange(0, [1, C], dtype=pl.INT32)
            relative = pl.cast(pl.sub(position, pl.cast(past + width, pl.INT32)), pl.FP32)
            visible = pl.cmp(relative, 0., cmp_type=2)
            raw = pl.add(raw, pl.sub(pl.mul(visible, 10000.), 10000.))
            z = pl.reshape(raw, [8, C // 8])
            maxima = pl.row_expand_add(pl.full([8, 8], dtype=pl.FP32, value=0.), pl.row_max(z))
            maximum = pl.row_max(pl.transpose(maxima, 0, 1))
            exponent = pl.exp(pl.row_expand_sub(z, maximum))
            partial = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(exponent))
            normalizer = pl.row_sum(pl.transpose(partial, 0, 1))
            result = pl.reshape(pl.row_expand_div(exponent, normalizer), [1, C])
            probability[row:row + 1, 0:C] = result
            rounded[row:row + 1, 0:C] = pl.cast(result, pl.BF16, mode="rint")
        return probability, rounded

    @kernel
    def vjp(probability: pl.Tensor, dy: pl.Tensor, dx: pl.Out[pl.Tensor]):
        with pl.spmd(rows, name_hint="orthrus_attention_softmax_backward"):
            row = pl.tile.get_block_idx()
            p = pl.reshape(pl.slice(probability, [1, C], [row, 0]), [8, C // 8])
            d = pl.reshape(pl.cast(pl.slice(dy, [1, C], [row, 0]), pl.FP32), [8, C // 8])
            partial = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(p, d)))
            dot = pl.row_sum(pl.transpose(partial, 0, 1))
            result = pl.mul(p, pl.row_expand_sub(d, dot))
            dx[row:row + 1, 0:C] = pl.cast(pl.reshape(result, [1, C]), pl.BF16, mode="rint")
        return dx

    if backward:
        @pl.jit.host
        def entry(probability: pl.Tensor, dy: pl.Tensor, dx: pl.Out[pl.Tensor]):
            vjp(probability, dy, dx, device=0)
    else:
        @pl.jit.host
        def entry(x: pl.Tensor, probability: pl.Out[pl.Tensor], rounded: pl.Out[pl.Tensor],
                  past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
            forward(x, probability, rounded, past, width, device=0)
    if inline:
        return vjp if backward else forward
    return entry
