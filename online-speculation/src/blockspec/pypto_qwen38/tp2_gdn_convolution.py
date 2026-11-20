"""Orthrus 的只读 K32 卷积：保留历史边界，批量计算块内独立位置。"""


def make_convolution(batch_size, *, backend="sequential", channel_tile=128):
    if batch_size not in (1, 2) or backend not in ("sequential", "parallel"):
        raise ValueError("one/two draft blocks and a supported convolution backend required")
    if channel_tile not in (64, 128):
        raise ValueError("channel tile must be 64 or 128")
    import pypto.language as pl
    C, T = 5120, channel_tile
    tasks = C // T * batch_size
    blocks = min(tasks, 60)
    parallel = backend == "parallel"
    prefix = 3 if parallel else 32

    @pl.jit.inline
    def convolution(projected: pl.Tensor, weight: pl.Tensor, history: pl.Tensor,
                    out: pl.Tensor, slot: pl.Scalar[pl.INT32], slot_second: pl.Scalar[pl.INT32],
                    layer: pl.Scalar[pl.INT32]):
        rows = pl.reshape(history, [32 * 48 * 4, C])
        with pl.spmd(blocks, name_hint="orthrus_tp2_conv"):
            for task in pl.range(pl.tile.get_block_idx(), tasks, blocks):
                seq = task // (C // T)
                col = task % (C // T) * T
                first = seq * 32
                request_slot = slot + seq * (slot_second - slot)
                base = (request_slot * 48 + layer) * 4
                p1 = pl.cast(pl.slice(rows, [1, T], [base + 1, col]), pl.FP32)
                p2 = pl.cast(pl.slice(rows, [1, T], [base + 2, col]), pl.FP32)
                p3 = pl.cast(pl.slice(rows, [1, T], [base + 3, col]), pl.FP32)
                w0 = pl.cast(pl.slice(weight, [1, T], [0, col]), pl.FP32)
                w1 = pl.cast(pl.slice(weight, [1, T], [1, col]), pl.FP32)
                w2 = pl.cast(pl.slice(weight, [1, T], [2, col]), pl.FP32)
                w3 = pl.cast(pl.slice(weight, [1, T], [3, col]), pl.FP32)
                # 前三行需要历史；后面的输入全部来自已经完成的 projected。
                for token in pl.range(prefix):
                    current = pl.cast(pl.slice(projected, [1, T], [first + token, col]), pl.FP32)
                    x = pl.add(pl.add(pl.mul(p1, w0), pl.mul(p2, w1)),
                               pl.add(pl.mul(p3, w2), pl.mul(current, w3)))
                    x = pl.cast(pl.cast(x, pl.BF16, mode="rint"), pl.FP32)
                    out[first + token:first + token + 1, col:col + T] = pl.cast(
                        pl.div(x, pl.add(pl.exp(pl.neg(x)), 1.), high_precision=True), pl.BF16, mode="rint")
                    p1 = p2
                    p2 = p3
                    p3 = current
                if parallel:
                    a = pl.cast(pl.slice(projected, [29, T], [first, col]), pl.FP32)
                    b = pl.cast(pl.slice(projected, [29, T], [first + 1, col]), pl.FP32)
                    c = pl.cast(pl.slice(projected, [29, T], [first + 2, col]), pl.FP32)
                    d = pl.cast(pl.slice(projected, [29, T], [first + 3, col]), pl.FP32)
                    x = pl.add(pl.add(pl.col_expand_mul(a, w0), pl.col_expand_mul(b, w1)),
                               pl.add(pl.col_expand_mul(c, w2), pl.col_expand_mul(d, w3)))
                    x = pl.cast(pl.cast(x, pl.BF16, mode="rint"), pl.FP32)
                    out[first + 3:first + 32, col:col + T] = pl.cast(
                        pl.div(x, pl.add(pl.exp(pl.neg(x)), 1.), high_precision=True), pl.BF16, mode="rint")
        return out

    return convolution


def make_entry(batch_size, *, backend="sequential", channel_tile=128):
    import pypto.language as pl
    convolution = make_convolution(batch_size, backend=backend, channel_tile=channel_tile)

    @pl.jit
    def kernel(projected: pl.Tensor, weight: pl.Tensor, history: pl.Tensor,
               out: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32], slot_second: pl.Scalar[pl.INT32],
               layer: pl.Scalar[pl.INT32]):
        return convolution(projected, weight, history, out, slot, slot_second, layer)

    @pl.jit.host
    def entry(projected: pl.Tensor, weight: pl.Tensor, history: pl.Tensor,
              out: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32], slot_second: pl.Scalar[pl.INT32],
              layer: pl.Scalar[pl.INT32]):
        kernel(projected, weight, history, out, slot, slot_second, layer, device=0)

    return entry
