"""K32 因果卷积：直接读取 AR 历史，输出本轮特征，不提交卷积状态。"""

from .baseline import native_module


def make_convolution(source, *, backend="fused", history_matrices=1):
    if backend not in {"native", "fused"} or type(history_matrices) is not int or history_matrices < 1:
        raise ValueError("native/fused convolution and positive history_matrices required")
    import pypto.language as pl
    native = native_module(source, "model").causal_conv_causal_conv_silu_inline
    C = 10240

    @pl.jit.inline
    def readonly(projected: pl.Tensor, weight: pl.Tensor, history: pl.Tensor, output: pl.Tensor,
                 base: pl.Scalar[pl.INT32]):
        rows = pl.reshape(history, [history_matrices * 4, C])
        with pl.spmd(80, name_hint="orthrus_gdn_readonly_conv32"):
            column = pl.tile.get_block_idx() * 128
            p1 = pl.cast(pl.slice(rows, [1, 128], [base + 1, column]), pl.FP32)
            p2 = pl.cast(pl.slice(rows, [1, 128], [base + 2, column]), pl.FP32)
            p3 = pl.cast(pl.slice(rows, [1, 128], [base + 3, column]), pl.FP32)
            w0 = pl.cast(pl.slice(weight, [1, 128], [0, column]), pl.FP32)
            w1 = pl.cast(pl.slice(weight, [1, 128], [1, column]), pl.FP32)
            w2 = pl.cast(pl.slice(weight, [1, 128], [2, column]), pl.FP32)
            w3 = pl.cast(pl.slice(weight, [1, 128], [3, column]), pl.FP32)
            for token in pl.range(32):
                current = pl.cast(pl.slice(projected, [1, 128], [token, column]), pl.FP32)
                x = pl.add(pl.add(pl.mul(p1, w0), pl.mul(p2, w1)), pl.add(pl.mul(p3, w2), pl.mul(current, w3)))
                x = pl.cast(pl.cast(x, pl.BF16, mode="rint"), pl.FP32)
                output[token:token + 1, column:column + 128] = pl.cast(
                    pl.div(x, pl.add(pl.exp(pl.neg(x)), 1.), high_precision=True), pl.BF16, mode="rint")
                p1 = p2
                p2 = p3
                p3 = current
        return output

    @pl.jit.inline
    def native_copy(projected: pl.Tensor, weight: pl.Tensor, history: pl.Tensor, output: pl.Tensor,
                    base: pl.Scalar[pl.INT32]):
        rows = pl.reshape(history, [history_matrices * 4, C])
        state = pl.create_tensor([4, C], dtype=pl.BF16)
        with pl.spmd(320, name_hint="orthrus_copy_conv"):
            task = pl.tile.get_block_idx()
            row = task // 80
            column = task % 80 * 128
            state[row:row + 1, column:column + 128] = pl.slice(rows, [1, 128], [base + row, column])
        output, state = native(projected, weight, state, output, 32)
        return output

    return readonly if backend == "fused" else native_copy
