"""draft K32 的共享 MLP，直接引用主模型权重，不修改目标模型执行。"""

from .projection_programs import make_norm, make_projection


def make_shared_tail(source):
    import pypto.language as pl
    norm, _ = make_norm(source)
    up_linear, _ = make_projection(source, 5120, 17408)
    down_linear, _ = make_projection(source, 17408, 5120)

    @pl.jit.inline
    def residual(left: pl.Tensor, right: pl.Tensor, result: pl.Tensor):
        with pl.spmd(32, name_hint="orthrus_residual32"):
            row = pl.tile.get_block_idx()
            a = pl.cast(pl.slice(left, [1, 5120], [row, 0]), pl.FP32)
            b = pl.cast(pl.slice(right, [1, 5120], [row, 0]), pl.FP32)
            result[row:row + 1, 0:5120] = pl.cast(pl.add(a, b), pl.BF16, mode="rint")
        return result

    @pl.jit.inline
    def tail(hidden: pl.Tensor, attention: pl.Tensor, post_gamma: pl.Tensor,
             gate_weight: pl.Tensor, up_weight: pl.Tensor, down_weight: pl.Tensor,
             result: pl.Tensor, layer: pl.Scalar[pl.INT32]):
        gamma: pl.Tensor[[1, 5120], pl.FP32] = pl.slice(post_gamma, [1, 5120], [layer, 0])
        wg: pl.Tensor[[17408, 5120], pl.BF16] = pl.slice(gate_weight, [17408, 5120], [layer * 17408, 0])
        wu: pl.Tensor[[17408, 5120], pl.BF16] = pl.slice(up_weight, [17408, 5120], [layer * 17408, 0])
        wd: pl.Tensor[[5120, 17408], pl.BF16] = pl.slice(down_weight, [5120, 17408], [layer * 5120, 0])
        skip = pl.create_tensor([32, 5120], dtype=pl.BF16)
        normalized = pl.create_tensor([32, 5120], dtype=pl.BF16)
        skip = residual(hidden, attention, skip)
        normalized = norm(skip, gamma, normalized)
        gate = pl.create_tensor([32, 17408], dtype=pl.BF16)
        up = pl.create_tensor([32, 17408], dtype=pl.BF16)
        middle = pl.create_tensor([32, 17408], dtype=pl.BF16)
        gate = up_linear(normalized, wg, gate)
        up = up_linear(normalized, wu, up)
        with pl.spmd(136, name_hint="orthrus_swiglu32"):
            column = pl.tile.get_block_idx() * 128
            g = pl.cast(pl.slice(gate, [32, 128], [0, column]), pl.FP32)
            u = pl.cast(pl.slice(up, [32, 128], [0, column]), pl.FP32)
            # 与原生相同：SiLU 和乘 up 后各作一次 BF16 舍入。
            silu = pl.cast(pl.div(g, pl.add(pl.exp(pl.neg(g)), 1.)), pl.BF16, mode="rint")
            middle[0:32, column:column + 128] = pl.cast(pl.mul(pl.cast(silu, pl.FP32), u), pl.BF16, mode="rint")
        output = pl.create_tensor([32, 5120], dtype=pl.BF16)
        output = down_linear(middle, wd, output)
        result = residual(skip, output, result)
        return result

    return tail
