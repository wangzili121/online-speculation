"""K32 draft 投影与 input norm；保留原生 16+16 供独立对照。"""

from .baseline import native_module


def make_projection(source, inputs, outputs, *, tile=None, native=False):
    if any(type(n) is not int or n < 1 for n in (inputs, outputs)):
        raise ValueError("positive input/output dimensions required")
    tile = (64 if outputs <= 1024 else 128) if tile is None else tile
    if type(tile) is not int or tile not in {64, 128} or inputs % 512 or outputs % tile or native and outputs % 128:
        raise ValueError("complete 512 input tiles and 64/128 output tiles required")
    import pypto.language as pl
    original = native_module(source, "model").linear_hidden_projection_inline
    N = min(tile, outputs)

    @pl.jit.inline
    def block(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
        with pl.spmd(outputs // N, name_hint="orthrus_projection32", optimizations=[pl.cross_core_slot(slot_num=1)]):
            column = pl.tile.get_block_idx() * N
            accumulator = pl.create_tensor([32, N], dtype=pl.FP32)
            for chunk in pl.pipeline(inputs // 512, stage=2):
                start = chunk * 512
                x = pl.slice(hidden, [32, 512], [0, start])
                w = pl.slice(weight, [N, 512], [column, start])
                accumulator = pl.matmul_acc(accumulator, x, w, b_trans=True, init_cond=(chunk == 0))
            result[0:32, column:column + N] = pl.cast(accumulator, pl.BF16, mode="rint")
        return result

    @pl.jit.inline
    def baseline(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Tensor):
        for half in pl.range(2):
            row = half * 16
            x: pl.Tensor[[16, inputs], pl.BF16] = pl.slice(hidden, [16, inputs], [row, 0])
            out = pl.create_tensor([16, outputs], dtype=pl.BF16)
            out = original(x, weight, out)
            result[row:row + 16, 0:outputs] = out
        return result

    implementation = baseline if native else block

    @pl.jit
    def linear(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Out[pl.Tensor]):
        return implementation(hidden, weight, result)

    @pl.jit.host
    def entry(hidden: pl.Tensor, weight: pl.Tensor, result: pl.Out[pl.Tensor]):
        linear(hidden, weight, result, device=0)

    return implementation, entry


def make_norm(source, *, row_tile=1, native=False):
    if type(row_tile) is not int or row_tile not in {1, 2}:
        raise ValueError("row_tile must be 1 or 2")
    import pypto.language as pl
    original = native_module(source, "model").rmsnorm_rmsnorm_inline
    R, D = row_tile, 5120

    @pl.jit.inline
    def block(hidden: pl.Tensor, gamma: pl.Tensor, result: pl.Tensor):
        with pl.spmd(32 // R, name_hint="orthrus_input_norm32"):
            begin = pl.tile.get_block_idx() * R
            for lane in pl.range(R):
                row = begin + lane
                source = pl.cast(pl.slice(hidden, [1, D], [row, 0]), pl.FP32)
                # 一行分成八段，仅为满足 A2 归约 tile 的 32 字节对齐。
                x = pl.reshape(source, [8, D // 8])
                partial = pl.row_sum(pl.mul(x, x))
                repeated = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.), partial)
                total = pl.row_sum(pl.transpose(repeated, 0, 1))
                inverse = pl.rsqrt(pl.add(pl.mul(total, 1. / D), 1e-6), high_precision=True)
                gamma_row = pl.slice(gamma, [1, D], [0, 0])
                weight = pl.reshape(gamma_row, [8, D // 8])
                normalized = pl.mul(pl.row_expand_mul(x, inverse), weight)
                result[row:row + 1, 0:D] = pl.reshape(pl.cast(normalized, pl.BF16, mode="rint"), [1, D])
        return result

    @pl.jit.inline
    def baseline(hidden: pl.Tensor, gamma: pl.Tensor, result: pl.Tensor):
        for half in pl.range(2):
            row = half * 16
            x: pl.Tensor[[16, D], pl.BF16] = pl.slice(hidden, [16, D], [row, 0])
            out = pl.create_tensor([16, D], dtype=pl.BF16)
            out = original(x, gamma, out)
            result[row:row + 16, 0:D] = out
        return result

    implementation = baseline if native else block

    @pl.jit
    def norm(hidden: pl.Tensor, gamma: pl.Tensor, result: pl.Out[pl.Tensor]):
        return implementation(hidden, gamma, result)

    @pl.jit.host
    def entry(hidden: pl.Tensor, gamma: pl.Tensor, result: pl.Out[pl.Tensor]):
        norm(hidden, gamma, result, device=0)

    return implementation, entry
