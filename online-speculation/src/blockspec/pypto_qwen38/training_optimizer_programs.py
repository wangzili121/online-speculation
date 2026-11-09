"""梯度有限性、全局裁剪与执行参数发布；状态候选确认后才切换版本。"""


def make_stats(shape):
    from math import prod
    import pypto.language as pl
    N = prod(shape)
    T = min(4096, N)
    if N % T:
        raise ValueError("aligned gradient tensor required")

    @pl.jit
    def stats(value: pl.Tensor, sums: pl.Out[pl.Tensor], flags: pl.Out[pl.Tensor]):
        x = pl.reshape(value, [N // T, T])
        with pl.spmd((N // T + 15) // 16, name_hint="orthrus_gradient_statistics"):
            base = pl.tile.get_block_idx() * 16
            for part in pl.range(16):
                row = base + part
                if row < N // T:
                    block = pl.reshape(pl.slice(x, [1, T], [row, 0]), [8, T // 8])
                    sums[row:row + 1, 0:8] = pl.reshape(pl.row_sum(pl.mul(block, block)), [1, 8])
                    flags[row:row + 1, 0:8] = pl.reshape(pl.row_min(pl.cmp(pl.abs(block), 3.4028234663852886e38, cmp_type=3)), [1, 8])
        return sums, flags

    @pl.jit.host
    def entry(value: pl.Tensor, sums: pl.Out[pl.Tensor], flags: pl.Out[pl.Tensor]):
        stats(value, sums, flags, device=0)
    return entry, N // T


def make_reduce(rows):
    import pypto.language as pl
    R = (rows + 63) // 64

    @pl.jit
    def reduce(sums: pl.Tensor, flags: pl.Tensor, out_sums: pl.Out[pl.Tensor], out_flags: pl.Out[pl.Tensor]):
        padded = pl.create_tensor([R * 64, 8], dtype=pl.FP32)
        good = pl.create_tensor([R * 64, 8], dtype=pl.FP32)
        with pl.spmd(R, name_hint="orthrus_gradient_pad"):
            row = pl.tile.get_block_idx() * 64
            padded[row:row + 64, 0:8] = pl.full([64, 8], dtype=pl.FP32, value=0.)
            good[row:row + 64, 0:8] = pl.full([64, 8], dtype=pl.FP32, value=1.)
        with pl.spmd(R, name_hint="orthrus_gradient_actual"):
            base = pl.tile.get_block_idx() * 64
            for part in pl.range(64):
                row = base + part
                if row < rows:
                    padded[row:row + 1, 0:8] = pl.slice(sums, [1, 8], [row, 0])
                    good[row:row + 1, 0:8] = pl.slice(flags, [1, 8], [row, 0])
        with pl.spmd(R, name_hint="orthrus_gradient_reduce"):
            row = pl.tile.get_block_idx()
            out_sums[row:row + 1, 0:8] = pl.col_sum(pl.slice(padded, [64, 8], [row * 64, 0]))
            out_flags[row:row + 1, 0:8] = pl.reshape(pl.row_min(pl.transpose(pl.slice(good, [64, 8], [row * 64, 0]), 0, 1)), [1, 8])
        return out_sums, out_flags

    @pl.jit.host
    def entry(sums: pl.Tensor, flags: pl.Tensor, out_sums: pl.Out[pl.Tensor], out_flags: pl.Out[pl.Tensor]):
        reduce(sums, flags, out_sums, out_flags, device=0)
    return entry, R


def make_publish(shape, *, gamma=False, inline=False):
    from math import prod
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    N = prod(shape)
    T = min(2048, N)
    if N % T:
        raise ValueError("aligned parameter required")
    if gamma:
        @kernel
        def publish(master: pl.Tensor, execution: pl.Out[pl.Tensor]):
            x = pl.reshape(master, [N // T, T])
            y = pl.reshape(execution, [N // T, T])
            with pl.spmd((N // T + 15) // 16, name_hint="orthrus_publish_norm"):
                base = pl.tile.get_block_idx() * 16
                for part in pl.range(16):
                    row = base + part
                    if row < N // T:
                        raw = pl.cast(pl.slice(x, [1, T], [row, 0]), pl.BF16, mode="rint")
                        y[row:row + 1, 0:T] = pl.add(pl.cast(raw, pl.FP32), 1.)
            return execution
    else:
        @kernel
        def publish(master: pl.Tensor, execution: pl.Out[pl.Tensor]):
            x = pl.reshape(master, [N // T, T])
            y = pl.reshape(execution, [N // T, T])
            with pl.spmd((N // T + 15) // 16, name_hint="orthrus_publish_projection"):
                base = pl.tile.get_block_idx() * 16
                for part in pl.range(16):
                    row = base + part
                    if row < N // T:
                        y[row:row + 1, 0:T] = pl.cast(pl.slice(x, [1, T], [row, 0]), pl.BF16, mode="rint")
            return execution

    @pl.jit.host
    def entry(master: pl.Tensor, execution: pl.Out[pl.Tensor]):
        publish(master, execution, device=0)
    return publish if inline else entry


def make_global_stats(max_norm):
    """六组参数统一求 norm/clip；只回传三个标量的八份副本。"""
    import pypto.language as pl

    @pl.jit
    def calculate(a: pl.Tensor, b: pl.Tensor, c: pl.Tensor, d: pl.Tensor, e: pl.Tensor, f: pl.Tensor,
                  va: pl.Tensor, vb: pl.Tensor, vc: pl.Tensor, vd: pl.Tensor, ve: pl.Tensor, vf: pl.Tensor,
                  output: pl.Out[pl.Tensor]):
        with pl.spmd(1, name_hint="orthrus_global_gradient_norm"):
            row = pl.tile.get_block_idx()
            total = pl.add(pl.add(pl.add(a, b), pl.add(c, d)), pl.add(e, f))
            repeated = pl.col_expand(pl.full([8, 8], dtype=pl.FP32, value=1.), total)
            norm = pl.row_expand(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.sqrt(pl.row_sum(repeated)))
            scale = pl.minimum(pl.mul(pl.recip(pl.add(norm, 1e-6)), max_norm), 1.)
            valid = pl.mul(pl.mul(pl.mul(va, vb), pl.mul(vc, vd)), pl.mul(ve, vf))
            valid_matrix = pl.col_expand(pl.full([8, 8], dtype=pl.FP32, value=1.), valid)
            all_valid = pl.row_expand(pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_min(valid_matrix))
            finite = pl.mul(all_valid, pl.cmp(norm, 3.4028234663852886e38, cmp_type=3))
            output[row:row + 1, 0:8] = pl.slice(norm, [1, 8], [0, 0])
            output[row + 1:row + 2, 0:8] = pl.slice(scale, [1, 8], [0, 0])
            output[row + 2:row + 3, 0:8] = pl.slice(finite, [1, 8], [0, 0])
        return output

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, c: pl.Tensor, d: pl.Tensor, e: pl.Tensor, f: pl.Tensor,
              va: pl.Tensor, vb: pl.Tensor, vc: pl.Tensor, vd: pl.Tensor, ve: pl.Tensor, vf: pl.Tensor,
              output: pl.Out[pl.Tensor]):
        calculate(a, b, c, d, e, f, va, vb, vc, vd, ve, vf, output, device=0)
    return entry


def make_accumulate(shape):
    from math import prod
    import pypto.language as pl
    N = prod(shape)
    T = min(2048, N)

    @pl.jit
    def add(total: pl.Tensor, gradient: pl.Tensor, output: pl.Out[pl.Tensor]):
        a = pl.reshape(total, [N // T, T])
        b = pl.reshape(gradient, [N // T, T])
        out = pl.reshape(output, [N // T, T])
        with pl.spmd((N // T + 15) // 16, name_hint="orthrus_accumulate_master_gradient"):
            base = pl.tile.get_block_idx() * 16
            for part in pl.range(16):
                row = base + part
                if row < N // T:
                    out[row:row + 1, 0:T] = pl.add(pl.slice(a, [1, T], [row, 0]), pl.slice(b, [1, T], [row, 0]))
        return output

    @pl.jit.host
    def entry(total: pl.Tensor, gradient: pl.Tensor, output: pl.Out[pl.Tensor]):
        add(total, gradient, output, device=0)
    return entry


def make_norm_equal(dimension):
    import pypto.language as pl
    D = dimension

    @pl.jit
    def compare(a: pl.Tensor, b: pl.Tensor, flags: pl.Out[pl.Tensor]):
        with pl.spmd(1, name_hint="orthrus_online_initial_norm_contract"):
            row = pl.tile.get_block_idx()
            equal = pl.cmp(pl.reshape(a, [8, D // 8]), pl.reshape(b, [8, D // 8]), cmp_type=0)
            flags[row:row + 1, 0:8] = pl.reshape(pl.row_min(equal), [1, 8])
        return flags

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, flags: pl.Out[pl.Tensor]):
        compare(a, b, flags, device=0)
    return entry
