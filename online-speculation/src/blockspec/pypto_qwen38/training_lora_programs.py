"""低秩增量与基础权重的融合合并；保持两次 BF16 舍入边界。"""


def make_weight_merge(rows, columns, rank, *, inline=False):
    if any(type(n) is not int or n < 64 or n % 64 for n in (rows, columns, rank)):
        raise ValueError("64-aligned projection and adapter dimensions required")
    import pypto.language as pl
    kernel = pl.jit.inline if inline else pl.jit
    tile = 64
    tasks = rows // tile * (columns // tile)
    group = min(16, max(1, tasks // 32))

    @kernel
    def merge(base: pl.Tensor, b: pl.Tensor, a: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.spmd((tasks + group - 1) // group, name_hint="orthrus_lora_merge"):
            first = pl.tile.get_block_idx() * group
            for part in pl.range(group):
                task = first + part
                if task < tasks:
                    row = task // (columns // tile) * tile
                    col = task % (columns // tile) * tile
                    if rank == tile:
                        left = pl.slice(b, [tile, tile], [row, 0])
                        right = pl.slice(a, [tile, tile], [0, col])
                        delta = pl.matmul(left, right, out_dtype=pl.FP32)
                    else:
                        acc = pl.create_tensor([tile, tile], dtype=pl.FP32)
                        for step in pl.pipeline(rank // tile, stage=2):
                            left = pl.slice(b, [tile, tile], [row, step * tile])
                            right = pl.slice(a, [tile, tile], [step * tile, col])
                            acc = pl.matmul_acc(acc, left, right, init_cond=(step == 0))
                        delta = acc
                    # 与分开执行的矩阵乘、加法完全相同：delta 先舍入，再加 W0。
                    rounded = pl.cast(pl.cast(delta, pl.BF16, mode="rint"), pl.FP32)
                    original = pl.cast(pl.slice(base, [tile, tile], [row, col]), pl.FP32)
                    output[row:row + tile, col:col + tile] = pl.cast(
                        pl.add(original, rounded), pl.BF16, mode="rint")
        return output

    @pl.jit.host
    def entry(base: pl.Tensor, b: pl.Tensor, a: pl.Tensor, output: pl.Out[pl.Tensor]):
        merge(base, b, a, output, device=0)
    return merge if inline else entry
