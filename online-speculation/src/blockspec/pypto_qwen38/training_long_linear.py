"""长词表 VJP：先分块转置行梯度，使用短步长输入执行同一矩阵乘。"""


def plan(k):
    if k < 65536 or k % 512:
        raise ValueError("large 512-aligned reduction required")
    chunk = max(c for c in range(512, 4097, 512) if k % c == 0)
    return chunk, k // chunk


def make_long_linear(m, k, n, *, inline=False, reduction_limit=512):
    if m != 32:
        raise ValueError("long-vocabulary VJP requires the pinned K32 row buffer")
    import pypto.language as pl
    from .training_programs import make_linear

    C, P = plan(k)
    # (M,K) @ (K,N) = transpose((K,M)) @ (K,N)，保留完整词表的同一矩阵乘。
    product = make_linear(
        (k, m), (k, n), transpose_a=True, transpose_b=False, inline=True,
        reduction_limit=reduction_limit,
    )
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def linear(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        packed = pl.create_tensor([P * m, C], dtype=pl.BF16)
        transposed = pl.create_tensor([k, m], dtype=pl.BF16)
        # 一次读取一行中的连续区间，避免对超长行做二维 DMA。
        with pl.spmd((P * m + 15) // 16, name_hint="orthrus_pack_vocab_gradient_rows"):
            base = pl.tile.get_block_idx() * 16
            for part in pl.range(16):
                index = base + part
                if index < P * m:
                    group = index // m
                    row = index % m
                    for piece in pl.range(C // 512):
                        begin = piece * 512
                        packed[index : index + 1, begin : begin + 512] = pl.slice(
                            a, [1, 512], [row, group * C + begin]
                        )
        with pl.spmd(k // 128, name_hint="orthrus_transpose_vocab_gradient_blocks"):
            begin = pl.tile.get_block_idx() * 128
            group = begin // C
            piece = begin % C
            values = pl.slice(packed, [m, 128], [group * m, piece])
            transposed[begin : begin + 128, 0:m] = pl.transpose(values, 0, 1)
        output = product(transposed, b, output)
        return output

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        linear(a, b, output, device=0)

    return linear if inline else entry
