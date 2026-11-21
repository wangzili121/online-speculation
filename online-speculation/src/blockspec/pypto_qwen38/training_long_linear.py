"""长词表 VJP：先分块转置行梯度，使用短步长输入执行同一矩阵乘。"""


def plan(k):
    if k < 65536 or k % 512:
        raise ValueError("large 512-aligned reduction required")
    chunk = max(c for c in range(512, 4097, 512) if k % c == 0)
    return chunk, k // chunk


def make_long_linear(m, k, n, *, inline=False, reduction_limit=512):
    if type(m) is not int or m not in (16, 32):
        raise ValueError("long-vocabulary VJP requires 16 or 32 physical rows")
    if m == 16:
        return _compact_input_vjp(k, n, inline=inline, reduction_limit=reduction_limit)
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


def _compact_input_vjp(k, n, *, inline, reduction_limit):
    """输入只保留 16 行，转置时补零到 32 列，复用已验证的宽行矩阵计算。"""
    import pypto.language as pl
    from .training_programs import make_linear

    C, P = plan(k)
    # [词表,16] 的窄步长在当前设备上明显更慢；补零列不改变有效行的梯度。
    product = make_linear((k, 32), (k, n), transpose_a=True, transpose_b=False,
                          inline=True, reduction_limit=reduction_limit)
    tile = 512 if n % 512 == 0 else 64
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def linear(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        packed = pl.create_tensor([P * 32, C], dtype=pl.BF16)
        transposed = pl.create_tensor([k, 32], dtype=pl.BF16)
        with pl.spmd((P * 32 + 15) // 16, name_hint="orthrus_pack_compact_vocab_gradient"):
            base = pl.tile.get_block_idx() * 16
            for part in pl.range(16):
                index = base + part
                if index < P * 32:
                    group = index // 32
                    row = index % 32
                    for piece in pl.range(C // 512):
                        begin = piece * 512
                        # 保持单行连续读取，避免原始大词表行上的超长二维 DMA 步长。
                        if row < 16:
                            packed[index:index + 1, begin:begin + 512] = pl.slice(
                                a, [1, 512], [row, group * C + begin])
                        else:
                            packed[index:index + 1, begin:begin + 512] = pl.full(
                                [1, 512], dtype=pl.BF16, value=0.)
        with pl.spmd(k // 128, name_hint="orthrus_transpose_compact_vocab_gradient"):
            begin = pl.tile.get_block_idx() * 128
            group = begin // C
            piece = begin % C
            values = pl.slice(packed, [32, 128], [group * 32, piece])
            transposed[begin:begin + 128, 0:32] = pl.transpose(values, 0, 1)
        temporary = pl.create_tensor([32, n], dtype=pl.BF16)
        temporary = product(transposed, b, temporary)
        with pl.spmd(n // tile, name_hint="orthrus_store_compact_vocab_vjp"):
            col = pl.tile.get_block_idx() * tile
            output[0:16, col:col + tile] = pl.slice(temporary, [16, tile], [0, col])
        return output

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
        linear(a, b, output, device=0)

    return linear if inline else entry
