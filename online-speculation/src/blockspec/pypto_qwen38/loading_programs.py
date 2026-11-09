"""权重加载阶段的设备转换；文件 I/O 和张量形状由主机提供。"""


def make_convert(source_shape, target_shape, *, source_dtype="bf16", target_dtype="bf16",
                 transpose_conv=False, add_one=False, copy_shape=None):
    """复制矩阵片段，或把 [channel,4] 卷积权重转换为 [4,channel]。"""
    if source_dtype not in {"bf16", "fp32"} or target_dtype not in {"bf16", "fp32"}:
        raise ValueError("BF16/FP32 weight formats required")
    if len(source_shape) != 2 or len(target_shape) != 2 or min(*source_shape, *target_shape) < 1:
        raise ValueError("positive matrix shapes required")
    if source_dtype == "fp32" and target_dtype != "fp32":
        raise ValueError("FP32 loading preserves FP32 parameters")
    import pypto.language as pl
    M, N = source_shape
    CM, CN = copy_shape or source_shape
    T = next((t for t in (2048, 1024, 512, 256, 128, 64) if N % t == 0 and CN % t == 0), 0)
    if transpose_conv:
        if N != 4 or M % 256 or source_dtype != "bf16" or target_dtype != "bf16":
            raise ValueError("convolution requires complete 256-channel BF16 groups")
        T = 256
        CM, CN = copy_shape or (4, M)
        if CM != 4 or CN % T:
            raise ValueError("complete convolution channel groups required")
        BLOCKS = 4 * (CN // T)
    else:
        if T == 0 or CN % T or min(CM, CN) < 1:
            raise ValueError("complete 64/128/256-column weight tiles required")
        BLOCKS = CM * (CN // T)
    R = 1 if transpose_conv else 8 if CM % 8 == 0 else 4 if CM % 4 == 0 else 1
    if not transpose_conv:
        BLOCKS //= R
    REGIONS = (BLOCKS + 63) // 64
    PAD = REGIONS * 64

    if source_dtype == "bf16" and target_dtype == "bf16":
        @pl.jit.inline
        def normal_part(source: pl.Tensor, columns: pl.Tensor, target: pl.Tensor, partial: pl.Tensor,
                        block: pl.Scalar[pl.INT32], source_row: pl.Scalar[pl.INT32], source_column: pl.Scalar[pl.INT32],
                        target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
                        rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
            row = block // (CN // T) * R
            col = block % (CN // T) * T
            if row < rows and col < cols:
                value = pl.slice(source, [R, T], [source_row + row, source_column + col])
                x = pl.cast(value, pl.FP32)
                if add_one:
                    output = pl.cast(pl.add(x, 1.), pl.BF16, mode="rint")
                else:
                    output = value
                target[target_row + row:target_row + row + R, target_column + col:target_column + col + T] = output
            return target, partial

    elif source_dtype == "bf16" and target_dtype == "fp32":
        @pl.jit.inline
        def normal_part(source: pl.Tensor, columns: pl.Tensor, target: pl.Tensor, partial: pl.Tensor,
                        block: pl.Scalar[pl.INT32], source_row: pl.Scalar[pl.INT32], source_column: pl.Scalar[pl.INT32],
                        target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
                        rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
            row = block // (CN // T) * R
            col = block % (CN // T) * T
            if row < rows and col < cols:
                value = pl.slice(source, [R, T], [source_row + row, source_column + col])
                x = pl.cast(value, pl.FP32)
                if add_one:
                    output = pl.add(x, 1.)
                else:
                    output = x
                target[target_row + row:target_row + row + R, target_column + col:target_column + col + T] = output
            return target, partial

    else:
        @pl.jit.inline
        def normal_part(source: pl.Tensor, columns: pl.Tensor, target: pl.Tensor, partial: pl.Tensor,
                        block: pl.Scalar[pl.INT32], source_row: pl.Scalar[pl.INT32], source_column: pl.Scalar[pl.INT32],
                        target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
                        rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
            row = block // (CN // T) * R
            col = block % (CN // T) * T
            if row < rows and col < cols:
                value = pl.slice(source, [R, T], [source_row + row, source_column + col])
                if add_one:
                    output = pl.add(value, 1.)
                else:
                    output = value
                target[target_row + row:target_row + row + R, target_column + col:target_column + col + T] = output
            return target, partial

    @pl.jit.inline
    def conv_part(source: pl.Tensor, columns: pl.Tensor, target: pl.Tensor, partial: pl.Tensor,
                  block: pl.Scalar[pl.INT32], source_row: pl.Scalar[pl.INT32], source_column: pl.Scalar[pl.INT32],
                  target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
                  rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
        row = block // (CN // T)
        col = block % (CN // T) * T
        if col < cols:
            index = pl.slice(columns, [1, T], [0, 0])
            index = pl.add(pl.mul(pl.add(index, pl.cast(source_row + col, pl.INT32)), 4), pl.cast(row, pl.INT32))
            gathered = pl.gather(source, index=index)
            value = pl.reinterpret_view(gathered, pl.BF16, shape=[1, T])
            target[target_row + row:target_row + row + 1, target_column + col:target_column + col + T] = value
        return target, partial

    if target_dtype == "bf16":
        @pl.jit.inline
        def finite_part(target: pl.Tensor, partial: pl.Tensor, block: pl.Scalar[pl.INT32],
                        target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
                        rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
            row = block // (CN // T) * R
            col = block % (CN // T) * T
            if row < rows and col < cols:
                value = pl.slice(target, [R, T], [target_row + row, target_column + col])
                x = pl.cast(value, pl.FP32)
                valid = pl.row_min(pl.reshape(pl.cmp(pl.abs(x), 3.4028234663852886e38, cmp_type=3), [8, R * T // 8]))
                partial[block:block + 1, 0:8] = pl.reshape(valid, [1, 8])
            return partial
    else:
        @pl.jit.inline
        def finite_part(target: pl.Tensor, partial: pl.Tensor, block: pl.Scalar[pl.INT32],
                        target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
                        rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
            row = block // (CN // T) * R
            col = block % (CN // T) * T
            if row < rows and col < cols:
                x = pl.slice(target, [R, T], [target_row + row, target_column + col])
                valid = pl.row_min(pl.reshape(pl.cmp(pl.abs(x), 3.4028234663852886e38, cmp_type=3), [8, R * T // 8]))
                partial[block:block + 1, 0:8] = pl.reshape(valid, [1, 8])
            return partial

    fragment = conv_part if transpose_conv else normal_part

    @pl.jit
    def convert(source: pl.Tensor, columns: pl.Tensor, target: pl.Out[pl.Tensor], flags: pl.Out[pl.Tensor],
                source_row: pl.Scalar[pl.INT32], source_column: pl.Scalar[pl.INT32],
                target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
                rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
        partial = pl.create_tensor([PAD, 8], dtype=pl.FP32)
        with pl.spmd(REGIONS, name_hint="orthrus_weight_flags_init"):
            start = pl.tile.get_block_idx() * 64
            partial[start:start + 64, 0:8] = pl.full([64, 8], dtype=pl.FP32, value=1.)
        with pl.spmd(BLOCKS, name_hint="orthrus_weight_convert"):
            block = pl.tile.get_block_idx()
            target, partial = fragment(source, columns, target, partial, block, source_row, source_column,
                                       target_row, target_column, rows, cols)
        # 复制和数值检查分阶段，检查已写出的 bank，避免跨 dtype 临时值复用。
        with pl.spmd(BLOCKS, name_hint="orthrus_weight_finite"):
            block = pl.tile.get_block_idx()
            partial = finite_part(target, partial, block, target_row, target_column, rows, cols)
        with pl.spmd(REGIONS, name_hint="orthrus_weight_finite_reduce"):
            start = pl.tile.get_block_idx()
            x = pl.transpose(pl.slice(partial, [64, 8], [start * 64, 0]), 0, 1)
            flags[start:start + 1, 0:8] = pl.reshape(pl.row_min(x), [1, 8])
        return target, flags

    # 第二级以上归约在宿主编排中重复调用小核，避免长矩阵超过 UB 容量。
    @pl.jit.host
    def entry(source: pl.Tensor, columns: pl.Tensor, target: pl.Out[pl.Tensor], flags: pl.Out[pl.Tensor],
              source_row: pl.Scalar[pl.INT32], source_column: pl.Scalar[pl.INT32],
              target_row: pl.Scalar[pl.INT32], target_column: pl.Scalar[pl.INT32],
              rows: pl.Scalar[pl.INT32], cols: pl.Scalar[pl.INT32]):
        convert(source, columns, target, flags, source_row, source_column,
                target_row, target_column, rows, cols, device=0)

    return entry, REGIONS


def make_reduce(rows):
    if type(rows) is not int or rows < 1:
        raise ValueError("positive flag row count required")
    import pypto.language as pl
    R = (rows + 63) // 64

    @pl.jit
    def reduce(source: pl.Tensor, result: pl.Out[pl.Tensor]):
        padded = pl.create_tensor([R * 64, 8], dtype=pl.FP32)
        with pl.spmd(R, name_hint="orthrus_weight_check_pad"):
            block = pl.tile.get_block_idx()
            padded[block * 64:block * 64 + 64, 0:8] = pl.full([64, 8], dtype=pl.FP32, value=1.)
        with pl.spmd(rows, name_hint="orthrus_weight_check_copy"):
            block = pl.tile.get_block_idx()
            padded[block:block + 1, 0:8] = pl.slice(source, [1, 8], [block, 0])
        with pl.spmd(R, name_hint="orthrus_weight_finite_final"):
            block = pl.tile.get_block_idx()
            # 显式补 1，再归约完整 tile；尾块不依赖 PadValue 的传递。
            x = pl.transpose(pl.slice(padded, [64, 8], [block * 64, 0]), 0, 1)
            result[block:block + 1, 0:8] = pl.reshape(pl.row_min(x), [1, 8])
        return result

    @pl.jit.host
    def entry(source: pl.Tensor, result: pl.Out[pl.Tensor]):
        reduce(source, result, device=0)

    return entry, R


def make_zero(shape, dtype="bf16"):
    if len(shape) != 2 or min(shape) < 1 or shape[1] % 64 or dtype not in {"bf16", "fp32"}:
        raise ValueError("aligned positive weight matrix required")
    import pypto.language as pl
    M, N = shape
    T = min(256, N)
    R = 16 if M % 16 == 0 else 1
    D = pl.BF16 if dtype == "bf16" else pl.FP32

    @pl.jit
    def zero(result: pl.Out[pl.Tensor]):
        with pl.spmd((M // R) * (N // T), name_hint="orthrus_weight_zero"):
            block = pl.tile.get_block_idx()
            row = block // (N // T) * R
            col = block % (N // T) * T
            result[row:row + R, col:col + T] = pl.full([R, T], dtype=D, value=0.)
        return result

    @pl.jit.host
    def entry(result: pl.Out[pl.Tensor]):
        zero(result, device=0)

    return entry


def make_equal(elements):
    if type(elements) is not int or elements not in {128, 256}:
        raise ValueError("replicated norm must contain 128 or 256 elements")
    import pypto.language as pl

    @pl.jit
    def equal(a: pl.Tensor, b: pl.Tensor, flags: pl.Out[pl.Tensor]):
        with pl.spmd(1, name_hint="orthrus_weight_replicas"):
            task = pl.tile.get_block_idx()
            x = pl.cast(pl.reshape(a, [8, elements // 8]), pl.FP32)
            y = pl.cast(pl.reshape(b, [8, elements // 8]), pl.FP32)
            flags[task:task + 1, 0:8] = pl.reshape(pl.row_min(pl.cmp(x, y, cmp_type=0)), [1, 8])
        return flags

    @pl.jit.host
    def entry(a: pl.Tensor, b: pl.Tensor, flags: pl.Out[pl.Tensor]):
        equal(a, b, flags, device=0)

    return entry
