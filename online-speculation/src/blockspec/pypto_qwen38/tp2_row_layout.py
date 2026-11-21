"""压紧双请求的计算行，保留每请求 32 行的输入、输出及训练接口。"""


def row_geometry(batch_size, physical_rows):
    if type(batch_size) is not int or batch_size not in (1, 2):
        raise ValueError("one or two draft blocks required")
    if type(physical_rows) is not int or physical_rows not in (16, 32):
        raise ValueError("physical draft rows must be 16 or 32")
    if physical_rows == 16 and batch_size != 2:
        raise ValueError("16-row layout requires a paired draft")
    return 32 * batch_size, physical_rows * batch_size


def make_pack(batch_size, physical_rows, columns):
    _, count = row_geometry(batch_size, physical_rows)
    import pypto.language as pl
    if physical_rows == 32:
        @pl.jit.inline
        def pack(source: pl.Tensor):
            return source
        return pack
    tile = next(t for t in (512, 256, 128, 64) if columns % t == 0)

    @pl.jit.inline
    def pack(source: pl.Tensor):
        result = pl.create_tensor([count, columns], dtype=pl.BF16)
        with pl.spmd(batch_size * (columns // tile), name_hint="orthrus_pack_paired_rows"):
            task = pl.tile.get_block_idx()
            seq = task // (columns // tile)
            col = task % (columns // tile) * tile
            result[seq * 16:seq * 16 + 16, col:col + tile] = pl.slice(
                source, [16, tile], [seq * 32, col])
        return result
    return pack


def make_expand(batch_size, physical_rows, columns):
    external, _ = row_geometry(batch_size, physical_rows)
    import pypto.language as pl
    if physical_rows == 32:
        @pl.jit.inline
        def expand(value: pl.Tensor, storage: pl.InOut[pl.Tensor]):
            return value
        return expand
    tile = next(t for t in (512, 256, 128, 64) if columns % t == 0)
    tasks = external // 16 * (columns // tile)
    blocks = min(64, tasks)

    @pl.jit.inline
    def expand(value: pl.Tensor, storage: pl.InOut[pl.Tensor]):
        with pl.spmd(blocks, name_hint="orthrus_expand_paired_rows"):
            for task in pl.range(pl.tile.get_block_idx(), tasks, blocks):
                row = task // (columns // tile) * 16
                col = task % (columns // tile) * tile
                seq = row // 32
                if row % 32 == 0:
                    storage[row:row + 16, col:col + tile] = pl.slice(value, [16, tile], [seq * 16, col])
                else:
                    storage[row:row + 16, col:col + tile] = pl.full([16, tile], dtype=pl.BF16, value=0.)
        return storage
    return expand


def make_feedback(physical_rows):
    row_geometry(2, physical_rows)
    import pypto.language as pl
    if physical_rows == 32:
        @pl.jit.inline
        def capture(hidden: pl.Tensor, feedback: pl.InOut[pl.Tensor], start: pl.Scalar[pl.INT32]):
            with pl.spmd(20, name_hint="orthrus_online_suffix_input"):
                tile = pl.tile.get_block_idx()
                row = tile // 10 * 16
                col = tile % 10 * 512
                feedback[row:row + 16, col:col + 512] = pl.slice(hidden, [16, 512], [start + row, col])
            return feedback
        return capture

    @pl.jit.inline
    def capture(hidden: pl.Tensor, feedback: pl.InOut[pl.Tensor], start: pl.Scalar[pl.INT32]):
        with pl.spmd(20, name_hint="orthrus_online_compact_input"):
            tile = pl.tile.get_block_idx()
            row = tile // 10 * 16
            col = tile % 10 * 512
            if row == 0:
                feedback[row:row + 16, col:col + 512] = pl.slice(hidden, [16, 512], [start // 32 * 16, col])
            else:
                feedback[row:row + 16, col:col + 512] = pl.full([16, 512], dtype=pl.BF16, value=0.)
        return feedback
    return capture
