"""Read committed history in the DSV4.1 A3 whole-graph cache layout."""

from dataclasses import dataclass

from .pypto_dsv41_swa import HEAD_DIM, WINDOW, swa_spans

PAGE_ROWS = 128
ROW_TILE = 16
PADDED_TILE = -2
SCATTER_TILE = -1


@dataclass(frozen=True)
class PackedRows:
    rows: tuple[int, ...]
    contiguous_starts: tuple[int, ...]


def _row_plan(rows):
    rows = tuple(rows)
    if len(rows) % ROW_TILE:
        raise ValueError("packed output capacity must be a multiple of 16")
    runs = []
    for offset in range(0, len(rows), ROW_TILE):
        tile = rows[offset:offset + ROW_TILE]
        start = tile[0]
        if start >= 0 and tile == tuple(range(start, start + ROW_TILE)):
            runs.append(start)
        elif all(row == -1 for row in tile):
            runs.append(PADDED_TILE)
        else:
            runs.append(SCATTER_TILE)
    return PackedRows(rows, tuple(runs))


def window_plan(page_table, anchor, physical_pages):
    """Plan one SWA gather shared by every query in the draft block."""
    rows = [-1] * WINDOW
    for dst, src, length in swa_spans(page_table, anchor, physical_pages):
        rows[dst:dst + length] = range(src, src + length)
    return _row_plan(rows)


def compressed_plan(page_table, anchor, ratio, physical_pages, selected):
    """Lower selected logical compressed IDs to the packed pool's physical rows."""
    if anchor < 0 or ratio not in (1, 2) or physical_pages <= 0:
        raise ValueError("invalid compressed history")
    committed = anchor // ratio
    rows = []
    for index in selected:
        if index == -1:
            rows.append(-1)
            continue
        if not 0 <= index < committed:
            raise ValueError("selected row extends beyond committed compressed history")
        page, offset = divmod(index, PAGE_ROWS)
        if page >= len(page_table) or not 0 <= page_table[page] < physical_pages:
            raise ValueError("committed compressed row has no physical page")
        rows.append(page_table[page] * PAGE_ROWS + offset)
    return _row_plan(rows)


def packed_history_lookup(kind):
    """Build a 1 KiB table to upload once per worker and reuse across layers."""
    import torch

    if kind == "fp8":
        codes = torch.arange(-128, 128, dtype=torch.int8).view(torch.float8_e4m3fn)
        return codes.float().view(torch.int32).reshape(1, 256)
    if kind == "fp4":
        magnitudes = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), dtype=torch.bfloat16)
        bits = torch.cat((magnitudes, -magnitudes)).view(torch.int16).to(torch.int32) & 65535
        byte = torch.arange(256)
        return (bits[byte & 15] | (bits[byte >> 4] << 16)).reshape(1, 256)
    raise ValueError("invalid packed cache codec")


def make_packed_history_gather(kind, physical_pages, selected_rows):
    """Read packed rows directly from the target bank and round to BF16 once."""
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.quant import decode_fp8_cache_values

    if kind not in ("fp8", "fp4") or physical_pages <= 0:
        raise ValueError("invalid packed cache")
    if selected_rows <= 0 or selected_rows % ROW_TILE:
        raise ValueError("invalid packed gather capacity")
    window = kind == "fp8"
    width = HEAD_DIM if window else HEAD_DIM // 2
    group = 32 if window else 16
    groups = HEAD_DIM // group
    tiles = selected_rows // ROW_TILE
    workers = min(tiles, 32)
    pair_count = ROW_TILE * (HEAD_DIM // 2)

    @pl.jit
    def gather(
        payload: pl.Tensor[[physical_pages, PAGE_ROWS, 1, width], pl.UINT8],
        scales: pl.Tensor[[physical_pages, PAGE_ROWS, 1, groups], pl.BF16],
        rows: pl.Tensor[[tiles, ROW_TILE], pl.INT32],
        starts: pl.Tensor[[tiles], pl.INT32],
        lookup: pl.Tensor[[1, 256], pl.INT32],
        output: pl.Out[pl.Tensor[[selected_rows, HEAD_DIM], pl.BF16]],
    ):
        packed = pl.reshape(payload, [physical_pages * PAGE_ROWS, width])
        factors = pl.reshape(scales, [physical_pages * PAGE_ROWS, groups])
        with pl.spmd(workers, name_hint="orthrus_v41_packed_history"):
            worker = pl.tile.get_block_idx()
            table = pl.tile.load(lookup, [0, 0], [1, 256])
            if not window:
                pair_scratch = pl.create_tile([1, pair_count], dtype=pl.INT32)
            for block in pl.range(worker, tiles, workers):
                start = pl.read(starts, [block])
                if start == PADDED_TILE:
                    zero_tile = pl.tile.full([ROW_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    pl.tile.store(zero_tile, [block * ROW_TILE, 0], output)
                else:
                    if start >= 0:
                        physical = pl.cast(start, pl.INDEX)
                        data = pl.tile.load(packed, [physical, 0], [ROW_TILE, width])
                        scale = pl.tile.load(factors, [physical, 0], [ROW_TILE, groups])
                        scale_column = pl.reshape(pl.cast(scale, pl.FP32), [ROW_TILE * groups, 1])
                    else:
                        ids = pl.tile.load(rows, [block, 0], [1, ROW_TILE])
                        data = pl.tile.create([ROW_TILE, width], dtype=pl.UINT8)
                        padded_scales = pl.tile.create([ROW_TILE, 32], dtype=pl.BF16)
                        for lane in pl.range(ROW_TILE):
                            physical = pl.cast(pl.max(pl.tile.read(ids, [0, lane]), 0), pl.INDEX)
                            data = pl.tile.gather_row(data, packed, [lane, 0], [physical, 0], [1, width])
                            padded_scales = pl.tile.gather_row(
                                padded_scales, factors, [lane, 0], [physical, 0], [1, groups]
                            )
                        scale_values = pl.cast(padded_scales, pl.FP32)
                        if window:
                            columns = pl.tile.arange(0, [1, ROW_TILE * groups], dtype=pl.INT32)
                            order = pl.add(columns, pl.shls(pl.shrs(columns, 4), 4))
                            bits = pl.reinterpret_view(scale_values, pl.INT32)
                            scratch = pl.tile.create([1, ROW_TILE * groups], dtype=pl.INT32)
                            compact = pl.tile.gather(bits, order, scratch)
                            unpadded = pl.reinterpret_view(compact, pl.FP32)
                            scale_column = pl.reshape(unpadded, [ROW_TILE * groups, 1])
                        else:
                            scale_column = pl.reshape(scale_values, [ROW_TILE * groups, 1])
                    if window:
                        values = decode_fp8_cache_values(data, table, ROW_TILE, HEAD_DIM)
                    else:
                        signed = pl.reinterpret_view(data, pl.INT8)
                        half = pl.cast(signed, pl.FP16)
                        byte_ids = pl.ands(pl.cast(half, pl.INT16, mode="rint"), 255)
                        wide_ids = pl.cast(pl.cast(byte_ids, pl.FP32), pl.INT32, mode="rint")
                        pair_ids = pl.reshape(wide_ids, [1, pair_count])
                        pairs = pl.tile.gather(table, pair_ids, pair_scratch)
                        bits = pl.reinterpret_view(pairs, pl.BF16, shape=[ROW_TILE, HEAD_DIM])
                        values = pl.cast(bits, pl.FP32)
                    grouped = pl.reshape(values, [ROW_TILE * groups, group])
                    expanded = pl.row_expand_mul(grouped, scale_column)
                    result = pl.reshape(pl.cast(expanded, pl.BF16, mode="rint"), [ROW_TILE, HEAD_DIM])
                    pl.tile.store(result, [block * ROW_TILE, 0], output)
                    if start < 0:
                        ids = pl.tile.load(rows, [block, 0], [1, ROW_TILE])
                        zero = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        for lane in pl.range(ROW_TILE):
                            if pl.tile.read(ids, [0, lane]) < 0:
                                pl.tile.store(zero, [block * ROW_TILE + lane, 0], output)
        return output

    return gather
