"""Physical writes made by the DSV4.1 provisional target verifier."""


def index_written_pages(slots):
    return sorted({page for page, row in slots.tolist() if page >= 0 and row >= 0})


def plane_write_extents(name, plane, *, native, pages, element_bytes, origin):
    """Map native publication controls to byte ranges in the shared arenas."""
    page_stride = plane["stride"][0] * element_bytes
    row_stride = plane["stride"][1] * element_bytes
    row_bytes = plane["shape"][-1] * element_bytes
    rows = plane["shape"][1]
    if name.endswith(".long_kv_cache"):
        slots = [(page, row) for page, row in native["slot_mapping"].tolist() if page >= 0]
    elif name.endswith(".indexer.k_cache"):
        dirty = index_written_pages(native["slot_mapping"])
        if any(page not in pages for page in dirty):
            raise ValueError("Verification index write leaves its cache owner")
        return [(page * page_stride + origin, rows * row_bytes) for page in dirty]
    else:
        starts = native["query_start_loc"].tolist()
        positions = native["positions"].tolist()
        table = native["block_table"].tolist()
        slots = [
            (table[request][0], position % rows)
            for request, (begin, end) in enumerate(zip(starts, starts[1:]))
            for position in positions[max(begin, end - rows):end]
        ]
    if any(page not in pages or not 0 <= row < rows for page, row in slots):
        raise ValueError("Verification write slot lies outside its physical cache plane")
    return [(page * page_stride + origin + row * row_stride, row_bytes) for page, row in slots]


def make_index_page_copy(blocks, payload, source_stride, destination_stride, dtype, *,
                         source_offset=0, destination_offset=0):
    """Copy selected physical pages without touching unrelated plane bytes."""
    import pypto.language as pl

    tile = min(payload, 4096)
    tiles_per_page = payload // tile

    @pl.jit
    def copy_pages(
        source: pl.Tensor[[blocks, source_stride], dtype],
        destination: pl.InOut[pl.Tensor[[blocks, destination_stride], dtype]],
        pages: pl.Tensor[[blocks], pl.INT32],
        count: pl.Scalar[pl.INT32],
    ):
        for task in pl.spmd(count * tiles_per_page, name_hint="orthrus_index_page_copy"):
            page = pl.cast(pl.read(pages, [task // tiles_per_page]), pl.INDEX)
            offset = (task % tiles_per_page) * tile
            source_column = source_offset + offset
            destination_column = destination_offset + offset
            value = source[page:page + 1, source_column:source_column + tile]
            destination[page:page + 1, destination_column:destination_column + tile] = value
        return destination

    return copy_pages
