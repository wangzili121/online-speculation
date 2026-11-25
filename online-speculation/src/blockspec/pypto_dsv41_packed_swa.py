"""Compose the A3 packed cache reader with Orthrus's resident SWA draft kernel."""

from .pypto_dsv41_packed_history import ROW_TILE, make_packed_history_gather, window_plan
from .pypto_dsv41_swa import HEAD_DIM, WINDOW, make_swa_attention


def packed_swa_controls(page_table, anchor, physical_pages):
    """Reuse these controls across all SWA layers of the same draft round."""
    import torch

    plan = window_plan(page_table, anchor, physical_pages)
    rows = torch.tensor(plan.rows, dtype=torch.int32).reshape(-1, ROW_TILE)
    starts = torch.tensor(plan.contiguous_starts, dtype=torch.int32)
    # The packed gather materializes a dense, zero-padded 128-row scratch
    # window. Cube can load it as one span.
    spans = torch.tensor(((0, 0, WINDOW), (0, 0, 0)), dtype=torch.int32)
    bias = torch.zeros(1, WINDOW, dtype=torch.float32)
    bias[:, min(anchor, WINDOW):] = -1e30
    return rows, starts, spans, bias


def packed_swa_views(runtime, rank_caches, layer, *, worker_id):
    """Borrow one layer from the native Prefill/Decode shared allocations."""
    views = []
    for name in ("window_cache", "window_scales"):
        bank = rank_caches[name]
        layers, *layer_shape = bank.shape
        if not 0 <= layer < layers:
            raise ValueError("SWA layer has no packed resident cache")
        views.append(runtime.view_tensor(
            bank, layer_shape, byte_offset=layer * (bank.nbytes // layers),
            worker_id=worker_id,
        ))
    return tuple(views)


def packed_swa_programs(physical_pages, heads, block_size):
    gather = make_packed_history_gather("fp8", physical_pages, WINDOW)
    attention = make_swa_attention(heads, token_major_query=True, block_size=block_size)
    return gather, attention


def make_packed_swa_chain(physical_pages, heads, block_size):
    """Fuse the packed gather and staged SWA into a single device dispatch."""
    import pypto.language as pl

    gather_entry, attention_entry = packed_swa_programs(physical_pages, heads, block_size)
    gather = pl.jit.inline(auto_scope=False)(gather_entry._func)
    attention = pl.jit.inline(attention_entry._func)
    tiles = WINDOW // ROW_TILE

    @pl.jit
    def attend(
        packed: pl.Tensor[[physical_pages, WINDOW, 1, HEAD_DIM], pl.UINT8],
        scales: pl.Tensor[[physical_pages, WINDOW, 1, HEAD_DIM // 32], pl.BF16],
        lookup: pl.Tensor[[1, 256], pl.INT32],
        rows: pl.Tensor[[tiles, ROW_TILE], pl.INT32],
        starts: pl.Tensor[[tiles], pl.INT32],
        spans: pl.Tensor,
        bias: pl.Tensor,
        query: pl.Tensor,
        temporary: pl.Tensor,
        sink: pl.Tensor,
        zeros: pl.Tensor,
        output: pl.Out[pl.Tensor],
        lse: pl.Out[pl.Tensor],
    ):
        scratch = pl.create_tensor([WINDOW, HEAD_DIM], dtype=pl.BF16)
        scratch = gather(packed, scales, rows, starts, lookup, scratch)
        return attention(query, scratch, spans, temporary, bias, sink, zeros, output, lse)

    return attend


def run_packed_swa(worker, programs, packed, scales, lookup, controls, scratch,
                   query, temporary, sink, zeros, output, lse, *, config):
    """Use one scratch allocation for all draft queries without host readback."""
    gather, attention = programs
    rows, starts, spans, bias = controls
    worker.run(gather, packed, scales, rows, starts, lookup, scratch, config=config)
    worker.run(attention, query, scratch, spans, temporary, bias, sink, zeros,
               output, lse, config=config)
