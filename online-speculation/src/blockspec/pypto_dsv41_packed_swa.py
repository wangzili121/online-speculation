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


def run_packed_swa(worker, programs, packed, scales, controls, scratch,
                   query, temporary, sink, zeros, output, lse, *, config):
    """Use one scratch allocation for all draft queries without host readback."""
    gather, attention = programs
    rows, starts, spans, bias = controls
    worker.run(gather, packed, scales, rows, starts, scratch, config=config)
    worker.run(attention, query, scratch, spans, temporary, bias, sink, zeros,
               output, lse, config=config)
