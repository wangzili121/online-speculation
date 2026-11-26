"""Compose A3 packed cache readers with resident Orthrus draft Attention."""

from .pypto_dsv41_packed_history import ROW_TILE, make_packed_history_gather, window_plan
from .pypto_dsv41_swa import HEAD_DIM, WINDOW, make_swa_attention
from .pypto_dsv41_sparse import HEAD_TILE, TOPK, make_sparse_attention


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


def make_packed_selection_gather(compressed_pages, block_size):
    import pypto.language as pl

    gather_entry = make_packed_history_gather("fp4", compressed_pages, block_size * TOPK, queries=block_size)
    gather = pl.jit.inline(auto_scope=False)(gather_entry._func)

    @pl.jit.inline(auto_scope=False)
    def gather_selection(
        compressed: pl.Tensor, compressed_scales: pl.Tensor, compressed_lookup: pl.Tensor,
        selected_rows: pl.Tensor[[block_size, TOPK], pl.INT32], lengths: pl.Tensor[[block_size], pl.INT32],
        bank: pl.Out[pl.Tensor[[block_size * TOPK, HEAD_DIM], pl.BF16]],
    ):
        tiles = pl.reshape(selected_rows, [block_size * TOPK // ROW_TILE, ROW_TILE])
        return gather(compressed, compressed_scales, tiles, lengths, compressed_lookup, bank)

    return gather_selection


def make_packed_gathered_attention(window_pages, block_size):
    """Use an already decoded compressed selection and a layer-local SWA window."""
    import pypto.language as pl

    swa_entry = make_packed_swa_chain(window_pages, HEAD_TILE, block_size)
    sparse_entry = make_sparse_attention(head_major_swa=True, block_size=block_size, gathered_selection=True)
    swa = pl.jit.inline(auto_scope=False)(swa_entry._func)
    sparse = pl.jit.inline(auto_scope=False)(sparse_entry._func)

    @pl.jit.inline(auto_scope=False)
    def attend(
        window: pl.Tensor, window_scales: pl.Tensor, window_lookup: pl.Tensor,
        window_rows: pl.Tensor, window_starts: pl.Tensor, spans: pl.Tensor, window_bias: pl.Tensor,
        bank: pl.Tensor[[block_size * TOPK, HEAD_DIM], pl.BF16],
        selected_rows: pl.Tensor[[block_size, TOPK], pl.INT32],
        lengths: pl.Tensor[[block_size], pl.INT32], selection_bias: pl.Tensor,
        query: pl.Tensor, temporary: pl.Tensor, sink: pl.Tensor, zeros: pl.Tensor,
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        local = pl.create_tensor([block_size * HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
        local_lse = pl.create_tensor([block_size * HEAD_TILE, 1], dtype=pl.FP32)
        local, local_lse = swa(window, window_scales, window_lookup, window_rows, window_starts,
                               spans, window_bias, query, temporary, sink, zeros, local, local_lse)
        flat = pl.reshape(query, [block_size * HEAD_TILE, HEAD_DIM])
        return sparse(flat, bank, selected_rows, lengths, selection_bias, zeros, local, local_lse, output, lse)

    return attend


def make_packed_compressed_chain(window_pages, compressed_pages, block_size):
    """Combine SWA, temporary KV and the device Indexer's packed-cache selection."""
    import pypto.language as pl

    gather = make_packed_selection_gather(compressed_pages, block_size)
    attention = make_packed_gathered_attention(window_pages, block_size)

    @pl.jit
    def attend(
        window: pl.Tensor, window_scales: pl.Tensor, window_lookup: pl.Tensor,
        window_rows: pl.Tensor, window_starts: pl.Tensor, spans: pl.Tensor, window_bias: pl.Tensor,
        compressed: pl.Tensor, compressed_scales: pl.Tensor, compressed_lookup: pl.Tensor,
        selected_rows: pl.Tensor[[block_size, TOPK], pl.INT32],
        lengths: pl.Tensor[[block_size], pl.INT32], selection_bias: pl.Tensor,
        query: pl.Tensor, temporary: pl.Tensor, sink: pl.Tensor, zeros: pl.Tensor,
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        bank = pl.create_tensor([block_size * TOPK, HEAD_DIM], dtype=pl.BF16)
        bank = gather(compressed, compressed_scales, compressed_lookup, selected_rows, lengths, bank)
        return attention(window, window_scales, window_lookup, window_rows, window_starts, spans, window_bias,
                         bank, selected_rows, lengths, selection_bias, query, temporary, sink, zeros, output, lse)

    return attend


def run_packed_swa(worker, programs, packed, scales, lookup, controls, scratch,
                   query, temporary, sink, zeros, output, lse, *, config):
    """Use one scratch allocation for all draft queries without host readback."""
    gather, attention = programs
    rows, starts, spans, bias = controls
    worker.run(gather, packed, scales, rows, starts, lookup, scratch, config=config)
    worker.run(attention, query, scratch, spans, temporary, bias, sink, zeros,
               output, lse, config=config)
