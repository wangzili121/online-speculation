"""Read-only Orthrus query selection over the 1122 packed index cache."""

from .pypto_dsv41_indexer import INDEX_CAPACITIES, INDEX_DIM, INDEX_HEADS
from .pypto_dsv41_sparse import TOPK
from .pypto_dsv41_swa import WINDOW


def packed_index_controls(page_table, anchor, ratio, block_size):
    """Prepare one committed prefix for all queries of a draft request."""
    import torch

    if ratio not in (1, 2) or not 0 <= anchor <= INDEX_CAPACITIES[-1]:
        raise ValueError("invalid packed index prefix")
    visible = anchor // ratio
    bucket = next(size for size in INDEX_CAPACITIES if size >= visible)
    count = (visible + WINDOW - 1) // WINDOW
    pages = torch.as_tensor(page_table, dtype=torch.int32)[:count]
    if pages.numel() != count or bool((pages < 0).any()):
        raise ValueError("committed index rows require allocated physical pages")
    table = torch.full((1, bucket // WINDOW), -1, dtype=torch.int32)
    table[0, :count] = pages
    capacity = max(16, block_size)
    return bucket, table, torch.zeros(capacity, dtype=torch.int32), torch.full((capacity,), visible, dtype=torch.int32)


def make_packed_draft_indexer(physical_pages, bucket, ratio, block_size):
    """Project draft queries, score the target prefix and emit physical Top-K rows.

    C1A Full and Reindex use the same complete-prefix scorer in this serving
    geometry. The candidate pool covers every 8-row block up to 16K. C2A
    shares the physical page ABI but the caller supplies half-length history.
    """
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.config import FLASH, INDEX_CACHE_GROUP
    from models.deepseek_v4_1_flash_a3.decode_indexer import (
        _write_c2a_short_topk, project_index_weights, quantize_index_query,
        score_c2a_full, score_full, select_c2a_index_topk, select_index_topk,
    )
    from models.deepseek_v4_1_flash_a3.gemm import gemm_bf16_to_bf16

    if ratio == 1 and bucket > FLASH.candidate_topk_blocks * FLASH.candidate_block_size:
        raise ValueError("C1A candidate pool must cover the committed prefix")
    capacity = max(16, block_size)
    hidden, latent = FLASH.hidden_size, FLASH.q_lora_rank
    index_width = INDEX_HEADS * INDEX_DIM
    scale_width = index_width // INDEX_CACHE_GROUP
    short = bucket <= TOPK
    c2a = ratio == 2

    @pl.jit
    def select(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        low_rank: pl.Tensor[[capacity, latent], pl.BF16],
        idx_wq: pl.Tensor[[latent, index_width], pl.BF16],
        idx_weight: pl.Tensor[[hidden, INDEX_HEADS], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        key: pl.Tensor[[physical_pages, WINDOW, 1, INDEX_DIM // 2], pl.UINT8],
        key_scale: pl.Tensor[[physical_pages, WINDOW, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8],
        table: pl.Tensor[[1, bucket // WINDOW], pl.INT32],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        visible: pl.Tensor[[capacity], pl.INT32],
        rows: pl.Out[pl.Tensor[[block_size, TOPK], pl.INT32]],
        lengths: pl.Out[pl.Tensor[[block_size], pl.INT32]],
        bias: pl.Out[pl.Tensor[[block_size, TOPK], pl.FP32]],
    ):
        selected = pl.create_tensor([capacity, TOPK], dtype=pl.INT32)
        if short:
            # Every committed row fits in Top-K. No projection or scores needed.
            with pl.spmd(block_size, name_hint="orthrus_short_index") as selected_ready:
                token = pl.tile.get_block_idx()
                _write_c2a_short_topk(table, request_ids, visible, selected, token)
        else:
            start = pl.system.task_dummy(deps=[])
            projected = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
            projected, projected_ready = gemm_bf16_to_bf16(
                low_rank, idx_wq, projected, block_size, start, latent, index_width,
                128, 256, 32, "orthrus_index_q", 0, 0, False, 16, 2,
            )
            query = pl.create_tensor([capacity, index_width // 2], dtype=pl.UINT8)
            scales = pl.create_tensor([capacity, scale_width], dtype=pl.UINT8)
            query_ready = quantize_index_query(projected, cosine, sine, query, scales, block_size, projected_ready)
            weights = pl.create_tensor([capacity, INDEX_HEADS], dtype=pl.BF16)
            weights_ready = project_index_weights(x, idx_weight, weights, block_size, start)
            scores = pl.create_tensor([capacity, bucket], dtype=pl.FP32)
            if c2a:
                scores_ready = score_c2a_full(
                    query, scales, weights, key, key_scale, table, request_ids, visible, scores,
                    block_size, start, query_ready, weights_ready,
                )
                selected_ready = select_c2a_index_topk(
                    scores, table, request_ids, visible, selected, block_size, scores_ready,
                )
            else:
                scores_ready = score_full(
                    query, scales, weights, key, key_scale, table, request_ids, visible, scores,
                    block_size, start, query_ready, weights_ready,
                )
                selected_ready = select_index_topk(scores, table, request_ids, selected, block_size, scores_ready)

        counts = pl.create_tensor([block_size, 8], dtype=pl.INT32)
        with pl.spmd(block_size, name_hint="orthrus_index_mask", deps=[selected_ready]) as masked:
            token = pl.tile.get_block_idx()
            indices = pl.load(selected, [token, 0], [1, TOPK])
            valid = pl.tile.cmps(pl.cast(indices, pl.FP32), -1.0, cmp_type=4)
            zeros = pl.tile.full([1, TOPK], dtype=pl.FP32, value=0)
            invalid = pl.tile.full([1, TOPK], dtype=pl.FP32, value=-1e30)
            mask = pl.tile.select(valid, zeros, invalid)
            ones = pl.tile.full([1, TOPK], dtype=pl.FP32, value=1)
            items = pl.reshape(pl.tile.select(valid, ones, zeros), [8, TOPK // 8])
            scratch = pl.create_tile([8, TOPK // 8], dtype=pl.FP32)
            partials = pl.reshape(pl.row_sum(items, tmp_tile=scratch), [1, 8])
            total = pl.tile.read(partials, [0, 0])
            for lane in pl.unroll(1, 8):
                total = total + pl.tile.read(partials, [0, lane])
            packed = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
            pl.tile.write(packed, [0, 0], pl.cast(total, pl.INT32))
            pl.store(indices, [token, 0], rows)
            pl.store(mask, [token, 0], bias)
            pl.store(packed, [token, 0], counts)
        result_lengths = pl.reshape(lengths, [1, block_size])
        with pl.spmd(block_size // 8, name_hint="orthrus_index_lengths", deps=[masked]):
            begin = pl.tile.get_block_idx() * 8
            values = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
            for lane in pl.range(8):
                pl.tile.write(values, [0, lane], pl.read(counts, [begin + lane, 0]))
            pl.store(values, [0, begin], result_lengths)
        return rows, lengths, bias

    return select
