"""Native Ascend QLI over a committed prefix, without a draft cache append."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class IndexPages:
    key: torch.Tensor  # [physical page, rows per page, 1, 128], INT8
    scale: torch.Tensor  # [physical page, rows per page, 1], FP16
    block_table: torch.Tensor


def select_projected_history(
    query: torch.Tensor,
    weights: torch.Tensor,
    anchors: torch.Tensor,
    pages: IndexPages,
    *,
    ratio: Literal[1, 2],
    topk: int,
    candidate_mode: Literal["plain", "produce", "consume"] = "plain",
    candidates: torch.Tensor | None = None,
    candidate_topk_blocks: int = 0,
    candidate_block_size: int = 0,
    maximum_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select from target index pages on A2/A3, using one anchor per request.

    Query is post-RoPE [B,K,H,128], and weights [B,K,H] include indexer
    scaling as returned by the native indexer's project_weights. Anchors
    give the committed token counts, excluding the anchor itself.
    """
    from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
    from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query

    batch, block, heads, dim = query.shape
    if ratio not in (1, 2) or not 1 <= block <= 32 or not 1 <= topk <= 2048:
        raise ValueError("invalid V4.1 ratio, draft width or Top-K")
    if weights.shape != query.shape[:3] or anchors.shape != (batch,) or pages.block_table.shape[0] != batch:
        raise ValueError("incompatible draft indexer batch")
    mode = {"produce": 1, "consume": 2, "plain": 3}[candidate_mode]
    if mode != 3 and (ratio != 1 or candidate_block_size != 8 or candidate_topk_blocks < 1):
        raise ValueError("C1A candidate modes require an 8-row block and a positive limit")
    candidate_shape = (batch * block, 1, candidate_topk_blocks)
    if mode == 2 and (candidates is None or candidates.shape != candidate_shape or candidates.dtype != torch.int32):
        raise ValueError("C1A Reindex requires the draft candidate pool for these query rows")

    # QLI sees logical index rows as its coordinates. The runtime has already
    # compressed the keys. Using a prefix length in rows makes the exact same
    # set visible for all queries, including the first row of an odd C2A prefix.
    lengths = (anchors // ratio).int()
    max_rows = int(lengths.max().item()) if maximum_length is None else maximum_length
    if max_rows < 0:
        raise ValueError("maximum index history length must be non-negative")
    if max_rows == 0:
        if mode == 1:
            candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
        return torch.full((batch, block, topk), -1, dtype=torch.int32, device=query.device), candidates

    quantized, query_scale = quantize_indexer_query(query.flatten(0, 1))
    kwargs = dict(
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=query.device) * block,
        seqused_k=lengths,
        max_seqlen_q=block,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
    )
    ops = torch.ops._C_ascend
    if hasattr(ops, "npu_quant_lightning_indexer_v3"):
        select_op = ops.npu_quant_lightning_indexer_v3
    else:
        select_op = ops.npu_quant_lightning_indexer_v2
    # The V3 launch uses the same metadata builder on A2/A3.
    metadata = ops.npu_quant_lightning_indexer_v2_metadata(
        heads, 1, dim, topk, 2, batch_size=batch, max_seqlen_k=max_rows, **kwargs,
    )
    selected, _, candidate_out = select_op(
        quantized, pages.key, weights.flatten(0, 1).to(torch.float16), query_scale, pages.scale,
        topk, 2, block_table=pages.block_table, metadata=metadata,
        candidate_topk_index=candidates if mode == 2 else None,
        candidate_mode=mode, candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size, **kwargs,
    )
    last_visible = (lengths - 1).repeat_interleave(block)
    selected = selected.squeeze(1)
    if mode == 2:
        # When the candidate domain is shorter than Top-K, QLI may fill a
        # slot with row 0. Check actual membership before sorting.
        pool = candidates.squeeze(1).sort(dim=-1).values
        blocks = selected // candidate_block_size
        locations = torch.searchsorted(pool, blocks.contiguous()).clamp_max(candidate_topk_blocks - 1)
        selected = selected.masked_fill(pool.gather(-1, locations.long()) != blocks, -1)
        selected = selected.sort(dim=-1).values
        duplicate = selected[:, 1:] == selected[:, :-1]
        selected = torch.cat((selected[:, :1], selected[:, 1:].masked_fill(duplicate, -1)), dim=-1)
    selected = prepare_indexer_indices(selected, last_visible, 1)
    return selected.reshape(batch, block, topk), candidate_out if mode == 1 else candidates
