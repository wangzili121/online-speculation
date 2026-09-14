"""Ascend-friendly page selection for sparse Orthrus draft attention."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PillarPagePlan:
    """A compact paged-KV view consumed by draft attention."""

    block_table: torch.Tensor
    sequence_lengths: list[int]
    selected_blocks: list[int]
    total_blocks: list[int]


def _block_counts(history_lengths: list[int], block_size: int) -> list[int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not history_lengths or any(length <= 0 for length in history_lengths):
        raise ValueError("history lengths must be positive")
    return [math.ceil(length / block_size) for length in history_lengths]


def _budget_blocks(
    history_lengths: list[int],
    block_size: int,
    *,
    budget_ratio: float,
    min_tokens: int,
    max_tokens: int,
    dense_below_tokens: int,
) -> tuple[list[int], list[int]]:
    if not 0 < budget_ratio <= 1:
        raise ValueError("budget_ratio must be in (0, 1]")
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("require 0 < min_tokens <= max_tokens")
    if dense_below_tokens < 0:
        raise ValueError("dense_below_tokens must be non-negative")
    counts = _block_counts(history_lengths, block_size)
    budgets = []
    for length, count in zip(history_lengths, counts):
        if length < dense_below_tokens:
            budgets.append(count)
            continue
        token_budget = min(max_tokens, max(min_tokens, math.ceil(
            length * budget_ratio)))
        budgets.append(min(count, max(1, math.ceil(token_budget / block_size))))
    return counts, budgets


def compact_verifier_queries(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Keep only the first and last verifier query for every request."""

    if query.ndim != 3:
        raise ValueError("query must have [tokens,heads,head_dim]")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("query_start_loc must contain request boundaries")
    boundaries = query_start_loc.to(device=query.device, dtype=torch.long)
    first = boundaries[:-1]
    last = boundaries[1:] - 1
    return torch.stack(
        (query.index_select(0, first), query.index_select(0, last)), dim=1)


def recent_page_plan(
    block_table: torch.Tensor,
    history_lengths: list[int],
    block_size: int,
    *,
    budget_ratio: float,
    min_tokens: int,
    max_tokens: int,
    dense_below_tokens: int,
) -> PillarPagePlan:
    """Select only the most recent pages at the requested sparse budget."""

    counts, budgets = _budget_blocks(
        history_lengths, block_size, budget_ratio=budget_ratio,
        min_tokens=min_tokens, max_tokens=max_tokens,
        dense_below_tokens=dense_below_tokens)
    width = max(budgets)
    selected = torch.full(
        (len(history_lengths), width), -1, dtype=torch.int32,
        device=block_table.device)
    lengths = []
    for row, (length, count, budget) in enumerate(zip(
            history_lengths, counts, budgets)):
        selected[row, :budget] = block_table[
            row, count - budget:count].to(torch.int32)
        final_page_tokens = length - (count - 1) * block_size
        lengths.append((budget - 1) * block_size + final_page_tokens)
    return PillarPagePlan(selected, lengths, budgets, counts)


def verifier_guided_page_plan(
    query_by_kv_head: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    history_lengths: list[int],
    *,
    budget_ratio: float,
    min_tokens: int,
    max_tokens: int,
    dense_below_tokens: int,
) -> PillarPagePlan:
    """Select old pages with a verifier-query proxy and retain the newest page.

    SparseSpec rematerializes every verifier attention score and selects tokens
    independently per layer and KV head. Ascend's Qwen GQA attention does not
    expose those scores, so this portable probe scores the final key of each
    page with the latest verifier query. One shared page set is produced for
    each request; the full verifier remains untouched.
    """

    if key_cache.ndim != 4:
        raise ValueError("key_cache must have [blocks,block,kv_heads,head_dim]")
    block_size = key_cache.shape[1]
    counts, budgets = _budget_blocks(
        history_lengths, block_size, budget_ratio=budget_ratio,
        min_tokens=min_tokens, max_tokens=max_tokens,
        dense_below_tokens=dense_below_tokens)
    batch = len(history_lengths)
    if query_by_kv_head.shape != (batch, key_cache.shape[2], key_cache.shape[3]):
        raise ValueError(
            "query_by_kv_head must have [batch,kv_heads,head_dim]")
    if block_table.shape[0] != batch or block_table.shape[1] < max(counts):
        raise ValueError("block_table does not cover every history")

    width = max(budgets)
    max_old_selected = max(budget - 1 for budget in budgets)
    logical_width = max(counts)
    table = block_table[:, :logical_width].to(torch.long)
    positions = torch.arange(logical_width, device=table.device)[None]
    count_tensor = torch.tensor(counts, device=table.device)
    budget_tensor = torch.tensor(budgets, device=table.device)

    selected = torch.full(
        (batch, width), -1, dtype=torch.int32, device=table.device)
    if max_old_selected:
        representatives = key_cache[
            table.clamp_min(0), block_size - 1]
        scores = (query_by_kv_head[:, None] * representatives).sum(
            dim=(-1, -2))
        old_mask = (table < 0) | (positions >= (count_tensor - 1)[:, None])
        scores = scores.masked_fill(old_mask, -torch.inf)
        # sorted=True keeps valid pages before the -inf padding of short rows.
        picked_positions = torch.topk(
            scores, k=max_old_selected, dim=1, sorted=True).indices
        picked_blocks = table.gather(1, picked_positions).to(torch.int32)
        selected[:, :max_old_selected] = picked_blocks
        selected[:, :max_old_selected].masked_fill_(
            positions[:, :max_old_selected]
            >= (budget_tensor - 1)[:, None], -1)

    newest_blocks = table.gather(1, (count_tensor - 1)[:, None]).to(torch.int32)
    selected.scatter_(1, (budget_tensor - 1)[:, None], newest_blocks)
    lengths = [
        (budget - 1) * block_size + length - (count - 1) * block_size
        for length, count, budget in zip(history_lengths, counts, budgets)
    ]
    return PillarPagePlan(selected, lengths, budgets, counts)


def vegas_page_plan(
    verifier_queries: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    history_lengths: list[int],
    *,
    budget_ratio: float,
    min_tokens: int,
    max_tokens: int,
    dense_below_tokens: int,
    page_chunk: int = 16,
) -> PillarPagePlan:
    """Select pages from per-layer verifier attention evidence.

    VEGAS selects individual draft tokens from attention weights emitted by
    the dense verifier. Ascend FIA does not expose those weights, so this
    portable reference recomputes QK scores for the verifier's first and last
    query. It scores every token in every old page, averages across verifier
    queries and GQA heads, and ranks a page by its most important token. The
    newest (possibly partial) page is always retained.

    Unlike :func:`verifier_guided_page_plan`, callers build this plan for each
    transformer layer. The implementation is intentionally chunked to bound
    temporary memory; a future fused FIA implementation can produce the same
    metadata without a second QK pass.
    """

    if key_cache.ndim != 4:
        raise ValueError("key_cache must have [blocks,block,kv_heads,head_dim]")
    if page_chunk <= 0:
        raise ValueError("page_chunk must be positive")
    batch = len(history_lengths)
    query_heads = verifier_queries.shape[2] if verifier_queries.ndim == 4 else -1
    kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    if (verifier_queries.ndim != 4
            or verifier_queries.shape[:2] != (batch, 2)
            or verifier_queries.shape[3] != head_dim
            or query_heads % kv_heads):
        raise ValueError(
            "verifier_queries must have [batch,2,query_heads,head_dim] "
            "with query_heads divisible by kv_heads")

    block_size = key_cache.shape[1]
    counts, budgets = _budget_blocks(
        history_lengths, block_size, budget_ratio=budget_ratio,
        min_tokens=min_tokens, max_tokens=max_tokens,
        dense_below_tokens=dense_below_tokens)
    if block_table.shape[0] != batch or block_table.shape[1] < max(counts):
        raise ValueError("block_table does not cover every history")

    selected = torch.full(
        (batch, max(budgets)), -1, dtype=torch.int32,
        device=block_table.device)
    repeats = query_heads // kv_heads
    logical_width = max(counts)
    table = block_table[:, :logical_width].to(torch.long)
    count_tensor = torch.tensor(counts, device=table.device)
    old_width = max(count - 1 for count in counts)
    old_budget_width = max(budget - 1 for budget in budgets)
    if old_budget_width:
        # Mean reductions are linear, so reduce first/last queries and the
        # GQA groups before QK. This is exactly equivalent to materializing
        # [query,page,token,kv_head,gqa_group] scores and then averaging them.
        queries = verifier_queries.float().view(
            batch, 2, kv_heads, repeats, head_dim).mean(dim=(1, 3))
        page_scores = []
        for start in range(0, old_width, page_chunk):
            stop = min(start + page_chunk, old_width)
            physical_blocks = table[:, start:stop].clamp_min(0)
            keys = key_cache[physical_blocks].float()
            token_scores = torch.einsum(
                "bhd,bpthd->bpt", queries, keys) / kv_heads
            scores = token_scores.amax(dim=2)
            positions = torch.arange(start, stop, device=table.device)[None]
            scores.masked_fill_(positions >= (count_tensor - 1)[:, None],
                                -torch.inf)
            page_scores.append(scores)
        scores = torch.cat(page_scores, dim=1)
        picked = torch.topk(
            scores, k=old_budget_width, dim=1, sorted=False).indices

    for row, (count, budget) in enumerate(zip(counts, budgets)):
        logical_blocks = table[row, :count]
        if budget == count:
            selected[row, :count] = logical_blocks.to(torch.int32)
            continue
        old_budget = budget - 1
        if old_budget:
            positions = picked[row, :old_budget].sort().values
            selected[row, :old_budget] = logical_blocks.index_select(
                0, positions).to(torch.int32)
        selected[row, old_budget] = logical_blocks[-1].to(torch.int32)

    lengths = [
        (budget - 1) * block_size + length - (count - 1) * block_size
        for length, count, budget in zip(history_lengths, counts, budgets)
    ]
    return PillarPagePlan(selected, lengths, budgets, counts)


def reuse_vegas_page_plan(
    cached_old_blocks: list[torch.Tensor | None],
    block_table: torch.Tensor,
    history_lengths: list[int],
    block_size: int,
    *,
    budget_ratio: float,
    min_tokens: int,
    max_tokens: int,
    dense_below_tokens: int,
) -> PillarPagePlan:
    """Reuse verifier-selected old pages and advance the newest page.

    Dense rows always use the current block table. Sparse rows reuse the old
    importance pages from a previous verifier step while taking their newest
    page from the current table, so newly committed Orthrus tokens remain
    visible without another QK scan.
    """

    counts, budgets = _budget_blocks(
        history_lengths, block_size, budget_ratio=budget_ratio,
        min_tokens=min_tokens, max_tokens=max_tokens,
        dense_below_tokens=dense_below_tokens)
    if len(cached_old_blocks) != len(history_lengths):
        raise ValueError("cached_old_blocks must contain one row per request")
    if (block_table.shape[0] != len(history_lengths)
            or block_table.shape[1] < max(counts)):
        raise ValueError("block_table does not cover every history")

    selected = torch.full(
        (len(history_lengths), max(budgets)), -1, dtype=torch.int32,
        device=block_table.device)
    for row, (count, budget, cached) in enumerate(zip(
            counts, budgets, cached_old_blocks)):
        logical_blocks = block_table[row, :count].to(torch.long)
        if budget == count:
            selected[row, :count] = logical_blocks.to(torch.int32)
            continue
        if cached is None or cached.numel() != budget - 1:
            raise ValueError("cached VEGAS row does not match its current budget")
        selected[row, :budget - 1] = cached.to(
            device=selected.device, dtype=torch.int32)
        selected[row, budget - 1] = logical_blocks[-1].to(torch.int32)

    lengths = [
        (budget - 1) * block_size + length - (count - 1) * block_size
        for length, count, budget in zip(history_lengths, counts, budgets)
    ]
    return PillarPagePlan(selected, lengths, budgets, counts)
