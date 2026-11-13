"""Torch reference for V4.1 Orthrus index selection and latent attention.

Inputs are logical views of dequantized target pages and post-RoPE queries.
The reference leaves native cache layout, projections and output RoPE to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn.functional as F

from .dsv41_contract import BLOCK_SIZE, V41Topology
from .dsv4_hybrid import gather_paged_cache, tail_logical_indices


@dataclass(frozen=True)
class PagedRows:
    values: torch.Tensor  # [physical page, row within page, channels]
    table: torch.Tensor  # [request, logical page], entries are physical page IDs

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        if self.values.ndim != 3 or self.values.shape[1] == 0:
            raise ValueError("expected [page, row, channel] cache with positive page width")
        if self.table.ndim != 2 or indices.ndim not in (2, 3) or self.table.shape[0] != indices.shape[0]:
            raise ValueError("page table and row IDs must describe the same requests")
        valid = indices >= 0
        if not bool(valid.any()):
            return self.values.new_zeros((*indices.shape, self.values.shape[-1]))
        logical = indices.clamp_min(0) // self.values.shape[1]
        if bool((logical >= self.table.shape[1]).any()):
            raise ValueError("logical row exceeds page table capacity")
        table = self.table.masked_fill(
            (self.table < 0) | (self.table >= self.values.shape[0]), -1
        )
        return gather_paged_cache(self.values, table, indices, validate=True)


@dataclass(frozen=True)
class DraftSelection:
    """Private, per-round logical indices. Never aliases the target selection buffers."""

    topk: Mapping[int, torch.Tensor] = field(default_factory=dict)
    candidates: torch.Tensor | None = None


def _candidate_rows(
    scores: torch.Tensor, lengths: torch.Tensor, block_size: int, limit: int
) -> torch.Tensor:
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    blocks = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    blocks = blocks.unflatten(-1, (-1, block_size)).amax(dim=-1)
    block_ids = torch.arange(blocks.shape[-1], device=scores.device)
    latest = (lengths - 1) // block_size
    blocks = blocks.masked_fill(block_ids == latest[:, None, None], torch.inf)
    ids = blocks.argsort(dim=-1, descending=True, stable=True)[..., :limit]
    eligible = blocks.gather(-1, ids) > -torch.inf
    chosen = torch.zeros_like(blocks, dtype=torch.bool).scatter(-1, ids, eligible)
    return chosen.repeat_interleave(block_size, dim=-1)[..., :width]


def _logical_topk(scores: torch.Tensor, limit: int) -> torch.Tensor:
    # Resolve exact ties by smaller logical row ID so physical page order is irrelevant.
    width = scores.shape[-1]
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :limit]
    ids = ids.masked_fill(scores.gather(-1, ids) == -torch.inf, width)
    ids = ids.sort(dim=-1).values
    ids = ids.masked_fill(ids == width, -1)
    return F.pad(ids, (0, limit - ids.shape[-1]), value=-1)


def select_history(
    topology: V41Topology,
    layer: int,
    anchors: torch.Tensor,
    state: DraftSelection,
    *,
    query: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    index_keys: Mapping[int, PagedRows] | None = None,
) -> tuple[torch.Tensor | None, DraftSelection]:
    """Select only complete target history rows preceding each anchor.

    Indexer query is [B,K,H,D]. Weights are [B,K,H] before index score
    scaling. Source layer IDs address the read-only index key pages.
    """
    role = topology.layers[layer]
    if role.mode == "swa":
        return None, state
    if role.mode == "reuse":
        if role.index_source not in state.topk:
            raise ValueError(f"draft index source {role.index_source} has not run")
        return state.topk[role.index_source], state
    if query is None or weights is None or index_keys is None or role.kv_source not in index_keys:
        raise ValueError(f"layer {layer} requires draft indexer inputs and target index pages")
    if query.ndim != 4 or query.shape[:3] != weights.shape or anchors.shape != query.shape[:1]:
        raise ValueError("incompatible draft indexer query and weights")
    if bool((anchors < 0).any()) or not 1 <= query.shape[1] <= BLOCK_SIZE:
        raise ValueError("invalid anchor or draft block width")
    if role.mode == "reindex" and state.candidates is None:
        raise ValueError(f"draft candidate source {role.candidate_source} has not run")

    # An incomplete compression group belongs to target staging state.
    lengths = anchors // role.ratio
    rows = torch.arange(int(lengths.max().item()), device=anchors.device)
    visible = rows < lengths[:, None]
    keys = index_keys[role.kv_source].gather(torch.where(visible, rows, -1))
    scores = torch.einsum("bqhd,brd->bqhr", query.float(), keys.float()).relu()
    scale = (query.shape[-1] * query.shape[-2]) ** -0.5
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2) * scale
    scores = scores.masked_fill(~visible[:, None, :], -torch.inf)

    candidates = state.candidates
    if layer == role.candidate_source:
        candidates = _candidate_rows(
            scores, lengths, topology.candidate_block_size, topology.candidate_blocks
        )
    elif role.mode == "reindex":
        scores = scores.masked_fill(~candidates, -torch.inf)
    selected = _logical_topk(scores, topology.topk)
    return selected, DraftSelection({**state.topk, layer: selected}, candidates)


def draft_attention(
    query: torch.Tensor,
    temporary_kv: torch.Tensor,
    swa: PagedRows,
    anchors: torch.Tensor,
    sink: torch.Tensor,
    *,
    window: int,
    global_kv: PagedRows | None = None,
    global_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference output and LSE for a joint SWA + sparse history + draft block softmax.

    Query is [B,K,H,D], temporary KV is [B,K,D], and the sink is [H].
    All K rows see the same SWA history and the entire temporary block.
    The global history selected by the indexer can differ by query.
    """
    batch, block, heads, dim = query.shape
    if not 1 <= block <= BLOCK_SIZE or temporary_kv.shape != (batch, block, dim):
        raise ValueError("incompatible temporary KV or draft block size")
    if sink.shape != (heads,) or anchors.shape != (batch,):
        raise ValueError("incompatible anchors or attention sink")
    if bool((anchors < 0).any()):
        raise ValueError("anchor positions must be non-negative")
    if (global_indices is None) != (global_kv is None):
        raise ValueError("global history requires both cache pages and indices")
    local_ids = tail_logical_indices(anchors, window)
    local = swa.gather(local_ids)
    parts = [local[:, None].expand(-1, block, -1, -1)]
    valid = [(local_ids >= 0)[:, None].expand(-1, block, -1)]
    if global_kv is not None:
        parts.append(global_kv.gather(global_indices))
        valid.append(global_indices >= 0)
    parts.append(temporary_kv[:, None].expand(-1, block, -1, -1))
    valid.append(torch.ones((batch, block, block), device=query.device, dtype=torch.bool))
    kv = torch.cat(parts, dim=2).float()
    logits = torch.einsum("bqhd,bqrd->bqhr", query.float(), kv) * dim**-0.5
    logits = logits.masked_fill(~torch.cat(valid, dim=-1).unsqueeze(2), -torch.inf)
    # The sink is a virtual row with a zero value. It contributes to the denominator once.
    sink_logits = sink.float().view(1, 1, heads, 1).expand(batch, block, -1, -1)
    lse = torch.logsumexp(torch.cat((logits, sink_logits), dim=-1), dim=-1, keepdim=True)
    output = torch.einsum("bqhr,bqrd->bqhd", (logits - lse).exp(), kv)
    return output, lse


def attend_layer(
    topology: V41Topology,
    layer: int,
    query: torch.Tensor,
    temporary_kv: torch.Tensor,
    swa: PagedRows,
    anchors: torch.Tensor,
    sink: torch.Tensor,
    *,
    global_pages: Mapping[int, PagedRows],
    selection: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the global KV owner independently of the most recent indexer."""
    role = topology.layers[layer]
    pages = None if role.kv_source is None else global_pages[role.kv_source]
    if selection is not None and (
        pages is None
        or selection.ndim != 3
        or selection.shape[:2] != query.shape[:2]
        or bool((selection >= (anchors // role.ratio)[:, None, None]).any())
    ):
        raise ValueError("global selection is outside this draft's committed history")
    return draft_attention(
        query, temporary_kv, swa, anchors, sink, window=topology.window,
        global_kv=pages, global_indices=selection,
    )
