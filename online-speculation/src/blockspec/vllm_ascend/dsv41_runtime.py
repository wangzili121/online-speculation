"""Read-only V4.1 target-history binding and one-layer Orthrus execution.

This module deliberately contains no vLLM imports.  The model adapter supplies
the live target layer and its already-built MRV2 metadata; this module turns
them into the exact cache views consumed by the native QLI/SMLA path.  Keeping
the boundary small makes cache ownership testable without constructing the
full V4.1 model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .dsv41_attention import PagedRows
from .dsv41_contract import LayerRole, V41Topology
from .dsv41_native_attention import native_draft_attention
from .dsv41_native_index import IndexPages, select_projected_history


@dataclass(frozen=True)
class V41CacheNames:
    """Target-owned cache resources read by one diffusion layer."""

    swa: str
    long_kv: str | None
    index_k: str | None


@dataclass(frozen=True)
class V41TargetHistory:
    """One layer's immutable target-cache views for a K-wide draft round."""

    swa: PagedRows
    anchors: torch.Tensor
    long_kv: PagedRows | None = None
    index_k: IndexPages | None = None
    global_lengths: torch.Tensor | None = None


@dataclass(frozen=True)
class V41NativeDraftState:
    """Draft-private QLI handoff; never aliases target selection buffers."""

    topk: Mapping[int, torch.Tensor] = field(default_factory=dict)
    candidates: torch.Tensor | None = None


def compose_draft_lookback(
    target_input_ids: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    previous_lookback: torch.Tensor,
    num_rejected: torch.Tensor,
) -> torch.Tensor:
    """Advance Engram history from the target step to the Orthrus anchor.

    ``previous_lookback`` is newest-first history immediately before the
    target's scheduled rows.  Rejected suffix rows are discarded, surviving
    target rows are prepended newest-first, and the fixed Engram depth is
    preserved.  The operation is entirely on device and introduces no host
    synchronization.
    """

    if target_input_ids.ndim != 1 or target_query_start_loc.ndim != 1:
        raise ValueError("target tokens and query boundaries must be one-dimensional")
    batch, depth = previous_lookback.shape
    if target_query_start_loc.shape != (batch + 1,) or num_rejected.shape != (batch,):
        raise ValueError("Engram lookback batch metadata is inconsistent")
    starts = target_query_start_loc[:-1].long()
    ends = target_query_start_loc[1:].long() - num_rejected.long()
    lengths = (ends - starts).clamp_min(0)
    offsets = torch.arange(depth, device=target_input_ids.device).view(1, -1)
    current_positions = ends[:, None] - 1 - offsets
    safe_current = current_positions.clamp(0, max(target_input_ids.numel() - 1, 0))
    if target_input_ids.numel():
        current = target_input_ids[safe_current]
    else:
        current = previous_lookback.new_full((batch, depth), -1)
    old_offsets = (offsets - lengths[:, None]).clamp(0, max(depth - 1, 0))
    old = previous_lookback.gather(1, old_offsets)
    return torch.where(offsets < lengths[:, None], current, old)


def cache_names(topology: V41Topology, layer: int, *, root: str = "model") -> V41CacheNames:
    """Return the live target resource names for a diffusion layer."""

    role = topology.layers[layer]
    prefix = f"{root}.layers.{layer}.self_attn"
    if role.kv_source is None:
        return V41CacheNames(f"{prefix}.swa_cache", None, None)
    source = f"{root}.layers.{role.kv_source}.self_attn"
    return V41CacheNames(
        f"{prefix}.swa_cache",
        f"{source}.long_kv_cache",
        f"{source}.indexer.k_cache",
    )


def target_cache_names(topology: V41Topology, *, root: str = "model") -> tuple[str, ...]:
    """Return each target cache resource once, in deterministic layer order."""

    names: list[str] = []
    seen: set[str] = set()
    for layer in range(len(topology.layers)):
        resources = cache_names(topology, layer, root=root)
        for name in (resources.swa, resources.long_kv, resources.index_k):
            if name is not None and name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)


def build_readonly_metadata(
    groups: list[Any],
    common_metadata: Any,
    block_tables: Mapping[int, torch.Tensor],
    *,
    num_query_heads: int,
) -> dict[str, Any]:
    """Build MRV2 metadata for a non-causal K block with every write disabled.

    vLLM-Ascend 0.32's V4.1 builder does not expose ``update_cache=False`` as
    a formal argument, while newer builders do.  Passing the flag is harmless
    on 0.32 (``build`` forwards optional keywords), and replacing every slot
    with PAD is the version-independent write barrier.
    """

    if num_query_heads < 1:
        raise ValueError("num_query_heads must be positive")
    slots = getattr(common_metadata, "slot_mapping", None)
    if not torch.is_tensor(slots):
        raise ValueError("common draft metadata has no slot mapping")
    shared_batch: dict[str, Any] = {}
    per_layer: dict[str, Any] = {}
    for group in groups:
        group_id = group.kv_cache_group_id
        block_table = block_tables.get(group_id)
        if not torch.is_tensor(block_table):
            raise ValueError(f"missing block table for V4.1 cache group {group_id}")
        metadata_input = copy.copy(common_metadata)
        metadata_input.block_table_tensor = block_table[: metadata_input.num_reqs]
        metadata_input.slot_mapping = torch.full_like(slots, -1)
        metadata_input.causal = False
        metadata = group.get_metadata_builder().build(
            0,
            metadata_input,
            common_v41_metadata={},
            common_v41_batch_metadata=shared_batch,
            build_query=True,
            update_cache=False,
            skip_ring_state_update=True,
            build_compressor_metadata=False,
            num_query_heads=num_query_heads,
        )
        if hasattr(metadata, "causal"):
            metadata.causal = False
        for layer_name in group.layer_names:
            if layer_name in per_layer:
                raise ValueError(f"duplicate V4.1 cache metadata for {layer_name}")
            per_layer[layer_name] = metadata
    return per_layer


def _cache_value(owner: Any, label: str) -> Any:
    values = getattr(owner, "kv_cache", None)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError(f"{label} is not bound to exactly one target cache")
    return values[0]


def _rows(value: Any, label: str) -> torch.Tensor:
    """Normalize the A2/A3 paged latent layout to [page,row,channel]."""

    if isinstance(value, (tuple, list)):
        raise NotImplementedError(
            f"{label} uses a packed/quantized cache ABI; the current A2/A3 "
            "Orthrus path requires a directly addressable latent cache"
        )
    if not torch.is_tensor(value):
        raise TypeError(f"{label} cache is not a tensor")
    if value.ndim == 4 and value.shape[2] == 1:
        value = value.squeeze(2)
    if value.ndim != 3:
        raise ValueError(f"{label} must have [page,row,channel] storage")
    return value


def _table(metadata: Any, label: str) -> torch.Tensor:
    table = getattr(metadata, "block_table", None)
    if not torch.is_tensor(table) or table.ndim != 2:
        raise ValueError(f"{label} metadata has no two-dimensional block table")
    return table


def _lengths(metadata: Any, label: str) -> torch.Tensor:
    lengths = getattr(metadata, "cache_seq_lens", None)
    if lengths is None:
        lengths = getattr(metadata, "seq_lens", None)
    if not torch.is_tensor(lengths) or lengths.ndim != 1:
        raise ValueError(f"{label} metadata has no one-dimensional cache lengths")
    return lengths


def bind_target_history(
    target_model: Any,
    metadata_by_prefix: Mapping[str, Any],
    topology: V41Topology,
    layer: int,
    anchors: torch.Tensor,
) -> V41TargetHistory:
    """Bind one diffusion layer to live target pages without exposing writes.

    ``anchors`` are committed target-token counts, one per request.  The
    metadata must have been built for those exact lengths with cache updates
    disabled by the vLLM adapter.
    """

    role = topology.layers[layer]
    if anchors.ndim != 1 or anchors.dtype not in (torch.int32, torch.int64):
        raise ValueError("anchors must be a one-dimensional integer tensor")
    inner = getattr(target_model, "model", target_model)
    layers = getattr(inner, "layers", None)
    if layers is None or len(layers) != len(topology.layers):
        raise ValueError("target model and V4.1 topology have different layers")

    attention = layers[layer].self_attn
    impl = getattr(attention, "v41_impl", None)
    if impl is None or not hasattr(impl, "_get_layer_metadata"):
        raise TypeError(f"target layer {layer} does not expose the V4.1 metadata contract")
    metadata = impl._get_layer_metadata(metadata_by_prefix)
    swa_meta = metadata.swa
    swa_lengths = _lengths(swa_meta, "SWA")[: anchors.numel()].to(anchors)
    if not torch.equal(swa_lengths, anchors):
        raise ValueError("SWA metadata does not describe the committed anchor history")
    swa_owner = attention.dsa_attn.swa_cache_layer
    swa = PagedRows(_rows(_cache_value(swa_owner, "SWA"), "SWA"), _table(swa_meta, "SWA"))

    if role.kv_source is None:
        return V41TargetHistory(swa=swa, anchors=anchors)

    source_attention = layers[role.kv_source].self_attn
    long_owner = getattr(source_attention, "long_kv_cache", None)
    indexer = getattr(source_attention, "indexer", None)
    index_owner = getattr(indexer, "k_cache", None)
    if long_owner is None or index_owner is None:
        raise RuntimeError(f"V4.1 source layer {role.kv_source} has incomplete target caches")
    if metadata.attention is None or metadata.indexer is None:
        raise RuntimeError(f"V4.1 layer {layer} has incomplete cache metadata")

    expected = anchors.div(role.ratio, rounding_mode="floor")
    long_lengths = _lengths(metadata.attention, "long KV")[: anchors.numel()].to(expected)
    index_lengths = _lengths(metadata.indexer.cache, "index K")[: anchors.numel()].to(expected)
    if not torch.equal(long_lengths, expected) or not torch.equal(index_lengths, expected):
        raise ValueError("compressed metadata includes incomplete or future target rows")

    long_kv = PagedRows(
        _rows(_cache_value(long_owner, "long KV"), "long KV"),
        _table(metadata.attention, "long KV"),
    )
    index_value = _cache_value(index_owner, "index K")
    if not isinstance(index_value, (tuple, list)) or len(index_value) != 2:
        raise ValueError("A2/A3 index K cache must contain INT8 keys and FP16 scales")
    key, scale = index_value
    if key.ndim != 4 or key.shape[2] != 1 or scale.ndim not in (3, 4):
        raise ValueError("unexpected index K cache layout")
    if scale.ndim == 4 and scale.shape[2:] == (1, 1):
        scale = scale.squeeze(2)
    if scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.ndim != 3 or scale.shape[-1] != 1:
        raise ValueError("index K scale must have one scalar per cache row")
    index_k = IndexPages(key, scale, _table(metadata.indexer.cache, "index K"))
    return V41TargetHistory(swa, anchors, long_kv, index_k, expected)


def run_native_layer(
    topology: V41Topology,
    layer: int,
    query: torch.Tensor,
    temporary_kv: torch.Tensor,
    sink: torch.Tensor,
    history: V41TargetHistory,
    state: V41NativeDraftState,
    *,
    index_query: torch.Tensor | None = None,
    index_weights: torch.Tensor | None = None,
    runtime_cmp_ratio: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, V41NativeDraftState]:
    """Run QLI when owned by this layer, then joint-history Orthrus attention."""

    role: LayerRole = topology.layers[layer]
    selection = None
    candidates = state.candidates
    topk = dict(state.topk)
    if role.kv_source is not None:
        if role.mode == "reuse":
            if role.index_source not in state.topk:
                raise ValueError(f"draft index source {role.index_source} has not run")
            selection = state.topk[role.index_source]
        else:
            if index_query is None or index_weights is None or history.index_k is None:
                raise ValueError(f"layer {layer} requires diffusion indexer projections")
            if role.mode == "reindex" and candidates is None:
                raise ValueError(f"layer {layer} requires the draft candidate pool")
            candidate_mode = (
                "produce" if layer == role.candidate_source else "consume" if role.mode == "reindex" else "plain"
            )
            selection, candidates = select_projected_history(
                index_query,
                index_weights,
                history.anchors,
                history.index_k,
                ratio=role.ratio,
                topk=topology.topk,
                candidate_mode=candidate_mode,
                candidates=candidates,
                candidate_topk_blocks=topology.candidate_blocks,
                candidate_block_size=topology.candidate_block_size,
            )
            topk[layer] = selection
    next_state = V41NativeDraftState(topk, candidates)

    output, lse = native_draft_attention(
        query,
        temporary_kv,
        history.swa,
        history.anchors,
        sink,
        window=topology.window,
        global_kv=history.long_kv,
        global_indices=selection,
        global_lengths=history.global_lengths,
        max_swa_history=int(history.anchors.max().item()) if history.anchors.numel() else 0,
        max_global_history=(
            int(history.global_lengths.max().item())
            if history.global_lengths is not None and history.global_lengths.numel()
            else 0
        ),
        runtime_cmp_ratio=runtime_cmp_ratio,
    )
    return output, lse, next_state
