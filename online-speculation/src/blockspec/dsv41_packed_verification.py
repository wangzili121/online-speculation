"""Greedy verification and cache commit for the packed A3 graph."""

from dataclasses import dataclass

from .pypto_dsv41_greedy import verified_prefix


def verify_packed_round(anchor_token, candidates, writes, *, forward, save, restore):
    """Verify a draft block whose anchor token is still pending in the cache.

    The caller binds the same request-local packed cache to save, forward and
    restore. The snapshot covers writes for the anchor and every candidate,
    including the rolling compressor state. A correction or bonus token
    becomes the next round's pending anchor.
    """
    candidates = tuple(candidates)
    inputs = (anchor_token, *candidates)
    journal = save(tuple(writes))
    try:
        target = tuple(forward(inputs))
        if len(target) != len(inputs):
            raise ValueError("the verifier must produce one prediction for each input row")
        emitted = verified_prefix(candidates, target)
    except Exception:
        restore(journal)
        raise
    accepted = len(emitted) - 1
    if accepted == len(candidates):
        return emitted, accepted

    restore(journal)
    try:
        forward(inputs[:accepted + 1])
    except Exception:
        restore(journal)
        raise
    return emitted, accepted


@dataclass(frozen=True)
class CacheWrite:
    """A contiguous byte interval inside a rank-local cache allocation."""

    name: str
    offset: int
    size: int


def verifier_cache_writes(
    model_config, caches, *, anchor, tokens, window_pages, c1a_pages, c2a_pages, state_block,
):
    """Plan a byte-exact journal for an A3 multi-token verifier call.

    Cache payloads and scales are separate allocations. The C2A compressor
    also overwrites a rolling prefix state, so shrinking a logical length
    alone cannot roll back a rejected candidate.
    """
    if anchor < 0 or tokens <= 0:
        raise ValueError("verification requires a nonempty token span")

    def intervals(name, source, start, end, table):
        value = caches[name]
        page_count, rows_per_page = value.shape[1:3]
        position = start
        while position < end:
            logical, row = divmod(position, rows_per_page)
            if logical >= len(table) or not 0 <= table[logical] < page_count:
                raise ValueError(f"{name} write leaves its physical page table")
            page = table[logical]
            count = min(rows_per_page - row, end - position)
            item_size = value.element_size()
            first = source * value.stride(0) + page * value.stride(1) + row * value.stride(2)
            yield CacheWrite(name, first * item_size, count * value.stride(2) * item_size)
            position += count

    end = anchor + tokens
    for layer in range(model_config.num_hidden_layers):
        for name in ("window_cache", "window_scales"):
            yield from intervals(name, layer, anchor, end, window_pages)

    state_source = 0
    for slot, layer in enumerate(model_config.kv_source_layer_ids):
        ratio = model_config.compress_ratios[layer]
        pages = c2a_pages if ratio == 2 else c1a_pages
        for name in ("compressed_cache", "compressed_scales", "index_cache", "index_scales"):
            yield from intervals(name, slot, anchor // ratio, end // ratio, pages)
        if ratio == 2:
            state = caches["state_cache"]
            if not 0 <= state_block < state.shape[1]:
                raise ValueError("C2A state write leaves its physical page table")
            # Four small rows per source. Saving all four covers chunk and
            # decode compressor variants with no per-position host work.
            item_size = state.element_size()
            first = state_source * state.stride(0) + state_block * state.stride(1)
            yield CacheWrite("state_cache", first * item_size, state.stride(1) * item_size)
            state_source += 1
