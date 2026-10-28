"""Exact read-only attention primitives for the DSV4 Orthrus backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def front_pack_indices(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move valid non-negative indices to the front of the last dimension."""
    if indices.ndim < 1:
        raise ValueError("indices must have at least one dimension")
    valid = indices >= 0
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    packed = torch.gather(indices, -1, order)
    packed_valid = torch.gather(valid, -1, order)
    return torch.where(packed_valid, packed, -1), packed_valid


def c4_history_indices(
    indices: torch.Tensor, history_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """QLI returns compressed row IDs, not original-token positions."""
    complete_rows = torch.div(history_lengths, 4, rounding_mode="floor")
    valid = (indices >= 0) & (indices < complete_rows[:, None, None])
    return front_pack_indices(torch.where(valid, indices, -1))


def merge_attention_segments(
    outputs: Sequence[torch.Tensor],
    logsumexp: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized attention segments exactly.

    Every output is ``sum(exp(score) * value) / sum(exp(score))`` for one
    segment and every LSE is ``log(sum(exp(score)))`` with a final singleton
    dimension. Empty segments must be omitted by the caller.
    """
    if not outputs or len(outputs) != len(logsumexp):
        raise ValueError("outputs and logsumexp must contain the same non-zero segments")
    shape = outputs[0].shape
    lse_shape = logsumexp[0].shape
    if lse_shape != shape[:-1] + (1,):
        raise ValueError(f"LSE shape {lse_shape} does not match output shape {shape}")
    if any(output.shape != shape for output in outputs):
        raise ValueError("all segment outputs must have the same shape")
    if any(lse.shape != lse_shape for lse in logsumexp):
        raise ValueError("all segment LSE tensors must have the same shape")

    lse_stack = torch.stack([lse.float() for lse in logsumexp], dim=0)
    merged_lse = torch.logsumexp(lse_stack, dim=0)
    weights = torch.exp(lse_stack - merged_lse.unsqueeze(0))
    merged = sum(
        output.float() * weight
        for output, weight in zip(outputs, weights, strict=True)
    )
    return merged.to(outputs[0].dtype), merged_lse


def merge_attention_outputs(
    outputs: Sequence[torch.Tensor],
    logsumexp: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Merge segmented attention outputs using the native NPU update op."""
    if not outputs or len(outputs) != len(logsumexp):
        raise ValueError("outputs and logsumexp must contain the same non-zero segments")
    shape = outputs[0].shape
    lse_shape = logsumexp[0].shape
    if len(shape) < 2 or lse_shape != shape[:-1] + (1,):
        raise ValueError(f"LSE shape {lse_shape} does not match output shape {shape}")
    if any(output.shape != shape for output in outputs):
        raise ValueError("all segment outputs must have the same shape")
    if any(lse.shape != lse_shape for lse in logsumexp):
        raise ValueError("all segment LSE tensors must have the same shape")

    if outputs[0].device.type == "npu":
        import torch_npu

        head_dim = shape[-1]
        merged, _ = torch_npu.npu_attention_update(
            [lse.float().reshape(-1) for lse in logsumexp],
            [output.reshape(-1, head_dim) for output in outputs],
            0,
        )
        return merged.view(shape)

    merged, _ = merge_attention_segments(outputs, logsumexp)
    return merged


def tail_logical_indices(seq_lens: torch.Tensor, width: int) -> torch.Tensor:
    """Build front-packed logical indices for each request's trailing window."""
    if seq_lens.ndim != 1 or width <= 0:
        raise ValueError("seq_lens must be rank one and width must be positive")
    lengths = torch.minimum(seq_lens, torch.full_like(seq_lens, width))
    starts = seq_lens - lengths
    offsets = torch.arange(width, device=seq_lens.device, dtype=seq_lens.dtype)
    indices = starts[:, None] + offsets[None, :]
    return torch.where(offsets[None, :] < lengths[:, None], indices, -1)


def history_and_block_mask(
    history_lengths: torch.Tensor, history_width: int, block_width: int
) -> torch.Tensor:
    """Mask padded history, while all queries see the entire temporary block."""
    offsets = torch.arange(history_width + block_width, device=history_lengths.device)
    masked = (offsets[None, :] >= history_lengths[:, None]) & (offsets[None, :] < history_width)
    return masked[:, None, None, :].expand(-1, 1, block_width, -1).contiguous()


def compressed_logical_indices(
    seq_lens: torch.Tensor, ratio: int, maximum: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complete C4/C128 cache rows and their per-request lengths.

    DSV4 keeps an incomplete compression group in the target state cache. It
    becomes a readable compressed KV row only after ``ratio`` source tokens.
    """
    if ratio not in (4, 128):
        raise ValueError("compressed history ratio must be 4 or 128")
    lengths = torch.div(seq_lens, ratio, rounding_mode="floor")
    width = int(maximum or max(int(lengths.max().item()), 1))
    offsets = torch.arange(width, device=seq_lens.device, dtype=seq_lens.dtype)
    indices = offsets[None, :].expand(seq_lens.shape[0], -1)
    return torch.where(offsets[None, :] < lengths[:, None], indices, -1), lengths


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    logical_indices: torch.Tensor,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """Gather logical token indices from ``[blocks, block, heads, dim]`` cache.

    ``logical_indices`` may be ``[batch, tokens]`` or
    ``[batch, queries, tokens]``. Invalid ``-1`` entries are returned as zero;
    their lengths must also be supplied to the attention kernel.
    """
    if cache.ndim not in (3, 4):
        raise ValueError(f"unsupported paged cache shape: {tuple(cache.shape)}")
    if block_table.ndim != 2 or logical_indices.ndim not in (2, 3):
        raise ValueError("block_table and logical_indices have invalid rank")
    if block_table.shape[0] != logical_indices.shape[0]:
        raise ValueError("cache gather batch dimensions differ")
    block_size = cache.shape[1]
    valid = logical_indices >= 0
    safe = logical_indices.clamp_min(0).long()
    logical_blocks = torch.div(safe, block_size, rounding_mode="floor")
    if validate and logical_blocks.numel():
        if int(logical_blocks.max().item()) >= block_table.shape[1]:
            raise ValueError("logical index exceeds block table capacity")
    batch_shape = logical_indices.shape
    batch_ids = torch.arange(
        batch_shape[0], device=logical_indices.device, dtype=torch.long
    ).view((batch_shape[0],) + (1,) * (logical_indices.ndim - 1))
    physical_blocks = block_table[batch_ids, logical_blocks]
    if validate and torch.any(valid & (physical_blocks < 0)):
        raise ValueError("logical index maps to an unallocated cache block")
    physical_blocks = physical_blocks.clamp_min(0).long()
    offsets = safe % block_size
    if logical_indices.ndim == 3 and logical_indices.shape[0] > 1 and cache.is_contiguous():
        # C4 has a different sparse list per query. A single row index avoids
        # materializing the much larger two-index gather on the NPU.
        rows = physical_blocks * block_size + offsets
        gathered = cache.flatten(0, 1).index_select(0, rows.flatten()).view(
            *logical_indices.shape, *cache.shape[2:])
        mask = ~valid.view(valid.shape + (1,) * (gathered.ndim - valid.ndim))
        return gathered.masked_fill(mask, 0)
    gathered = cache[physical_blocks, offsets]
    mask = valid.view(valid.shape + (1,) * (gathered.ndim - valid.ndim))
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def cache_checksum(cache: torch.Tensor, *, full: bool = False) -> torch.Tensor:
    """Snapshot a bounded sample, or all elements for explicit debug checks."""
    flat = cache.reshape(-1)
    if full:
        return flat.clone()
    max_samples = 65_536
    stride = max((flat.numel() + max_samples - 1) // max_samples, 1)
    return flat[::stride][:max_samples].clone()


def assert_cache_unchanged(
    before: torch.Tensor, cache: torch.Tensor, name: str
) -> None:
    after = cache.reshape(-1) if before.numel() == cache.numel() else cache_checksum(cache)
    equal = before == after
    if before.is_floating_point():
        equal = equal | (torch.isnan(before) & torch.isnan(after))
    if not bool(torch.all(equal).item()):
        changed = ~equal
        changed_count = int(changed.sum().item())
        finite = changed & torch.isfinite(before) & torch.isfinite(after)
        max_abs = 0.0
        if bool(torch.any(finite).item()):
            max_abs = float(
                (before[finite].float() - after[finite].float()).abs().max().item()
            )
        raise RuntimeError(
            f"Orthrus draft modified target cache {name}: "
            f"sample_changed={changed_count}/{before.numel()}, max_abs={max_abs}"
        )
