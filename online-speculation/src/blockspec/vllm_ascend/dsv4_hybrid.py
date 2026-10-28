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


def tail_logical_indices(seq_lens: torch.Tensor, width: int) -> torch.Tensor:
    """Build front-packed logical indices for each request's trailing window."""
    if seq_lens.ndim != 1 or width <= 0:
        raise ValueError("seq_lens must be rank one and width must be positive")
    lengths = torch.minimum(seq_lens, torch.full_like(seq_lens, width))
    starts = seq_lens - lengths
    offsets = torch.arange(width, device=seq_lens.device, dtype=seq_lens.dtype)
    indices = starts[:, None] + offsets[None, :]
    return torch.where(offsets[None, :] < lengths[:, None], indices, -1)


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
    gathered = cache[physical_blocks.clamp_min(0).long(), safe % block_size]
    mask = valid.view(valid.shape + (1,) * (gathered.ndim - valid.ndim))
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def cache_checksum(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded sampled fingerprint for draft cache debug checks."""
    flat = cache.reshape(-1)
    max_samples = 65_536
    stride = max((flat.numel() + max_samples - 1) // max_samples, 1)
    sample = flat[::stride][:max_samples].float()
    weights = torch.arange(
        1, sample.numel() + 1, device=sample.device, dtype=torch.float32
    )
    return sample.sum(), (sample * weights).sum()


def assert_cache_unchanged(
    before: tuple[torch.Tensor, torch.Tensor], cache: torch.Tensor, name: str
) -> None:
    after = cache_checksum(cache)
    if not torch.equal(before[0], after[0]) or not torch.equal(before[1], after[1]):
        raise RuntimeError(f"Orthrus draft modified target cache {name}")
