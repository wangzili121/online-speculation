"""Model-independent normalization, resampling, and reservoir partitioning for SMC."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import isfinite
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def normalize_smc_log_weights(
    log_weights: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    if not log_weights:
        raise ValueError("SMC requires at least one branch weight")
    values = np.asarray(log_weights, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("SMC log weights must be finite")
    weights = np.exp(values - float(np.max(values)))
    total = float(weights.sum())
    if not isfinite(total) or total <= 0:
        raise ValueError("SMC weights cannot be normalized")
    probabilities = weights / total
    ess = float(1.0 / np.square(probabilities).sum())
    return tuple(float(value) for value in probabilities), ess


def systematic_resample(
    probabilities: Sequence[float],
    count: int,
    generator: np.random.Generator,
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("resampling count must be positive")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("resampling probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("resampling probabilities must have positive mass")
    values /= total
    start = float(generator.random()) / count
    positions = start + np.arange(count, dtype=np.float64) / count
    cumulative = np.cumsum(values, dtype=np.float64)
    cumulative[-1] = 1.0
    return tuple(
        int(value) for value in np.searchsorted(cumulative, positions, side="right")
    )


def partition_resampled_reservoirs(
    reservoirs: Sequence[Sequence[T]],
    selected: Sequence[int],
) -> tuple[tuple[T, ...], ...]:
    """Split each finite reservoir among its resampled copies without duplication."""

    if not selected:
        raise ValueError("at least one resampled index is required")
    if any(index < 0 or index >= len(reservoirs) for index in selected):
        raise ValueError("a resampled index lies outside the reservoir set")
    occurrences: dict[int, list[int]] = defaultdict(list)
    for output_index, source_index in enumerate(selected):
        occurrences[int(source_index)].append(output_index)
    outputs: list[tuple[T, ...] | None] = [None] * len(selected)
    for source_index, output_positions in occurrences.items():
        buckets: list[list[T]] = [[] for _ in output_positions]
        for item_index, item in enumerate(reservoirs[source_index]):
            buckets[item_index % len(buckets)].append(item)
        for output_position, bucket in zip(output_positions, buckets, strict=True):
            outputs[output_position] = tuple(bucket)
    if any(output is None for output in outputs):
        raise RuntimeError("SMC reservoir partition omitted an output")
    return tuple(output for output in outputs if output is not None)


__all__ = [
    "normalize_smc_log_weights",
    "partition_resampled_reservoirs",
    "systematic_resample",
]
