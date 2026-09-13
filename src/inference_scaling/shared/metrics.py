"""Distributional and sampling diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import fsum
from typing import Hashable, TypeVar

import numpy as np

T = TypeVar("T", bound=Hashable)


def normalize_counts(counts: Mapping[T, int]) -> dict[T, float]:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("at least one sample is required")
    return {value: count / total for value, count in counts.items()}


def empirical_distribution(samples: Iterable[T]) -> dict[T, float]:
    return normalize_counts(Counter(samples))


def total_variation(left: Mapping[T, float], right: Mapping[T, float]) -> float:
    support = set(left) | set(right)
    return 0.5 * fsum(abs(left.get(value, 0.0) - right.get(value, 0.0)) for value in support)


def importance_effective_sample_size(log_weights: Sequence[float]) -> float:
    if not log_weights:
        return 0.0
    values = np.asarray(log_weights, dtype=np.float64)
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return 0.0
    weights = np.exp(values - maximum)
    denominator = float(np.dot(weights, weights))
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


def autocorrelation_ess(values: Sequence[float]) -> float:
    """Initial-positive-sequence estimate of Markov-chain ESS."""

    array = np.asarray(values, dtype=np.float64)
    count = array.size
    if count < 2:
        return float(count)
    centered = array - array.mean()
    variance = float(np.dot(centered, centered) / count)
    if variance == 0:
        return float(count)
    correlations: list[float] = []
    for lag in range(1, count):
        covariance = float(np.dot(centered[:-lag], centered[lag:]) / (count - lag))
        correlation = covariance / variance
        if lag % 2 == 0 and correlations and correlations[-1] + correlation <= 0:
            break
        correlations.append(correlation)
    integrated_time = max(1.0, 1.0 + 2.0 * sum(correlations))
    return min(float(count), float(count / integrated_time))


__all__ = [
    "autocorrelation_ess",
    "empirical_distribution",
    "importance_effective_sample_size",
    "normalize_counts",
    "total_variation",
]
