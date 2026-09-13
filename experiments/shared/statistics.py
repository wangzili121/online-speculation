"""Model-independent statistical estimators used by experiment reports."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

from inference_scaling.shared.metrics import normalize_counts, total_variation


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> list[float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and positive trials")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * (
        (
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        ** 0.5
    ) / denominator
    return [center - radius, center + radius]


def estimated_pass_at_k(correct: int, draws: int, k: int) -> float:
    """Unbiased pass@k estimator from ``draws`` samples without replacement."""

    if not 0 <= correct <= draws:
        raise ValueError("correct must lie between zero and draws")
    if draws <= 0 or not 1 <= k <= draws:
        raise ValueError("k must lie between one and the number of draws")
    if draws - correct < k:
        return 1.0
    return 1.0 - math.comb(draws - correct, k) / math.comb(draws, k)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int = 10_000,
) -> list[float]:
    if not values or replicates <= 0:
        raise ValueError("bootstrap values and replicate count must be positive")
    numeric = tuple(float(value) for value in values)
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(numeric[rng.randrange(len(numeric))] for _ in numeric)
        for _ in range(replicates)
    ]
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def clustered_paired_binary_difference(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str,
    outcome_key: str,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Compare paired binary outcomes while retaining repeated draws per cluster."""

    if not left_records or not right_records or replicates <= 0:
        raise ValueError("paired records and replicate count must be positive")

    def grouped(
        records: Sequence[Mapping[str, Any]],
    ) -> dict[int, list[float]]:
        result: dict[int, list[float]] = {}
        for record in records:
            result.setdefault(int(record[cluster_key]), []).append(
                float(bool(record[outcome_key]))
            )
        return result

    left = grouped(left_records)
    right = grouped(right_records)
    if left.keys() != right.keys():
        raise ValueError("paired methods must contain the same clusters")
    indices = sorted(left)
    differences: list[float] = []
    for index in indices:
        if len(left[index]) != len(right[index]):
            raise ValueError("paired methods must contain the same draws per cluster")
        differences.append(
            statistics.fmean(left[index]) - statistics.fmean(right[index])
        )
    rng = random.Random(seed)
    bootstrap = [
        statistics.fmean(
            differences[rng.randrange(len(differences))] for _ in differences
        )
        for _ in range(replicates)
    ]
    return {
        "difference": statistics.fmean(differences),
        "clustered_bootstrap_95": [
            quantile(bootstrap, 0.025),
            quantile(bootstrap, 0.975),
        ],
    }


def probability_distribution(counts: Mapping[str, int]) -> dict[str, float]:
    return normalize_counts(counts)


def total_variation_distance(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    return total_variation(left, right)


def jensen_shannon_bits(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    support = set(left) | set(right)
    midpoint = {
        key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2 for key in support
    }

    def divergence(distribution: Mapping[str, float]) -> float:
        return sum(
            probability * math.log2(probability / midpoint[key])
            for key, probability in distribution.items()
            if probability > 0
        )

    return 0.5 * (divergence(left) + divergence(right))


def bootstrap_answer_distance(
    left: Mapping[int, Sequence[str]],
    right: Mapping[int, Sequence[str]],
    problem_indices: Sequence[int],
    *,
    seed: int = 0,
    replicates: int = 2_000,
) -> dict[str, list[float]]:
    if not problem_indices or replicates <= 0:
        raise ValueError("problem indices and replicate count must be positive")
    rng = random.Random(seed)
    tv_samples: list[float] = []
    js_samples: list[float] = []
    for _ in range(replicates):
        televisions: list[float] = []
        divergences: list[float] = []
        for problem_index in problem_indices:
            left_answers = left[problem_index]
            right_answers = right[problem_index]
            if not left_answers or not right_answers:
                raise ValueError("every problem requires observations from both methods")
            left_sample = Counter(
                left_answers[rng.randrange(len(left_answers))] for _ in left_answers
            )
            right_sample = Counter(
                right_answers[rng.randrange(len(right_answers))] for _ in right_answers
            )
            left_distribution = probability_distribution(left_sample)
            right_distribution = probability_distribution(right_sample)
            televisions.append(
                total_variation_distance(left_distribution, right_distribution)
            )
            divergences.append(
                jensen_shannon_bits(left_distribution, right_distribution)
            )
        tv_samples.append(statistics.fmean(televisions))
        js_samples.append(statistics.fmean(divergences))
    return {
        "mean_total_variation_bootstrap_95": [
            quantile(tv_samples, 0.025),
            quantile(tv_samples, 0.975),
        ],
        "mean_jensen_shannon_bits_bootstrap_95": [
            quantile(js_samples, 0.025),
            quantile(js_samples, 0.975),
        ],
    }


__all__ = [
    "bootstrap_answer_distance",
    "bootstrap_mean_interval",
    "clustered_paired_binary_difference",
    "estimated_pass_at_k",
    "jensen_shannon_bits",
    "probability_distribution",
    "quantile",
    "total_variation_distance",
    "wilson_interval",
]
