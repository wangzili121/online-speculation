"""Frozen-design streaming importance sampling.

Historical contributions may be inspected before the fresh budget is frozen.
After :meth:`FrozenStreamingISEstimator.freeze`, only the declared fresh sample
ids can enter the estimator.  Arrival order may change wall-clock behavior but
cannot change the final statistic.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.importance import logmeanexp
from inference_scaling.shared.metrics import importance_effective_sample_size


def ordinary_importance_log_weight(
    *,
    reward: float,
    reward_temperature: float,
    target_logprob: float,
    behavior_logprob: float,
) -> float:
    """Return the unclipped log contribution for ordinary importance sampling."""

    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    values = (reward, target_logprob, behavior_logprob)
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("importance-sampling inputs must be finite")
    return float(reward) / reward_temperature + float(target_logprob) - float(
        behavior_logprob
    )


@dataclass(frozen=True, slots=True)
class ISContribution:
    sample_id: str
    candidate_index: int
    log_weight: float
    source: str

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("IS sample id cannot be empty")
        if self.candidate_index < 0:
            raise ValueError("candidate index must be non-negative")
        if not isfinite(self.log_weight):
            raise ValueError("IS log weight must be finite")
        if self.source not in {"history", "fresh"}:
            raise ValueError("IS contribution source must be history or fresh")


@dataclass(frozen=True, slots=True)
class FrozenISSnapshot:
    frozen: bool
    complete: bool
    expected_fresh: int
    received_fresh: int
    history_count: int
    log_weights: tuple[float, ...]
    effective_sample_sizes: tuple[float, ...]
    contribution_counts: tuple[int, ...]


ISUpdateCallback = Callable[[FrozenISSnapshot, ISContribution], None]


class FrozenStreamingISEstimator:
    """Accumulate history immediately and fresh evaluations in any order."""

    def __init__(
        self,
        candidate_count: int,
        *,
        on_update: ISUpdateCallback | None = None,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        self.candidate_count = int(candidate_count)
        self._lock = threading.RLock()
        self._on_update = on_update
        self._contributions: list[list[ISContribution]] = [
            [] for _ in range(candidate_count)
        ]
        self._seen_ids: set[str] = set()
        self._expected_fresh: dict[str, int] | None = None
        self._received_fresh: set[str] = set()

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._expected_fresh is not None

    @property
    def complete(self) -> bool:
        with self._lock:
            return self._expected_fresh is not None and len(
                self._received_fresh
            ) == len(self._expected_fresh)

    def _check_candidate(self, candidate_index: int) -> None:
        if not 0 <= candidate_index < self.candidate_count:
            raise ValueError("candidate index is outside the frozen estimator")

    def _append(self, contribution: ISContribution) -> None:
        self._check_candidate(contribution.candidate_index)
        if contribution.sample_id in self._seen_ids:
            raise ValueError(f"duplicate IS sample id {contribution.sample_id!r}")
        self._seen_ids.add(contribution.sample_id)
        group = self._contributions[contribution.candidate_index]
        group.append(contribution)
        try:
            if self._on_update is not None:
                self._on_update(self.snapshot(), contribution)
        except BaseException:
            group.pop()
            self._seen_ids.remove(contribution.sample_id)
            raise

    def add_history(
        self, sample_id: str, candidate_index: int, log_weight: float
    ) -> None:
        with self._lock:
            if self.frozen:
                raise RuntimeError("history cannot be revealed after the fresh design is frozen")
            self._append(
                ISContribution(sample_id, int(candidate_index), float(log_weight), "history")
            )

    def freeze(self, fresh_sample_ids: Sequence[Sequence[str]]) -> FrozenISSnapshot:
        with self._lock:
            if self.frozen:
                raise RuntimeError("fresh IS design is already frozen")
            if len(fresh_sample_ids) != self.candidate_count:
                raise ValueError("fresh design needs one id list per candidate")
            expected: dict[str, int] = {}
            for candidate_index, sample_ids in enumerate(fresh_sample_ids):
                if not sample_ids and not self._contributions[candidate_index]:
                    raise ValueError("every candidate needs history or a fresh evaluation")
                for sample_id in sample_ids:
                    if not sample_id:
                        raise ValueError("fresh IS sample id cannot be empty")
                    if sample_id in expected or sample_id in self._seen_ids:
                        raise ValueError(f"duplicate frozen IS sample id {sample_id!r}")
                    expected[sample_id] = candidate_index
            self._expected_fresh = expected
            return self.snapshot()

    def consume_fresh(
        self, sample_id: str, candidate_index: int, log_weight: float
    ) -> None:
        with self._lock:
            if not self.frozen:
                raise RuntimeError("fresh contribution arrived before the design was frozen")
            expected_index = (self._expected_fresh or {}).get(sample_id)
            if expected_index is None:
                raise ValueError(f"fresh sample id {sample_id!r} is not in the frozen design")
            if expected_index != candidate_index:
                raise ValueError("fresh contribution belongs to a different candidate")
            contribution = ISContribution(
                sample_id, int(candidate_index), float(log_weight), "fresh"
            )
            if sample_id in self._received_fresh:
                raise ValueError(f"duplicate IS sample id {sample_id!r}")
            self._received_fresh.add(sample_id)
            try:
                self._append(contribution)
            except BaseException:
                self._received_fresh.remove(sample_id)
                raise

    def snapshot(self) -> FrozenISSnapshot:
        with self._lock:
            log_weights = tuple(
                logmeanexp([item.log_weight for item in group])
                if group
                else float("-inf")
                for group in self._contributions
            )
            return FrozenISSnapshot(
                frozen=self.frozen,
                complete=self.complete,
                expected_fresh=len(self._expected_fresh or {}),
                received_fresh=len(self._received_fresh),
                history_count=sum(
                    item.source == "history"
                    for group in self._contributions
                    for item in group
                ),
                log_weights=log_weights,
                effective_sample_sizes=tuple(
                    importance_effective_sample_size(
                        [item.log_weight for item in group]
                    )
                    for group in self._contributions
                ),
                contribution_counts=tuple(len(group) for group in self._contributions),
            )

    def final_log_weights(self) -> tuple[float, ...]:
        with self._lock:
            if not self.complete:
                raise RuntimeError("fresh IS design is not complete")
            return self.snapshot().log_weights

    def select(self, seeds: SeedStream, *labels: object) -> int:
        log_weights = np.asarray(self.final_log_weights(), dtype=np.float64)
        if np.any(~np.isfinite(log_weights)):
            raise RuntimeError("every candidate needs a finite final IS estimate")
        weights = np.exp(log_weights - float(np.max(log_weights)))
        probabilities = weights / weights.sum()
        return int(seeds.generator("frozen-streaming-is", *labels).choice(
            self.candidate_count, p=probabilities
        ))

    def contributions(self, candidate_index: int) -> tuple[ISContribution, ...]:
        with self._lock:
            self._check_candidate(candidate_index)
            return tuple(self._contributions[candidate_index])
