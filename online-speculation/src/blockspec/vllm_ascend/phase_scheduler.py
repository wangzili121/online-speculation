"""Request-level phase admission for Orthrus speculative serving."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftAdmissionDecision:
    selected_indices: list[int]
    deferred_indices: list[int]
    dominant_length_bucket: int | None
    queue_ages: dict[str, int]


class StaggeredDraftAdmission:
    """Bound K32 fan-in while preserving fair request progress.

    A deferred request executes the target model normally for this iteration and
    competes for Orthrus drafting again on its next token. This preserves the
    target distribution and bounds the next iteration's speculative verify batch.
    """

    def __init__(self, *, max_batch: int, length_bucket: int,
                 max_delay_steps: int):
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        if length_bucket <= 0:
            raise ValueError("length_bucket must be positive")
        if max_delay_steps <= 0:
            raise ValueError("max_delay_steps must be positive")
        self.max_batch = max_batch
        self.length_bucket = length_bucket
        self.max_delay_steps = max_delay_steps
        self._ages: dict[str, int] = {}

    def select(self, request_ids: list[str], history_lengths: list[int],
               active_indices: list[int]) -> DraftAdmissionDecision:
        if len(request_ids) != len(history_lengths):
            raise ValueError("request_ids and history_lengths must align")
        active_ids = {request_ids[index] for index in active_indices}
        for request_id in list(self._ages):
            if request_id not in active_ids:
                del self._ages[request_id]
        for index in active_indices:
            request_id = request_ids[index]
            self._ages[request_id] = self._ages.get(request_id, 0) + 1

        if len(active_indices) <= self.max_batch:
            selected = list(active_indices)
            dominant = None
        else:
            buckets = {
                index: history_lengths[index] // self.length_bucket
                for index in active_indices
            }
            # Weight a bucket by occupancy and accumulated waiting. This favors
            # stable attention shapes without starving a smaller bucket.
            weights = Counter()
            for index in active_indices:
                request_id = request_ids[index]
                weights[buckets[index]] += 1 + self._ages[request_id]
            dominant = max(weights, key=lambda bucket: (weights[bucket], -bucket))

            def priority(index: int) -> tuple[int, int, int, int]:
                age = self._ages[request_ids[index]]
                return (
                    int(age >= self.max_delay_steps),
                    age,
                    int(buckets[index] == dominant),
                    -index,
                )

            selected = sorted(active_indices, key=priority, reverse=True)[
                :self.max_batch]

        selected_set = set(selected)
        deferred = [index for index in active_indices
                    if index not in selected_set]
        for index in selected:
            self._ages[request_ids[index]] = 0
        return DraftAdmissionDecision(
            selected_indices=selected,
            deferred_indices=deferred,
            dominant_length_bucket=dominant,
            queue_ages={request_ids[index]: self._ages[request_ids[index]]
                        for index in active_indices},
        )
