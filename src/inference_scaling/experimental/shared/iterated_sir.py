"""Model-independent iterated sampling-importance-resampling kernel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

import numpy as np

from inference_scaling.shared.stepwise import StepwiseCandidate, normalize_log_weights


StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class IteratedSIRTransition(Generic[StateT]):
    """One i-SIR transition with the previous weighted state in pool position zero."""

    previous: StepwiseCandidate[StateT]
    pool: tuple[StepwiseCandidate[StateT], ...]
    probabilities: tuple[float, ...]
    selected_index: int

    def __post_init__(self) -> None:
        if len(self.pool) < 2:
            raise ValueError("an i-SIR pool requires at least two states")
        if self.pool[0] is not self.previous:
            raise ValueError("pool position zero must retain the previous state")
        if len(self.probabilities) != len(self.pool):
            raise ValueError("each i-SIR pool state requires one probability")
        if not 0 <= self.selected_index < len(self.pool):
            raise ValueError("selected_index lies outside the i-SIR pool")

    @property
    def selected(self) -> StepwiseCandidate[StateT]:
        return self.pool[self.selected_index]

    @property
    def retained_previous(self) -> bool:
        return self.selected_index == 0


def iterated_sir_transition(
    current: StepwiseCandidate[StateT],
    fresh: Sequence[StepwiseCandidate[StateT]],
    *,
    rng: np.random.Generator,
) -> IteratedSIRTransition[StateT]:
    """Apply one independent-proposal i-SIR transition.

    ``current`` includes every auxiliary random variable used to compute its
    non-negative weight.  The caller must draw each element of ``fresh``
    independently from the same extended proposal.  Retaining the complete
    state is what makes this a finite-pool invariant transition rather than a
    new self-normalized SIR approximation at every update.
    """

    proposed = tuple(fresh)
    if not proposed:
        raise ValueError("an i-SIR transition requires at least one fresh state")
    pool = (current, *proposed)
    probabilities = normalize_log_weights([state.log_weight for state in pool])
    selected_index = int(rng.choice(len(pool), p=probabilities))
    return IteratedSIRTransition(
        previous=current,
        pool=pool,
        probabilities=probabilities,
        selected_index=selected_index,
    )


def iterated_sir_tv_bound(
    *,
    pool_size: int,
    updates: int,
    normalized_weight_supremum: float,
) -> float:
    """Return the standard finite-update i-SIR total-variation upper bound.

    Let ``w`` be the target/proposal weight and let
    ``normalized_weight_supremum = sup(w) / E_proposal[w]``.  For a pool of
    size ``N``, the minorization constant is
    ``(N - 1) / (2 L + N - 2)``.  The returned bound is the corresponding
    contraction factor raised to ``updates``.  The condition ``L < infinity``
    is explicit because the bound is not available for unbounded weights.
    """

    if isinstance(pool_size, bool) or pool_size < 2:
        raise ValueError("pool_size must be an integer of at least two")
    if isinstance(updates, bool) or updates < 0:
        raise ValueError("updates must be a non-negative integer")
    bound = float(normalized_weight_supremum)
    if not isfinite(bound) or bound < 1:
        raise ValueError("normalized_weight_supremum must be finite and at least one")
    minorization = (pool_size - 1) / (2 * bound + pool_size - 2)
    return float((1 - minorization) ** updates)


__all__ = [
    "IteratedSIRTransition",
    "iterated_sir_transition",
    "iterated_sir_tv_bound",
]
