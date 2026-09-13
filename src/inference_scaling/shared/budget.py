"""Model-independent variance--cost allocation for replay rollouts."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class VarianceCostEstimate:
    history_std: float
    fresh_std: float
    history_cost: float = 1.0
    fresh_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.history_std < 0 or self.fresh_std < 0:
            raise ValueError("standard deviations must be non-negative")
        if not all(
            isfinite(value)
            for value in (
                self.history_std,
                self.fresh_std,
                self.history_cost,
                self.fresh_cost,
            )
        ):
            raise ValueError("variance and cost estimates must be finite")
        if self.history_cost <= 0 or self.fresh_cost <= 0:
            raise ValueError("per-sample costs must be positive")


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    history_count: int
    fresh_count: int
    continuous_history: float
    continuous_fresh: float
    estimated_cost: float


def allocate_fresh_rollout_budget(
    *,
    outer_ratios: Sequence[float],
    standard_deviations: Sequence[float],
    costs: Sequence[float],
    rollout_budget: float,
    minimum_fresh: int | Sequence[int] = 1,
) -> tuple[BudgetAllocation, ...]:
    """Specialize the common allocator to independent fresh rollouts only."""

    count = len(outer_ratios)
    if len(standard_deviations) != count or len(costs) != count:
        raise ValueError("fresh allocation inputs must have the same length")
    groups = tuple(range(count))
    return allocate_variance_cost_budget(
        outer_ratios=outer_ratios,
        statistics=tuple(
            VarianceCostEstimate(
                history_std=0.0,
                fresh_std=deviation,
                fresh_cost=cost,
            )
            for deviation, cost in zip(standard_deviations, costs, strict=True)
        ),
        history_capacities=(0,) * count,
        history_groups=groups,
        group_capacities={group: 0 for group in groups},
        rollout_budget=rollout_budget,
        minimum_fresh=minimum_fresh,
    )


def _proportional_capped_counts(
    total: float,
    coefficients: npt.NDArray[np.float64],
    caps: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    result = np.zeros_like(coefficients, dtype=np.float64)
    active = {index for index, coefficient in enumerate(coefficients) if coefficient > 0}
    remaining = min(float(total), float(caps.sum()))
    while active and remaining > 1e-12:
        coefficient_sum = sum(float(coefficients[index]) for index in active)
        if coefficient_sum <= 0:
            break
        proposed = {
            index: remaining * float(coefficients[index]) / coefficient_sum
            for index in active
        }
        saturated = [
            index
            for index in active
            if proposed[index] >= float(caps[index]) - result[index] - 1e-12
        ]
        if not saturated:
            for index, value in proposed.items():
                result[index] += value
            break
        for index in saturated:
            addition = max(0.0, float(caps[index]) - result[index])
            result[index] += addition
            remaining -= addition
            active.remove(index)
    return result


def allocate_variance_cost_budget(
    *,
    outer_ratios: Sequence[float],
    statistics: Sequence[VarianceCostEstimate],
    history_capacities: Sequence[int],
    history_groups: Sequence[Hashable],
    group_capacities: Mapping[Hashable, int],
    rollout_budget: float,
    minimum_fresh: int | Sequence[int] = 1,
) -> tuple[BudgetAllocation, ...]:
    """Solve the continuous allocation and round within all hard constraints."""

    count = len(outer_ratios)
    if not (
        len(statistics) == count
        and len(history_capacities) == count
        and len(history_groups) == count
    ):
        raise ValueError("all candidate-level allocation inputs must have the same length")
    if count == 0:
        raise ValueError("at least one candidate is required")
    if rollout_budget <= 0 or not isfinite(rollout_budget):
        raise ValueError("rollout_budget must be positive and finite")
    if isinstance(minimum_fresh, int):
        minimum = np.full(count, minimum_fresh, dtype=np.int64)
    else:
        if len(minimum_fresh) != count:
            raise ValueError("minimum_fresh must have one value per candidate")
        minimum = np.asarray(minimum_fresh, dtype=np.int64)
    if np.any(minimum < 0):
        raise ValueError("minimum fresh counts must be non-negative")

    ratios = np.asarray(outer_ratios, dtype=np.float64)
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("outer probability ratios must be positive and finite")
    history_caps = np.asarray(history_capacities, dtype=np.int64)
    if np.any(history_caps < 0):
        raise ValueError("history capacities must be non-negative")
    missing_groups = set(history_groups) - set(group_capacities)
    if missing_groups:
        raise ValueError(
            f"history groups are missing capacities: {sorted(map(repr, missing_groups))}"
        )
    history_costs = np.asarray(
        [item.history_cost for item in statistics], dtype=np.float64
    )
    fresh_costs = np.asarray(
        [item.fresh_cost for item in statistics], dtype=np.float64
    )
    history_coefficients = np.asarray(
        [
            ratio * item.history_std / sqrt(item.history_cost)
            for ratio, item in zip(ratios, statistics, strict=True)
        ],
        dtype=np.float64,
    )
    fresh_coefficients = np.asarray(
        [
            ratio * item.fresh_std / sqrt(item.fresh_cost)
            for ratio, item in zip(ratios, statistics, strict=True)
        ],
        dtype=np.float64,
    )
    history_coefficients[history_caps == 0] = 0.0

    mandatory_cost = float(np.dot(minimum, fresh_costs))
    if mandatory_cost > rollout_budget + 1e-9:
        raise ValueError(
            "rollout budget cannot cover the required fresh sample for every candidate"
        )
    remaining_budget = rollout_budget - mandatory_cost
    continuous_history = np.zeros(count, dtype=np.float64)
    continuous_fresh_extra = np.zeros(count, dtype=np.float64)
    active_history = {
        index
        for index in range(count)
        if history_caps[index] > 0 and history_coefficients[index] > 0
    }
    group_remaining = {
        group: float(capacity) for group, capacity in group_capacities.items()
    }
    for group, capacity in group_remaining.items():
        if capacity < 0:
            raise ValueError(f"history group {group!r} has a negative capacity")

    while remaining_budget > 1e-12:
        active_fresh = {
            index for index in range(count) if fresh_coefficients[index] > 0
        }
        denominator = sum(
            float(history_coefficients[index] * history_costs[index])
            for index in active_history
        ) + sum(
            float(fresh_coefficients[index] * fresh_costs[index])
            for index in active_fresh
        )
        if denominator <= 0:
            active_fresh = {index for index in range(count) if minimum[index] > 0}
            if not active_fresh:
                break
            temporary_fresh_coefficients = 1.0 / fresh_costs
            denominator = float(
                sum(
                    temporary_fresh_coefficients[index] * fresh_costs[index]
                    for index in active_fresh
                )
            )
        else:
            temporary_fresh_coefficients = fresh_coefficients

        proposed_history = {
            index: remaining_budget * history_coefficients[index] / denominator
            for index in active_history
        }
        proposed_fresh = {
            index: remaining_budget * temporary_fresh_coefficients[index] / denominator
            for index in active_fresh
        }

        violated_group: Hashable | None = None
        for group in dict.fromkeys(history_groups):
            members = [
                index for index in active_history if history_groups[index] == group
            ]
            if members and sum(proposed_history[index] for index in members) > (
                group_remaining[group] + 1e-12
            ):
                violated_group = group
                break
        if violated_group is not None:
            members = [
                index
                for index in active_history
                if history_groups[index] == violated_group
            ]
            coefficients = np.asarray(
                [history_coefficients[index] for index in members], dtype=np.float64
            )
            caps = np.asarray(
                [
                    history_caps[index] - continuous_history[index]
                    for index in members
                ],
                dtype=np.float64,
            )
            fixed = _proportional_capped_counts(
                group_remaining[violated_group], coefficients, caps
            )
            fixed_cost = sum(
                value * history_costs[member]
                for member, value in zip(members, fixed, strict=True)
            )
            if fixed_cost > remaining_budget + 1e-12:
                fixed *= remaining_budget / fixed_cost
            for member, value in zip(members, fixed, strict=True):
                continuous_history[member] += value
                remaining_budget -= value * history_costs[member]
                active_history.discard(member)
            group_remaining[violated_group] = 0.0
            continue

        violated_index = next(
            (
                index
                for index in sorted(active_history)
                if proposed_history[index]
                > history_caps[index] - continuous_history[index] + 1e-12
            ),
            None,
        )
        if violated_index is not None:
            addition = float(
                history_caps[violated_index] - continuous_history[violated_index]
            )
            continuous_history[violated_index] += addition
            remaining_budget -= addition * history_costs[violated_index]
            group = history_groups[violated_index]
            group_remaining[group] = max(
                0.0, group_remaining[group] - addition
            )
            active_history.remove(violated_index)
            continue

        for index, value in proposed_history.items():
            continuous_history[index] += value
        for index, value in proposed_fresh.items():
            continuous_fresh_extra[index] += value
        remaining_budget = 0.0

    continuous_fresh = minimum.astype(np.float64) + continuous_fresh_extra
    history_integer = np.floor(continuous_history + 1e-12).astype(np.int64)
    fresh_integer = minimum + np.floor(continuous_fresh_extra + 1e-12).astype(
        np.int64
    )
    integer_cost = float(
        np.dot(history_integer, history_costs) + np.dot(fresh_integer, fresh_costs)
    )
    budget_left = rollout_budget - integer_cost
    group_used: dict[Hashable, int] = {}
    for index, group in enumerate(history_groups):
        group_used[group] = group_used.get(group, 0) + int(history_integer[index])

    remainders: list[tuple[float, int, str]] = []
    for index in range(count):
        history_fraction = float(continuous_history[index] - history_integer[index])
        fresh_fraction = float(continuous_fresh[index] - fresh_integer[index])
        if history_fraction > 1e-12:
            remainders.append((history_fraction, index, "history"))
        if fresh_fraction > 1e-12:
            remainders.append((fresh_fraction, index, "fresh"))
    remainders.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, index, source in remainders:
        cost = history_costs[index] if source == "history" else fresh_costs[index]
        if cost > budget_left + 1e-9:
            continue
        if source == "history":
            group = history_groups[index]
            if history_integer[index] >= history_caps[index]:
                continue
            if group_used.get(group, 0) >= group_capacities[group]:
                continue
            history_integer[index] += 1
            group_used[group] = group_used.get(group, 0) + 1
        else:
            fresh_integer[index] += 1
        budget_left -= float(cost)

    allocations = []
    for index in range(count):
        cost = (
            history_integer[index] * history_costs[index]
            + fresh_integer[index] * fresh_costs[index]
        )
        allocations.append(
            BudgetAllocation(
                history_count=int(history_integer[index]),
                fresh_count=int(fresh_integer[index]),
                continuous_history=float(continuous_history[index]),
                continuous_fresh=float(continuous_fresh[index]),
                estimated_cost=float(cost),
            )
        )
    return tuple(allocations)


__all__ = [
    "BudgetAllocation",
    "VarianceCostEstimate",
    "allocate_fresh_rollout_budget",
    "allocate_variance_cost_budget",
]
