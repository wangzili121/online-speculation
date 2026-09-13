from itertools import product

import numpy as np
import pytest

from inference_scaling.experimental.shared.iterated_sir import (
    iterated_sir_transition,
    iterated_sir_tv_bound,
)
from inference_scaling.shared.stepwise import StepwiseCandidate


def _transition_matrix(target, proposal, pool_size):
    target = np.asarray(target, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    weights = target / proposal
    states = range(len(target))
    matrix = np.zeros((len(target), len(target)), dtype=np.float64)
    for current in states:
        for fresh in product(states, repeat=pool_size - 1):
            probability = float(np.prod([proposal[index] for index in fresh]))
            pool = (current, *fresh)
            pool_weights = np.asarray([weights[index] for index in pool])
            normalized = pool_weights / pool_weights.sum()
            for position, selected in enumerate(pool):
                matrix[current, selected] += probability * normalized[position]
    return matrix


def test_iterated_sir_kernel_satisfies_detailed_balance_exactly() -> None:
    target = np.asarray([0.2, 0.5, 0.3])
    proposal = np.asarray([0.6, 0.3, 0.1])
    matrix = _transition_matrix(target, proposal, pool_size=3)

    assert matrix.sum(axis=1) == pytest.approx(np.ones(3))
    assert target @ matrix == pytest.approx(target)
    flow = target[:, None] * matrix
    assert flow == pytest.approx(flow.T)


def test_iterated_sir_transition_retains_complete_current_state() -> None:
    current = StepwiseCandidate({"tokens": (1,), "rollouts": (2, 3)}, -0.2)
    fresh = (
        StepwiseCandidate({"tokens": (0,), "rollouts": (4,)}, -1.0),
        StepwiseCandidate({"tokens": (2,), "rollouts": (5,)}, -0.4),
    )
    transition = iterated_sir_transition(
        current,
        fresh,
        rng=np.random.default_rng(7),
    )

    assert transition.pool[0] is current
    assert transition.previous.value["rollouts"] == (2, 3)
    assert sum(transition.probabilities) == pytest.approx(1.0)
    assert transition.selected in transition.pool


def test_iterated_sir_tv_bound_is_explicit_in_update_count() -> None:
    bounds = [
        iterated_sir_tv_bound(
            pool_size=4,
            updates=updates,
            normalized_weight_supremum=3.0,
        )
        for updates in range(5)
    ]
    assert bounds[0] == 1.0
    assert all(left > right for left, right in zip(bounds, bounds[1:]))
    assert bounds[4] == pytest.approx((1 - 3 / 8) ** 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pool_size": 1, "updates": 1, "normalized_weight_supremum": 2},
        {"pool_size": 2, "updates": -1, "normalized_weight_supremum": 2},
        {"pool_size": 2, "updates": 1, "normalized_weight_supremum": 0.9},
    ],
)
def test_iterated_sir_tv_bound_validates_its_assumptions(kwargs) -> None:
    with pytest.raises(ValueError):
        iterated_sir_tv_bound(**kwargs)
