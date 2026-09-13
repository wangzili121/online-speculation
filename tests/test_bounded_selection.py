from itertools import product

import numpy as np
import pytest

from inference_scaling.experimental.shared.bounded_selection import (
    invariant_categorical_index,
)
from inference_scaling.shared.stepwise import categorical_index_from_uniform


def test_explicit_categorical_uniform_matches_numpy_choice() -> None:
    probabilities = (
        (0.1, 0.2, 0.7),
        (0.25, 0.25, 0.25, 0.25),
        (1e-10, 0.3, 0.6999999999),
    )
    for values in probabilities:
        for seed in range(1000):
            expected = int(
                np.random.default_rng(seed).choice(len(values), p=values)
            )
            uniform = float(np.random.default_rng(seed).random())
            assert categorical_index_from_uniform(values, uniform) == expected


def test_exact_intervals_reproduce_categorical_selection() -> None:
    weights = (0.3, 2.0, 0.7, 1.1)
    for uniform in np.linspace(0.0, 0.999, 101):
        expected = categorical_index_from_uniform(weights, float(uniform))
        assert (
            invariant_categorical_index(weights, weights, uniform=float(uniform))
            == expected
        )


def test_interval_decision_is_complete_over_all_vertices() -> None:
    generator = np.random.default_rng(91)
    for _ in range(500):
        lower = generator.uniform(0.05, 2.0, size=4)
        upper = lower + generator.uniform(0.0, 3.0, size=4)
        uniform = float(generator.random())
        vertex_indices = {
            categorical_index_from_uniform(vertex, uniform)
            for vertex in product(*zip(lower, upper, strict=True))
        }
        decision = invariant_categorical_index(lower, upper, uniform=uniform)
        assert decision == (vertex_indices.pop() if len(vertex_indices) == 1 else None)


@pytest.mark.parametrize(
    "lower,upper,uniform",
    [
        ((), (), 0.5),
        ((1.0,), (1.0, 2.0), 0.5),
        ((0.0,), (1.0,), 0.5),
        ((2.0,), (1.0,), 0.5),
        ((1.0,), (1.0,), 1.0),
    ],
)
def test_invalid_weight_intervals_fail_early(lower, upper, uniform) -> None:
    with pytest.raises(ValueError):
        invariant_categorical_index(lower, upper, uniform=uniform)
