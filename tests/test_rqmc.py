import numpy as np
import pytest

from inference_scaling.experimental.shared.rqmc import (
    randomized_lattice_uniforms,
    scrambled_sobol_uniforms,
)


def test_randomized_lattice_is_seeded_bounded_and_evenly_spaced() -> None:
    count = 8
    first = randomized_lattice_uniforms(count, seed=17)
    repeated = randomized_lattice_uniforms(count, seed=17)
    different = randomized_lattice_uniforms(count, seed=18)

    assert first == repeated
    assert first != different
    assert all(0.0 <= value < 1.0 for value in first)
    ordered = sorted(first)
    circular_gaps = [
        (ordered[(index + 1) % count] - ordered[index]) % 1.0 for index in range(count)
    ]
    assert circular_gaps == pytest.approx([1.0 / count] * count)


@pytest.mark.parametrize("count,seed", [(0, 0), (1, -1)])
def test_randomized_lattice_rejects_invalid_arguments(count, seed) -> None:
    with pytest.raises(ValueError):
        randomized_lattice_uniforms(count, seed=seed)


def test_each_randomized_lattice_coordinate_is_marginally_uniform() -> None:
    points = np.asarray(
        [randomized_lattice_uniforms(4, seed=seed) for seed in range(4096)]
    )

    assert np.all(np.abs(points.mean(axis=0) - 0.5) < 0.015)
    assert np.all(np.abs(points.var(axis=0) - 1.0 / 12.0) < 0.005)


def test_scrambled_sobol_points_are_seeded_and_bounded() -> None:
    first = scrambled_sobol_uniforms(8, 5, seed=17)
    repeated = scrambled_sobol_uniforms(8, 5, seed=17)
    different = scrambled_sobol_uniforms(8, 5, seed=18)

    assert first == repeated
    assert first != different
    assert len(first) == 8
    assert all(len(point) == 5 for point in first)
    assert all(0.0 <= value < 1.0 for point in first for value in point)


@pytest.mark.parametrize("count,dimension,seed", [(0, 1, 0), (1, 0, 0), (1, 1, -1)])
def test_scrambled_sobol_rejects_invalid_shapes(count, dimension, seed) -> None:
    with pytest.raises(ValueError):
        scrambled_sobol_uniforms(count, dimension, seed=seed)


def test_scrambled_sobol_reduces_randomization_variance_for_linear_integrand() -> None:
    count = 8
    repetitions = 128
    sobol_means = np.asarray(
        [
            np.mean(scrambled_sobol_uniforms(count, 1, seed=seed))
            for seed in range(repetitions)
        ]
    )
    iid_means = np.asarray(
        [
            np.random.default_rng(seed).random(count).mean()
            for seed in range(repetitions)
        ]
    )

    assert abs(float(sobol_means.mean()) - 0.5) < 0.01
    assert np.var(sobol_means) < 0.1 * np.var(iid_means)
