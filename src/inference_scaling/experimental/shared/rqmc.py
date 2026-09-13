"""Randomized quasi-Monte Carlo point sets for autoregressive sampling."""

from __future__ import annotations

import numpy as np


def randomized_lattice_uniforms(
    count: int,
    *,
    seed: int,
) -> tuple[float, ...]:
    """Return a randomly shifted one-dimensional lattice in ``[0, 1)``.

    Every labelled point is marginally uniform because the shared shift is
    uniform.  The points are dependent and exactly ``1 / count`` apart on the
    unit circle.  An autoregressive backend can turn each point into a complete
    sequence with arithmetic inverse-CDF sampling.
    """

    if count <= 0:
        raise ValueError("lattice count must be positive")
    if seed < 0:
        raise ValueError("lattice seed must be non-negative")
    shift = float(np.random.default_rng(seed).random())
    return tuple((shift + index / count) % 1.0 for index in range(count))


def scrambled_sobol_uniforms(
    count: int,
    dimension: int,
    *,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    """Return a digitally scrambled Sobol point set in ``[0, 1)^dimension``."""

    if count <= 0 or dimension <= 0:
        raise ValueError("Sobol count and dimension must be positive")
    if seed < 0:
        raise ValueError("Sobol seed must be non-negative")
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - torch is required
        raise ModuleNotFoundError("scrambled Sobol rollouts require torch") from error
    engine = torch.quasirandom.SobolEngine(
        dimension=dimension,
        scramble=True,
        seed=int(seed % (2**31 - 1)),
    )
    points = engine.draw(count, dtype=torch.float64).cpu().numpy()
    points = np.clip(points, 0.0, np.nextafter(1.0, 0.0))
    return tuple(tuple(float(value) for value in row) for row in points)


__all__ = ["randomized_lattice_uniforms", "scrambled_sobol_uniforms"]
