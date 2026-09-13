"""Exact categorical decisions from bounded positive weights."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite


def invariant_categorical_index(
    lower_weights: Sequence[float],
    upper_weights: Sequence[float],
    *,
    uniform: float,
) -> int | None:
    """Return the selected index when it is fixed over every weight interval.

    Selection uses the first normalized cumulative weight strictly greater than
    ``uniform``.  ``None`` means that at least two outcomes remain possible; it
    does not approximate or randomize the decision.
    """

    lower = tuple(float(value) for value in lower_weights)
    upper = tuple(float(value) for value in upper_weights)
    if not lower or len(lower) != len(upper):
        raise ValueError("weight intervals must be non-empty and aligned")
    if not isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("uniform must be finite and lie in [0, 1)")
    if any(
        not isfinite(low)
        or not isfinite(high)
        or low <= 0.0
        or high < low
        for low, high in zip(lower, upper, strict=True)
    ):
        raise ValueError("weight intervals must be finite, positive and ordered")

    one_minus_uniform = 1.0 - uniform
    for index in range(len(lower)):
        # Every admissible vector must place the cumulative mass before index
        # at or below the fixed uniform draw.
        maximum_before_minus_threshold = (
            one_minus_uniform * sum(upper[:index])
            - uniform * sum(lower[index:])
        )
        # Every admissible vector must place the cumulative mass through index
        # strictly above the fixed uniform draw.
        minimum_through_minus_threshold = (
            one_minus_uniform * sum(lower[: index + 1])
            - uniform * sum(upper[index + 1 :])
        )
        if (
            maximum_before_minus_threshold <= 0.0
            and minimum_through_minus_threshold > 0.0
        ):
            return index
    return None


__all__ = ["invariant_categorical_index"]
