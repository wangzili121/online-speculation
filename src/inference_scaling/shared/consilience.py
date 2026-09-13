"""Pointwise confidence-trajectory arithmetic, without model or dataset access."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite


@dataclass(frozen=True, slots=True)
class ConfidenceWindows:
    skipped: int
    window: int
    initial: float
    final: float
    score: float


def confidence_windows(
    values: Sequence[float],
    *,
    window_fraction: float = 0.2,
    window_tokens: int | None = None,
    skip_fraction: float = 0.05,
    initial_penalty: float = 3.0,
) -> ConfidenceWindows:
    """Compute the score on a complete, already isolated thinking trajectory."""
    if not isfinite(window_fraction) or not 0 < window_fraction <= 1:
        raise ValueError("Consilience window_fraction must be in (0, 1]")
    if window_tokens is not None and window_tokens <= 0:
        raise ValueError("Consilience window_tokens must be positive")
    if not isfinite(skip_fraction) or not 0 <= skip_fraction < 1:
        raise ValueError("Consilience skip_fraction must be in [0, 1)")
    if not isfinite(initial_penalty) or initial_penalty < 0:
        raise ValueError("Consilience initial_penalty must be finite and non-negative")
    if not values or any(not isfinite(value) for value in values):
        raise ValueError("Consilience requires a nonempty finite confidence trajectory")
    length = len(values)
    skipped = min(length - 1, int(length * skip_fraction))
    window = min(
        window_tokens if window_tokens is not None else max(1, int(length * window_fraction)),
        length - skipped,
    )
    initial = fsum(values[skipped : skipped + window]) / window
    final = fsum(values[-window:]) / window
    score = final - initial_penalty * initial
    if not isfinite(score):
        raise ValueError("Consilience score overflowed")
    return ConfidenceWindows(skipped, window, initial, final, score)


__all__ = ["ConfidenceWindows", "confidence_windows"]
