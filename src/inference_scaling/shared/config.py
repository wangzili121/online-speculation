"""Runtime configuration shared by every model family."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def canonical_float(value: float) -> str:
    """Return a stable, round-trippable identifier for one finite float."""

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"a policy parameter must be finite, got {value!r}")
    return repr(numeric)


def require_finite(name: str, value: int | float) -> None:
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def require_positive(name: str, value: int | float) -> None:
    require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def require_nonnegative(name: str, value: int | float) -> None:
    require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def require_probability(
    name: str,
    value: float,
    *,
    include_zero: bool = True,
) -> None:
    require_finite(name, value)
    lower_valid = value >= 0 if include_zero else value > 0
    if not lower_valid or value > 1:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must lie in {interval}, got {value!r}")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int = 0
    max_batch_size: int = 32
    max_batch_tokens: int = 4096
    deterministic: bool = True

    def __post_init__(self) -> None:
        require_nonnegative("seed", self.seed)
        require_positive("max_batch_size", self.max_batch_size)
        require_positive("max_batch_tokens", self.max_batch_tokens)


@dataclass(frozen=True, slots=True)
class SMCForestConfig:
    """Model-independent particle and rollout layout for an SMC forest."""

    particle_count: int = 8
    branch_factor: int = 2
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    reward_workers: int = 4
    reuse_rollout_forest: bool = True

    def __post_init__(self) -> None:
        for name in (
            "particle_count",
            "branch_factor",
            "rollout_count",
            "block_size",
            "total_length",
            "reward_temperature",
            "reward_workers",
        ):
            require_positive(name, getattr(self, name))
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


__all__ = [
    "RuntimeConfig",
    "SMCForestConfig",
    "canonical_float",
    "require_finite",
    "require_nonnegative",
    "require_positive",
    "require_probability",
]
