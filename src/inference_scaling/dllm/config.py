"""Configuration for masked diffusion language-model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from inference_scaling.shared.config import (
    canonical_float,
    require_nonnegative,
    require_positive,
    require_probability,
)

RemaskingStrategy = Literal[
    "low_confidence",
    "low_confidence_static",
    "low_confidence_dynamic",
    "random",
    "sequential",
]

@dataclass(frozen=True, slots=True)
class DiffusionSamplingConfig:
    """One reverse-diffusion sampling policy.

    ``steps_per_block`` is used instead of a total step count so the same policy
    can generate a candidate block or a longer rollout without changing its
    per-block transition kernel.
    """

    block_length: int = 32
    steps_per_block: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    cfg_scale: float = 0.0
    remasking: RemaskingStrategy = "low_confidence"
    confidence_threshold: float = 0.85
    mask_token_id: int | None = None

    def __post_init__(self) -> None:
        require_positive("block_length", self.block_length)
        require_positive("steps_per_block", self.steps_per_block)
        if self.steps_per_block > self.block_length:
            raise ValueError("steps_per_block cannot exceed block_length")
        require_nonnegative("temperature", self.temperature)
        require_nonnegative("top_k", self.top_k)
        require_probability("top_p", self.top_p, include_zero=False)
        require_nonnegative("cfg_scale", self.cfg_scale)
        if self.remasking not in (
            "low_confidence",
            "low_confidence_static",
            "low_confidence_dynamic",
            "random",
            "sequential",
        ):
            raise ValueError(f"unsupported remasking strategy {self.remasking!r}")
        require_probability(
            "confidence_threshold",
            self.confidence_threshold,
            include_zero=False,
        )
        if self.mask_token_id is not None and self.mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")

    @property
    def policy_id(self) -> str:
        return (
            f"block={self.block_length};steps={self.steps_per_block};"
            f"temperature={canonical_float(self.temperature)};top_k={self.top_k};"
            f"top_p={canonical_float(self.top_p)};"
            f"cfg={canonical_float(self.cfg_scale)};remasking={self.remasking};"
            f"threshold={canonical_float(self.confidence_threshold)};"
            f"mask={self.mask_token_id}"
        )

    @property
    def has_exact_trajectory_density(self) -> bool:
        """Whether committed transitions have a tractable normalized density."""

        return self.temperature > 0 and self.remasking in {"random", "sequential"}

    def validate_generation_length(
        self,
        generation_length: int,
        *,
        prefix_length: int | None = None,
    ) -> None:
        require_positive("generation_length", generation_length)
        if prefix_length is not None and prefix_length < 0:
            raise ValueError("prefix_length must be non-negative")
        if generation_length % self.block_length:
            raise ValueError("generation_length must be divisible by block_length")

    def total_steps(
        self,
        generation_length: int,
        *,
        prefix_length: int = 0,
    ) -> int:
        self.validate_generation_length(
            generation_length, prefix_length=prefix_length
        )
        return generation_length // self.block_length * self.steps_per_block


@dataclass(frozen=True, slots=True)
class DiffusionISConfig:
    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 32
    total_length: int = 128
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.importance_log_ratio_clip is not None:
            require_positive("importance_log_ratio_clip", self.importance_log_ratio_clip)


@dataclass(frozen=True, slots=True)
class DiffusionMHConfig:
    total_length: int = 128
    updates: int = 8
    reward_temperature: float = 1.0

    def __post_init__(self) -> None:
        require_positive("total_length", self.total_length)
        require_positive("updates", self.updates)
        require_positive("reward_temperature", self.reward_temperature)


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHConfig:
    """Finite-step sharpening of an exact reverse-trajectory policy."""

    total_length: int = 128
    decision_block_size: int = 32
    updates_per_stage: int = 2
    alpha: float = 2.0

    def __post_init__(self) -> None:
        for name in ("total_length", "decision_block_size", "updates_per_stage"):
            require_positive(name, getattr(self, name))
        if self.decision_block_size > self.total_length:
            raise ValueError("decision_block_size cannot exceed total_length")
        require_positive("alpha", self.alpha)


@dataclass(frozen=True, slots=True)
class DiffusionBlockBeamConfig:
    """Sampled diffusion-block search used as the counterpart of token beam search."""

    total_length: int = 128
    decision_block_size: int = 32
    width: int = 8
    branching_factor: int = 2

    def __post_init__(self) -> None:
        for name in ("total_length", "decision_block_size", "width", "branching_factor"):
            require_positive(name, getattr(self, name))
        if self.decision_block_size > self.total_length:
            raise ValueError("decision_block_size cannot exceed total_length")


def diffusion_decision_stage_lengths(
    *,
    prompt_length: int,
    total_length: int,
    decision_block_size: int,
    sampling: DiffusionSamplingConfig,
) -> tuple[int, ...]:
    """Partition a continuation without splitting a diffusion block."""

    for name, value in (
        ("total_length", total_length),
        ("decision_block_size", decision_block_size),
    ):
        require_positive(name, value)
    if decision_block_size > total_length:
        raise ValueError("decision_block_size cannot exceed total_length")
    sampling.validate_generation_length(total_length, prefix_length=prompt_length)
    if prompt_length < 0:
        raise ValueError("prompt_length must be non-negative")
    sampling.validate_generation_length(decision_block_size)
    first = decision_block_size
    lengths: list[int] = []
    remaining = total_length
    next_length = first
    while remaining:
        length = min(next_length, remaining)
        lengths.append(length)
        remaining -= length
        next_length = decision_block_size
    offset = 0
    for length in lengths:
        sampling.validate_generation_length(
            length,
            prefix_length=prompt_length + offset,
        )
        offset += length
    return tuple(lengths)


@dataclass(frozen=True, slots=True)
class VRPOSamplingConfig:
    """Monte Carlo layout for the masked-diffusion ELBO estimator."""

    timestep_samples: int = 8
    masks_per_timestep: int = 1
    antithetic: bool = True

    def __post_init__(self) -> None:
        require_positive("timestep_samples", self.timestep_samples)
        require_positive("masks_per_timestep", self.masks_per_timestep)

    @property
    def forward_passes(self) -> int:
        return self.timestep_samples * self.masks_per_timestep


__all__ = [
    "DiffusionBlockBeamConfig",
    "DiffusionISConfig",
    "DiffusionMHConfig",
    "DiffusionPowerMHConfig",
    "DiffusionSamplingConfig",
    "RemaskingStrategy",
    "VRPOSamplingConfig",
    "diffusion_decision_stage_lengths",
]
