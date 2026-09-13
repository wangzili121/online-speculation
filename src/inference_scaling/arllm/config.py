"""Validated configuration objects shared by all algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from inference_scaling.shared.generation import DEFAULT_MAX_NEW_TOKENS

from inference_scaling.shared.config import (
    RuntimeConfig as RuntimeConfig,
    SMCForestConfig as SMCForestConfig,
    canonical_float,
    require_finite,
    require_positive,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """The complete stochastic policy used for one autoregressive request."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        require_positive("temperature", self.temperature)
        require_probability("top_p", self.top_p, include_zero=False)
        if self.top_k is not None:
            require_positive("top_k", self.top_k)
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")

    @property
    def policy_id(self) -> str:
        return (
            f"temperature={canonical_float(self.temperature)};"
            f"top_p={canonical_float(self.top_p)};"
            f"top_k={self.top_k};eos={self.eos_token_id}"
        )


@dataclass(frozen=True, slots=True)
class MHConfig:
    alpha: float = 4.0
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    block_size: int = 32
    steps_per_block: int = 10
    chains: int = 1
    suffix_schedule: str = "uniform"
    iterations: int | None = None

    def __post_init__(self) -> None:
        require_finite("alpha", self.alpha)
        if self.alpha < 1:
            raise ValueError("alpha must be at least one")
        for name in ("total_length", "block_size", "steps_per_block", "chains"):
            require_positive(name, getattr(self, name))
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale"}:
            raise ValueError("unknown MH suffix_schedule")
        if self.iterations is not None:
            require_positive("iterations", self.iterations)

    @property
    def stages(self) -> tuple[int, ...]:
        if self.iterations is not None:
            return (self.total_length,)
        lengths = tuple(range(self.block_size, self.total_length + 1, self.block_size))
        return lengths if lengths and lengths[-1] == self.total_length else (*lengths, self.total_length)

    @property
    def stage_updates(self) -> int:
        return self.steps_per_block if self.iterations is None else self.iterations


@dataclass(frozen=True, slots=True)
class RewardMHConfig:
    """Full-sequence MH budget for a base-times-exponentiated-reward target."""

    total_length: int = DEFAULT_MAX_NEW_TOKENS
    block_size: int = 32
    steps_per_block: int = 10
    reward_temperature: float = 0.1
    suffix_schedule: str = "uniform"
    iterations: int | None = None

    def __post_init__(self) -> None:
        for name in ("total_length", "block_size", "steps_per_block"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale"}:
            raise ValueError("unknown MH suffix_schedule")
        if self.iterations is not None:
            require_positive("iterations", self.iterations)

    @property
    def updates(self) -> int:
        if self.iterations is not None:
            return self.iterations
        blocks = (self.total_length + self.block_size - 1) // self.block_size
        return blocks * self.steps_per_block


@dataclass(frozen=True, slots=True)
class ConditionalISConfig:
    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    apply_importance_correction: bool = True
    rollout_design: str = "iid"
    exact_rollout_early_stop: bool = False
    rollout_log_weight_bounds: tuple[float, float] | None = None
    rollout_evaluation_batch_size: int = 1

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            require_positive(
                "importance_log_ratio_clip",
                self.importance_log_ratio_clip,
            )
        if (
            not self.apply_importance_correction
            and self.importance_log_ratio_clip is not None
        ):
            raise ValueError(
                "importance_log_ratio_clip requires apply_importance_correction=True"
            )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.rollout_design not in {
            "iid",
            "scrambled_sobol",
            "arithmetic_lattice",
        }:
            raise ValueError("unknown rollout_design")
        require_positive(
            "rollout_evaluation_batch_size",
            self.rollout_evaluation_batch_size,
        )
        if self.rollout_log_weight_bounds is not None:
            if len(self.rollout_log_weight_bounds) != 2:
                raise ValueError("rollout_log_weight_bounds requires two values")
            lower, upper = self.rollout_log_weight_bounds
            require_finite("rollout_log_weight_lower_bound", lower)
            require_finite("rollout_log_weight_upper_bound", upper)
            if lower > upper:
                raise ValueError("rollout log-weight bounds must be ordered")
        if self.exact_rollout_early_stop:
            if self.rollout_log_weight_bounds is None:
                raise ValueError(
                    "exact rollout early stopping requires log-weight bounds"
                )
            if self.rollout_design != "iid":
                raise ValueError(
                    "exact rollout early stopping currently requires iid rollouts"
                )
        elif self.rollout_log_weight_bounds is not None:
            raise ValueError(
                "rollout_log_weight_bounds require exact_rollout_early_stop=True"
            )


@dataclass(frozen=True, slots=True)
class IteratedConditionalISConfig:
    """Finite-pool i-SIR updates for each autoregressive candidate block."""

    pool_size: int = 3
    updates: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    apply_importance_correction: bool = True

    def __post_init__(self) -> None:
        for name in (
            "pool_size",
            "updates",
            "rollout_count",
            "block_size",
            "total_length",
        ):
            require_positive(name, getattr(self, name))
        if self.pool_size < 2:
            raise ValueError("pool_size must be at least two")
        require_positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            require_positive(
                "importance_log_ratio_clip",
                self.importance_log_ratio_clip,
            )
        if (
            not self.apply_importance_correction
            and self.importance_log_ratio_clip is not None
        ):
            raise ValueError(
                "importance_log_ratio_clip requires apply_importance_correction=True"
            )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")

    @property
    def fresh_candidate_evaluations(self) -> int:
        """Distinct extended states evaluated at one generation step."""

        return 1 + self.updates * (self.pool_size - 1)

    @property
    def pool_candidate_uses(self) -> int:
        return self.updates * self.pool_size


@dataclass(frozen=True, slots=True)
class ProgressiveISConfig:
    """Pilot/evaluation split for cost-aware conditional-weight estimation."""

    candidate_count: int = 4
    pilot_rollouts_per_candidate: int = 2
    evaluation_cost_budget: float = 16.0
    minimum_evaluation_per_candidate: int = 1
    block_size: int = 16
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    reward_workers: int = 4
    run_ahead_rollouts_per_candidate: int = 0
    evaluation_reference_rollouts_per_candidate: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate_count",
            "pilot_rollouts_per_candidate",
            "minimum_evaluation_per_candidate",
            "block_size",
            "total_length",
            "reward_workers",
        ):
            require_positive(name, getattr(self, name))
        require_positive("evaluation_cost_budget", self.evaluation_cost_budget)
        require_positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            require_positive(
                "importance_log_ratio_clip", self.importance_log_ratio_clip
            )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.run_ahead_rollouts_per_candidate < 0:
            raise ValueError("run_ahead_rollouts_per_candidate must be non-negative")
        if (
            self.evaluation_reference_rollouts_per_candidate is not None
            and self.evaluation_reference_rollouts_per_candidate <= 0
        ):
            raise ValueError(
                "evaluation_reference_rollouts_per_candidate must be positive"
            )


@dataclass(frozen=True, slots=True)
class BaseReplayConfig:
    candidate_count: int = 4
    block_size: int = 16
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    reward_temperature: float = 1.0
    max_history_per_candidate: int = 8
    fresh_rollouts: int = 2
    truncation: float = 8.0
    reserve_rollouts: int = 0

    def __post_init__(self) -> None:
        for name in ("candidate_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.max_history_per_candidate < 0:
            raise ValueError("max_history_per_candidate must be non-negative")
        require_positive("fresh_rollouts", self.fresh_rollouts)
        require_positive("truncation", self.truncation)
        if self.reserve_rollouts < 0:
            raise ValueError("reserve_rollouts must be non-negative")


@dataclass(frozen=True, slots=True)
class DynamicISConfig:
    candidate_count: int = 4
    block_size: int = 16
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    reward_temperature: float = 1.0
    max_history_per_candidate: int = 8
    truncation: float = 8.0
    reserve_rollouts: int = 0
    rollout_budget: float = 64.0
    auxiliary_mixture: float = 0.25
    minimum_fresh_per_candidate: int = 1

    def __post_init__(self) -> None:
        for name in ("candidate_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.max_history_per_candidate < 0:
            raise ValueError("max_history_per_candidate must be non-negative")
        require_positive("truncation", self.truncation)
        if self.reserve_rollouts < 0:
            raise ValueError("reserve_rollouts must be non-negative")
        require_positive("rollout_budget", self.rollout_budget)
        require_probability("auxiliary_mixture", self.auxiliary_mixture)
        if self.auxiliary_mixture >= 1:
            raise ValueError("auxiliary_mixture must lie in [0, 1)")
        require_positive(
            "minimum_fresh_per_candidate", self.minimum_fresh_per_candidate
        )
