"""Importance-weight identities shared by ARLLM and dLLM replay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, log1p
from typing import Generic, Literal, TypeVar

import numpy as np

from inference_scaling.shared.metrics import importance_effective_sample_size

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class RolloutObservation(Generic[PayloadT]):
    """One terminal completion before its IS or replay weight is applied."""

    reward: float
    target_logprob: float | None = None
    proposal_logprob: float | None = None
    payload: PayloadT | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.reward):
            raise ValueError("rollout reward must be finite")
        for name, value in (
            ("target_logprob", self.target_logprob),
            ("proposal_logprob", self.proposal_logprob),
        ):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when provided")


@dataclass(frozen=True, slots=True)
class WeightedRollout(Generic[PayloadT]):
    observation: RolloutObservation[PayloadT]
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float | None
    log_weight: float


@dataclass(frozen=True, slots=True)
class MonteCarloWeightEstimate(Generic[PayloadT]):
    log_weight: float
    rollouts: tuple[WeightedRollout[PayloadT], ...]


class MonteCarloRolloutWeightProvider(Generic[PayloadT]):
    """Weights on-policy, off-policy, or deliberately uncorrected rollouts."""

    def __init__(
        self,
        *,
        reward_temperature: float,
        correction: Literal["importance", "identity", "none"] = "importance",
        log_ratio_clip: float | None = None,
    ) -> None:
        if reward_temperature <= 0:
            raise ValueError("reward_temperature must be positive")
        if correction not in {"importance", "identity", "none"}:
            raise ValueError(f"unknown rollout correction mode {correction!r}")
        if log_ratio_clip is not None and log_ratio_clip <= 0:
            raise ValueError("log_ratio_clip must be positive when provided")
        if correction != "importance" and log_ratio_clip is not None:
            raise ValueError("log-ratio clipping requires importance correction")
        self.reward_temperature = float(reward_temperature)
        self.correction = correction
        self.log_ratio_clip = log_ratio_clip

    def weight(self, observation: RolloutObservation[PayloadT]) -> WeightedRollout[PayloadT]:
        if self.correction == "importance":
            if observation.target_logprob is None or observation.proposal_logprob is None:
                raise ValueError(
                    "importance correction requires target and proposal log-probabilities"
                )
            ratio = (
                observation.target_logprob - observation.proposal_logprob
            )
            raw_ratio: float | None = ratio
            applied_ratio: float | None = ratio
            if self.log_ratio_clip is not None:
                applied_ratio = min(
                    max(ratio, -self.log_ratio_clip), self.log_ratio_clip
                )
        elif self.correction == "identity":
            raw_ratio = 0.0
            applied_ratio = 0.0
        else:
            raw_ratio = None
            applied_ratio = None
        log_weight = observation.reward / self.reward_temperature
        if applied_ratio is not None:
            log_weight += applied_ratio
        return WeightedRollout(
            observation=observation,
            raw_log_importance_ratio=raw_ratio,
            applied_log_importance_ratio=applied_ratio,
            log_weight=log_weight,
        )

    def estimate(
        self, observations: Sequence[RolloutObservation[PayloadT]]
    ) -> MonteCarloWeightEstimate[PayloadT]:
        weighted = tuple(self.weight(observation) for observation in observations)
        if not weighted:
            raise ValueError("at least one rollout is required to estimate a conditional weight")
        return MonteCarloWeightEstimate(
            log_weight=logmeanexp([rollout.log_weight for rollout in weighted]),
            rollouts=weighted,
        )


@dataclass(frozen=True, slots=True)
class ProbabilityObservation:
    target_logprob: float
    behavior_logprob: float
    reward: float


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


def corrected_replay_log_weight(
    history: Sequence[ProbabilityObservation],
    fresh: Sequence[ProbabilityObservation],
    *,
    truncation: float,
    reward_temperature: float,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Truncated history estimator plus its exact fresh-sample tail correction."""

    if not history:
        raise ValueError("corrected replay requires at least one history observation")
    if not fresh:
        raise ValueError("corrected replay requires at least one fresh observation")
    if truncation <= 0 or reward_temperature <= 0:
        raise ValueError("truncation and reward_temperature must be positive")
    log_truncation = log(truncation)
    history_terms: list[float] = []
    for observation in history:
        log_ratio = observation.target_logprob - observation.behavior_logprob
        if not isfinite(log_ratio):
            raise ValueError("history behavior support must lie inside target support")
        history_terms.append(
            min(log_truncation, log_ratio)
            + observation.reward / reward_temperature
        )

    fresh_terms: list[float] = []
    for observation in fresh:
        if observation.behavior_logprob == float("-inf"):
            log_tail = 0.0
        else:
            log_ratio = observation.target_logprob - observation.behavior_logprob
            if log_ratio <= log_truncation:
                log_tail = float("-inf")
            else:
                log_tail = log1p(-exp(log_truncation - log_ratio))
        fresh_terms.append(log_tail + observation.reward / reward_temperature)

    history_mean = logmeanexp(history_terms)
    finite_fresh = [value for value in fresh_terms if value != float("-inf")]
    fresh_mean = (
        logmeanexp(finite_fresh) + log(len(finite_fresh) / len(fresh_terms))
        if finite_fresh
        else float("-inf")
    )
    return (
        float(np.logaddexp(history_mean, fresh_mean)),
        tuple(history_terms),
        tuple(fresh_terms),
    )


@dataclass(frozen=True, slots=True)
class ReplayWeightEstimate:
    log_weight: float
    history_log_terms: tuple[float, ...]
    fresh_log_terms: tuple[float, ...]

    @property
    def history_count(self) -> int:
        return len(self.history_log_terms)

    @property
    def fresh_count(self) -> int:
        return len(self.fresh_log_terms)

    @property
    def history_ess(self) -> float:
        return importance_effective_sample_size(self.history_log_terms)

    @property
    def fresh_ess(self) -> float:
        finite = tuple(value for value in self.fresh_log_terms if isfinite(value))
        return importance_effective_sample_size(finite)


class TruncatedReplayRolloutWeightProvider:
    """Unbiased truncated-history estimator with a fresh target-policy tail."""

    def __init__(self, *, truncation: float, reward_temperature: float) -> None:
        if truncation <= 0 or reward_temperature <= 0:
            raise ValueError("truncation and reward_temperature must be positive")
        self.truncation = float(truncation)
        self.reward_temperature = float(reward_temperature)

    def estimate(
        self,
        history: Sequence[ProbabilityObservation],
        fresh: Sequence[ProbabilityObservation],
    ) -> ReplayWeightEstimate:
        log_weight, history_terms, fresh_terms = corrected_replay_log_weight(
            history,
            fresh,
            truncation=self.truncation,
            reward_temperature=self.reward_temperature,
        )
        return ReplayWeightEstimate(log_weight, history_terms, fresh_terms)


__all__ = [
    "MonteCarloWeightEstimate",
    "MonteCarloRolloutWeightProvider",
    "ProbabilityObservation",
    "ReplayWeightEstimate",
    "RolloutObservation",
    "TruncatedReplayRolloutWeightProvider",
    "WeightedRollout",
    "corrected_replay_log_weight",
    "logmeanexp",
]
