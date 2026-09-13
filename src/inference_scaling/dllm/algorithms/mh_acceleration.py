"""Exact acceleration variants for whole-continuation diffusion MH."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, log
from typing import Hashable, Sequence

import numpy as np

from inference_scaling.dllm.algorithms.mh import (
    DiffusionRewardBatchFunction,
    DiffusionRewardFunction,
    _evaluate_mh_rewards,
    _mh_requests,
    _sample_mh_requests,
)
from inference_scaling.dllm.config import DiffusionMHConfig, DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionBackend, DiffusionSample
from inference_scaling.shared.mh import decide_metropolis_hastings
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class DelayedDiffusionMHStep:
    update: int
    stage_one_accepted: bool
    exact_reward_evaluated: bool
    accepted: bool
    stage_one_log_acceptance: float
    stage_two_log_acceptance: float | None


@dataclass(frozen=True, slots=True)
class DelayedDiffusionMHResult:
    prompt: TokenSequence
    initial: DiffusionSample
    final: DiffusionSample
    final_reward: float
    final_surrogate_reward: float
    steps: tuple[DelayedDiffusionMHStep, ...]
    exact_reward_evaluations: int
    surrogate_reward_evaluations: int

    @property
    def acceptance_rate(self) -> float:
        return (
            sum(step.accepted for step in self.steps) / len(self.steps)
            if self.steps
            else 0.0
        )


def run_diffusion_reward_mh_delayed(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionMHConfig,
    sampling: DiffusionSamplingConfig,
    reward: DiffusionRewardFunction | None = None,
    reward_batch: DiffusionRewardBatchFunction | None = None,
    surrogate_reward: DiffusionRewardFunction | None = None,
    surrogate_reward_batch: DiffusionRewardBatchFunction | None = None,
    proposal_batch_size: int | None = None,
    seed: int = 0,
) -> DelayedDiffusionMHResult:
    """Use a fixed surrogate for stage one and correct it exactly at stage two."""

    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one exact reward callback")
    if (surrogate_reward is None) == (surrogate_reward_batch is None):
        raise ValueError("provide exactly one surrogate reward callback")
    sampling.validate_generation_length(config.total_length, prefix_length=len(prompt))
    seeds = SeedStream(seed)
    requests = _mh_requests(prompt, config, sampling, seeds)
    samples = _sample_mh_requests(backend, requests, proposal_batch_size)
    surrogate_values = _evaluate_mh_rewards(
        prompt, samples, surrogate_reward, surrogate_reward_batch
    )
    current = samples[0]
    current_surrogate = surrogate_values[0]
    current_reward = _evaluate_mh_rewards(
        prompt, (current,), reward, reward_batch
    )[0]
    exact_evaluations = 1
    steps: list[DelayedDiffusionMHStep] = []
    for update, (proposal, proposed_surrogate) in enumerate(
        zip(samples[1:], surrogate_values[1:], strict=True), start=1
    ):
        stage_one = decide_metropolis_hastings(
            current_target_log_density=current_surrogate / config.reward_temperature,
            proposed_target_log_density=proposed_surrogate
            / config.reward_temperature,
            uniform=float(
                seeds.generator("dllm-mh", "delayed", update, "stage-one").random()
            ),
        )
        proposed_reward: float | None = None
        stage_two_log_acceptance: float | None = None
        accepted = False
        if stage_one.accepted:
            proposed_reward = _evaluate_mh_rewards(
                prompt, (proposal,), reward, reward_batch
            )[0]
            exact_evaluations += 1
            stage_two = decide_metropolis_hastings(
                current_target_log_density=(
                    current_reward - current_surrogate
                )
                / config.reward_temperature,
                proposed_target_log_density=(
                    proposed_reward - proposed_surrogate
                )
                / config.reward_temperature,
                uniform=float(
                    seeds.generator(
                        "dllm-mh", "delayed", update, "stage-two"
                    ).random()
                ),
            )
            stage_two_log_acceptance = stage_two.log_acceptance
            accepted = stage_two.accepted
        if accepted:
            assert proposed_reward is not None
            current = proposal
            current_reward = proposed_reward
            current_surrogate = proposed_surrogate
        steps.append(
            DelayedDiffusionMHStep(
                update=update,
                stage_one_accepted=stage_one.accepted,
                exact_reward_evaluated=stage_one.accepted,
                accepted=accepted,
                stage_one_log_acceptance=stage_one.log_acceptance,
                stage_two_log_acceptance=stage_two_log_acceptance,
            )
        )
    return DelayedDiffusionMHResult(
        prompt=prompt,
        initial=samples[0],
        final=current,
        final_reward=current_reward,
        final_surrogate_reward=current_surrogate,
        steps=tuple(steps),
        exact_reward_evaluations=exact_evaluations,
        surrogate_reward_evaluations=len(samples),
    )


def _trajectory_key(sample: DiffusionSample) -> Hashable:
    return (
        sample.token_ids,
        tuple(
            (
                step.block_index,
                step.step_index,
                step.positions,
                step.token_ids,
            )
            for step in sample.trace
        ),
    )


@dataclass(frozen=True, slots=True)
class ReplayMixtureDiffusionMHStep:
    update: int
    proposal_source: str
    accepted: bool
    log_acceptance: float


@dataclass(frozen=True, slots=True)
class ReplayMixtureDiffusionMHResult:
    prompt: TokenSequence
    final: DiffusionSample
    final_reward: float
    steps: tuple[ReplayMixtureDiffusionMHStep, ...]
    base_draws: int
    history_draws: int

    @property
    def acceptance_rate(self) -> float:
        return (
            sum(step.accepted for step in self.steps) / len(self.steps)
            if self.steps
            else 0.0
        )


def run_diffusion_replay_mixture_mh(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionMHConfig,
    sampling: DiffusionSamplingConfig,
    history: Sequence[DiffusionSample],
    history_probability: float,
    reward: DiffusionRewardFunction | None = None,
    reward_batch: DiffusionRewardBatchFunction | None = None,
    seed: int = 0,
) -> ReplayMixtureDiffusionMHResult:
    """Use a frozen empirical trajectory cache inside an exact independence proposal."""

    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one reward callback")
    if not 0 <= history_probability < 1:
        raise ValueError("history_probability must lie in [0, 1)")
    if history_probability > 0 and not history:
        raise ValueError("positive history probability requires a frozen cache")
    if not sampling.has_exact_trajectory_density:
        raise ValueError("replay-mixture MH requires exact trajectory probabilities")
    for sample in history:
        if (
            sample.prefix != prompt
            or sample.trajectory_logprob is None
            or sample.policy_id != sampling.policy_id
            or sample.model_id != backend.model_id
        ):
            raise ValueError("cached trajectories must match the prompt and exact policy")
    seeds = SeedStream(seed)
    source_rng = seeds.generator("dllm-replay-mh", "sources")
    use_history = [False] + [
        bool(value)
        for value in source_rng.random(config.updates) < history_probability
    ]
    samples: list[DiffusionSample | None] = [None] * (config.updates + 1)
    base_positions = [index for index, cached in enumerate(use_history) if not cached]
    requests = _mh_requests(prompt, config, sampling, seeds)
    base_requests = [requests[index] for index in base_positions]
    base_samples = _sample_mh_requests(backend, base_requests, None)
    for index, sample in zip(base_positions, base_samples, strict=True):
        samples[index] = sample
    history_draws = 0
    for index, cached in enumerate(use_history):
        if not cached:
            continue
        history_index = int(
            seeds.generator("dllm-replay-mh", "history", index).integers(len(history))
        )
        samples[index] = history[history_index]
        history_draws += 1
    resolved = tuple(sample for sample in samples if sample is not None)
    if len(resolved) != config.updates + 1:
        raise RuntimeError("replay-mixture proposal routing omitted a sample")
    rewards = _evaluate_mh_rewards(prompt, resolved, reward, reward_batch)
    history_counts = Counter(_trajectory_key(sample) for sample in history)

    def proposal_logprob(sample: DiffusionSample) -> float:
        if sample.trajectory_logprob is None:
            raise ValueError("an MH proposal omitted its trajectory probability")
        if history_probability == 0:
            return float(sample.trajectory_logprob)
        empirical_count = history_counts.get(_trajectory_key(sample), 0)
        history_logprob = (
            log(empirical_count / len(history)) if empirical_count else float("-inf")
        )
        return float(
            np.logaddexp(
                log(1 - history_probability) + sample.trajectory_logprob,
                log(history_probability) + history_logprob,
            )
        )

    current = resolved[0]
    current_reward = rewards[0]
    current_q = proposal_logprob(current)
    steps = []
    for update, (proposal, proposed_reward, cached) in enumerate(
        zip(resolved[1:], rewards[1:], use_history[1:], strict=True), start=1
    ):
        if proposal.trajectory_logprob is None or current.trajectory_logprob is None:
            raise ValueError("replay-mixture MH requires exact trajectory scores")
        proposed_q = proposal_logprob(proposal)
        decision = decide_metropolis_hastings(
            current_target_log_density=(
                current.trajectory_logprob
                + current_reward / config.reward_temperature
            ),
            proposed_target_log_density=(
                proposal.trajectory_logprob
                + proposed_reward / config.reward_temperature
            ),
            forward_proposal_log_probability=proposed_q,
            reverse_proposal_log_probability=current_q,
            uniform=float(
                seeds.generator("dllm-replay-mh", "accept", update).random()
            ),
        )
        if decision.accepted:
            current = proposal
            current_reward = proposed_reward
            current_q = proposed_q
        steps.append(
            ReplayMixtureDiffusionMHStep(
                update=update,
                proposal_source="history" if cached else "base",
                accepted=decision.accepted,
                log_acceptance=decision.log_acceptance,
            )
        )
    if not isfinite(current_reward):
        raise ValueError("reward must be finite")
    return ReplayMixtureDiffusionMHResult(
        prompt=prompt,
        final=current,
        final_reward=current_reward,
        steps=tuple(steps),
        base_draws=len(base_positions),
        history_draws=history_draws,
    )


__all__ = [
    "DelayedDiffusionMHResult",
    "DelayedDiffusionMHStep",
    "ReplayMixtureDiffusionMHResult",
    "ReplayMixtureDiffusionMHStep",
    "run_diffusion_replay_mixture_mh",
    "run_diffusion_reward_mh_delayed",
]
