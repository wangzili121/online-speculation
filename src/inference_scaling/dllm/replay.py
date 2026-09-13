"""Exact trajectory replay for block-diffusion rollout weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTrajectoryScoreRequest,
)
from inference_scaling.shared.importance import (
    ProbabilityObservation,
    ReplayWeightEstimate,
    TruncatedReplayRolloutWeightProvider,
    logmeanexp,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.stepwise import normalize_log_weights
from inference_scaling.shared.types import TokenSequence
from inference_scaling.shared.verifier import TokenBatchReward

DiffusionReplayRewardBatch = TokenBatchReward


def _validate_exact_pair(
    target: DiffusionSamplingConfig,
    behavior: DiffusionSamplingConfig,
) -> None:
    if not target.has_exact_trajectory_density or not behavior.has_exact_trajectory_density:
        raise ValueError("replay correction requires exact target and behavior trajectories")
    target_schedule = (
        target.block_length,
        target.steps_per_block,
        target.remasking,
        target.mask_token_id,
    )
    behavior_schedule = (
        behavior.block_length,
        behavior.steps_per_block,
        behavior.remasking,
        behavior.mask_token_id,
    )
    if target_schedule != behavior_schedule:
        raise ValueError("target and replay behavior trajectory schedules must match")


@dataclass(frozen=True, slots=True)
class DiffusionReplayRecord:
    rollout_prefix: TokenSequence
    sample: DiffusionSample
    target_trajectory_logprob: float
    behavior_trajectory_logprob: float
    reward: float

    def __post_init__(self) -> None:
        if self.sample.prefix != self.rollout_prefix:
            raise ValueError("replay sample prefix does not match its cache key")
        if not isfinite(self.target_trajectory_logprob):
            raise ValueError("target trajectory probability must be finite")
        if not isfinite(self.behavior_trajectory_logprob):
            raise ValueError("behavior trajectory probability must be finite")

    @property
    def observation(self) -> ProbabilityObservation:
        return ProbabilityObservation(
            self.target_trajectory_logprob,
            self.behavior_trajectory_logprob,
            self.reward,
        )


@dataclass(frozen=True, slots=True)
class DiffusionReplayHistory:
    rollout_prefix: TokenSequence
    records: tuple[DiffusionReplayRecord, ...]

    def __post_init__(self) -> None:
        if any(record.rollout_prefix != self.rollout_prefix for record in self.records):
            raise ValueError("replay history mixes different rollout prefixes")


@dataclass(frozen=True, slots=True)
class DiffusionReplayCandidate:
    sample: DiffusionSample
    estimate: ReplayWeightEstimate
    outer_log_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class DiffusionReplaySelection:
    candidates: tuple[DiffusionReplayCandidate, ...]
    probabilities: tuple[float, ...]
    selected_index: int
    fresh_records: tuple[tuple[DiffusionReplayRecord, ...], ...]

    @property
    def selected(self) -> DiffusionReplayCandidate:
        return self.candidates[self.selected_index]


def build_diffusion_replay_history(
    *,
    target_backend: DiffusionBackend,
    behavior_backend: DiffusionBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[DiffusionSample],
    rollout_length: int,
    count_per_candidate: int,
    target_sampling: DiffusionSamplingConfig,
    behavior_sampling: DiffusionSamplingConfig,
    reward_batch: DiffusionReplayRewardBatch,
    seed: int,
) -> tuple[DiffusionReplayHistory, ...]:
    """Generate immutable off-policy trajectories and precompute target scores."""

    _validate_exact_pair(target_sampling, behavior_sampling)
    if count_per_candidate <= 0:
        raise ValueError("count_per_candidate must be positive")
    if rollout_length <= 0:
        return tuple(
            DiffusionReplayHistory(prompt + generated_prefix + candidate.token_ids, ())
            for candidate in candidates
        )
    seeds = SeedStream(seed)
    requests: list[DiffusionGenerationRequest] = []
    owners: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        rollout_prefix = prompt + generated_prefix + candidate.token_ids
        for history_index in range(count_per_candidate):
            requests.append(
                DiffusionGenerationRequest(
                    prefix=rollout_prefix,
                    generation_length=rollout_length,
                    sampling=behavior_sampling,
                    seed=seeds.derive("dllm-replay", candidate_index, history_index),
                    request_id=f"dllm-replay:{candidate_index}:{history_index}",
                )
            )
            owners.append(candidate_index)
    samples = behavior_backend.sample_batch(requests)
    if len(samples) != len(requests):
        raise RuntimeError("behavior backend returned an invalid replay sample count")
    target_scores = target_backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, target_sampling) for sample in samples]
    )
    if len(target_scores) != len(samples):
        raise RuntimeError("target backend returned an invalid replay score count")
    continuations = [
        generated_prefix + candidates[owner].token_ids + sample.token_ids
        for owner, sample in zip(owners, samples, strict=True)
    ]
    rewards = [float(value) for value in reward_batch(prompt, continuations)]
    if len(rewards) != len(samples):
        raise RuntimeError("reward evaluator returned an invalid replay count")
    grouped: list[list[DiffusionReplayRecord]] = [[] for _ in candidates]
    for owner, sample, target_score, reward in zip(
        owners, samples, target_scores, rewards, strict=True
    ):
        if sample.trajectory_logprob is None:
            raise RuntimeError("behavior backend omitted replay trajectory probability")
        grouped[owner].append(
            DiffusionReplayRecord(
                rollout_prefix=sample.prefix,
                sample=sample,
                target_trajectory_logprob=float(target_score),
                behavior_trajectory_logprob=float(sample.trajectory_logprob),
                reward=reward,
            )
        )
    return tuple(
        DiffusionReplayHistory(
            prompt + generated_prefix + candidate.token_ids,
            tuple(records),
        )
        for candidate, records in zip(candidates, grouped, strict=True)
    )


def select_diffusion_candidates_with_replay(
    *,
    target_backend: DiffusionBackend,
    behavior_backend: DiffusionBackend | None,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[DiffusionSample],
    histories: Sequence[DiffusionReplayHistory] | None,
    rollout_length: int,
    fresh_count: int | Sequence[int],
    target_sampling: DiffusionSamplingConfig,
    behavior_sampling: DiffusionSamplingConfig | None,
    reward_batch: DiffusionReplayRewardBatch,
    reward_temperature: float,
    truncation: float,
    seed: int,
    candidate_log_ratios: Sequence[float] | None = None,
) -> DiffusionReplaySelection:
    """Estimate each rollout weight from fresh data or corrected replay plus a fresh tail."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if isinstance(fresh_count, int):
        fresh_counts = (fresh_count,) * len(candidates)
    else:
        fresh_counts = tuple(int(value) for value in fresh_count)
    if len(fresh_counts) != len(candidates) or any(value <= 0 for value in fresh_counts):
        raise ValueError("fresh_count must be positive for every candidate")
    if reward_temperature <= 0 or truncation <= 0:
        raise ValueError("reward_temperature and truncation must be positive")
    outer_log_ratios = tuple(candidate_log_ratios or (0.0,) * len(candidates))
    if len(outer_log_ratios) != len(candidates) or any(
        not isfinite(value) for value in outer_log_ratios
    ):
        raise ValueError("candidate log-ratios must be finite and match the candidates")
    histories = histories or tuple(
        DiffusionReplayHistory(prompt + generated_prefix + candidate.token_ids, ())
        for candidate in candidates
    )
    if len(histories) != len(candidates):
        raise ValueError("one replay history is required per candidate")
    replay_enabled = any(history.records for history in histories)
    if replay_enabled:
        if behavior_backend is None or behavior_sampling is None:
            raise ValueError("replay records require their behavior backend and policy")
        _validate_exact_pair(target_sampling, behavior_sampling)

    if rollout_length == 0:
        continuations = [generated_prefix + candidate.token_ids for candidate in candidates]
        rewards = [float(value) for value in reward_batch(prompt, continuations)]
        if len(rewards) != len(candidates):
            raise RuntimeError("reward evaluator returned an invalid terminal count")
        replay_candidates = tuple(
            DiffusionReplayCandidate(
                sample=candidate,
                estimate=ReplayWeightEstimate(
                    reward / reward_temperature,
                    (),
                    (reward / reward_temperature,),
                ),
                outer_log_ratio=outer_log_ratio,
            )
            for candidate, reward, outer_log_ratio in zip(
                candidates, rewards, outer_log_ratios, strict=True
            )
        )
        probabilities = normalize_log_weights(
            [
                candidate.estimate.log_weight + candidate.outer_log_ratio
                for candidate in replay_candidates
            ]
        )
        selected = int(np.random.default_rng(seed).choice(len(candidates), p=probabilities))
        return DiffusionReplaySelection(
            replay_candidates,
            probabilities,
            selected,
            tuple(() for _ in candidates),
        )

    seeds = SeedStream(seed)
    requests: list[DiffusionGenerationRequest] = []
    owners: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        expected_prefix = prompt + generated_prefix + candidate.token_ids
        if histories[candidate_index].rollout_prefix != expected_prefix:
            raise ValueError("replay history does not belong to its candidate")
        for fresh_index in range(fresh_counts[candidate_index]):
            requests.append(
                DiffusionGenerationRequest(
                    prefix=expected_prefix,
                    generation_length=rollout_length,
                    sampling=target_sampling,
                    seed=seeds.derive("dllm-replay-fresh", candidate_index, fresh_index),
                    request_id=f"dllm-replay-fresh:{candidate_index}:{fresh_index}",
                )
            )
            owners.append(candidate_index)
    samples = target_backend.sample_batch(requests)
    if len(samples) != len(requests):
        raise RuntimeError("target backend returned an invalid fresh rollout count")
    behavior_scores: list[float | None]
    if replay_enabled:
        assert behavior_backend is not None and behavior_sampling is not None
        resolved = behavior_backend.score_trajectories(
            [DiffusionTrajectoryScoreRequest(sample, behavior_sampling) for sample in samples]
        )
        if len(resolved) != len(samples):
            raise RuntimeError("behavior backend returned an invalid fresh score count")
        behavior_scores = [float(value) for value in resolved]
    else:
        behavior_scores = [None] * len(samples)
    continuations = [
        generated_prefix + candidates[owner].token_ids + sample.token_ids
        for owner, sample in zip(owners, samples, strict=True)
    ]
    rewards = [float(value) for value in reward_batch(prompt, continuations)]
    if len(rewards) != len(samples):
        raise RuntimeError("reward evaluator returned an invalid fresh rollout count")
    grouped_fresh: list[list[DiffusionReplayRecord]] = [[] for _ in candidates]
    for owner, sample, behavior_score, reward in zip(
        owners, samples, behavior_scores, rewards, strict=True
    ):
        if sample.trajectory_logprob is None:
            raise RuntimeError("target backend omitted fresh trajectory probability")
        grouped_fresh[owner].append(
            DiffusionReplayRecord(
                rollout_prefix=sample.prefix,
                sample=sample,
                target_trajectory_logprob=float(sample.trajectory_logprob),
                behavior_trajectory_logprob=(
                    float(behavior_score)
                    if behavior_score is not None
                    else float(sample.trajectory_logprob)
                ),
                reward=reward,
            )
        )

    replay_candidates: list[DiffusionReplayCandidate] = []
    for candidate, history, fresh_records, outer_log_ratio in zip(
        candidates, histories, grouped_fresh, outer_log_ratios, strict=True
    ):
        if history.records:
            shared_estimate = TruncatedReplayRolloutWeightProvider(
                truncation=truncation,
                reward_temperature=reward_temperature,
            ).estimate(
                [record.observation for record in history.records],
                [record.observation for record in fresh_records],
            )
            log_weight = shared_estimate.log_weight
            history_terms = shared_estimate.history_log_terms
            fresh_terms = shared_estimate.fresh_log_terms
        else:
            fresh_terms = tuple(
                record.reward / reward_temperature for record in fresh_records
            )
            history_terms = ()
            log_weight = logmeanexp(fresh_terms)
        replay_candidates.append(
            DiffusionReplayCandidate(
                sample=candidate,
                estimate=ReplayWeightEstimate(
                    log_weight,
                    tuple(history_terms),
                    tuple(fresh_terms),
                ),
                outer_log_ratio=outer_log_ratio,
            )
        )
    probabilities = normalize_log_weights(
        [
            candidate.estimate.log_weight + candidate.outer_log_ratio
            for candidate in replay_candidates
        ]
    )
    selected = int(
        seeds.generator("dllm-replay-select").choice(
            len(replay_candidates), p=probabilities
        )
    )
    return DiffusionReplaySelection(
        tuple(replay_candidates),
        probabilities,
        selected,
        tuple(tuple(records) for records in grouped_fresh),
    )


__all__ = [
    "DiffusionReplayCandidate",
    "DiffusionReplayHistory",
    "DiffusionReplayRecord",
    "DiffusionReplayRewardBatch",
    "DiffusionReplaySelection",
    "build_diffusion_replay_history",
    "select_diffusion_candidates_with_replay",
]
