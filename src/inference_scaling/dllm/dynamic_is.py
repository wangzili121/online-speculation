"""Defensive candidate proposals and variance--cost replay allocation for dLLMs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Literal, Sequence

import numpy as np

from inference_scaling.dllm.config import (
    DiffusionISConfig,
    DiffusionSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.replay import (
    DiffusionReplayHistory,
    DiffusionReplayRewardBatch,
    DiffusionReplaySelection,
    build_diffusion_replay_history,
    select_diffusion_candidates_with_replay,
)
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTrajectoryScoreRequest,
)
from inference_scaling.shared.budget import (
    BudgetAllocation,
    VarianceCostEstimate,
    allocate_variance_cost_budget,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence

DynamicDiffusionArm = Literal[
    "base_candidate_fixed",
    "trajectory_replay_aware_fixed",
    "trajectory_replay_aware_optimal",
]


@dataclass(frozen=True, slots=True)
class DynamicDiffusionDraw:
    sample: DiffusionSample
    source: str
    target_logprob: float
    auxiliary_logprob: float
    mixture_logprob: float
    outer_log_ratio: float


@dataclass(frozen=True, slots=True)
class DynamicDiffusionStep:
    generated_length_before: int
    draws: tuple[DynamicDiffusionDraw, ...]
    allocations: tuple[BudgetAllocation, ...]
    selection: DiffusionReplaySelection
    design_rollouts: int
    evaluation_history_rollouts: int


@dataclass(frozen=True, slots=True)
class DynamicDiffusionResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    arm: DynamicDiffusionArm
    steps: tuple[DynamicDiffusionStep, ...]

    @property
    def rollout_reuse_rate(self) -> float:
        history = sum(
            candidate.estimate.history_count
            for step in self.steps
            for candidate in step.selection.candidates
        )
        fresh = sum(
            candidate.estimate.fresh_count
            for step in self.steps
            for candidate in step.selection.candidates
        )
        return history / (history + fresh) if history + fresh else 0.0


def _score_subset(
    backend: DiffusionBackend,
    samples: Sequence[DiffusionSample],
    indices: Sequence[int],
    sampling: DiffusionSamplingConfig,
) -> dict[int, float]:
    if not indices:
        return {}
    scores = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(samples[index], sampling) for index in indices]
    )
    if len(scores) != len(indices):
        raise RuntimeError("backend returned an invalid candidate score count")
    return {
        index: float(score) for index, score in zip(indices, scores, strict=True)
    }


def draw_defensive_diffusion_candidates(
    *,
    target_backend: DiffusionBackend,
    auxiliary_backend: DiffusionBackend,
    prefix: TokenSequence,
    generation_length: int,
    count: int,
    sampling: DiffusionSamplingConfig,
    auxiliary_probability: float,
    seed: int,
    stage_index: int,
) -> tuple[DynamicDiffusionDraw, ...]:
    """Draw from a base/auxiliary mixture and compute the exact outer ratio."""

    if count <= 0 or not 0 < auxiliary_probability < 1:
        raise ValueError("count must be positive and mixture probability must lie in (0, 1)")
    if not sampling.has_exact_trajectory_density:
        raise ValueError("dynamic dLLM candidates require an exact trajectory policy")
    seeds = SeedStream(seed)
    source_rng = seeds.generator("dllm-dynamic", stage_index, "sources")
    use_auxiliary = [
        bool(value)
        for value in source_rng.random(count) < auxiliary_probability
    ]
    target_requests: list[DiffusionGenerationRequest] = []
    target_indices: list[int] = []
    auxiliary_requests: list[DiffusionGenerationRequest] = []
    auxiliary_indices: list[int] = []
    for candidate_index, auxiliary in enumerate(use_auxiliary):
        request = DiffusionGenerationRequest(
            prefix=prefix,
            generation_length=generation_length,
            sampling=sampling,
            seed=seeds.derive(
                "dllm-dynamic", stage_index, "candidate", candidate_index
            ),
            request_id=f"dllm-dynamic:{stage_index}:candidate:{candidate_index}",
        )
        if auxiliary:
            auxiliary_requests.append(request)
            auxiliary_indices.append(candidate_index)
        else:
            target_requests.append(request)
            target_indices.append(candidate_index)

    samples: list[DiffusionSample | None] = [None] * count
    for indices, sampled in (
        (target_indices, target_backend.sample_batch(target_requests)),
        (auxiliary_indices, auxiliary_backend.sample_batch(auxiliary_requests)),
    ):
        if len(indices) != len(sampled):
            raise RuntimeError("backend returned an invalid dynamic candidate count")
        for index, sample in zip(indices, sampled, strict=True):
            if sample.trajectory_logprob is None:
                raise RuntimeError("dynamic proposal omitted its trajectory probability")
            samples[index] = sample
    resolved = tuple(sample for sample in samples if sample is not None)
    if len(resolved) != count:
        raise RuntimeError("dynamic candidate routing omitted an output")

    target_scores = {
        index: float(resolved[index].trajectory_logprob)
        for index in target_indices
    }
    target_scores.update(
        _score_subset(target_backend, resolved, auxiliary_indices, sampling)
    )
    auxiliary_scores = {
        index: float(resolved[index].trajectory_logprob)
        for index in auxiliary_indices
    }
    auxiliary_scores.update(
        _score_subset(auxiliary_backend, resolved, target_indices, sampling)
    )
    log_target_mix = log(1 - auxiliary_probability)
    log_auxiliary_mix = log(auxiliary_probability)
    draws = []
    for index, sample in enumerate(resolved):
        target_logprob = target_scores[index]
        auxiliary_logprob = auxiliary_scores[index]
        mixture_logprob = float(
            np.logaddexp(
                log_target_mix + target_logprob,
                log_auxiliary_mix + auxiliary_logprob,
            )
        )
        draws.append(
            DynamicDiffusionDraw(
                sample=sample,
                source="auxiliary" if use_auxiliary[index] else "target",
                target_logprob=target_logprob,
                auxiliary_logprob=auxiliary_logprob,
                mixture_logprob=mixture_logprob,
                outer_log_ratio=target_logprob - mixture_logprob,
            )
        )
    return tuple(draws)


def _draw_target_candidates(
    *,
    backend: DiffusionBackend,
    prefix: TokenSequence,
    generation_length: int,
    count: int,
    sampling: DiffusionSamplingConfig,
    seed: int,
    stage_index: int,
) -> tuple[DynamicDiffusionDraw, ...]:
    seeds = SeedStream(seed)
    samples = backend.sample_batch(
        [
            DiffusionGenerationRequest(
                prefix=prefix,
                generation_length=generation_length,
                sampling=sampling,
                seed=seeds.derive(
                    "dllm-dynamic-base", stage_index, candidate_index
                ),
                request_id=f"dllm-dynamic-base:{stage_index}:{candidate_index}",
            )
            for candidate_index in range(count)
        ]
    )
    if len(samples) != count:
        raise RuntimeError("backend returned an invalid base candidate count")
    return tuple(
        DynamicDiffusionDraw(
            sample=sample,
            source="target",
            target_logprob=(
                float(sample.trajectory_logprob)
                if sample.trajectory_logprob is not None
                else 0.0
            ),
            auxiliary_logprob=(
                float(sample.trajectory_logprob)
                if sample.trajectory_logprob is not None
                else 0.0
            ),
            mixture_logprob=(
                float(sample.trajectory_logprob)
                if sample.trajectory_logprob is not None
                else 0.0
            ),
            outer_log_ratio=0.0,
        )
        for sample in samples
    )


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    result = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    return result if isfinite(result) and result >= 0 else 1.0


def _design_statistics(
    histories: Sequence[DiffusionReplayHistory],
    fresh_histories: Sequence[DiffusionReplayHistory],
    *,
    reward_temperature: float,
    truncation: float,
    history_cost: float,
    fresh_cost: float,
) -> tuple[VarianceCostEstimate, ...]:
    if len(histories) != len(fresh_histories):
        raise ValueError("history and fresh design samples must align by candidate")
    log_truncation = log(truncation)
    reward_shift = max(
        (
            record.reward / reward_temperature
            for design in (*histories, *fresh_histories)
            for record in design.records
        ),
        default=0.0,
    )
    result: list[VarianceCostEstimate] = []
    for history, fresh in zip(histories, fresh_histories, strict=True):
        history_contributions = [
            exp(
                min(
                    log_truncation,
                    record.target_trajectory_logprob
                    - record.behavior_trajectory_logprob,
                )
                + record.reward / reward_temperature
                - reward_shift
            )
            for record in history.records
        ]
        fresh_contributions = [
            max(
                0.0,
                1.0
                - exp(
                    min(
                        0.0,
                        log_truncation
                        + record.target_trajectory_logprob
                        - record.behavior_trajectory_logprob,
                    )
                ),
            )
            * exp(record.reward / reward_temperature - reward_shift)
            for record in fresh.records
        ]
        result.append(
            VarianceCostEstimate(
                history_std=_sample_standard_deviation(history_contributions),
                fresh_std=_sample_standard_deviation(fresh_contributions),
                history_cost=history_cost,
                fresh_cost=fresh_cost,
            )
        )
    return tuple(result)


def run_dynamic_diffusion_is(
    *,
    arm: DynamicDiffusionArm,
    target_backend: DiffusionBackend,
    auxiliary_backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionISConfig,
    sampling: DiffusionSamplingConfig,
    reward_batch: DiffusionReplayRewardBatch,
    history_rollouts: int,
    fresh_rollouts: int,
    truncation: float,
    auxiliary_probability: float = 0.5,
    history_cost: float = 0.05,
    fresh_cost: float = 1.0,
    design_rollouts: int = 2,
    seed: int = 0,
) -> DynamicDiffusionResult:
    """Run one of the three paired dynamic-IS arms with independent design data."""

    allowed = {
        "base_candidate_fixed",
        "trajectory_replay_aware_fixed",
        "trajectory_replay_aware_optimal",
    }
    if arm not in allowed:
        raise ValueError(f"unknown dynamic dLLM arm {arm!r}")
    for name, value in (
        ("history_rollouts", history_rollouts),
        ("fresh_rollouts", fresh_rollouts),
        ("design_rollouts", design_rollouts),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if history_cost <= 0 or fresh_cost <= 0:
        raise ValueError("rollout costs must be positive")
    stage_lengths = diffusion_decision_stage_lengths(
        prompt_length=len(prompt),
        total_length=config.total_length,
        decision_block_size=config.block_size,
        sampling=sampling,
    )
    seeds = SeedStream(seed)
    generated: TokenSequence = ()
    steps: list[DynamicDiffusionStep] = []
    for stage_index, candidate_length in enumerate(stage_lengths):
        candidate_prefix = prompt + generated
        if arm == "base_candidate_fixed":
            draws = _draw_target_candidates(
                backend=target_backend,
                prefix=candidate_prefix,
                generation_length=candidate_length,
                count=config.candidate_count,
                sampling=sampling,
                seed=seeds.derive("dynamic", arm, "candidates"),
                stage_index=stage_index,
            )
        else:
            draws = draw_defensive_diffusion_candidates(
                target_backend=target_backend,
                auxiliary_backend=auxiliary_backend,
                prefix=candidate_prefix,
                generation_length=candidate_length,
                count=config.candidate_count,
                sampling=sampling,
                auxiliary_probability=auxiliary_probability,
                seed=seeds.derive("dynamic", arm, "candidates"),
                stage_index=stage_index,
            )
        candidates = tuple(draw.sample for draw in draws)
        rollout_length = config.total_length - len(generated) - candidate_length
        evaluation_histories: tuple[DiffusionReplayHistory, ...] | None = None
        design_count = 0
        evaluation_count = 0
        if rollout_length == 0:
            fresh_counts = (1,) * len(candidates)
            allocations = tuple(
                BudgetAllocation(0, 0, 0.0, 0.0, 0.0) for _ in candidates
            )
        elif arm == "base_candidate_fixed":
            fresh_counts = (history_rollouts + fresh_rollouts,) * len(candidates)
            allocations = tuple(
                BudgetAllocation(0, count, 0.0, float(count), float(count))
                for count in fresh_counts
            )
        else:
            evaluation_histories = build_diffusion_replay_history(
                target_backend=target_backend,
                behavior_backend=auxiliary_backend,
                prompt=prompt,
                generated_prefix=generated,
                candidates=candidates,
                rollout_length=rollout_length,
                count_per_candidate=history_rollouts,
                target_sampling=sampling,
                behavior_sampling=sampling,
                reward_batch=reward_batch,
                seed=seeds.derive("dynamic", arm, "evaluation-history", stage_index),
            )
            evaluation_count = sum(len(history.records) for history in evaluation_histories)
            if arm == "trajectory_replay_aware_fixed":
                fresh_counts = (fresh_rollouts,) * len(candidates)
                allocations = tuple(
                    BudgetAllocation(
                        len(history.records),
                        fresh_rollouts,
                        float(len(history.records)),
                        float(fresh_rollouts),
                        len(history.records) * history_cost
                        + fresh_rollouts * fresh_cost,
                    )
                    for history in evaluation_histories
                )
            else:
                design_histories = build_diffusion_replay_history(
                    target_backend=target_backend,
                    behavior_backend=auxiliary_backend,
                    prompt=prompt,
                    generated_prefix=generated,
                    candidates=candidates,
                    rollout_length=rollout_length,
                    count_per_candidate=design_rollouts,
                    target_sampling=sampling,
                    behavior_sampling=sampling,
                    reward_batch=reward_batch,
                    seed=seeds.derive("dynamic", arm, "design-history", stage_index),
                )
                # A second, independent pilot is drawn from the target policy.
                # Reversing target/behavior in the history builder gives both
                # exact log densities needed for the fresh-tail variance.
                fresh_design_histories = build_diffusion_replay_history(
                    target_backend=auxiliary_backend,
                    behavior_backend=target_backend,
                    prompt=prompt,
                    generated_prefix=generated,
                    candidates=candidates,
                    rollout_length=rollout_length,
                    count_per_candidate=design_rollouts,
                    target_sampling=sampling,
                    behavior_sampling=sampling,
                    reward_batch=reward_batch,
                    seed=seeds.derive("dynamic", arm, "design-fresh", stage_index),
                )
                design_count = sum(
                    len(history.records)
                    for design_set in (design_histories, fresh_design_histories)
                    for history in design_set
                )
                statistics = _design_statistics(
                    design_histories,
                    fresh_design_histories,
                    reward_temperature=config.reward_temperature,
                    truncation=truncation,
                    history_cost=history_cost,
                    fresh_cost=fresh_cost,
                )
                groups = tuple(history.rollout_prefix for history in evaluation_histories)
                capacities = tuple(len(history.records) for history in evaluation_histories)
                group_capacities: dict[TokenSequence, int] = {}
                for group, capacity in zip(groups, capacities, strict=True):
                    group_capacities[group] = group_capacities.get(group, 0) + capacity
                rollout_budget = len(candidates) * (
                    history_rollouts * history_cost + fresh_rollouts * fresh_cost
                )
                allocations = allocate_variance_cost_budget(
                    outer_ratios=[exp(draw.outer_log_ratio) for draw in draws],
                    statistics=statistics,
                    history_capacities=capacities,
                    history_groups=groups,
                    group_capacities=group_capacities,
                    rollout_budget=rollout_budget,
                    minimum_fresh=1,
                )
                evaluation_histories = tuple(
                    DiffusionReplayHistory(
                        history.rollout_prefix,
                        history.records[: allocation.history_count],
                    )
                    for history, allocation in zip(
                        evaluation_histories, allocations, strict=True
                    )
                )
                fresh_counts = tuple(allocation.fresh_count for allocation in allocations)

        selection = select_diffusion_candidates_with_replay(
            target_backend=target_backend,
            behavior_backend=(
                auxiliary_backend if evaluation_histories is not None else None
            ),
            prompt=prompt,
            generated_prefix=generated,
            candidates=candidates,
            histories=evaluation_histories,
            rollout_length=rollout_length,
            fresh_count=fresh_counts,
            target_sampling=sampling,
            behavior_sampling=(sampling if evaluation_histories is not None else None),
            reward_batch=reward_batch,
            reward_temperature=config.reward_temperature,
            truncation=truncation,
            seed=seeds.derive("dynamic", arm, "selection", stage_index),
            candidate_log_ratios=[draw.outer_log_ratio for draw in draws],
        )
        steps.append(
            DynamicDiffusionStep(
                generated_length_before=len(generated),
                draws=draws,
                allocations=allocations,
                selection=selection,
                design_rollouts=design_count,
                evaluation_history_rollouts=evaluation_count,
            )
        )
        generated += selection.selected.sample.token_ids
    return DynamicDiffusionResult(prompt, generated, arm, tuple(steps))


__all__ = [
    "DynamicDiffusionArm",
    "DynamicDiffusionDraw",
    "DynamicDiffusionResult",
    "DynamicDiffusionStep",
    "draw_defensive_diffusion_candidates",
    "run_dynamic_diffusion_is",
]
