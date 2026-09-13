"""Pilot/evaluation-separated adaptive rollout allocation for diffusion IS."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Sequence

import numpy as np

from inference_scaling.dllm.config import (
    DiffusionISConfig,
    DiffusionSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.replay import (
    DiffusionReplayRewardBatch,
    DiffusionReplaySelection,
    select_diffusion_candidates_with_replay,
)
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
)
from inference_scaling.shared.budget import (
    BudgetAllocation,
    allocate_fresh_rollout_budget,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class ProgressiveDiffusionISStep:
    generated_length_before: int
    candidates: tuple[DiffusionSample, ...]
    allocations: tuple[BudgetAllocation, ...]
    pilot_rollouts: int
    selection: DiffusionReplaySelection


@dataclass(frozen=True, slots=True)
class ProgressiveDiffusionISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ProgressiveDiffusionISStep, ...]


def _sample_candidates(
    backend: DiffusionBackend,
    *,
    prefix: TokenSequence,
    length: int,
    count: int,
    sampling: DiffusionSamplingConfig,
    seeds: SeedStream,
    step_index: int,
) -> tuple[DiffusionSample, ...]:
    samples = backend.sample_batch(
        [
            DiffusionGenerationRequest(
                prefix=prefix,
                generation_length=length,
                sampling=sampling,
                seed=seeds.derive(
                    "dllm-progressive", step_index, "candidate", candidate_index
                ),
                request_id=(
                    f"dllm-progressive:{step_index}:candidate:{candidate_index}"
                ),
            )
            for candidate_index in range(count)
        ]
    )
    if len(samples) != count:
        raise RuntimeError("backend returned an invalid progressive candidate count")
    return tuple(samples)


def _pilot_standard_deviations(
    backend: DiffusionBackend,
    *,
    prompt: TokenSequence,
    generated: TokenSequence,
    candidates: Sequence[DiffusionSample],
    rollout_length: int,
    count: int,
    sampling: DiffusionSamplingConfig,
    reward_batch: DiffusionReplayRewardBatch,
    reward_temperature: float,
    seeds: SeedStream,
    step_index: int,
) -> tuple[float, ...]:
    requests = []
    owners = []
    for candidate_index, candidate in enumerate(candidates):
        for rollout_index in range(count):
            requests.append(
                DiffusionGenerationRequest(
                    prefix=prompt + generated + candidate.token_ids,
                    generation_length=rollout_length,
                    sampling=sampling,
                    seed=seeds.derive(
                        "dllm-progressive",
                        step_index,
                        "pilot",
                        candidate_index,
                        rollout_index,
                    ),
                    request_id=(
                        f"dllm-progressive:{step_index}:candidate:{candidate_index}:"
                        f"pilot:{rollout_index}"
                    ),
                )
            )
            owners.append(candidate_index)
    samples = backend.sample_batch(requests)
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid progressive pilot count")
    continuations = [
        generated + candidates[owner].token_ids + sample.token_ids
        for owner, sample in zip(owners, samples, strict=True)
    ]
    rewards = [float(value) for value in reward_batch(prompt, continuations)]
    if len(rewards) != len(samples) or any(not isfinite(value) for value in rewards):
        raise RuntimeError("reward evaluator returned an invalid progressive pilot count")
    grouped: list[list[float]] = [[] for _ in candidates]
    for owner, reward in zip(owners, rewards, strict=True):
        grouped[owner].append(reward / reward_temperature)
    # A common shift prevents overflow without changing the relative standard
    # deviations used by the cross-candidate allocation problem.
    shift = max((value for values in grouped for value in values), default=0.0)
    deviations = []
    for values in grouped:
        contributions = [exp(value - shift) for value in values]
        deviation = (
            float(np.std(np.asarray(contributions, dtype=np.float64), ddof=1))
            if len(contributions) >= 2
            else 1.0
        )
        deviations.append(deviation if isfinite(deviation) and deviation >= 0 else 1.0)
    return tuple(deviations)


def run_progressive_diffusion_is(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionISConfig,
    sampling: DiffusionSamplingConfig,
    reward_batch: DiffusionReplayRewardBatch,
    pilot_rollouts_per_candidate: int,
    evaluation_rollout_budget: int,
    seed: int = 0,
) -> ProgressiveDiffusionISResult:
    """Freeze an adaptive fresh-rollout budget using independent pilot samples."""

    if pilot_rollouts_per_candidate <= 0:
        raise ValueError("pilot_rollouts_per_candidate must be positive")
    if evaluation_rollout_budget < config.candidate_count:
        raise ValueError("evaluation budget must cover one rollout per candidate")
    stage_lengths = diffusion_decision_stage_lengths(
        prompt_length=len(prompt),
        total_length=config.total_length,
        decision_block_size=config.block_size,
        sampling=sampling,
    )
    seeds = SeedStream(seed)
    generated: TokenSequence = ()
    steps = []
    for step_index, candidate_length in enumerate(stage_lengths):
        candidates = _sample_candidates(
            backend,
            prefix=prompt + generated,
            length=candidate_length,
            count=config.candidate_count,
            sampling=sampling,
            seeds=seeds,
            step_index=step_index,
        )
        rollout_length = config.total_length - len(generated) - candidate_length
        if rollout_length == 0:
            allocations = tuple(
                BudgetAllocation(0, 0, 0.0, 0.0, 0.0) for _ in candidates
            )
            fresh_counts = (1,) * len(candidates)
            pilot_count = 0
        else:
            deviations = _pilot_standard_deviations(
                backend,
                prompt=prompt,
                generated=generated,
                candidates=candidates,
                rollout_length=rollout_length,
                count=pilot_rollouts_per_candidate,
                sampling=sampling,
                reward_batch=reward_batch,
                reward_temperature=config.reward_temperature,
                seeds=seeds,
                step_index=step_index,
            )
            pilot_count = len(candidates) * pilot_rollouts_per_candidate
            allocations = allocate_fresh_rollout_budget(
                outer_ratios=(1.0,) * len(candidates),
                standard_deviations=deviations,
                costs=(1.0,) * len(candidates),
                rollout_budget=float(evaluation_rollout_budget),
                minimum_fresh=1,
            )
            fresh_counts = tuple(item.fresh_count for item in allocations)
        selection = select_diffusion_candidates_with_replay(
            target_backend=backend,
            behavior_backend=None,
            prompt=prompt,
            generated_prefix=generated,
            candidates=candidates,
            histories=None,
            rollout_length=rollout_length,
            fresh_count=fresh_counts,
            target_sampling=sampling,
            behavior_sampling=None,
            reward_batch=reward_batch,
            reward_temperature=config.reward_temperature,
            truncation=1.0,
            seed=seeds.derive("dllm-progressive", step_index, "selection"),
        )
        steps.append(
            ProgressiveDiffusionISStep(
                generated_length_before=len(generated),
                candidates=candidates,
                allocations=allocations,
                pilot_rollouts=pilot_count,
                selection=selection,
            )
        )
        generated += selection.selected.sample.token_ids
    return ProgressiveDiffusionISResult(prompt, generated, tuple(steps))


__all__ = [
    "ProgressiveDiffusionISResult",
    "ProgressiveDiffusionISStep",
    "run_progressive_diffusion_is",
]
