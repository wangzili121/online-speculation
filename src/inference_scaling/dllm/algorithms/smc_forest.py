"""Block-diffusion SMC with reusable conditional rollout suffixes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.replay import DiffusionReplayRewardBatch
from inference_scaling.dllm.types import DiffusionBackend, DiffusionGenerationRequest
from inference_scaling.shared.config import SMCForestConfig
from inference_scaling.shared.importance import logmeanexp
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.smc import (
    normalize_smc_log_weights,
    partition_resampled_reservoirs,
    systematic_resample,
)
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class DiffusionForestRollout:
    token_ids: TokenSequence
    reward: float


@dataclass(frozen=True, slots=True)
class DiffusionSMCParticle:
    token_ids: TokenSequence
    log_lookahead: float
    reservoir: tuple[DiffusionForestRollout, ...]
    ancestor: int


@dataclass(frozen=True, slots=True)
class DiffusionSMCBranch:
    parent_index: int
    token_ids: TokenSequence
    full_prefix: TokenSequence
    log_lookahead: float
    incremental_log_weight: float
    reservoir: tuple[DiffusionForestRollout, ...]
    reused_rollouts: int
    fresh_rollouts: int


@dataclass(frozen=True, slots=True)
class DiffusionSMCStep:
    generated_length_before: int
    branches: tuple[DiffusionSMCBranch, ...]
    normalized_weights: tuple[float, ...]
    effective_sample_size: float
    selected_branch_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DiffusionSMCResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    final_particles: tuple[DiffusionSMCParticle, ...]
    steps: tuple[DiffusionSMCStep, ...]
    reused_rollouts: int
    fresh_rollouts: int


def _candidate_blocks(
    backend: DiffusionBackend,
    prompt: TokenSequence,
    particles: Sequence[DiffusionSMCParticle],
    *,
    block_length: int,
    branch_factor: int,
    sampling: DiffusionSamplingConfig,
    seeds: SeedStream,
    step_index: int,
    reuse: bool,
) -> tuple[list[tuple[int, TokenSequence]], list[list[DiffusionForestRollout]]]:
    branches: list[tuple[int, TokenSequence]] = []
    inherited: list[list[DiffusionForestRollout]] = []
    requests: list[DiffusionGenerationRequest] = []
    request_positions: list[int] = []
    for parent_index, particle in enumerate(particles):
        reservoirs_by_block: dict[TokenSequence, list[DiffusionForestRollout]] = (
            defaultdict(list)
        )
        if reuse:
            for rollout in particle.reservoir:
                block = rollout.token_ids[:block_length]
                if block:
                    reservoirs_by_block[block].append(rollout)
        reused_blocks = [
            rollout.token_ids[:block_length]
            for rollout in particle.reservoir
            if rollout.token_ids[:block_length]
        ][:branch_factor]
        start = len(branches)
        for branch_index in range(branch_factor):
            position = len(branches)
            if branch_index < len(reused_blocks):
                branches.append((parent_index, reused_blocks[branch_index]))
                inherited.append([])
            else:
                branches.append((parent_index, ()))
                inherited.append([])
                requests.append(
                    DiffusionGenerationRequest(
                        prefix=prompt + particle.token_ids,
                        generation_length=block_length,
                        sampling=sampling,
                        seed=seeds.derive(
                            "dllm-smc",
                            step_index,
                            "candidate",
                            parent_index,
                            branch_index,
                        ),
                        request_id=(
                            f"dllm-smc:{step_index}:parent:{parent_index}:"
                            f"branch:{branch_index}"
                        ),
                    )
                )
                request_positions.append(position)
        positions_by_block: dict[TokenSequence, list[int]] = defaultdict(list)
        for position in range(start, len(branches)):
            block = branches[position][1]
            if block:
                positions_by_block[block].append(position)
        for block, rollouts in reservoirs_by_block.items():
            positions = positions_by_block.get(block, ())
            for offset, rollout in enumerate(rollouts):
                if positions:
                    inherited[positions[offset % len(positions)]].append(
                        DiffusionForestRollout(
                            rollout.token_ids[len(block) :], rollout.reward
                        )
                    )
    samples = backend.sample_batch(requests) if requests else []
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid SMC candidate count")
    for position, sample in zip(request_positions, samples, strict=True):
        parent_index, _ = branches[position]
        branches[position] = (parent_index, sample.token_ids)
    return branches, inherited


def _evaluate_branches(
    backend: DiffusionBackend,
    prompt: TokenSequence,
    particles: Sequence[DiffusionSMCParticle],
    branch_draws: Sequence[tuple[int, TokenSequence]],
    inherited: Sequence[Sequence[DiffusionForestRollout]],
    *,
    remaining_after_block: int,
    config: SMCForestConfig,
    sampling: DiffusionSamplingConfig,
    reward_batch: DiffusionReplayRewardBatch,
    seeds: SeedStream,
    step_index: int,
) -> tuple[DiffusionSMCBranch, ...]:
    reservoirs = [list(values[: config.rollout_count]) for values in inherited]
    requests = []
    request_branches = []
    for branch_index, ((parent_index, block), reservoir) in enumerate(
        zip(branch_draws, reservoirs, strict=True)
    ):
        if remaining_after_block == 0:
            continue
        needed = config.rollout_count - len(reservoir)
        for rollout_index in range(needed):
            requests.append(
                DiffusionGenerationRequest(
                    prefix=prompt + particles[parent_index].token_ids + block,
                    generation_length=remaining_after_block,
                    sampling=sampling,
                    seed=seeds.derive(
                        "dllm-smc",
                        step_index,
                        "lookahead",
                        branch_index,
                        rollout_index,
                    ),
                    request_id=(
                        f"dllm-smc:{step_index}:branch:{branch_index}:"
                        f"lookahead:{rollout_index}"
                    ),
                )
            )
            request_branches.append(branch_index)
    samples = backend.sample_batch(requests) if requests else []
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid SMC rollout count")
    continuations = [
        particles[branch_draws[branch_index][0]].token_ids
        + branch_draws[branch_index][1]
        + sample.token_ids
        for branch_index, sample in zip(request_branches, samples, strict=True)
    ]
    rewards = [float(value) for value in reward_batch(prompt, continuations)]
    if len(rewards) != len(samples) or any(not isfinite(value) for value in rewards):
        raise RuntimeError("reward evaluator returned an invalid SMC rollout batch")
    for branch_index, sample, reward in zip(
        request_branches, samples, rewards, strict=True
    ):
        reservoirs[branch_index].append(
            DiffusionForestRollout(sample.token_ids, reward)
        )
    if remaining_after_block == 0:
        terminal_continuations = [
            particles[parent_index].token_ids + block
            for parent_index, block in branch_draws
        ]
        terminal_rewards = [
            float(value) for value in reward_batch(prompt, terminal_continuations)
        ]
        if len(terminal_rewards) != len(branch_draws):
            raise RuntimeError("reward evaluator returned an invalid SMC terminal batch")
        reservoirs = [
            [DiffusionForestRollout((), reward)] for reward in terminal_rewards
        ]

    branches = []
    for branch_index, ((parent_index, block), reservoir) in enumerate(
        zip(branch_draws, reservoirs, strict=True)
    ):
        if not reservoir:
            raise RuntimeError("every SMC branch requires a lookahead estimate")
        log_lookahead = logmeanexp(
            [rollout.reward / config.reward_temperature for rollout in reservoir]
        )
        reused = min(len(inherited[branch_index]), len(reservoir))
        branches.append(
            DiffusionSMCBranch(
                parent_index=parent_index,
                token_ids=block,
                full_prefix=particles[parent_index].token_ids + block,
                log_lookahead=log_lookahead,
                incremental_log_weight=(
                    log_lookahead - particles[parent_index].log_lookahead
                ),
                reservoir=tuple(reservoir),
                reused_rollouts=reused,
                fresh_rollouts=(
                    0 if remaining_after_block == 0 else len(reservoir) - reused
                ),
            )
        )
    return tuple(branches)


def _split_reservoirs(
    branches: Sequence[DiffusionSMCBranch], selected: Sequence[int]
) -> tuple[DiffusionSMCParticle, ...]:
    buckets = partition_resampled_reservoirs(
        [branch.reservoir for branch in branches], selected
    )
    return tuple(
        DiffusionSMCParticle(
            branches[branch_index].full_prefix,
            branches[branch_index].log_lookahead,
            bucket,
            branch_index,
        )
        for branch_index, bucket in zip(selected, buckets, strict=True)
    )


def run_diffusion_smc_rollout_forest(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: SMCForestConfig,
    sampling: DiffusionSamplingConfig,
    reward_batch: DiffusionReplayRewardBatch,
    seed: int = 0,
) -> DiffusionSMCResult:
    """Run the same telescoping lookahead SMC construction over diffusion blocks."""

    sampling.validate_generation_length(config.total_length, prefix_length=len(prompt))
    sampling.validate_generation_length(config.block_size)
    seeds = SeedStream(seed)
    particles = tuple(
        DiffusionSMCParticle((), 0.0, (), -1)
        for _ in range(config.particle_count)
    )
    steps = []
    total_reused = 0
    total_fresh = 0
    while len(particles[0].token_ids) < config.total_length:
        generated_before = len(particles[0].token_ids)
        block_length = min(config.block_size, config.total_length - generated_before)
        draws, inherited = _candidate_blocks(
            backend,
            prompt,
            particles,
            block_length=block_length,
            branch_factor=config.branch_factor,
            sampling=sampling,
            seeds=seeds,
            step_index=len(steps),
            reuse=config.reuse_rollout_forest,
        )
        branches = _evaluate_branches(
            backend,
            prompt,
            particles,
            draws,
            inherited,
            remaining_after_block=config.total_length
            - generated_before
            - block_length,
            config=config,
            sampling=sampling,
            reward_batch=reward_batch,
            seeds=seeds,
            step_index=len(steps),
        )
        probabilities, effective_sample_size = normalize_smc_log_weights(
            [branch.incremental_log_weight for branch in branches]
        )
        selected = systematic_resample(
            probabilities,
            config.particle_count,
            seeds.generator("dllm-smc", len(steps), "resample"),
        )
        particles = _split_reservoirs(branches, selected)
        total_reused += sum(branch.reused_rollouts for branch in branches)
        total_fresh += sum(branch.fresh_rollouts for branch in branches)
        steps.append(
            DiffusionSMCStep(
                generated_length_before=generated_before,
                branches=branches,
                normalized_weights=tuple(float(value) for value in probabilities),
                effective_sample_size=effective_sample_size,
                selected_branch_indices=selected,
            )
        )
    output_index = int(seeds.generator("dllm-smc", "final").integers(len(particles)))
    return DiffusionSMCResult(
        prompt,
        particles[output_index].token_ids,
        particles,
        tuple(steps),
        total_reused,
        total_fresh,
    )


__all__ = [
    "DiffusionForestRollout",
    "DiffusionSMCBranch",
    "DiffusionSMCParticle",
    "DiffusionSMCResult",
    "DiffusionSMCStep",
    "run_diffusion_smc_rollout_forest",
]
