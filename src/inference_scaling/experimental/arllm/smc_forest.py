"""Sequential Monte Carlo with a reusable rollout forest.

At each block, particles are extended by base-model candidate blocks.  A
positive rollout estimate of

    h(prefix) = E_base[exp(reward / temperature) | prefix]

twists the particle weights.  The incremental weight is the child's estimate
divided by its parent's estimate, so the factors telescope to the final target.
Suffixes already sampled for a parent's estimate are valid base-conditional
rollouts after a matching child block and can be carried forward.  Duplicated
particles split, rather than copy, their finite reservoir; missing entries are
topped up with fresh independent rollouts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from inference_scaling.arllm.acceleration import StreamingRewardEvaluator
from inference_scaling.arllm.algorithms.conditional_is import (
    RewardBatchFunction,
    RewardFunction,
    _validate_base_sampling,
)
from inference_scaling.arllm.config import SMCForestConfig, SamplingConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.importance import logmeanexp
from inference_scaling.shared.smc import (
    normalize_smc_log_weights,
    partition_resampled_reservoirs,
    systematic_resample,
)
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    TokenSequence,
)


@dataclass(frozen=True, slots=True)
class ForestRollout:
    """A base-conditional suffix and its already-computed terminal reward."""

    token_ids: TokenSequence
    reward: float


@dataclass(frozen=True, slots=True)
class SMCParticle:
    token_ids: TokenSequence
    log_lookahead: float
    reservoir: tuple[ForestRollout, ...]
    ancestor: int


@dataclass(frozen=True, slots=True)
class SMCBranch:
    parent_index: int
    token_ids: TokenSequence
    full_prefix: TokenSequence
    log_lookahead: float
    incremental_log_weight: float
    reservoir: tuple[ForestRollout, ...]
    reused_rollouts: int
    fresh_rollouts: int


@dataclass(frozen=True, slots=True)
class SMCForestStep:
    generated_length_before: int
    branches: tuple[SMCBranch, ...]
    normalized_weights: tuple[float, ...]
    effective_sample_size: float
    selected_branch_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SMCForestResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    final_particles: tuple[SMCParticle, ...]
    steps: tuple[SMCForestStep, ...]
    reused_rollouts: int
    fresh_rollouts: int


def _block_from_suffix(suffix: TokenSequence, block_length: int) -> TokenSequence:
    return suffix[:block_length]


def _candidate_blocks(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    particles: Sequence[SMCParticle],
    *,
    block_length: int,
    branch_factor: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    reuse: bool,
) -> tuple[list[tuple[int, TokenSequence]], list[list[ForestRollout]]]:
    branches: list[tuple[int, TokenSequence]] = []
    inherited: list[list[ForestRollout]] = []
    requests: list[GenerationRequest] = []
    request_positions: list[int] = []
    for parent_index, particle in enumerate(particles):
        eos = sampling.eos_token_id
        if eos is not None and eos in particle.token_ids:
            for _ in range(branch_factor):
                branches.append((parent_index, ()))
                inherited.append(list(particle.reservoir))
            continue
        reservoirs_by_block: dict[TokenSequence, list[ForestRollout]] = defaultdict(list)
        if reuse:
            for rollout in particle.reservoir:
                block = _block_from_suffix(rollout.token_ids, block_length)
                if block:
                    reservoirs_by_block[block].append(rollout)
        reused_blocks = [
            _block_from_suffix(rollout.token_ids, block_length)
            for rollout in particle.reservoir
            if _block_from_suffix(rollout.token_ids, block_length)
        ][:branch_factor]
        for branch_index in range(branch_factor):
            position = len(branches)
            if branch_index < len(reused_blocks):
                block = reused_blocks[branch_index]
                branches.append((parent_index, block))
                inherited.append([])
                continue
            branches.append((parent_index, ()))
            inherited.append([])
            requests.append(
                GenerationRequest(
                    prefix=prompt + particle.token_ids,
                    max_new_tokens=block_length,
                    sampling=sampling,
                    seed=seeds.derive(
                        "smc_forest", step_index, "candidate", parent_index, branch_index
                    ),
                    request_id=(
                        f"smc-forest:step:{step_index}:parent:{parent_index}:"
                        f"branch:{branch_index}"
                    ),
                )
            )
            request_positions.append(position)
        # Partition every compatible suffix across duplicate child blocks.  This
        # prevents one finite rollout from being counted twice in the same stage.
        positions_by_block: dict[TokenSequence, list[int]] = defaultdict(list)
        start = len(branches) - branch_factor
        for position in range(start, len(branches)):
            block = branches[position][1]
            if block:
                positions_by_block[block].append(position)
        for block, rollouts in reservoirs_by_block.items():
            positions = positions_by_block.get(block, [])
            if not positions:
                continue
            for offset, rollout in enumerate(rollouts):
                position = positions[offset % len(positions)]
                inherited[position].append(
                    ForestRollout(rollout.token_ids[len(block) :], rollout.reward)
                )
    if requests:
        samples = backend.sample_batch(requests)
        if len(samples) != len(requests):
            raise RuntimeError("base backend returned an invalid candidate count")
        for position, sample in zip(request_positions, samples, strict=True):
            if not sample.token_ids:
                raise RuntimeError("SMC candidate blocks must be nonempty")
            parent_index, _ = branches[position]
            branches[position] = (parent_index, sample.token_ids)
    return branches, inherited


def _evaluate_branches(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    particles: Sequence[SMCParticle],
    branch_draws: Sequence[tuple[int, TokenSequence]],
    inherited: Sequence[Sequence[ForestRollout]],
    *,
    remaining_after_block: int,
    config: SMCForestConfig,
    sampling: SamplingConfig,
    reward: RewardFunction | None,
    reward_batch: RewardBatchFunction | None,
    seeds: SeedStream,
    step_index: int,
    streaming_evaluator: StreamingRewardEvaluator | None,
) -> tuple[SMCBranch, ...]:
    reservoirs = [list(values[: config.rollout_count]) for values in inherited]
    requests: list[GenerationRequest] = []
    request_branches: list[int] = []
    reward_inputs: list[tuple[TokenSequence, TokenSequence]] = []
    eos = sampling.eos_token_id
    terminal: set[int] = set()
    for branch_index, ((parent_index, block), reservoir) in enumerate(
        zip(branch_draws, reservoirs, strict=True)
    ):
        full_prefix = particles[parent_index].token_ids + block
        if remaining_after_block == 0 or (
            eos is not None
            and (eos in particles[parent_index].token_ids or eos in block)
        ):
            terminal.add(branch_index)
            continue
        needed = config.rollout_count - len(reservoir)
        for rollout_index in range(needed):
            requests.append(
                GenerationRequest(
                    prefix=prompt + full_prefix,
                    max_new_tokens=remaining_after_block,
                    sampling=sampling,
                    seed=seeds.derive(
                        "smc_forest",
                        step_index,
                        "lookahead",
                        branch_index,
                        rollout_index,
                    ),
                    request_id=(
                        f"smc-forest:step:{step_index}:branch:{branch_index}:"
                        f"rollout:{rollout_index}"
                    ),
                )
            )
            request_branches.append(branch_index)
            reward_inputs.append((prompt, full_prefix))

    if requests and streaming_evaluator is not None:
        if reward is None or reward_batch is not None:
            raise ValueError("streaming SMC rewards require a scalar reward callback")
        samples, rewards, _ = streaming_evaluator.sample_and_score(
            backend, requests, reward_inputs, reward
        )
    else:
        samples = backend.sample_batch(requests) if requests else []
        generated = [
            prefix + sample.token_ids
            for (_, prefix), sample in zip(reward_inputs, samples, strict=True)
        ]
        if reward_batch is not None:
            rewards = tuple(float(value) for value in reward_batch(prompt, generated))
        else:
            if reward is None:
                raise ValueError("a reward callback is required")
            rewards = tuple(float(reward(prompt, value)) for value in generated)
    if len(samples) != len(requests) or len(rewards) != len(requests):
        raise RuntimeError("SMC rollout or reward count is invalid")
    if any(not isfinite(value) for value in rewards):
        raise ValueError("reward must be finite")
    for branch_index, sample, value in zip(
        request_branches, samples, rewards, strict=True
    ):
        reservoirs[branch_index].append(ForestRollout(sample.token_ids, float(value)))

    for branch_index in terminal:
        parent_index, block = branch_draws[branch_index]
        generated = particles[parent_index].token_ids + block
        if reward_batch is not None:
            values = tuple(float(value) for value in reward_batch(prompt, [generated]))
            if len(values) != 1:
                raise ValueError("reward_batch returned an invalid terminal count")
            value = values[0]
        else:
            if reward is None:
                raise ValueError("a reward callback is required")
            value = float(reward(prompt, generated))
        if not isfinite(value):
            raise ValueError("reward must be finite")
        reservoirs[branch_index] = [ForestRollout((), value)]

    branches: list[SMCBranch] = []
    for branch_index, ((parent_index, block), reservoir) in enumerate(
        zip(branch_draws, reservoirs, strict=True)
    ):
        if not reservoir:
            raise RuntimeError("each SMC branch needs a positive lookahead estimate")
        log_lookahead = logmeanexp(
            [item.reward / config.reward_temperature for item in reservoir]
        )
        inherited_count = min(len(inherited[branch_index]), len(reservoir))
        branches.append(
            SMCBranch(
                parent_index=parent_index,
                token_ids=block,
                full_prefix=particles[parent_index].token_ids + block,
                log_lookahead=log_lookahead,
                incremental_log_weight=(
                    log_lookahead - particles[parent_index].log_lookahead
                ),
                reservoir=tuple(reservoir),
                reused_rollouts=inherited_count,
                fresh_rollouts=(0 if branch_index in terminal else len(reservoir) - inherited_count),
            )
        )
    return tuple(branches)


def _split_selected_reservoirs(
    branches: Sequence[SMCBranch],
    selected: Sequence[int],
) -> tuple[SMCParticle, ...]:
    buckets = partition_resampled_reservoirs(
        [branch.reservoir for branch in branches], selected
    )
    return tuple(
        SMCParticle(
            token_ids=branches[branch_index].full_prefix,
            log_lookahead=branches[branch_index].log_lookahead,
            reservoir=bucket,
            ancestor=branch_index,
        )
        for branch_index, bucket in zip(selected, buckets, strict=True)
    )


def run_smc_rollout_forest(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: SMCForestConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
    streaming_rewards: bool = True,
) -> SMCForestResult:
    """Run the backend-independent rollout forest on Transformers or vLLM."""

    sampling = base_sampling or SamplingConfig()
    _validate_base_sampling(sampling)
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    particles = tuple(
        SMCParticle((), 0.0, (), -1) for _ in range(config.particle_count)
    )
    steps: list[SMCForestStep] = []
    evaluator = (
        StreamingRewardEvaluator(workers=config.reward_workers)
        if streaming_rewards and reward is not None and reward_batch is None
        else None
    )
    total_reused = 0
    total_fresh = 0
    try:
        while (
            max(len(particle.token_ids) for particle in particles) < config.total_length
            and not (
                sampling.eos_token_id is not None
                and all(
                    sampling.eos_token_id in particle.token_ids
                    for particle in particles
                )
            )
        ):
            generated_before = max(len(particle.token_ids) for particle in particles)
            block_length = min(config.block_size, config.total_length - generated_before)
            draws, inherited = _candidate_blocks(
                base_backend,
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
                base_backend,
                prompt,
                particles,
                draws,
                inherited,
                remaining_after_block=max(
                    0, config.total_length - generated_before - block_length
                ),
                config=config,
                sampling=sampling,
                reward=reward,
                reward_batch=reward_batch,
                seeds=seeds,
                step_index=len(steps),
                streaming_evaluator=evaluator,
            )
            probabilities, ess = normalize_smc_log_weights(
                [branch.incremental_log_weight for branch in branches]
            )
            selected = systematic_resample(
                probabilities,
                config.particle_count,
                seeds.generator("smc_forest", len(steps), "resample"),
            )
            particles = _split_selected_reservoirs(branches, selected)
            total_reused += sum(branch.reused_rollouts for branch in branches)
            total_fresh += sum(branch.fresh_rollouts for branch in branches)
            steps.append(
                SMCForestStep(
                    generated_length_before=generated_before,
                    branches=branches,
                    normalized_weights=tuple(float(value) for value in probabilities),
                    effective_sample_size=ess,
                    selected_branch_indices=selected,
                )
            )
            eos = sampling.eos_token_id
            if eos is not None and all(eos in particle.token_ids for particle in particles):
                break
    finally:
        if evaluator is not None:
            evaluator.close()
    selected_particle = int(
        seeds.generator("smc_forest", "final").integers(len(particles))
    )
    output = particles[selected_particle].token_ids
    eos = sampling.eos_token_id
    if eos is not None and eos in output:
        output = output[: output.index(eos) + 1]
    return SMCForestResult(
        prompt=prompt,
        token_ids=output,
        final_particles=particles,
        steps=tuple(steps),
        reused_rollouts=total_reused,
        fresh_rollouts=total_fresh,
    )
