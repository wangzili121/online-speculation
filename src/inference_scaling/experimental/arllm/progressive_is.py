"""Progressive conditional IS with an independent pilot/evaluation split.

Pilot rollouts estimate variance and relative compute cost, then the allocation is
frozen.  Only newly drawn evaluation rollouts enter the conditional-weight
estimator.  Pilot trajectories remain useful as verified draft-cache material,
but never become a second statistical observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite

import numpy as np

from inference_scaling.arllm.acceleration import (
    StreamingRewardEvaluator,
    StreamingRewardSnapshot,
)
from inference_scaling.arllm.algorithms.conditional_is import (
    RewardBatchFunction,
    RewardFunction,
    RolloutEvaluation,
    _sample_candidates,
    _score_samples,
    _validate_base_sampling,
    _validate_rollout_sampling,
)
from inference_scaling.arllm.config import ProgressiveISConfig, SamplingConfig
from inference_scaling.shared.budget import allocate_fresh_rollout_budget
from inference_scaling.shared.importance import logmeanexp
from inference_scaling.shared.rng import SeedStream
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    SequenceSample,
    TokenSequence,
)


@dataclass(frozen=True, slots=True)
class PilotSummary:
    rollout_count: int
    normalized_weight_std: float
    normalized_cost: float
    streaming: StreamingRewardSnapshot | None


@dataclass(frozen=True, slots=True)
class ProgressiveCandidate:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    pilot: PilotSummary
    evaluation_rollouts: tuple[RolloutEvaluation, ...]
    evaluation_count: int
    log_weight: float


@dataclass(frozen=True, slots=True)
class ProgressiveISStep:
    generated_length_before: int
    candidates: tuple[ProgressiveCandidate, ...]
    selected_index: int
    frozen_evaluation_cost: float

    @property
    def selected(self) -> ProgressiveCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class ProgressiveISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ProgressiveISStep, ...]


def _parameter_count(backend: AutoregressiveBackend) -> float:
    value = getattr(backend, "parameter_count", 1)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 1.0
    return converted if isfinite(converted) and converted > 0 else 1.0


def _draw_rollouts(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[SequenceSample],
    counts: Sequence[int],
    rollout_length: int,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward_temperature: float,
    importance_log_ratio_clip: float | None,
    reward: RewardFunction | None,
    reward_batch: RewardBatchFunction | None,
    seeds: SeedStream,
    step_index: int,
    phase: str,
    streaming_evaluator: StreamingRewardEvaluator | None,
    run_ahead_rollouts_per_candidate: int,
) -> tuple[tuple[RolloutEvaluation, ...], ...] | tuple:
    if len(counts) != len(candidates):
        raise ValueError("rollout counts must match candidates")
    requests: list[GenerationRequest] = []
    request_candidates: list[int] = []
    rollout_prefixes: list[TokenSequence] = []
    reward_inputs: list[tuple[TokenSequence, TokenSequence]] = []
    terminal: set[int] = set()
    eos = rollout_sampling.eos_token_id
    for candidate_index, (candidate, count) in enumerate(
        zip(candidates, counts, strict=True)
    ):
        if count < 0:
            raise ValueError("rollout counts must be non-negative")
        generated = generated_prefix + candidate.token_ids
        if rollout_length == 0 or (eos is not None and candidate.token_ids[-1] == eos):
            terminal.add(candidate_index)
            continue
        prefix = prompt + generated
        for rollout_index in range(count):
            requests.append(
                GenerationRequest(
                    prefix=prefix,
                    max_new_tokens=rollout_length,
                    sampling=rollout_sampling,
                    seed=seeds.derive(
                        "progressive_is",
                        step_index,
                        phase,
                        candidate_index,
                        rollout_index,
                    ),
                    request_id=(
                        f"progressive-is:{phase}:step:{step_index}:"
                        f"candidate:{candidate_index}:rollout:{rollout_index}"
                    ),
                )
            )
            request_candidates.append(candidate_index)
            rollout_prefixes.append(prefix)
            reward_inputs.append((prompt, generated))

    stream_snapshot = None
    if requests and streaming_evaluator is not None:
        if reward is None or reward_batch is not None:
            raise ValueError("streaming rewards require the scalar reward callback")
        run_ahead_requests: list[GenerationRequest] = []
        submit_run_ahead = getattr(base_backend, "submit_run_ahead", None)
        if callable(submit_run_ahead) and run_ahead_rollouts_per_candidate:
            for candidate_index, candidate in enumerate(candidates):
                generated = generated_prefix + candidate.token_ids
                if rollout_length == 0 or (
                    eos is not None and candidate.token_ids[-1] == eos
                ):
                    continue
                for rollout_index in range(run_ahead_rollouts_per_candidate):
                    run_ahead_requests.append(
                        GenerationRequest(
                            prefix=prompt + generated,
                            max_new_tokens=rollout_length,
                            sampling=base_sampling,
                            seed=seeds.derive(
                                "progressive_is",
                                step_index,
                                phase,
                                "run_ahead",
                                candidate_index,
                                rollout_index,
                            ),
                            request_id=(
                                f"progressive-is:{phase}:run-ahead:step:{step_index}:"
                                f"candidate:{candidate_index}:rollout:{rollout_index}"
                            ),
                        )
                    )
        samples, rewards, stream_snapshot = streaming_evaluator.sample_and_score(
            rollout_backend,
            requests,
            reward_inputs,
            reward,
            on_generation_complete=(
                (lambda _samples: submit_run_ahead(run_ahead_requests))
                if run_ahead_requests
                else None
            ),
        )
    else:
        samples = rollout_backend.sample_batch(requests) if requests else []
        generated_sequences = [
            generated + sample.token_ids
            for (_, generated), sample in zip(reward_inputs, samples, strict=True)
        ]
        if reward_batch is not None:
            rewards = tuple(
                float(value) for value in reward_batch(prompt, generated_sequences)
            )
        else:
            if reward is None:
                raise ValueError("a reward callback is required")
            rewards = tuple(float(reward(prompt, value)) for value in generated_sequences)
    if len(samples) != len(requests) or len(rewards) != len(requests):
        raise RuntimeError("rollout backend or reward callback returned an invalid count")
    if any(not isfinite(value) for value in rewards):
        raise ValueError("reward must be finite")
    if rollout_backend is not base_backend:
        observe = getattr(base_backend, "observe_draft_samples", None)
        if callable(observe):
            observe(samples)

    on_policy = (
        rollout_backend.model_id == base_backend.model_id
        and rollout_sampling == base_sampling
    )
    base_totals = (
        [sample.logprob for sample in samples]
        if on_policy
        else _score_samples(base_backend, rollout_prefixes, samples, base_sampling)
    )
    grouped: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    for candidate_index in terminal:
        generated = generated_prefix + candidates[candidate_index].token_ids
        if reward_batch is not None:
            terminal_reward = tuple(float(value) for value in reward_batch(prompt, [generated]))
            if len(terminal_reward) != 1:
                raise ValueError("reward_batch returned an invalid terminal count")
            value = terminal_reward[0]
        else:
            if reward is None:
                raise ValueError("a reward callback is required")
            value = float(reward(prompt, generated))
        grouped[candidate_index].append(
            RolloutEvaluation(
                token_ids=(),
                reward=value,
                base_logprob=0.0,
                proposal_logprob=0.0,
                raw_log_importance_ratio=0.0,
                applied_log_importance_ratio=0.0,
                log_weight=value / reward_temperature,
                proposal_model_id=rollout_backend.model_id,
                proposal_policy_id=rollout_sampling.policy_id,
            )
        )
    for candidate_index, sample, base_logprob, reward_value in zip(
        request_candidates, samples, base_totals, rewards, strict=True
    ):
        raw_ratio = float(base_logprob - sample.logprob)
        applied_ratio = raw_ratio
        if importance_log_ratio_clip is not None:
            applied_ratio = max(
                -importance_log_ratio_clip,
                min(importance_log_ratio_clip, raw_ratio),
            )
        grouped[candidate_index].append(
            RolloutEvaluation(
                token_ids=sample.token_ids,
                reward=float(reward_value),
                base_logprob=float(base_logprob),
                proposal_logprob=sample.logprob,
                raw_log_importance_ratio=raw_ratio,
                applied_log_importance_ratio=applied_ratio,
                log_weight=float(reward_value) / reward_temperature + applied_ratio,
                proposal_model_id=sample.model_id,
                proposal_policy_id=sample.policy_id,
            )
        )
    # One snapshot describes the whole physical batch.  It is attached to each
    # candidate by the caller without treating it as an independent measurement.
    return tuple(tuple(values) for values in grouped), stream_snapshot


def _pilot_statistics(
    pilot: Sequence[Sequence[RolloutEvaluation]],
    *,
    rollout_backend: AutoregressiveBackend,
    base_backend: AutoregressiveBackend,
    on_policy: bool,
) -> tuple[list[float], list[float]]:
    finite_logs = [item.log_weight for group in pilot for item in group]
    maximum = max(finite_logs) if finite_logs else 0.0
    deviations: list[float] = []
    raw_costs: list[float] = []
    proposal_parameters = _parameter_count(rollout_backend)
    base_parameters = _parameter_count(base_backend)
    for group in pilot:
        values = np.asarray(
            [exp(item.log_weight - maximum) for item in group], dtype=np.float64
        )
        deviations.append(
            float(np.std(values, ddof=1))
            if len(values) >= 2
            else max(float(values[0]) if len(values) else 0.0, 1e-12)
        )
        average_tokens = max(
            1.0,
            float(np.mean([max(1, len(item.token_ids)) for item in group]))
            if group
            else 1.0,
        )
        raw_costs.append(
            average_tokens
            * (proposal_parameters + (0.0 if on_policy else base_parameters))
        )
    minimum = min(raw_costs) if raw_costs else 1.0
    return deviations, [max(1.0, value / minimum) for value in raw_costs]


def progressive_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: ProgressiveISConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
    streaming_evaluator: StreamingRewardEvaluator | None = None,
) -> ProgressiveISStep:
    _validate_base_sampling(base_sampling)
    _validate_rollout_sampling(rollout_sampling)
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    remaining = config.total_length - len(generated_prefix)
    if remaining <= 0:
        raise ValueError("generated prefix has already reached total_length")
    block_length = min(config.block_size, remaining)
    candidates = _sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.candidate_count,
        block_length,
        base_sampling,
        seeds,
        step_index,
    )
    rollout_length = max(0, remaining - block_length)
    eos = rollout_sampling.eos_token_id
    terminal = [
        rollout_length == 0 or (eos is not None and candidate.token_ids[-1] == eos)
        for candidate in candidates
    ]
    pilot_counts = [
        0 if is_terminal else config.pilot_rollouts_per_candidate
        for is_terminal in terminal
    ]
    pilot, stream_snapshot = _draw_rollouts(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        generated_prefix=generated_prefix,
        candidates=candidates,
        counts=pilot_counts,
        rollout_length=rollout_length,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward_temperature=config.reward_temperature,
        importance_log_ratio_clip=config.importance_log_ratio_clip,
        reward=reward,
        reward_batch=reward_batch,
        seeds=seeds,
        step_index=step_index,
        phase="pilot",
        streaming_evaluator=streaming_evaluator,
        run_ahead_rollouts_per_candidate=config.run_ahead_rollouts_per_candidate,
    )
    on_policy = (
        rollout_backend.model_id == base_backend.model_id
        and rollout_sampling == base_sampling
    )
    deviations, costs = _pilot_statistics(
        pilot,
        rollout_backend=rollout_backend,
        base_backend=base_backend,
        on_policy=on_policy,
    )
    active_indices = [index for index, value in enumerate(terminal) if not value]
    evaluation_counts = [0] * len(candidates)
    frozen_cost = 0.0
    if active_indices:
        minimum_cost = sum(
            config.minimum_evaluation_per_candidate * costs[index]
            for index in active_indices
        )
        evaluation_cost_budget = config.evaluation_cost_budget
        if config.evaluation_reference_rollouts_per_candidate is not None:
            # Convert a familiar per-candidate count into a cost budget using
            # only the already-frozen pilot costs.  Allocation may still move
            # samples between candidates, but its total estimated cost matches
            # that fixed-count reference workload and always covers the minimum.
            reference_count = (
                len(active_indices)
                * config.evaluation_reference_rollouts_per_candidate
            )
            mean_cost = sum(costs[index] for index in active_indices) / len(
                active_indices
            )
            evaluation_cost_budget = max(
                minimum_cost, reference_count * mean_cost
            )
        if evaluation_cost_budget + 1e-12 < minimum_cost:
            raise ValueError(
                "evaluation_cost_budget cannot cover the minimum independent "
                "evaluation rollouts"
            )
        allocations = allocate_fresh_rollout_budget(
            outer_ratios=[1.0] * len(active_indices),
            standard_deviations=[
                max(deviations[index], 1e-12) for index in active_indices
            ],
            costs=[costs[index] for index in active_indices],
            rollout_budget=evaluation_cost_budget,
            minimum_fresh=config.minimum_evaluation_per_candidate,
        )
        for index, allocation in zip(active_indices, allocations, strict=True):
            evaluation_counts[index] = allocation.fresh_count
            frozen_cost += allocation.estimated_cost
    evaluation, _ = _draw_rollouts(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        generated_prefix=generated_prefix,
        candidates=candidates,
        counts=evaluation_counts,
        rollout_length=rollout_length,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward_temperature=config.reward_temperature,
        importance_log_ratio_clip=config.importance_log_ratio_clip,
        reward=reward,
        reward_batch=reward_batch,
        seeds=seeds,
        step_index=step_index,
        phase="evaluation",
        streaming_evaluator=streaming_evaluator,
        run_ahead_rollouts_per_candidate=config.run_ahead_rollouts_per_candidate,
    )
    progressive: list[ProgressiveCandidate] = []
    for index, candidate in enumerate(candidates):
        evaluations = evaluation[index]
        if terminal[index]:
            # The terminal weight is deterministic; the pilot call already evaluated
            # it once, while no generated rollout is counted against the budget.
            evaluations = pilot[index]
        if not evaluations:
            raise RuntimeError("each candidate needs an independent weight estimate")
        progressive.append(
            ProgressiveCandidate(
                token_ids=candidate.token_ids,
                base_token_logprobs=candidate.token_logprobs,
                pilot=PilotSummary(
                    rollout_count=pilot_counts[index],
                    normalized_weight_std=deviations[index],
                    normalized_cost=costs[index],
                    streaming=stream_snapshot,
                ),
                evaluation_rollouts=tuple(evaluations),
                evaluation_count=(1 if terminal[index] else evaluation_counts[index]),
                log_weight=logmeanexp([item.log_weight for item in evaluations]),
            )
        )
    log_weights = np.asarray([item.log_weight for item in progressive], dtype=np.float64)
    weights = np.exp(log_weights - float(np.max(log_weights)))
    probabilities = weights / weights.sum()
    selected = int(
        seeds.generator("progressive_is", step_index, "select").choice(
            len(progressive), p=probabilities
        )
    )
    return ProgressiveISStep(
        generated_length_before=len(generated_prefix),
        candidates=tuple(progressive),
        selected_index=selected,
        frozen_evaluation_cost=frozen_cost,
    )


def run_progressive_conditional_is(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ProgressiveISConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
    streaming_rewards: bool = True,
) -> ProgressiveISResult:
    base_sampling = base_sampling or SamplingConfig()
    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")
    evaluator = (
        StreamingRewardEvaluator(workers=config.reward_workers)
        if streaming_rewards and reward is not None and reward_batch is None
        else None
    )
    generated: list[int] = []
    steps: list[ProgressiveISStep] = []
    try:
        while len(generated) < config.total_length:
            step = progressive_is_step(
                base_backend=base_backend,
                rollout_backend=rollout_backend,
                prompt=prompt,
                generated_prefix=tuple(generated),
                config=config,
                base_sampling=base_sampling,
                rollout_sampling=rollout_sampling,
                reward=reward,
                reward_batch=reward_batch,
                seeds=seeds,
                step_index=len(steps),
                streaming_evaluator=evaluator,
            )
            generated.extend(step.selected.token_ids)
            steps.append(step)
            eos = base_sampling.eos_token_id
            if eos is not None and eos in step.selected.token_ids:
                generated = generated[: generated.index(eos) + 1]
                break
    finally:
        if evaluator is not None:
            evaluator.close()
    return ProgressiveISResult(prompt, tuple(generated), tuple(steps))
