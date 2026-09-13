"""Benchmark replay-aware candidate proposals and variance--cost allocation.

For the replay-aware methods, a small model first produces candidate blocks and
two hidden evaluation rollouts for each block.  The base-candidate control does
not construct or query that cache.  The replay-aware methods
sample from a defensive base/small-model mixture and correct candidate weights by
the exact outer p/q ratio.  The final method estimates allocation statistics from
an independent design pool before evaluation records are revealed.
"""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import copy
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from math import exp, log, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.arllm.scoped_execution import fixed_experiment_reward, with_output_scope
from experiments.arllm.runtime import validate_model_artifacts
from experiments.shared.artifacts import load_jsonl as _load_records

from experiments.arllm.gsm8k_reproduction import (
    _configured_verifier_reward,
    _file_sha256,
    _implementation_hashes,
    _fingerprint,
    _fraction_text,
    _load_backend,
    _prompt_tokens,
    _snapshot_delta,
    _timed,
)
from experiments.arllm.summarize_gsm8k_dynamic_is import METHODS, build_summary
from inference_scaling.arllm.algorithms.base_replay import _score_base
from inference_scaling.experimental.arllm.dynamic_is import (
    CandidateProposal,
    DesignStatisticsContext,
    RolloutBudgetContext,
    VarianceCostEstimate,
    dynamic_is_step,
    empirical_design_statistics,
)
from inference_scaling.arllm.backends import (
    BACKEND_CHOICES,
    CachedCandidateBackend,
    ScoreCachingBackend,
    close_backend,
    set_backend_override,
)
from inference_scaling.arllm.config import DynamicISConfig, SamplingConfig
from inference_scaling.shared.evaluation import extract_numeric_answer, load_gsm8k, select_problems
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    ReplaySampleRequest,
    sample_replay_records,
    validate_record_probabilities,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import replace_verifier_from_file
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, SequenceSample
IMPLEMENTATION_FILES = (
    "experiments/arllm/gsm8k_dynamic_is_benchmark.py",
    "experiments/arllm/summarize_gsm8k_dynamic_is.py",
    "src/inference_scaling/experimental/arllm/dynamic_is.py",
    "src/inference_scaling/arllm/algorithms/base_replay.py",
    "src/inference_scaling/arllm/backends/candidate_cache.py",
    "src/inference_scaling/arllm/backends/cache.py",
    "src/inference_scaling/arllm/backends/loader.py",
    "src/inference_scaling/arllm/backends/transformers_backend.py",
    "src/inference_scaling/arllm/backends/vllm_backend.py",
    "src/inference_scaling/arllm/replay.py",
    "src/inference_scaling/arllm/config.py",
    "src/inference_scaling/arllm/types.py",
)
def _sum_delta(
    left: dict[str, int | float], right: dict[str, int | float]
) -> dict[str, int | float]:
    return {name: left[name] + right[name] for name in left}


def _subtract_delta(
    left: dict[str, int | float], right: dict[str, int | float]
) -> dict[str, int | float]:
    result = {name: left[name] - right[name] for name in left}
    if any(float(value) < -1e-9 for value in result.values()):
        raise RuntimeError("a design-pool backend delta exceeded the enclosing online delta")
    return result


def _zero_delta(snapshot) -> dict[str, int | float]:
    return _snapshot_delta(snapshot, snapshot)


def _compute_fields(
    base_delta: dict[str, int | float],
    proposal_delta: dict[str, int | float],
) -> dict[str, int]:
    def slots(delta: dict[str, int | float]) -> int:
        return int(delta["generation_forward_token_slots"]) + int(
            delta["score_forward_token_slots"]
        )

    return {
        "base_forward_token_slots": slots(base_delta),
        "proposal_forward_token_slots": slots(proposal_delta),
        "forward_token_slots": slots(base_delta) + slots(proposal_delta),
        "base_estimated_dense_forward_flops": int(
            base_delta["estimated_dense_forward_flops"]
        ),
        "proposal_estimated_dense_forward_flops": int(
            proposal_delta["estimated_dense_forward_flops"]
        ),
        "estimated_dense_forward_flops": int(
            base_delta["estimated_dense_forward_flops"]
        )
        + int(proposal_delta["estimated_dense_forward_flops"]),
    }


@dataclass(slots=True)
class MatchedProxyBudget:
    history_cost: float
    fresh_cost: float
    rollouts_per_candidate: int
    budgets: list[float] = field(default_factory=list)

    def __call__(self, context: RolloutBudgetContext) -> float:
        history_targets = _fixed_history_targets(
            keys=context.keys,
            terminal=context.terminal,
            history_capacities=context.history_capacities,
            group_capacities=context.group_capacities,
            rollouts_per_candidate=self.rollouts_per_candidate,
        )
        budget = 0.0
        for terminal, history in zip(
            context.terminal, history_targets, strict=True
        ):
            if terminal:
                continue
            fresh = self.rollouts_per_candidate - history
            budget += self.history_cost * history + self.fresh_cost * fresh
        # The allocator requires a positive scalar even when every candidate is terminal.
        budget = budget if budget > 0 else min(self.history_cost, self.fresh_cost)
        self.budgets.append(budget)
        return budget


def _fixed_history_targets(
    *,
    keys: Sequence[ReplayKey],
    terminal: Sequence[bool],
    history_capacities: Sequence[int],
    group_capacities: Mapping[ReplayKey, int],
    rollouts_per_candidate: int,
) -> tuple[int, ...]:
    """Freeze a deterministic fixed control without reusing shared records."""

    if not (len(keys) == len(terminal) == len(history_capacities)):
        raise ValueError("fixed-allocation metadata must have matching lengths")
    if rollouts_per_candidate < 1:
        raise ValueError("rollouts_per_candidate must be positive")
    remaining = dict(group_capacities)
    if any(value < 0 for value in remaining.values()):
        raise ValueError("shared history capacities must be non-negative")
    targets: list[int] = []
    for key, is_terminal, capacity in zip(
        keys, terminal, history_capacities, strict=True
    ):
        if capacity < 0:
            raise ValueError("candidate history capacities must be non-negative")
        if is_terminal:
            targets.append(0)
            continue
        history = min(
            capacity,
            rollouts_per_candidate - 1,
            remaining.get(key, 0),
        )
        targets.append(history)
        remaining[key] = remaining.get(key, 0) - history
    return tuple(targets)


class FixedPerCandidateStatistics:
    """Encode a group-feasible H cached + (K-H) fresh control."""

    def __init__(
        self,
        *,
        proposal_backend,
        proposal_sampling: SamplingConfig,
        mixture: float,
        history_cost: float,
        fresh_cost: float,
        max_history: int,
        rollouts_per_candidate: int,
    ) -> None:
        self.proposal_backend = proposal_backend
        self.proposal_sampling = proposal_sampling
        self.mixture = mixture
        self.history_cost = history_cost
        self.fresh_cost = fresh_cost
        self.max_history = max_history
        self.rollouts_per_candidate = rollouts_per_candidate
        self._history_targets: dict[int, int] = {}

    def prepare(self, contexts: tuple[DesignStatisticsContext, ...]) -> None:
        group_capacities: dict[ReplayKey, int] = {}
        for context in contexts:
            previous = group_capacities.setdefault(
                context.key, context.available_history
            )
            if previous != context.available_history:
                raise RuntimeError("duplicate replay keys exposed inconsistent inventory")
        targets = _fixed_history_targets(
            keys=tuple(context.key for context in contexts),
            terminal=(False,) * len(contexts),
            history_capacities=tuple(
                min(self.max_history, context.available_history)
                for context in contexts
            ),
            group_capacities=group_capacities,
            rollouts_per_candidate=self.rollouts_per_candidate,
        )
        self._history_targets = {
            id(context): target
            for context, target in zip(contexts, targets, strict=True)
        }

    def _outer_log_ratio(self, context: DesignStatisticsContext) -> float:
        if self.mixture == 0:
            return 0.0
        prefix = context.key.prompt + context.key.generated_prefix
        candidate = context.key.candidate
        base_scores = context.base_policy.backend.score_batch(
            [ScoreRequest(prefix, (candidate,), None)]
        )[0]
        proposal_scores = self.proposal_backend.score_batch(
            [ScoreRequest(prefix, (candidate,), self.proposal_sampling)]
        )[0]
        base_logprob = float(sum(base_scores))
        proposal_logprob = float(sum(proposal_scores))
        mixture_logprob = float(
            np.logaddexp(
                log(1.0 - self.mixture) + base_logprob,
                log(self.mixture) + proposal_logprob,
            )
        )
        return base_logprob - mixture_logprob

    def __call__(self, context: DesignStatisticsContext) -> VarianceCostEstimate:
        try:
            history = self._history_targets[id(context)]
        except KeyError as error:
            raise RuntimeError(
                "fixed allocation statistics were read before preparation"
            ) from error
        fresh_extra = self.rollouts_per_candidate - 1 - history
        if fresh_extra < 0:
            raise ValueError("fixed allocation has more history than its rollout target")
        # The allocator multiplies these standard deviations by the outer ratio and
        # divides by sqrt(cost).  Cancelling those factors makes the continuous target
        # exactly (history, fresh_extra), without reading evaluation values.
        log_inverse_ratio = min(700.0, -self._outer_log_ratio(context))
        inverse_ratio = exp(log_inverse_ratio)
        return VarianceCostEstimate(
            history_std=history * sqrt(self.history_cost) * inverse_ratio,
            fresh_std=fresh_extra * sqrt(self.fresh_cost) * inverse_ratio,
            history_cost=self.history_cost,
            fresh_cost=self.fresh_cost,
        )


class BatchedDesignPool:
    """Build independent design records in batches and cache their statistics."""

    def __init__(
        self,
        *,
        method: str,
        step_index: int,
        rollout_length: int,
        count_per_source: int,
        base_policy: BehaviorPolicy,
        history_policy: BehaviorPolicy,
        store: InMemoryReplayStore,
        reward,
        seeds: SeedStream,
        history_cost: float,
        fresh_cost: float,
        base_backend,
        proposal_backend,
    ) -> None:
        self.method = method
        self.step_index = step_index
        self.rollout_length = rollout_length
        self.count_per_source = count_per_source
        self.base_policy = base_policy
        self.history_policy = history_policy
        self.store = store
        self.reward = reward
        self.seeds = seeds
        self.history_cost = history_cost
        self.fresh_cost = fresh_cost
        self.base_backend = base_backend
        self.proposal_backend = proposal_backend
        self.statistics: dict[ReplayKey, VarianceCostEstimate] = {}
        self.raw_statistics: list[dict[str, Any]] = []
        self.seconds = 0.0
        self.records_generated = 0
        self.base_delta = _zero_delta(base_backend.snapshot())
        self.proposal_delta = _zero_delta(proposal_backend.snapshot())

    def prepare(self, contexts: tuple[DesignStatisticsContext, ...]) -> None:
        unique: list[DesignStatisticsContext] = []
        seen: set[ReplayKey] = set()
        for context in contexts:
            if context.key not in seen:
                seen.add(context.key)
                unique.append(context)
        base_before = self.base_backend.snapshot()
        proposal_before = self.proposal_backend.snapshot()

        def build() -> None:
            base_requests: list[ReplaySampleRequest] = []
            history_requests: list[ReplaySampleRequest] = []
            for key_index, context in enumerate(unique):
                for draw_index in range(self.count_per_source):
                    base_requests.append(
                        ReplaySampleRequest(
                            key=context.key,
                            max_new_tokens=self.rollout_length,
                            seed=self.seeds.derive(
                                "dynamic-design",
                                self.step_index,
                                key_index,
                                "base",
                                draw_index,
                            ),
                            record_id=(
                                f"design:{self.method}:{self.step_index}:{key_index}:"
                                f"base:{draw_index}"
                            ),
                        )
                    )
                    if context.available_history:
                        history_requests.append(
                            ReplaySampleRequest(
                                key=context.key,
                                max_new_tokens=self.rollout_length,
                                seed=self.seeds.derive(
                                    "dynamic-design",
                                    self.step_index,
                                    key_index,
                                    "history",
                                    draw_index,
                                ),
                                record_id=(
                                    f"design:{self.method}:{self.step_index}:{key_index}:"
                                    f"history:{draw_index}"
                                ),
                            )
                        )
            records = list(
                sample_replay_records(self.base_policy, base_requests, self.reward)
            )
            records.extend(
                sample_replay_records(
                    self.history_policy, history_requests, self.reward
                )
            )
            for record in records:
                self.store.add_design(record)
            self.records_generated += len(records)
            records_by_key: dict[ReplayKey, list[Any]] = {}
            for record in records:
                records_by_key.setdefault(record.key, []).append(record)
            history_score_requests = []
            proposal_score_requests = []
            for context in unique:
                if not context.available_history:
                    continue
                key_records = records_by_key.get(context.key, [])
                history_completions = tuple(
                    record.completion
                    for record in key_records
                    if record.behavior_id == self.history_policy.behavior_id
                )
                all_completions = tuple(record.completion for record in key_records)
                if history_completions:
                    history_score_requests.append(
                        ScoreRequest(
                            context.key.rollout_prefix,
                            history_completions,
                            self.base_policy.sampling,
                        )
                    )
                if all_completions:
                    proposal_score_requests.append(
                        ScoreRequest(
                            context.key.rollout_prefix,
                            all_completions,
                            self.history_policy.sampling,
                        )
                    )
            if history_score_requests:
                self.base_policy.backend.score_batch(history_score_requests)
            if proposal_score_requests:
                self.history_policy.backend.score_batch(proposal_score_requests)
            for context in unique:
                raw = empirical_design_statistics(context)
                estimate = VarianceCostEstimate(
                    history_std=raw.history_std,
                    fresh_std=raw.fresh_std,
                    history_cost=self.history_cost,
                    fresh_cost=self.fresh_cost,
                )
                self.statistics[context.key] = estimate
                self.raw_statistics.append(
                    {
                        "available_history": context.available_history,
                        "history_std": estimate.history_std,
                        "fresh_std": estimate.fresh_std,
                        "raw_history_cost": raw.history_cost,
                        "raw_fresh_cost": raw.fresh_cost,
                    }
                )

        _, seconds = _timed(build)
        self.seconds += seconds
        self.base_delta = _sum_delta(
            self.base_delta,
            _snapshot_delta(base_before, self.base_backend.snapshot()),
        )
        self.proposal_delta = _sum_delta(
            self.proposal_delta,
            _snapshot_delta(proposal_before, self.proposal_backend.snapshot()),
        )

    def __call__(self, context: DesignStatisticsContext) -> VarianceCostEstimate:
        try:
            return self.statistics[context.key]
        except KeyError as error:
            raise RuntimeError("design statistics were read before batched preparation") from error


def _prepare_replay_cache(
    *,
    cached_base,
    cached_proposal,
    base_sampling: SamplingConfig,
    proposal_sampling: SamplingConfig,
    history_policy: BehaviorPolicy,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    prompt: tuple[int, ...],
    generated_prefix: tuple[int, ...],
    candidate_count: int,
    block_length: int,
    rollout_length: int,
    history_count: int,
    reward,
    reward_version: str,
    seeds: SeedStream,
    step_index: int,
    method: str,
) -> tuple[tuple[SequenceSample, ...], int]:
    prefix = prompt + generated_prefix
    candidate_requests = [
        GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=proposal_sampling,
            seed=seeds.derive("dynamic_is", step_index, index, "sample"),
            request_id=f"dynamic-is:step:{step_index}:candidate:{index}",
        )
        for index in range(candidate_count)
    ]
    samples = tuple(cached_proposal.sample_batch(candidate_requests))
    if len(samples) != candidate_count or any(not sample.token_ids for sample in samples):
        raise RuntimeError("cache proposal returned an invalid candidate batch")
    eos = base_sampling.eos_token_id
    replay_requests: list[ReplaySampleRequest] = []
    for candidate_index, sample in enumerate(samples):
        terminal = rollout_length == 0 or (
            eos is not None and sample.token_ids[-1] == eos
        )
        if terminal:
            continue
        key = ReplayKey(prompt, generated_prefix, sample.token_ids, reward_version)
        for history_index in range(history_count):
            replay_requests.append(
                ReplaySampleRequest(
                    key=key,
                    max_new_tokens=rollout_length,
                    seed=seeds.derive(
                        "dynamic-cache-history",
                        step_index,
                        candidate_index,
                        history_index,
                    ),
                    record_id=(
                        f"cache:{method}:{step_index}:{candidate_index}:{history_index}"
                    ),
                )
            )
    records = sample_replay_records(history_policy, replay_requests, reward)
    by_key: dict[ReplayKey, list[tuple[int, ...]]] = {}
    for record in records:
        by_key.setdefault(record.key, []).append(record.completion)
    if by_key:
        cached_proposal.score_batch(
            [
                ScoreRequest(key.rollout_prefix, tuple(completions), proposal_sampling)
                for key, completions in by_key.items()
            ]
        )
        cached_base.score_batch(
            [
                ScoreRequest(key.rollout_prefix, tuple(completions), base_sampling)
                for key, completions in by_key.items()
            ]
        )
    validate_record_probabilities(records, registry)
    for key, completions in by_key.items():
        _score_base(cached_base, key, completions, base_sampling)
    for record in records:
        store.add_evaluation(record)

    candidate_tokens = tuple(sample.token_ids for sample in samples)
    cached_base.score_batch([ScoreRequest(prefix, candidate_tokens, None)])
    cached_proposal.score_batch(
        [ScoreRequest(prefix, candidate_tokens, proposal_sampling)]
    )
    return samples, len(records)


@with_output_scope
def _run_method(
    *,
    method: str,
    backend,
    proposal_backend,
    prompt: tuple[int, ...],
    reward,
    reward_version: str,
    config: dict[str, Any],
    seeds: SeedStream,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    section = config["conditional_is"]
    extension = config["dynamic_extension"]
    candidate_count = int(section["candidate_count"])
    block_size = int(section["block_size"])
    total_length = int(config["generation"]["max_new_tokens"])
    history_count = int(extension["cache_history_rollouts"])
    design_count = int(extension["design_rollouts_per_source"])
    rollouts_per_candidate = int(extension["rollouts_per_candidate"])
    mixture = (
        0.0
        if method == "base_candidate_fixed"
        else float(extension["auxiliary_mixture"])
    )
    temperature = float(config.get("sampling", {}).get("temperature", 1.0))
    base_sampling = SamplingConfig(
        temperature=temperature, eos_token_id=backend.tokenizer.eos_token_id
    )
    proposal_sampling = SamplingConfig(
        temperature=temperature, eos_token_id=proposal_backend.tokenizer.eos_token_id
    )
    cached_base = ScoreCachingBackend(backend)
    cached_proposal = ScoreCachingBackend(proposal_backend)
    base_policy = BehaviorPolicy.for_backend(cached_base, base_sampling, label="base")
    history_policy = BehaviorPolicy.for_backend(
        cached_proposal, proposal_sampling, label="small-model-history"
    )
    registry = BehaviorRegistry([base_policy, history_policy])
    store = InMemoryReplayStore()
    history_cost = 1.0
    fresh_cost = 1.0 + proposal_backend.parameter_count / backend.parameter_count

    generated: list[int] = []
    steps = []
    cache_build_seconds = 0.0
    design_seconds = 0.0
    online_total_seconds = 0.0
    history_generated = 0
    design_records_generated = 0
    candidate_reproduction_all = True
    cache_base_delta = _zero_delta(backend.snapshot())
    cache_proposal_delta = _zero_delta(proposal_backend.snapshot())
    design_base_delta = _zero_delta(backend.snapshot())
    design_proposal_delta = _zero_delta(proposal_backend.snapshot())
    online_base_delta = _zero_delta(backend.snapshot())
    online_proposal_delta = _zero_delta(proposal_backend.snapshot())
    proxy_budget = 0.0
    outer_weight_ess_sum = 0.0
    final_weight_ess_sum = 0.0
    auxiliary_candidates = 0
    nonterminal_candidates = 0
    candidate_cache_hits = 0
    candidates_using_history = 0
    step_diagnostics: list[dict[str, Any]] = []
    step_index = 0

    while len(generated) < total_length:
        remaining = total_length - len(generated)
        block_length = min(block_size, remaining)
        rollout_length = max(0, remaining - block_length)
        cache_base_before = backend.snapshot()
        cache_proposal_before = proposal_backend.snapshot()
        if method == "base_candidate_fixed":
            cache_samples: tuple[SequenceSample, ...] = ()
            generated_count = 0
            cache_seconds = 0.0
        else:
            (cache_samples, generated_count), cache_seconds = _timed(
                lambda generated_prefix=tuple(generated),
                block_length=block_length,
                rollout_length=rollout_length,
                step_index=step_index: _prepare_replay_cache(
                    cached_base=cached_base,
                    cached_proposal=cached_proposal,
                    base_sampling=base_sampling,
                    proposal_sampling=proposal_sampling,
                    history_policy=history_policy,
                    registry=registry,
                    store=store,
                    prompt=prompt,
                    generated_prefix=generated_prefix,
                    candidate_count=candidate_count,
                    block_length=block_length,
                    rollout_length=rollout_length,
                    history_count=history_count,
                    reward=reward,
                    reward_version=reward_version,
                    seeds=seeds,
                    step_index=step_index,
                    method=method,
                )
            )
        cache_build_seconds += cache_seconds
        history_generated += generated_count
        cache_base_delta = _sum_delta(
            cache_base_delta,
            _snapshot_delta(cache_base_before, backend.snapshot()),
        )
        cache_proposal_delta = _sum_delta(
            cache_proposal_delta,
            _snapshot_delta(cache_proposal_before, proposal_backend.snapshot()),
        )

        budget = MatchedProxyBudget(
            history_cost=history_cost,
            fresh_cost=fresh_cost,
            rollouts_per_candidate=rollouts_per_candidate,
        )
        design: BatchedDesignPool | None = None
        if method == "replay_aware_optimal":
            design = BatchedDesignPool(
                method=method,
                step_index=step_index,
                rollout_length=rollout_length,
                count_per_source=design_count,
                base_policy=base_policy,
                history_policy=history_policy,
                store=store,
                reward=reward,
                seeds=seeds,
                history_cost=history_cost,
                fresh_cost=fresh_cost,
                base_backend=backend,
                proposal_backend=proposal_backend,
            )
            statistics_provider = design
            design_prepare = design.prepare
        else:
            statistics_provider = FixedPerCandidateStatistics(
                proposal_backend=cached_proposal,
                proposal_sampling=proposal_sampling,
                mixture=mixture,
                history_cost=history_cost,
                fresh_cost=fresh_cost,
                max_history=history_count,
                rollouts_per_candidate=rollouts_per_candidate,
            )
            design_prepare = statistics_provider.prepare

        algorithm = DynamicISConfig(
            candidate_count=candidate_count,
            block_size=block_size,
            total_length=total_length,
            reward_temperature=float(section["reward_temperature"]),
            max_history_per_candidate=history_count,
            truncation=float(config["replay"]["truncation"]),
            rollout_budget=float(candidate_count * rollouts_per_candidate),
            auxiliary_mixture=mixture,
            minimum_fresh_per_candidate=1,
        )
        step_proposal = (
            CandidateProposal.for_backend(
                CachedCandidateBackend(cached_proposal, cache_samples),
                proposal_sampling,
                label="replay-cache-source",
            )
            if mixture
            else None
        )
        online_base_before = backend.snapshot()
        online_proposal_before = proposal_backend.snapshot()
        step, seconds = _timed(
            lambda generated_prefix=tuple(generated),
            algorithm=algorithm,
            step_index=step_index,
            step_proposal=step_proposal,
            statistics_provider=statistics_provider,
            design_prepare=design_prepare,
            budget=budget: dynamic_is_step(
                base_backend=cached_base,
                registry=registry,
                store=store,
                prompt=prompt,
                generated_prefix=generated_prefix,
                config=algorithm,
                base_sampling=base_sampling,
                reward=reward,
                reward_version=reward_version,
                seeds=seeds,
                step_index=step_index,
                auxiliary_proposal=step_proposal,
                statistics_provider=statistics_provider,
                design_prepare=design_prepare,
                rollout_budget_provider=budget,
            )
        )
        online_total_seconds += seconds
        online_base_delta = _sum_delta(
            online_base_delta,
            _snapshot_delta(online_base_before, backend.snapshot()),
        )
        online_proposal_delta = _sum_delta(
            online_proposal_delta,
            _snapshot_delta(online_proposal_before, proposal_backend.snapshot()),
        )
        if design is not None:
            design_seconds += design.seconds
            design_records_generated += design.records_generated
            design_base_delta = _sum_delta(design_base_delta, design.base_delta)
            design_proposal_delta = _sum_delta(
                design_proposal_delta, design.proposal_delta
            )
        proxy_budget += budget.budgets[-1]

        cache_by_index = [sample.token_ids for sample in cache_samples]
        reproduced = all(
            candidate.draw.source != "auxiliary"
            or candidate.token_ids == cache_by_index[index]
            for index, candidate in enumerate(step.candidates)
        )
        if not reproduced:
            raise RuntimeError("replay-aware proposal did not reproduce its cached candidate")
        candidate_reproduction_all &= reproduced
        eos = base_sampling.eos_token_id
        cache_keys = {
            ReplayKey(prompt, tuple(generated), sample.token_ids, reward_version)
            for sample in cache_samples
            if rollout_length > 0 and (eos is None or sample.token_ids[-1] != eos)
        }
        terminal = [
            rollout_length == 0
            or (eos is not None and candidate.token_ids[-1] == eos)
            for candidate in step.candidates
        ]
        if method != "replay_aware_optimal":
            for candidate, is_terminal in zip(step.candidates, terminal, strict=True):
                if not is_terminal and (
                    candidate.allocation.history_count + candidate.allocation.fresh_count
                    != rollouts_per_candidate
                ):
                    raise RuntimeError("fixed control did not preserve its per-candidate budget")

        auxiliary_candidates += sum(
            candidate.draw.source == "auxiliary" for candidate in step.candidates
        )
        nonterminal_candidates += sum(not value for value in terminal)
        available_hits = sum(
            not is_terminal
            and ReplayKey(
                prompt, tuple(generated), candidate.token_ids, reward_version
            )
            in cache_keys
            for candidate, is_terminal in zip(step.candidates, terminal, strict=True)
        )
        candidate_cache_hits += available_hits
        candidates_using_history += sum(
            candidate.allocation.history_count > 0 for candidate in step.candidates
        )
        outer_ess = importance_effective_sample_size(
            [candidate.draw.outer_log_ratio for candidate in step.candidates]
        )
        final_ess = importance_effective_sample_size(
            [candidate.log_weight for candidate in step.candidates]
        )
        outer_weight_ess_sum += outer_ess
        final_weight_ess_sum += final_ess
        step_diagnostics.append(
            {
                "step_index": step_index,
                "generated_length_before": len(generated),
                "candidate_count": len(step.candidates),
                "auxiliary_candidates": sum(
                    candidate.draw.source == "auxiliary"
                    for candidate in step.candidates
                ),
                "available_cache_hits": available_hits,
                "history_used": sum(
                    candidate.allocation.history_count for candidate in step.candidates
                ),
                "fresh_used": sum(
                    candidate.allocation.fresh_count for candidate in step.candidates
                ),
                "proxy_budget": budget.budgets[-1],
                "proxy_cost_used": sum(
                    candidate.allocation.estimated_cost for candidate in step.candidates
                ),
                "outer_weight_ess": outer_ess,
                "final_weight_ess": final_ess,
                "selected_index": step.selected_index,
                "design_records_generated": (
                    design.records_generated if design is not None else 0
                ),
                "design_statistics": (
                    design.raw_statistics if design is not None else []
                ),
            }
        )
        print(
            f"  method={method} step={step_index} hits={available_hits} "
            f"history={step_diagnostics[-1]['history_used']} "
            f"fresh={step_diagnostics[-1]['fresh_used']}",
            flush=True,
        )
        generated.extend(step.selected.token_ids)
        steps.append(step)
        if eos is not None and eos in step.selected.token_ids:
            generated = generated[: generated.index(eos) + 1]
            break
        step_index += 1

    online_steady_base_delta = _subtract_delta(
        online_base_delta, design_base_delta
    )
    online_steady_proposal_delta = _subtract_delta(
        online_proposal_delta, design_proposal_delta
    )
    cache_compute = _compute_fields(cache_base_delta, cache_proposal_delta)
    design_compute = _compute_fields(design_base_delta, design_proposal_delta)
    steady_compute = _compute_fields(
        online_steady_base_delta, online_steady_proposal_delta
    )
    online_compute = _compute_fields(online_base_delta, online_proposal_delta)
    history_used = sum(
        candidate.allocation.history_count for step in steps for candidate in step.candidates
    )
    fresh_used = sum(
        candidate.allocation.fresh_count for step in steps for candidate in step.candidates
    )
    proxy_cost_used = sum(
        candidate.allocation.estimated_cost
        for step in steps
        for candidate in step.candidates
    )
    return tuple(generated), {
        "steps": len(steps),
        "candidate_count": sum(len(step.candidates) for step in steps),
        "auxiliary_candidates": auxiliary_candidates,
        "nonterminal_candidates": nonterminal_candidates,
        "candidate_cache_hits": candidate_cache_hits,
        "candidates_using_history": candidates_using_history,
        "history_generated": history_generated,
        "history_used": history_used,
        "fresh_used": fresh_used,
        "design_records_generated": design_records_generated,
        "proxy_budget": proxy_budget,
        "proxy_cost_used": proxy_cost_used,
        "outer_weight_ess_sum": outer_weight_ess_sum,
        "final_weight_ess_sum": final_weight_ess_sum,
        "cache_build_seconds": cache_build_seconds,
        "design_seconds": design_seconds,
        "steady_online_seconds": online_total_seconds - design_seconds,
        "online_total_seconds": online_total_seconds,
        "one_shot_seconds": cache_build_seconds + online_total_seconds,
        "cache_build_forward_token_slots": cache_compute["forward_token_slots"],
        "design_forward_token_slots": design_compute["forward_token_slots"],
        "steady_online_forward_token_slots": steady_compute["forward_token_slots"],
        "online_total_forward_token_slots": online_compute["forward_token_slots"],
        "cache_build_estimated_dense_forward_flops": cache_compute[
            "estimated_dense_forward_flops"
        ],
        "design_estimated_dense_forward_flops": design_compute[
            "estimated_dense_forward_flops"
        ],
        "steady_online_estimated_dense_forward_flops": steady_compute[
            "estimated_dense_forward_flops"
        ],
        "online_total_estimated_dense_forward_flops": online_compute[
            "estimated_dense_forward_flops"
        ],
        "one_shot_estimated_dense_forward_flops": cache_compute[
            "estimated_dense_forward_flops"
        ]
        + online_compute["estimated_dense_forward_flops"],
        "cache_build_base_estimated_dense_forward_flops": cache_compute[
            "base_estimated_dense_forward_flops"
        ],
        "cache_build_proposal_estimated_dense_forward_flops": cache_compute[
            "proposal_estimated_dense_forward_flops"
        ],
        "steady_online_base_estimated_dense_forward_flops": steady_compute[
            "base_estimated_dense_forward_flops"
        ],
        "steady_online_proposal_estimated_dense_forward_flops": steady_compute[
            "proposal_estimated_dense_forward_flops"
        ],
        "candidate_reproduction_all": candidate_reproduction_all,
        "evaluation_records_remaining": store.evaluation_count,
        "design_records_final": store.design_count,
        "base_score_cache": {
            "entries": cached_base.snapshot().entries,
            "hits": cached_base.snapshot().hits,
            "misses": cached_base.snapshot().misses,
        },
        "proposal_score_cache": {
            "entries": cached_proposal.snapshot().entries,
            "hits": cached_proposal.snapshot().hits,
            "misses": cached_proposal.snapshot().misses,
        },
        "step_diagnostics": step_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_3090_aligned.toml")
    )
    parser.add_argument(
        "--extension-config",
        type=Path,
        default=Path("configs/gsm8k_3090_dynamic_is.toml"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verifier-config", type=Path)
    add_model_output_arguments(parser)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    apply_model_output_overrides(config, args)
    replace_verifier_from_file(config, args.verifier_config)
    set_backend_override(config, args.backend)
    with args.extension_config.open("rb") as source:
        extension_config = tomllib.load(source)
    if "dynamic_extension" not in extension_config:
        raise ValueError("extension configuration is missing [dynamic_extension]")
    config["dynamic_extension"] = extension_config["dynamic_extension"]
    if args.limit is not None:
        config["run"]["sample_count"] = args.limit
    problems = select_problems(
        load_gsm8k(args.data),
        int(config["run"]["sample_count"]),
        seed=int(config["run"]["subset_seed"]),
    )
    run_dir = (
        args.output_root
        / str(config["run"]["name"])
        / f"dynamic-is-comparison-{args.tag}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    input_artifacts = validate_model_artifacts(config, {"base", "proposal"})
    base_hash = input_artifacts["weight_sha256"]["base"]
    proposal_hash = input_artifacts["weight_sha256"]["proposal"]
    effective = {
        "config": config,
        "tag": args.tag,
        "extension_config": {
            "path": str(args.extension_config),
            "sha256": _file_sha256(args.extension_config),
        },
        "problem_indices": [problem.index for problem in problems],
        "settings": {
            "methods": list(METHODS),
            "candidate_count": int(config["conditional_is"]["candidate_count"]),
            "block_size": int(config["conditional_is"]["block_size"]),
            "total_length": int(config["generation"]["max_new_tokens"]),
            "reward_temperature": float(
                config["conditional_is"]["reward_temperature"]
            ),
            "truncation": float(config["replay"]["truncation"]),
            **config["dynamic_extension"],
            "allocation_cost_proxy": (
                "history=one base-score equivalent per completion token; "
                "fresh=one base generation plus one small-model score equivalent"
            ),
            "verifier": config["verifier"],
        },
        "input_weight_sha256": {"base": base_hash, "proposal": proposal_hash},
        "input_metadata_sha256": input_artifacts["metadata_sha256"],
        "implementation_sha256": _implementation_hashes(
            Path(__file__).resolve().parents[2],
            entrypoints=IMPLEMENTATION_FILES,
        ),
    }
    fingerprint = _fingerprint(effective)
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "benchmark": "GSM8K replay-aware candidate proposal and allocation",
        "effective": effective,
    }
    existing_records = _load_records(records_path)
    completed = {int(record["problem_index"]) for record in existing_records}
    pending = [problem for problem in problems if problem.index not in completed]
    if not pending:
        if not manifest_path.is_file():
            raise ValueError(f"{run_dir} contains records without a manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_effective = copy.deepcopy(previous["effective"])
        previous_effective["settings"].pop("history_cost", None)
        previous_effective["settings"].pop("fresh_cost", None)
        if previous_effective != effective:
            raise ValueError(
                f"{run_dir} contains a different benchmark; choose a new --tag"
            )
        selected = {problem.index for problem in problems}
        records = [
            record
            for record in existing_records
            if int(record["problem_index"]) in selected
        ]
        summary = build_summary(previous, records)
        serialized = (
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        summary_path.write_text(serialized, encoding="utf-8")
        if args.aggregate_output is not None:
            args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
            args.aggregate_output.write_text(serialized, encoding="utf-8")
        print(serialized)
        return

    backend = _load_backend(str(config["models"]["base"]), config, role="base")
    proposal_backend = _load_backend(str(config["models"]["proposal"]), config, role="proposal")
    if backend.tokenizer.get_vocab() != proposal_backend.tokenizer.get_vocab():
        raise ValueError("base and proposal tokenizers must match")
    manifest["models"] = {
        "base": {
            "path": str(config["models"]["base"]),
            "parameter_count": backend.parameter_count,
        },
        "proposal": {
            "path": str(config["models"]["proposal"]),
            "parameter_count": proposal_backend.parameter_count,
        },
    }
    manifest["effective"]["settings"]["history_cost"] = 1.0
    manifest["effective"]["settings"]["fresh_cost"] = (
        1.0 + proposal_backend.parameter_count / backend.parameter_count
    )
    # The parameter counts affect the allocation proxy, so include them in the final fingerprint.
    manifest["fingerprint"] = _fingerprint(manifest["effective"])
    fingerprint = manifest["fingerprint"]
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError(f"{run_dir} contains a different benchmark; choose a new --tag")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    warm_prompt = _prompt_tokens(backend, pending[0], config)
    warm_sampling = SamplingConfig(eos_token_id=backend.tokenizer.eos_token_id)
    backend.sample_batch(
        [GenerationRequest(warm_prompt, 2, warm_sampling, 1, "base-warmup")]
    )
    proposal_backend.sample_batch(
        [GenerationRequest(warm_prompt, 2, warm_sampling, 2, "proposal-warmup")]
    )

    with records_path.open("a", encoding="utf-8", buffering=1) as sink:
        for ordinal, problem in enumerate(pending, 1):
            prompt = _prompt_tokens(backend, problem, config)
            verifier_reward, reward_version = fixed_experiment_reward(backend, problem, config, _configured_verifier_reward)
            method_results: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                seed = SeedStream(
                    SeedStream(int(config["run"]["seed"])).derive(
                        "dynamic-extension", problem.index
                    )
                )
                tokens, info = _run_method(
                    method=method,
                    backend=backend,
                    proposal_backend=proposal_backend,
                    prompt=prompt,
                    reward=verifier_reward,
                    reward_version=reward_version,
                    config=config,
                    seeds=seed,
                )
                output = backend.decode(tokens)
                prediction = extract_numeric_answer(info["output_segments"]["content_text"])
                method_results[method] = {
                    "prediction": _fraction_text(prediction),
                    "correct": prediction == problem.gold_answer,
                    "output": output,
                    **info,
                }
            record = {
                "schema_version": 1,
                "manifest_fingerprint": fingerprint,
                "problem_index": problem.index,
                "question_sha256": hashlib.sha256(problem.question.encode()).hexdigest(),
                "gold_answer": _fraction_text(problem.gold_answer),
                "methods": method_results,
            }
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"[{ordinal}/{len(pending)}] gsm8k_index={problem.index} "
                f"base={int(method_results['base_candidate_fixed']['correct'])} "
                f"dynamic={int(method_results['replay_aware_fixed']['correct'])} "
                f"optimal={int(method_results['replay_aware_optimal']['correct'])} "
                f"hit={method_results['replay_aware_fixed']['candidate_cache_hits']}/"
                f"{method_results['replay_aware_fixed']['nonterminal_candidates']}",
                flush=True,
            )

    selected = {problem.index for problem in problems}
    records = [
        record
        for record in _load_records(records_path)
        if int(record["problem_index"]) in selected
    ]
    summary = build_summary(manifest, records)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(serialized, encoding="utf-8")
    if args.aggregate_output is not None:
        args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_output.write_text(serialized, encoding="utf-8")
    print(serialized)
    close_backend(proposal_backend)
    close_backend(backend)


if __name__ == "__main__":
    main()
