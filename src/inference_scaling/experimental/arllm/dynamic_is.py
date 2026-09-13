"""Dynamic candidate proposals with replay-corrected conditional weights.

This module adds three independent mechanisms to ``base_replay``: a defensive
candidate mixture, an outer base/proposal probability ratio, and a frozen
variance--cost allocation.  Completion-level replay correction is reused
without changing its estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np

from inference_scaling.arllm.algorithms.base_replay import (
    ReplayWeightEstimate,
    build_fresh_replay_requests,
    estimate_replay_weight,
    write_reserve_records,
)
from inference_scaling.arllm.algorithms.conditional_is import (
    RewardFunction,
    _validate_base_sampling,
)
from inference_scaling.arllm.config import DynamicISConfig, SamplingConfig
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    ReplaySampleRequest,
    mixture_logprobabilities,
    score_continuations,
    sample_replay_records,
    validate_record_probabilities,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.budget import (
    BudgetAllocation,
    VarianceCostEstimate,
    allocate_variance_cost_budget,
)
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    proposal_id: str
    backend: AutoregressiveBackend
    sampling: SamplingConfig

    @classmethod
    def for_backend(
        cls,
        backend: AutoregressiveBackend,
        sampling: SamplingConfig,
        *,
        label: str | None = None,
    ) -> "CandidateProposal":
        return cls(label or f"{backend.model_id}|{sampling.policy_id}", backend, sampling)


@dataclass(frozen=True, slots=True)
class DynamicCandidateDraw:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    base_logprob: float
    auxiliary_logprob: float
    proposal_logprob: float
    outer_log_ratio: float
    source: str
    auxiliary_proposal_id: str | None


CandidateProposalFactory = Callable[
    [int, int, tuple[DynamicCandidateDraw, ...]], CandidateProposal
]
CandidateProposalSource = CandidateProposal | CandidateProposalFactory | None


@dataclass(frozen=True, slots=True)
class DesignStatisticsContext:
    key: ReplayKey
    evaluation_inventory: tuple[tuple[str, int], ...]
    store: InMemoryReplayStore
    registry: BehaviorRegistry
    base_policy: BehaviorPolicy
    reward_temperature: float
    truncation: float

    @property
    def available_history(self) -> int:
        return sum(count for _, count in self.evaluation_inventory)


DesignStatisticsProvider = Callable[[DesignStatisticsContext], VarianceCostEstimate]
DesignPreparation = Callable[[tuple[DesignStatisticsContext, ...]], None]


@dataclass(frozen=True, slots=True)
class RolloutBudgetContext:
    """Metadata available when freezing one step's rollout budget.

    The context intentionally exposes per-candidate and shared-key replay inventory
    counts but not evaluation completions or rewards.  This permits a cost-matched
    budget to depend on the candidates that were actually drawn and on cache hits
    without leaking the values later used by the estimator.
    """

    draws: tuple[DynamicCandidateDraw, ...]
    keys: tuple[ReplayKey, ...]
    terminal: tuple[bool, ...]
    history_capacities: tuple[int, ...]
    group_capacities: Mapping[ReplayKey, int]


RolloutBudgetProvider = Callable[[RolloutBudgetContext], float]


def constant_design_statistics(context: DesignStatisticsContext) -> VarianceCostEstimate:
    """Safe cold-start statistics; callers can replace this with design-pool estimates."""

    return VarianceCostEstimate(
        history_std=1.0 if context.available_history else 0.0,
        fresh_std=1.0,
    )


def _empirical_standard_deviation(values: Sequence[float], fallback: float) -> float:
    if len(values) < 2:
        return fallback
    result = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    return result if isfinite(result) else fallback


def empirical_design_statistics(context: DesignStatisticsContext) -> VarianceCostEstimate:
    """Estimate allocation inputs using only records already in the design pool."""

    inventory = dict(context.evaluation_inventory)
    design_records = context.store.design_records(context.key)
    behavior_ids = set(inventory)
    history_records = [
        record for record in design_records if record.behavior_id in behavior_ids
    ]
    base_records = [
        record
        for record in design_records
        if record.behavior_id == context.base_policy.behavior_id
    ]
    log_tau = log(context.truncation)

    history_values: list[float] = []
    if inventory and history_records:
        validate_record_probabilities(history_records, context.registry)
        completions = [record.completion for record in history_records]
        base_scores = score_continuations(context.base_policy, context.key, completions)
        mixture_scores = mixture_logprobabilities(
            context.registry, context.key, inventory, completions
        )
        history_values = [
            exp(
                min(log_tau, base_logprob - mixture_logprob)
                + record.reward / context.reward_temperature
            )
            for record, base_logprob, mixture_logprob in zip(
                history_records, base_scores, mixture_scores, strict=True
            )
        ]

    fresh_values: list[float] = []
    if base_records:
        if inventory:
            mixture_scores = mixture_logprobabilities(
                context.registry,
                context.key,
                inventory,
                [record.completion for record in base_records],
            )
            for record, mixture_logprob in zip(base_records, mixture_scores, strict=True):
                log_ratio = record.behavior_logprob - mixture_logprob
                tail = 0.0 if log_ratio <= log_tau else 1.0 - exp(log_tau - log_ratio)
                fresh_values.append(
                    tail * exp(record.reward / context.reward_temperature)
                )
        else:
            fresh_values = [
                exp(record.reward / context.reward_temperature) for record in base_records
            ]

    behavior_score_multiplier = 1 + len(inventory)
    history_cost = (
        max(
            1.0,
            behavior_score_multiplier
            * sum(len(record.completion) for record in history_records)
            / len(history_records),
        )
        if history_records
        else 1.0
    )
    fresh_cost = (
        max(
            1.0,
            behavior_score_multiplier
            * sum(len(record.completion) for record in base_records)
            / len(base_records),
        )
        if base_records
        else 1.0
    )
    return VarianceCostEstimate(
        history_std=(
            _empirical_standard_deviation(history_values, 1.0)
            if context.available_history
            else 0.0
        ),
        fresh_std=_empirical_standard_deviation(fresh_values, 1.0),
        history_cost=history_cost,
        fresh_cost=fresh_cost,
    )



@dataclass(frozen=True, slots=True)
class DynamicCandidate:
    draw: DynamicCandidateDraw
    allocation: BudgetAllocation
    estimate: ReplayWeightEstimate
    log_weight: float

    @property
    def token_ids(self) -> TokenSequence:
        return self.draw.token_ids


@dataclass(frozen=True, slots=True)
class DynamicISStep:
    generated_length_before: int
    candidates: tuple[DynamicCandidate, ...]
    selected_index: int

    @property
    def selected(self) -> DynamicCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class DynamicISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[DynamicISStep, ...]
    reserve_records_written: int


def _score_candidate(
    backend: AutoregressiveBackend,
    sampling: SamplingConfig | None,
    prefix: TokenSequence,
    candidate: TokenSequence,
) -> tuple[float, ...]:
    return _score_candidates(backend, sampling, prefix, (candidate,))[0]


def _score_candidates(
    backend: AutoregressiveBackend,
    sampling: SamplingConfig | None,
    prefix: TokenSequence,
    candidates: Sequence[TokenSequence],
) -> tuple[tuple[float, ...], ...]:
    if not candidates:
        return ()
    scored = backend.score_batch([ScoreRequest(prefix, tuple(candidates), sampling)])
    if len(scored) != len(candidates) or any(
        len(token_scores) != len(candidate)
        for token_scores, candidate in zip(scored, candidates, strict=True)
    ):
        raise RuntimeError("candidate proposal returned an invalid score shape")
    return tuple(scored)


def _resolve_auxiliary(
    source: CandidateProposalSource,
    step_index: int,
    candidate_index: int,
    previous: tuple[DynamicCandidateDraw, ...],
) -> CandidateProposal | None:
    if source is None or isinstance(source, CandidateProposal):
        return source
    proposal = source(step_index, candidate_index, previous)
    if not isinstance(proposal, CandidateProposal):
        raise TypeError("candidate proposal factory must return CandidateProposal")
    return proposal


def _draw_candidates(
    *,
    base_backend: AutoregressiveBackend,
    base_sampling: SamplingConfig,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    mixture: float,
    auxiliary_source: CandidateProposalSource,
    seeds: SeedStream,
    step_index: int,
) -> tuple[DynamicCandidateDraw, ...]:
    if auxiliary_source is None or isinstance(auxiliary_source, CandidateProposal):
        auxiliary = auxiliary_source
        if mixture > 0 and auxiliary is None:
            raise ValueError("a positive auxiliary mixture requires a candidate proposal")
        if (
            auxiliary is not None
            and auxiliary.sampling.eos_token_id != base_sampling.eos_token_id
        ):
            raise ValueError(
                "base and auxiliary candidate policies must agree on eos_token_id"
            )

        request_groups: dict[
            tuple[int, SamplingConfig],
            tuple[AutoregressiveBackend, list[tuple[int, GenerationRequest]]],
        ] = {}
        used_auxiliary: list[bool] = []
        for candidate_index in range(count):
            use_auxiliary = bool(
                mixture > 0
                and seeds.generator(
                    "dynamic_is", step_index, candidate_index, "component"
                ).random()
                < mixture
            )
            used_auxiliary.append(use_auxiliary)
            backend = (
                auxiliary.backend
                if use_auxiliary and auxiliary is not None
                else base_backend
            )
            sampling = (
                auxiliary.sampling
                if use_auxiliary and auxiliary is not None
                else base_sampling
            )
            request = GenerationRequest(
                prefix=prefix,
                max_new_tokens=block_length,
                sampling=sampling,
                seed=seeds.derive("dynamic_is", step_index, candidate_index, "sample"),
                request_id=f"dynamic-is:step:{step_index}:candidate:{candidate_index}",
            )
            key = (id(backend), sampling)
            if key not in request_groups:
                request_groups[key] = (backend, [])
            request_groups[key][1].append((candidate_index, request))

        ordered_samples: list[SequenceSample | None] = [None] * count
        for backend, indexed_requests in request_groups.values():
            sampled = backend.sample_batch([request for _, request in indexed_requests])
            if len(sampled) != len(indexed_requests):
                raise RuntimeError("candidate proposal returned an invalid sample count")
            for (candidate_index, request), sample in zip(
                indexed_requests, sampled, strict=True
            ):
                if not sample.token_ids:
                    raise RuntimeError("candidate proposal must return a non-empty candidate")
                if (
                    sample.model_id != backend.model_id
                    or sample.policy_id != request.sampling.policy_id
                ):
                    raise RuntimeError("candidate sample does not match its declared proposal")
                ordered_samples[candidate_index] = sample
        if any(sample is None for sample in ordered_samples):
            raise RuntimeError("candidate proposal omitted a sample")
        samples = tuple(sample for sample in ordered_samples if sample is not None)
        token_sequences = tuple(sample.token_ids for sample in samples)
        base_scores = _score_candidates(base_backend, None, prefix, token_sequences)
        auxiliary_scores = (
            _score_candidates(
                auxiliary.backend, auxiliary.sampling, prefix, token_sequences
            )
            if auxiliary is not None
            else None
        )

        draws: list[DynamicCandidateDraw] = []
        for candidate_index, (sample, base_token_scores) in enumerate(
            zip(samples, base_scores, strict=True)
        ):
            base_logprob = float(sum(base_token_scores))
            if not isfinite(base_logprob):
                raise ValueError("auxiliary candidate proposal generated outside base support")
            auxiliary_logprob = float("-inf")
            auxiliary_id: str | None = None
            if auxiliary is not None:
                assert auxiliary_scores is not None
                auxiliary_logprob = float(sum(auxiliary_scores[candidate_index]))
                auxiliary_id = auxiliary.proposal_id
                if used_auxiliary[candidate_index] and not isfinite(auxiliary_logprob):
                    raise RuntimeError(
                        "auxiliary sampler generated a zero-probability candidate"
                    )
            proposal_logprob = (
                base_logprob
                if mixture == 0
                else float(
                    np.logaddexp(
                        log(1.0 - mixture) + base_logprob,
                        log(mixture) + auxiliary_logprob,
                    )
                )
            )
            draws.append(
                DynamicCandidateDraw(
                    token_ids=sample.token_ids,
                    base_token_logprobs=tuple(base_token_scores),
                    base_logprob=base_logprob,
                    auxiliary_logprob=auxiliary_logprob,
                    proposal_logprob=proposal_logprob,
                    outer_log_ratio=base_logprob - proposal_logprob,
                    source="auxiliary" if used_auxiliary[candidate_index] else "base",
                    auxiliary_proposal_id=auxiliary_id,
                )
            )
        return tuple(draws)

    draws: list[DynamicCandidateDraw] = []
    for candidate_index in range(count):
        auxiliary = _resolve_auxiliary(
            auxiliary_source, step_index, candidate_index, tuple(draws)
        )
        if mixture > 0 and auxiliary is None:
            raise ValueError("a positive auxiliary mixture requires a candidate proposal")
        if auxiliary is not None and auxiliary.sampling.eos_token_id != base_sampling.eos_token_id:
            raise ValueError("base and auxiliary candidate policies must agree on eos_token_id")
        use_auxiliary = bool(
            mixture > 0
            and seeds.generator("dynamic_is", step_index, candidate_index, "component").random()
            < mixture
        )
        backend = auxiliary.backend if use_auxiliary and auxiliary is not None else base_backend
        sampling = auxiliary.sampling if use_auxiliary and auxiliary is not None else base_sampling
        request = GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=sampling,
            seed=seeds.derive("dynamic_is", step_index, candidate_index, "sample"),
            request_id=f"dynamic-is:step:{step_index}:candidate:{candidate_index}",
        )
        samples = backend.sample_batch([request])
        if len(samples) != 1 or not samples[0].token_ids:
            raise RuntimeError("candidate proposal must return one non-empty candidate")
        sample = samples[0]
        if sample.model_id != backend.model_id or sample.policy_id != sampling.policy_id:
            raise RuntimeError("candidate sample does not match its declared proposal")
        base_token_scores = _score_candidate(base_backend, None, prefix, sample.token_ids)
        base_logprob = float(sum(base_token_scores))
        if not isfinite(base_logprob):
            raise ValueError("auxiliary candidate proposal generated outside base support")
        auxiliary_logprob = float("-inf")
        auxiliary_id: str | None = None
        if auxiliary is not None:
            auxiliary_scores = _score_candidate(
                auxiliary.backend, auxiliary.sampling, prefix, sample.token_ids
            )
            auxiliary_logprob = float(sum(auxiliary_scores))
            auxiliary_id = auxiliary.proposal_id
            if use_auxiliary and not isfinite(auxiliary_logprob):
                raise RuntimeError("auxiliary sampler generated a zero-probability candidate")
        if mixture == 0:
            proposal_logprob = base_logprob
        else:
            proposal_logprob = float(
                np.logaddexp(
                    log(1.0 - mixture) + base_logprob,
                    log(mixture) + auxiliary_logprob,
                )
            )
        draws.append(
            DynamicCandidateDraw(
                token_ids=sample.token_ids,
                base_token_logprobs=tuple(base_token_scores),
                base_logprob=base_logprob,
                auxiliary_logprob=auxiliary_logprob,
                proposal_logprob=proposal_logprob,
                outer_log_ratio=base_logprob - proposal_logprob,
                source="auxiliary" if use_auxiliary else "base",
                auxiliary_proposal_id=auxiliary_id,
            )
        )
    return tuple(draws)


def dynamic_is_step(
    *,
    base_backend: AutoregressiveBackend,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: DynamicISConfig,
    base_sampling: SamplingConfig,
    reward: RewardFunction,
    reward_version: str,
    seeds: SeedStream,
    step_index: int,
    auxiliary_proposal: CandidateProposalSource = None,
    statistics_provider: DesignStatisticsProvider = constant_design_statistics,
    design_prepare: DesignPreparation | None = None,
    rollout_budget_provider: RolloutBudgetProvider | None = None,
) -> DynamicISStep:
    """Run one two-phase dynamic-candidate decision."""

    _validate_base_sampling(base_sampling)
    remaining = config.total_length - len(generated_prefix)
    if remaining <= 0:
        raise ValueError("generated prefix has already reached total_length")
    block_length = min(config.block_size, remaining)
    prefix = prompt + generated_prefix
    draws = _draw_candidates(
        base_backend=base_backend,
        base_sampling=base_sampling,
        prefix=prefix,
        count=config.candidate_count,
        block_length=block_length,
        mixture=config.auxiliary_mixture,
        auxiliary_source=auxiliary_proposal,
        seeds=seeds,
        step_index=step_index,
    )
    base_policy = BehaviorPolicy.for_backend(base_backend, base_sampling, label="base")
    registry.register(base_policy)
    rollout_length = max(0, remaining - block_length)
    eos = base_sampling.eos_token_id
    keys = [
        ReplayKey(prompt, generated_prefix, draw.token_ids, reward_version) for draw in draws
    ]
    terminal = [
        rollout_length == 0 or (eos is not None and draw.token_ids[-1] == eos)
        for draw in draws
    ]

    inventory_by_key = {key: store.inventory(key) for key in dict.fromkeys(keys)}
    group_capacities = {
        key: sum(inventory.values()) for key, inventory in inventory_by_key.items()
    }
    contexts: list[DesignStatisticsContext | None] = []
    history_capacities: list[int] = []
    minimum_fresh: list[int] = []
    for key, is_terminal in zip(keys, terminal, strict=True):
        inventory = inventory_by_key[key]
        if is_terminal:
            contexts.append(None)
            history_capacities.append(0)
            minimum_fresh.append(0)
            continue
        context = DesignStatisticsContext(
            key=key,
            evaluation_inventory=tuple(sorted(inventory.items())),
            store=store,
            registry=registry,
            base_policy=base_policy,
            reward_temperature=config.reward_temperature,
            truncation=config.truncation,
        )
        contexts.append(context)
        history_capacities.append(
            min(config.max_history_per_candidate, context.available_history)
        )
        minimum_fresh.append(config.minimum_fresh_per_candidate)

    if design_prepare is not None:
        design_prepare(tuple(context for context in contexts if context is not None))
    statistics = [
        VarianceCostEstimate(0.0, 0.0)
        if context is None
        else statistics_provider(context)
        for context in contexts
    ]

    outer_log_ratios = np.asarray([draw.outer_log_ratio for draw in draws], dtype=np.float64)
    relative_outer_ratios = np.exp(
        np.maximum(
            outer_log_ratios - float(np.max(outer_log_ratios)),
            log(np.finfo(np.float64).tiny),
        )
    )
    rollout_budget = (
        config.rollout_budget
        if rollout_budget_provider is None
        else float(
            rollout_budget_provider(
                RolloutBudgetContext(
                    draws=draws,
                    keys=tuple(keys),
                    terminal=tuple(terminal),
                    history_capacities=tuple(history_capacities),
                    group_capacities=dict(group_capacities),
                )
            )
        )
    )
    if rollout_budget <= 0 or not isfinite(rollout_budget):
        raise ValueError("rollout budget provider must return a positive finite value")
    allocations = allocate_variance_cost_budget(
        # A common scale cancels from every continuous allocation formula.
        outer_ratios=relative_outer_ratios,
        statistics=statistics,
        history_capacities=history_capacities,
        history_groups=keys,
        group_capacities=group_capacities,
        rollout_budget=rollout_budget,
        minimum_fresh=minimum_fresh,
    )

    claims = []
    for key, allocation, is_terminal in zip(keys, allocations, terminal, strict=True):
        if is_terminal:
            claims.append(None)
            continue
        claim = store.freeze_claims([key], allocation.history_count)[0]
        if claim.count != allocation.history_count:
            raise RuntimeError("frozen replay inventory changed during allocation")
        claims.append(claim)

    fresh_requests: list[ReplaySampleRequest] = []
    fresh_ranges: list[tuple[int, int] | None] = []
    for candidate_index, (key, allocation, is_terminal) in enumerate(
        zip(keys, allocations, terminal, strict=True)
    ):
        if is_terminal:
            fresh_ranges.append(None)
            continue
        start = len(fresh_requests)
        fresh_requests.extend(
            build_fresh_replay_requests(
                key=key,
                count=allocation.fresh_count,
                rollout_length=rollout_length,
                seeds=seeds,
                step_index=step_index,
                candidate_index=candidate_index,
            )
        )
        fresh_ranges.append((start, len(fresh_requests)))
    batched_fresh = sample_replay_records(base_policy, fresh_requests, reward)

    candidates: list[DynamicCandidate] = []
    for candidate_index, (
        draw,
        key,
        allocation,
        claim,
        is_terminal,
        fresh_range,
    ) in enumerate(
        zip(draws, keys, allocations, claims, terminal, fresh_ranges, strict=True)
    ):
        if is_terminal:
            reward_value = float(reward(prompt, generated_prefix + draw.token_ids))
            estimate = ReplayWeightEstimate(
                log_weight=reward_value / config.reward_temperature,
                history_log_terms=(),
                fresh_log_terms=(reward_value / config.reward_temperature,),
                history_record_ids=(),
                behavior_counts=(),
            )
        else:
            assert claim is not None
            assert fresh_range is not None
            estimate = estimate_replay_weight(
                base_backend=base_backend,
                base_policy=base_policy,
                registry=registry,
                store=store,
                claim=claim,
                fresh_count=allocation.fresh_count,
                rollout_length=rollout_length,
                reward=reward,
                reward_temperature=config.reward_temperature,
                truncation=config.truncation,
                seeds=seeds,
                step_index=step_index,
                candidate_index=candidate_index,
                precomputed_fresh_records=batched_fresh[
                    fresh_range[0] : fresh_range[1]
                ],
            )
        candidates.append(
            DynamicCandidate(
                draw=draw,
                allocation=allocation,
                estimate=estimate,
                log_weight=draw.outer_log_ratio + estimate.log_weight,
            )
        )

    log_weights = np.asarray([candidate.log_weight for candidate in candidates])
    weights = np.exp(log_weights - float(np.max(log_weights)))
    probabilities = weights / weights.sum()
    selected_index = int(
        seeds.generator("dynamic_is", step_index, "select").choice(
            len(candidates), p=probabilities
        )
    )
    return DynamicISStep(
        generated_length_before=len(generated_prefix),
        candidates=tuple(candidates),
        selected_index=selected_index,
    )


def run_dynamic_is(
    base_backend: AutoregressiveBackend,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    prompt: TokenSequence,
    config: DynamicISConfig,
    reward: RewardFunction,
    reward_version: str,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    auxiliary_proposal: CandidateProposalSource = None,
    statistics_provider: DesignStatisticsProvider = constant_design_statistics,
    design_prepare: DesignPreparation | None = None,
    rollout_budget_provider: RolloutBudgetProvider | None = None,
    reserve_policy: BehaviorPolicy | None = None,
) -> DynamicISResult:
    base_sampling = base_sampling or SamplingConfig()
    _validate_base_sampling(base_sampling)
    base_policy = BehaviorPolicy.for_backend(base_backend, base_sampling, label="base")
    registry.register(base_policy)
    reserve_policy = reserve_policy or base_policy
    registry.register(reserve_policy)
    generated: list[int] = []
    steps: list[DynamicISStep] = []
    reserve_written = 0
    step_index = 0
    while len(generated) < config.total_length:
        step = dynamic_is_step(
            base_backend=base_backend,
            registry=registry,
            store=store,
            prompt=prompt,
            generated_prefix=tuple(generated),
            config=config,
            base_sampling=base_sampling,
            reward=reward,
            reward_version=reward_version,
            seeds=seeds,
            step_index=step_index,
            auxiliary_proposal=auxiliary_proposal,
            statistics_provider=statistics_provider,
            design_prepare=design_prepare,
            rollout_budget_provider=rollout_budget_provider,
        )
        generated.extend(step.selected.token_ids)
        steps.append(step)
        eos = base_sampling.eos_token_id
        if eos is not None and eos in step.selected.token_ids:
            generated = generated[: generated.index(eos) + 1]
            break
        reserve_written += write_reserve_records(
            base_backend=base_backend,
            base_sampling=base_sampling,
            reserve_policy=reserve_policy,
            store=store,
            prompt=prompt,
            generated_prefix=tuple(generated),
            config=config,
            reward=reward,
            reward_version=reward_version,
            seeds=seeds,
            step_index=step_index,
        )
        step_index += 1
    return DynamicISResult(
        prompt=prompt,
        token_ids=tuple(generated),
        steps=tuple(steps),
        reserve_records_written=reserve_written,
    )
