"""Conditional importance sampling.

Candidate blocks are always sampled from the base model in this module.  A
completion may be sampled on-policy or from a full-support off-policy proposal.
Only the completion suffix receives the ``p_base / q`` correction.  This is the
finite-candidate, finite-rollout sampling-importance-resampling algorithm used as
the foundation for the replay extensions.  Optional symmetric clipping of the
sequence log-ratio is recorded explicitly; it is a biased variance-control
setting, while the default ``None`` retains the exact importance ratio.  An
explicit uncorrected ablation skips target-model rescoring and instead estimates
each candidate's future reward weighting under the rollout proposal itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log

from inference_scaling.arllm.config import ConditionalISConfig, SamplingConfig
from inference_scaling.shared.importance import (
    MonteCarloRolloutWeightProvider,
    RolloutObservation,
    logmeanexp,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.stepwise import (
    StepwiseCandidate,
    categorical_index_from_uniform,
    normalize_log_weights,
    run_stepwise_generation,
    stepwise_generation_step,
)
from inference_scaling.shared.verifier import TokenBatchReward, TokenReward
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

RewardFunction = TokenReward
RewardBatchFunction = TokenBatchReward


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    base_logprob: float | None
    proposal_logprob: float
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float | None
    log_weight: float
    proposal_model_id: str
    proposal_policy_id: str


@dataclass(frozen=True, slots=True)
class ConditionalCandidate:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    rollouts: tuple[RolloutEvaluation, ...]
    log_weight: float
    planned_rollout_count: int = 0
    log_weight_lower_bound: float | None = None
    log_weight_upper_bound: float | None = None


@dataclass(frozen=True, slots=True)
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate, ...]
    selected_index: int
    rollout_evaluations_planned: int = 0
    rollout_evaluations_performed: int = 0
    rollout_evaluations_skipped: int = 0
    rollout_evaluation_batches: int = 0
    exact_early_stop: bool = False
    selection_invariant_verified: bool = False

    @property
    def selected(self) -> ConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]


def _validate_base_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use a full-support autoregressive policy: "
            "top_p=1 and top_k=None"
        )


def _validate_rollout_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "off-policy IS requires proposal support wherever the base weighted target is positive; "
            "hard top-k/top-p truncation is not accepted"
        )


def _score_samples(
    base_backend: AutoregressiveBackend,
    prefixes: Sequence[TokenSequence],
    samples: Sequence[SequenceSample],
    base_sampling: SamplingConfig,
) -> list[float]:
    requests = [
        ScoreRequest(prefix, (sample.token_ids,), base_sampling)
        for prefix, sample in zip(prefixes, samples, strict=True)
    ]
    token_scores = base_backend.score_batch(requests)
    if len(token_scores) != len(samples):
        raise RuntimeError("backend returned an invalid number of base scores")
    totals: list[float] = []
    for sample, scores in zip(samples, token_scores, strict=True):
        if len(scores) != len(sample.token_ids):
            raise RuntimeError("backend returned an invalid base token score shape")
        total = float(sum(scores))
        if not isfinite(total):
            raise ValueError(
                "rollout proposal generated a completion outside base-model support"
            )
        totals.append(total)
    return totals


def _sample_candidates(
    base_backend: AutoregressiveBackend,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
) -> list[SequenceSample]:
    requests = [
        GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=sampling,
            seed=seeds.derive(
                "conditional_is", step_index, "candidate", candidate_index
            ),
            request_id=f"conditional-is:step:{step_index}:candidate:{candidate_index}",
        )
        for candidate_index in range(count)
    ]
    candidates = base_backend.sample_batch(requests)
    if len(candidates) != count:
        raise RuntimeError("backend returned an invalid number of candidates")
    for candidate in candidates:
        if not candidate.token_ids:
            raise RuntimeError("a candidate block must contain at least one token")
        if (
            candidate.model_id != base_backend.model_id
            or candidate.policy_id != sampling.policy_id
        ):
            raise RuntimeError(
                "candidate was not sampled and scored by the requested base policy"
            )
    return candidates


def estimate_conditional_weights(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[SequenceSample],
    rollout_length: int,
    rollout_count: int,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward_temperature: float,
    importance_log_ratio_clip: float | None,
    apply_importance_correction: bool,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
    rollout_design: str = "iid",
    rollout_index_offset: int = 0,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's conditional weight with on/off-policy rollouts."""

    _validate_rollout_sampling(rollout_sampling)
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    if rollout_design not in {
        "iid",
        "scrambled_sobol",
        "arithmetic_lattice",
    }:
        raise ValueError("unknown rollout_design")
    if rollout_index_offset < 0:
        raise ValueError("rollout_index_offset must be non-negative")
    if rollout_index_offset and rollout_design != "iid":
        raise ValueError("staged rollout offsets currently require iid rollouts")
    if rollout_design != "iid" and reward_batch is not None:
        raise ValueError(
            "randomized QMC rollouts require a fixed pointwise reward; "
            "batch-coupled rewards change when rollout dependence changes"
        )

    requests: list[GenerationRequest] = []
    request_candidates: list[int] = []
    rollout_prefixes: list[TokenSequence] = []
    terminal_candidates: set[int] = set()
    eos = rollout_sampling.eos_token_id

    for candidate_index, candidate in enumerate(candidates):
        full_generated_candidate = generated_prefix + candidate.token_ids
        terminal = rollout_length == 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        if terminal:
            terminal_candidates.add(candidate_index)
            continue
        rollout_prefix = prompt + full_generated_candidate
        if rollout_design == "scrambled_sobol":
            from inference_scaling.experimental.shared.rqmc import (
                scrambled_sobol_uniforms,
            )

            token_uniforms = scrambled_sobol_uniforms(
                rollout_count,
                rollout_length,
                seed=seeds.derive(
                    "conditional_is",
                    step_index,
                    "candidate",
                    candidate_index,
                    "scrambled_sobol",
                ),
            )
        else:
            token_uniforms = (None,) * rollout_count
        if rollout_design == "arithmetic_lattice":
            from inference_scaling.experimental.shared.rqmc import (
                randomized_lattice_uniforms,
            )

            arithmetic_uniforms = randomized_lattice_uniforms(
                rollout_count,
                seed=seeds.derive(
                    "conditional_is",
                    step_index,
                    "candidate",
                    candidate_index,
                    "arithmetic_lattice",
                ),
            )
        else:
            arithmetic_uniforms = (None,) * rollout_count
        for rollout_index in range(rollout_count):
            global_rollout_index = rollout_index_offset + rollout_index
            requests.append(
                GenerationRequest(
                    prefix=rollout_prefix,
                    max_new_tokens=rollout_length,
                    sampling=rollout_sampling,
                    seed=seeds.derive(
                        "conditional_is",
                        step_index,
                        "candidate",
                        candidate_index,
                        "rollout",
                        global_rollout_index,
                    ),
                    request_id=(
                        "conditional-is:"
                        f"step:{step_index}:candidate:{candidate_index}:"
                        f"rollout:{global_rollout_index}"
                    ),
                    uniforms=token_uniforms[rollout_index],
                    arithmetic_uniform=arithmetic_uniforms[rollout_index],
                )
            )
            request_candidates.append(candidate_index)
            rollout_prefixes.append(rollout_prefix)

    samples = rollout_backend.sample_batch(requests) if requests else []
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of rollouts")
    if rollout_backend is not base_backend:
        observe = getattr(base_backend, "observe_draft_samples", None)
        if callable(observe):
            observe(samples)
    rollout_is_base_policy = (
        rollout_backend.model_id == base_backend.model_id
        and rollout_sampling == base_sampling
    )
    if rollout_is_base_policy:
        base_totals: list[float | None] = [sample.logprob for sample in samples]
    elif apply_importance_correction:
        base_totals = (
            _score_samples(
                base_backend,
                rollout_prefixes,
                samples,
                base_sampling,
            )
            if samples
            else []
        )
    else:
        # This is a deliberate biased ablation, not an IS estimate of the base
        # continuation distribution.  Keep the score absent so diagnostics and
        # backend accounting cannot mistake it for an evaluated zero log-ratio.
        base_totals = [None for _ in samples]

    pending_by_candidate: list[
        list[tuple[TokenSequence, float, float, str, str, TokenSequence]]
    ] = [[] for _ in candidates]
    for candidate_index in terminal_candidates:
        generated = generated_prefix + candidates[candidate_index].token_ids
        pending_by_candidate[candidate_index].append(
            (
                (),
                0.0,
                0.0,
                rollout_backend.model_id,
                rollout_sampling.policy_id,
                generated,
            )
        )
    for candidate_index, sample, base_logprob in zip(
        request_candidates, samples, base_totals, strict=True
    ):
        generated = (
            generated_prefix + candidates[candidate_index].token_ids + sample.token_ids
        )
        proposal_logprob = sample.logprob
        pending_by_candidate[candidate_index].append(
            (
                sample.token_ids,
                base_logprob,
                proposal_logprob,
                sample.model_id,
                sample.policy_id,
                generated,
            )
        )

    pending = [item for group in pending_by_candidate for item in group]
    generated_sequences = [item[-1] for item in pending]
    if reward_batch is not None:
        rewards = tuple(
            float(value) for value in reward_batch(prompt, generated_sequences)
        )
        if len(rewards) != len(pending):
            raise ValueError("reward_batch returned an invalid number of rewards")
    else:
        assert reward is not None
        rewards = tuple(
            float(reward(prompt, generated)) for generated in generated_sequences
        )
    if any(not isfinite(value) for value in rewards):
        raise ValueError("reward must be finite")

    importance_weights = MonteCarloRolloutWeightProvider[
        tuple[TokenSequence, str, str]
    ](
        reward_temperature=reward_temperature,
        correction="importance",
        log_ratio_clip=importance_log_ratio_clip,
    )
    reward_only_weights = MonteCarloRolloutWeightProvider[
        tuple[TokenSequence, str, str]
    ](
        reward_temperature=reward_temperature,
        correction="none",
    )
    by_candidate: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    reward_index = 0
    for candidate_index, group in enumerate(pending_by_candidate):
        for token_ids, base_logprob, proposal_logprob, model_id, policy_id, _ in group:
            reward_value = rewards[reward_index]
            reward_index += 1
            observation = RolloutObservation(
                reward=reward_value,
                target_logprob=base_logprob,
                proposal_logprob=proposal_logprob,
                payload=(token_ids, model_id, policy_id),
            )
            weighted = (
                importance_weights.weight(observation)
                if base_logprob is not None
                else reward_only_weights.weight(observation)
            )
            by_candidate[candidate_index].append(
                RolloutEvaluation(
                    token_ids=token_ids,
                    reward=reward_value,
                    base_logprob=base_logprob,
                    proposal_logprob=proposal_logprob,
                    raw_log_importance_ratio=weighted.raw_log_importance_ratio,
                    applied_log_importance_ratio=weighted.applied_log_importance_ratio,
                    log_weight=weighted.log_weight,
                    proposal_model_id=model_id,
                    proposal_policy_id=policy_id,
                )
            )

    evaluated: list[ConditionalCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations = by_candidate[candidate_index]
        if not evaluations:
            raise RuntimeError(
                "each candidate must have at least one weight contribution"
            )
        candidate_log_weight = logmeanexp([item.log_weight for item in evaluations])
        evaluated.append(
            ConditionalCandidate(
                token_ids=candidate.token_ids,
                base_token_logprobs=candidate.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=candidate_log_weight,
                planned_rollout_count=len(evaluations),
                log_weight_lower_bound=candidate_log_weight,
                log_weight_upper_bound=candidate_log_weight,
            )
        )
    return tuple(evaluated)


class AutoregressiveStepwiseAdapter:
    """Expose conditional AR generation through the common stepwise protocol."""

    def __init__(
        self,
        *,
        base_backend: AutoregressiveBackend,
        rollout_backend: AutoregressiveBackend,
        prompt: TokenSequence,
        config: ConditionalISConfig,
        base_sampling: SamplingConfig,
        rollout_sampling: SamplingConfig,
        reward: RewardFunction | None,
        reward_batch: RewardBatchFunction | None = None,
    ) -> None:
        self.base_backend = base_backend
        self.rollout_backend = rollout_backend
        self.prompt = prompt
        self.config = config
        self.base_sampling = base_sampling
        self.rollout_sampling = rollout_sampling
        self.reward = reward
        self.reward_batch = reward_batch

    @property
    def initial_state(self) -> TokenSequence:
        return ()

    def is_terminal(self, state: TokenSequence) -> bool:
        eos = self.base_sampling.eos_token_id
        return len(state) >= self.config.total_length or (
            eos is not None and eos in state
        )

    def propose(
        self,
        state: TokenSequence,
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[SequenceSample]:
        _validate_base_sampling(self.base_sampling)
        remaining = self.config.total_length - len(state)
        if remaining <= 0:
            raise ValueError("generated prefix has already reached total_length")
        return _sample_candidates(
            self.base_backend,
            self.prompt + state,
            self.config.candidate_count,
            min(self.config.block_size, remaining),
            self.base_sampling,
            seeds,
            step_index,
        )

    def evaluate(
        self,
        state: TokenSequence,
        proposals: Sequence[SequenceSample],
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[StepwiseCandidate[ConditionalCandidate]]:
        remaining = self.config.total_length - len(state)
        candidate_length = len(proposals[0].token_ids)
        evaluated = estimate_conditional_weights(
            base_backend=self.base_backend,
            rollout_backend=self.rollout_backend,
            prompt=self.prompt,
            generated_prefix=state,
            candidates=proposals,
            rollout_length=max(0, remaining - candidate_length),
            rollout_count=self.config.rollout_count,
            base_sampling=self.base_sampling,
            rollout_sampling=self.rollout_sampling,
            reward_temperature=self.config.reward_temperature,
            importance_log_ratio_clip=self.config.importance_log_ratio_clip,
            apply_importance_correction=self.config.apply_importance_correction,
            reward=self.reward,
            seeds=seeds,
            step_index=step_index,
            reward_batch=self.reward_batch,
            rollout_design=self.config.rollout_design,
        )
        return tuple(
            StepwiseCandidate(candidate, candidate.log_weight)
            for candidate in evaluated
        )

    def advance(
        self,
        state: TokenSequence,
        selected: ConditionalCandidate,
        step_index: int,
    ) -> TokenSequence:
        del step_index
        generated = state + selected.token_ids
        eos = self.base_sampling.eos_token_id
        if eos is not None and eos in generated:
            generated = generated[: generated.index(eos) + 1]
        return generated


def _bounded_conditional_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: ConditionalISConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None,
) -> ConditionalISStep:
    """Evaluate rollout batches until the fixed categorical choice is known."""

    if reward is None or reward_batch is not None:
        raise ValueError(
            "exact rollout early stopping requires a fixed pointwise reward"
        )
    if config.rollout_log_weight_bounds is None:
        raise ValueError("exact rollout early stopping requires log-weight bounds")
    if config.rollout_design != "iid":
        raise ValueError("exact rollout early stopping currently requires iid rollouts")
    _validate_base_sampling(base_sampling)
    remaining_length = config.total_length - len(generated_prefix)
    if remaining_length <= 0:
        raise ValueError("generated prefix has already reached total_length")
    candidate_length = min(config.block_size, remaining_length)
    proposals = _sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.candidate_count,
        candidate_length,
        base_sampling,
        seeds,
        step_index,
    )
    rollout_length = max(0, remaining_length - len(proposals[0].token_ids))
    eos = rollout_sampling.eos_token_id
    terminal = tuple(
        rollout_length == 0 or (eos is not None and proposal.token_ids[-1] == eos)
        for proposal in proposals
    )
    planned = tuple(
        1 if is_terminal else config.rollout_count for is_terminal in terminal
    )
    planned_total = sum(planned)
    selection_uniform = float(
        seeds.generator("conditional_is", step_index, "select").random()
    )
    lower_log_weight, upper_log_weight = config.rollout_log_weight_bounds
    try:
        minimum_contribution = exp(lower_log_weight)
        maximum_contribution = exp(upper_log_weight)
    except OverflowError as error:
        raise ValueError("rollout log-weight bounds cannot be exponentiated") from error
    if (
        not isfinite(minimum_contribution)
        or not isfinite(maximum_contribution)
        or minimum_contribution <= 0.0
    ):
        raise ValueError(
            "rollout log-weight bounds must map to finite positive weights"
        )

    collected: list[list[RolloutEvaluation]] = [[] for _ in proposals]
    lower_candidate_weights: list[float] = []
    upper_candidate_weights: list[float] = []
    invariant_index: int | None = None
    rollout_offset = 0
    evaluation_batches = 0
    while rollout_offset < config.rollout_count:
        batch_size = min(
            config.rollout_evaluation_batch_size,
            config.rollout_count - rollout_offset,
        )
        batch = estimate_conditional_weights(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=generated_prefix,
            candidates=proposals,
            rollout_length=rollout_length,
            rollout_count=batch_size,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward_temperature=config.reward_temperature,
            importance_log_ratio_clip=config.importance_log_ratio_clip,
            apply_importance_correction=config.apply_importance_correction,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            rollout_design="iid",
            rollout_index_offset=rollout_offset,
        )
        evaluation_batches += 1
        for candidate_index, evaluated in enumerate(batch):
            if terminal[candidate_index]:
                if not collected[candidate_index]:
                    collected[candidate_index].append(evaluated.rollouts[0])
                continue
            for rollout in evaluated.rollouts:
                if not lower_log_weight <= rollout.log_weight <= upper_log_weight:
                    raise ValueError(
                        "observed rollout log-weight lies outside the declared bounds"
                    )
                collected[candidate_index].append(rollout)
        rollout_offset += batch_size

        lower_candidate_weights = []
        upper_candidate_weights = []
        for candidate_index, evaluations in enumerate(collected):
            contributions = [exp(item.log_weight) for item in evaluations]
            if terminal[candidate_index]:
                exact_weight = contributions[0]
                lower_candidate_weights.append(exact_weight)
                upper_candidate_weights.append(exact_weight)
                continue
            unseen = config.rollout_count - len(evaluations)
            lower_candidate_weights.append(
                (sum(contributions) + unseen * minimum_contribution)
                / config.rollout_count
            )
            upper_candidate_weights.append(
                (sum(contributions) + unseen * maximum_contribution)
                / config.rollout_count
            )
        from inference_scaling.experimental.shared.bounded_selection import (
            invariant_categorical_index,
        )

        invariant_index = invariant_categorical_index(
            lower_candidate_weights,
            upper_candidate_weights,
            uniform=selection_uniform,
        )
        if invariant_index is not None:
            break

    evaluated_candidates: list[ConditionalCandidate] = []
    for candidate_index, proposal in enumerate(proposals):
        evaluations = collected[candidate_index]
        if not evaluations:
            raise RuntimeError("bounded evaluation omitted a candidate")
        lower_weight = lower_candidate_weights[candidate_index]
        upper_weight = upper_candidate_weights[candidate_index]
        representative_weight = (lower_weight + upper_weight) / 2.0
        evaluated_candidates.append(
            ConditionalCandidate(
                token_ids=proposal.token_ids,
                base_token_logprobs=proposal.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=log(representative_weight),
                planned_rollout_count=planned[candidate_index],
                log_weight_lower_bound=log(lower_weight),
                log_weight_upper_bound=log(upper_weight),
            )
        )
    probabilities = normalize_log_weights(
        [candidate.log_weight for candidate in evaluated_candidates]
    )
    selected_index = categorical_index_from_uniform(
        probabilities,
        selection_uniform,
    )
    if invariant_index is not None and selected_index != invariant_index:
        raise RuntimeError(
            "bounded categorical proof disagrees with representative weights"
        )
    performed_total = sum(len(candidate.rollouts) for candidate in evaluated_candidates)
    skipped_total = planned_total - performed_total
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=tuple(evaluated_candidates),
        selected_index=selected_index,
        rollout_evaluations_planned=planned_total,
        rollout_evaluations_performed=performed_total,
        rollout_evaluations_skipped=skipped_total,
        rollout_evaluation_batches=evaluation_batches,
        exact_early_stop=skipped_total > 0,
        selection_invariant_verified=skipped_total > 0 and invariant_index is not None,
    )


def conditional_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: ConditionalISConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
) -> ConditionalISStep:
    if config.exact_rollout_early_stop:
        return _bounded_conditional_is_step(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=generated_prefix,
            config=config,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            reward_batch=reward_batch,
        )
    adapter = AutoregressiveStepwiseAdapter(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward=reward,
        reward_batch=reward_batch,
    )
    selection = stepwise_generation_step(
        adapter,
        generated_prefix,
        step_index,
        seeds,
        selection_namespace=("conditional_is",),
    )
    evaluated_candidates = tuple(candidate.value for candidate in selection.candidates)
    performed = sum(len(candidate.rollouts) for candidate in evaluated_candidates)
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=evaluated_candidates,
        selected_index=selection.selected_index,
        rollout_evaluations_planned=performed,
        rollout_evaluations_performed=performed,
        rollout_evaluation_batches=1,
    )


def run_conditional_is(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalISConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
) -> ConditionalISResult:
    """Generate a sequence by repeatedly applying finite conditional-IS steps."""

    base_sampling = base_sampling or SamplingConfig()
    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    _validate_base_sampling(base_sampling)
    _validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")

    if config.exact_rollout_early_stop:
        generated: TokenSequence = ()
        steps: list[ConditionalISStep] = []
        step_index = 0
        eos = base_sampling.eos_token_id
        while len(generated) < config.total_length and (
            eos is None or eos not in generated
        ):
            step = conditional_is_step(
                base_backend=base_backend,
                rollout_backend=rollout_backend,
                prompt=prompt,
                generated_prefix=generated,
                config=config,
                base_sampling=base_sampling,
                rollout_sampling=rollout_sampling,
                reward=reward,
                seeds=seeds,
                step_index=step_index,
                reward_batch=reward_batch,
            )
            generated += step.selected.token_ids
            if eos is not None and eos in generated:
                generated = generated[: generated.index(eos) + 1]
            steps.append(step)
            step_index += 1
        return ConditionalISResult(
            prompt=prompt,
            token_ids=generated,
            steps=tuple(steps),
        )

    adapter = AutoregressiveStepwiseAdapter(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward=reward,
        reward_batch=reward_batch,
    )
    generic = run_stepwise_generation(
        adapter,
        seeds,
        selection_namespace=("conditional_is",),
    )
    steps: list[ConditionalISStep] = []
    for step in generic.steps:
        candidates = tuple(candidate.value for candidate in step.candidates)
        performed = sum(len(candidate.rollouts) for candidate in candidates)
        steps.append(
            ConditionalISStep(
                generated_length_before=len(step.state_before),
                candidates=candidates,
                selected_index=step.selected_index,
                rollout_evaluations_planned=performed,
                rollout_evaluations_performed=performed,
                rollout_evaluation_batches=1,
            )
        )
    return ConditionalISResult(
        prompt=prompt,
        token_ids=generic.final_state,
        steps=tuple(steps),
    )
