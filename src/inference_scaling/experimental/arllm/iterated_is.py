"""Finite-pool iterated SIR for conditional autoregressive generation.

Each extended state contains a base-model candidate block and all completion
rollouts used to estimate its conditional target weight.  The previous state is
inserted unchanged into the next pool, so its rollout estimate is reused
without being treated as a fresh independent observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_scaling.arllm.algorithms.conditional_is import (
    ConditionalCandidate,
    RewardBatchFunction,
    RewardFunction,
    _sample_candidates,
    _validate_base_sampling,
    _validate_rollout_sampling,
    estimate_conditional_weights,
)
from inference_scaling.arllm.config import (
    IteratedConditionalISConfig,
    SamplingConfig,
)
from inference_scaling.arllm.types import AutoregressiveBackend, TokenSequence
from inference_scaling.experimental.shared.iterated_sir import (
    IteratedSIRTransition,
    iterated_sir_transition,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.stepwise import StepwiseCandidate


@dataclass(frozen=True, slots=True)
class IteratedConditionalISStep:
    generated_length_before: int
    evaluated_candidates: tuple[ConditionalCandidate, ...]
    transitions: tuple[IteratedSIRTransition[ConditionalCandidate], ...]

    def __post_init__(self) -> None:
        if not self.evaluated_candidates:
            raise ValueError("an iterated conditional-IS step requires candidates")
        if not self.transitions:
            raise ValueError("an iterated conditional-IS step requires transitions")

    @property
    def candidates(self) -> tuple[ConditionalCandidate, ...]:
        """All distinct candidate-rollout states evaluated for this block."""

        return self.evaluated_candidates

    @property
    def selected(self) -> ConditionalCandidate:
        return self.transitions[-1].selected.value

    @property
    def selected_index(self) -> int:
        """Index in the evaluated states, including equal-valued distinct draws."""
        return next(
            index for index, candidate in enumerate(self.evaluated_candidates)
            if candidate is self.selected
        )

    @property
    def retained_previous_updates(self) -> int:
        return sum(transition.retained_previous for transition in self.transitions)


@dataclass(frozen=True, slots=True)
class IteratedConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[IteratedConditionalISStep, ...]

    @property
    def fresh_candidate_evaluations(self) -> int:
        return sum(len(step.evaluated_candidates) for step in self.steps)

    @property
    def reused_pool_entries(self) -> int:
        return sum(len(step.transitions) for step in self.steps)


def iterated_conditional_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: IteratedConditionalISConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
) -> IteratedConditionalISStep:
    """Draw all fresh states in one batch, then apply finite-pool i-SIR updates.

    The reward must be a fixed pointwise function of a completed sequence.  A
    batched callback may vectorize that same function.  Batch-normalized or
    history-dependent rewards change the target while the chain runs and
    therefore do not satisfy the i-SIR invariance argument.
    """

    _validate_base_sampling(base_sampling)
    _validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")
    remaining = config.total_length - len(generated_prefix)
    if remaining <= 0:
        raise ValueError("generated prefix has already reached total_length")
    block_length = min(config.block_size, remaining)
    candidates = _sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.fresh_candidate_evaluations,
        block_length,
        base_sampling,
        seeds,
        step_index,
    )
    evaluated = estimate_conditional_weights(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        generated_prefix=generated_prefix,
        candidates=candidates,
        rollout_length=max(0, remaining - block_length),
        rollout_count=config.rollout_count,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward_temperature=config.reward_temperature,
        importance_log_ratio_clip=config.importance_log_ratio_clip,
        apply_importance_correction=config.apply_importance_correction,
        reward=reward,
        reward_batch=reward_batch,
        seeds=seeds,
        step_index=step_index,
    )

    current = StepwiseCandidate(evaluated[0], evaluated[0].log_weight)
    transitions: list[IteratedSIRTransition[ConditionalCandidate]] = []
    offset = 1
    for update_index in range(config.updates):
        fresh_values = evaluated[offset : offset + config.pool_size - 1]
        if len(fresh_values) != config.pool_size - 1:
            raise RuntimeError("candidate grouping does not match the i-SIR configuration")
        fresh = tuple(
            StepwiseCandidate(candidate, candidate.log_weight)
            for candidate in fresh_values
        )
        transition = iterated_sir_transition(
            current,
            fresh,
            rng=seeds.generator(
                "iterated_conditional_is",
                step_index,
                update_index,
                "select",
            ),
        )
        transitions.append(transition)
        current = transition.selected
        offset += config.pool_size - 1
    if offset != len(evaluated):
        raise RuntimeError("not every evaluated i-SIR state was assigned to an update")
    return IteratedConditionalISStep(
        generated_length_before=len(generated_prefix),
        evaluated_candidates=evaluated,
        transitions=tuple(transitions),
    )


def run_iterated_conditional_is(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: IteratedConditionalISConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
) -> IteratedConditionalISResult:
    """Generate a sequence with i-SIR at every autoregressive block."""

    base_sampling = base_sampling or SamplingConfig()
    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    _validate_base_sampling(base_sampling)
    _validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")

    generated: TokenSequence = ()
    steps: list[IteratedConditionalISStep] = []
    while len(generated) < config.total_length:
        eos = base_sampling.eos_token_id
        if eos is not None and eos in generated:
            break
        step = iterated_conditional_is_step(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=generated,
            config=config,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward=reward,
            reward_batch=reward_batch,
            seeds=seeds,
            step_index=len(steps),
        )
        generated += step.selected.token_ids
        if eos is not None and eos in generated:
            generated = generated[: generated.index(eos) + 1]
        steps.append(step)
    return IteratedConditionalISResult(prompt, generated, tuple(steps))


__all__ = [
    "IteratedConditionalISResult",
    "IteratedConditionalISStep",
    "iterated_conditional_is_step",
    "run_iterated_conditional_is",
]
