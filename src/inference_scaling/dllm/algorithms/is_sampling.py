"""Reward reweighting and conditional importance sampling for masked dLLMs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np

from inference_scaling.dllm.config import (
    DiffusionISConfig,
    DiffusionSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTrajectoryScoreRequest,
)
from inference_scaling.shared.importance import (
    MonteCarloRolloutWeightProvider,
    RolloutObservation,
)
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.stepwise import (
    StepwiseCandidate,
    normalize_log_weights,
    run_stepwise_generation,
)
from inference_scaling.shared.types import TokenSequence
from inference_scaling.shared.verifier import TokenBatchReward, TokenReward

DiffusionRewardFunction = TokenReward
DiffusionRewardBatchFunction = TokenBatchReward


@dataclass(frozen=True, slots=True)
class DiffusionSIRItem:
    sample: DiffusionSample
    reward: float
    target_trajectory_logprob: float | None
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float | None
    log_weight: float


@dataclass(frozen=True, slots=True)
class DiffusionSIRResult:
    items: tuple[DiffusionSIRItem, ...]
    probabilities: tuple[float, ...]
    selected_index: int
    effective_sample_size: float

    @property
    def selected(self) -> DiffusionSIRItem:
        return self.items[self.selected_index]


def resample_diffusion_candidates(
    *,
    samples: Sequence[DiffusionSample],
    rewards: Sequence[float],
    reward_temperature: float,
    rng: np.random.Generator,
    target_trajectory_logprobs: Sequence[float] | None = None,
    importance_log_ratio_clip: float | None = None,
) -> DiffusionSIRResult:
    """Sample once from reward-reweighted on-policy or trajectory-IS weights."""

    if len(samples) != len(rewards) or not samples:
        raise ValueError("samples and rewards must have the same positive length")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if target_trajectory_logprobs is not None and len(target_trajectory_logprobs) != len(samples):
        raise ValueError("one target trajectory score is required per sample")

    provider = MonteCarloRolloutWeightProvider[DiffusionSample](
        reward_temperature=reward_temperature,
        correction=("importance" if target_trajectory_logprobs is not None else "none"),
        log_ratio_clip=(
            importance_log_ratio_clip
            if target_trajectory_logprobs is not None
            else None
        ),
    )
    items: list[DiffusionSIRItem] = []
    for index, (sample, reward_value) in enumerate(zip(samples, rewards, strict=True)):
        target_logprob: float | None = None
        if target_trajectory_logprobs is not None:
            if sample.trajectory_logprob is None:
                raise ValueError("off-policy IS requires an exact proposal trajectory density")
            target_logprob = float(target_trajectory_logprobs[index])
        weighted = provider.weight(
            RolloutObservation(
                reward=float(reward_value),
                target_logprob=target_logprob,
                proposal_logprob=sample.trajectory_logprob,
                payload=sample,
            )
        )
        items.append(
            DiffusionSIRItem(
                sample=sample,
                reward=float(reward_value),
                target_trajectory_logprob=target_logprob,
                raw_log_importance_ratio=weighted.raw_log_importance_ratio,
                applied_log_importance_ratio=weighted.applied_log_importance_ratio,
                log_weight=weighted.log_weight,
            )
        )

    log_weights = [item.log_weight for item in items]
    probabilities = normalize_log_weights(log_weights)
    selected_index = int(rng.choice(len(items), p=probabilities))
    return DiffusionSIRResult(
        items=tuple(items),
        probabilities=probabilities,
        selected_index=selected_index,
        effective_sample_size=importance_effective_sample_size(log_weights),
    )


@dataclass(frozen=True, slots=True)
class DiffusionRolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    proposal_trajectory_logprob: float | None
    target_trajectory_logprob: float | None
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float | None
    log_weight: float
    proposal_model_id: str
    proposal_policy_id: str


@dataclass(frozen=True, slots=True)
class DiffusionConditionalCandidate:
    token_ids: TokenSequence
    rollouts: tuple[DiffusionRolloutEvaluation, ...]
    log_weight: float


@dataclass(frozen=True, slots=True)
class DiffusionConditionalISStep:
    generated_length_before: int
    candidates: tuple[DiffusionConditionalCandidate, ...]
    probabilities: tuple[float, ...]
    selected_index: int

    @property
    def selected(self) -> DiffusionConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class DiffusionConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[DiffusionConditionalISStep, ...]


def _evaluate_rewards(
    *,
    prompt: TokenSequence,
    continuations: Sequence[TokenSequence],
    reward: DiffusionRewardFunction | None,
    reward_batch: DiffusionRewardBatchFunction | None,
) -> list[float]:
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    if reward_batch is not None:
        values = list(reward_batch(prompt, continuations))
    else:
        assert reward is not None
        values = [reward(prompt, continuation) for continuation in continuations]
    if len(values) != len(continuations):
        raise RuntimeError("reward evaluator returned an invalid number of values")
    return [float(value) for value in values]


def _needs_trajectory_correction(
    *,
    rollout_backend: DiffusionBackend,
    rollout_sampling: DiffusionSamplingConfig,
    target_backend: DiffusionBackend | None,
    target_sampling: DiffusionSamplingConfig | None,
    apply_importance_correction: bool,
) -> bool:
    if not apply_importance_correction:
        return False
    if target_backend is None or target_sampling is None:
        return False
    return not (
        rollout_backend.model_id == target_backend.model_id
        and rollout_sampling.policy_id == target_sampling.policy_id
    )


class DiffusionStepwiseAdapter:
    """Expose conditional diffusion generation through the common stepwise protocol."""

    def __init__(
        self,
        *,
        base_backend: DiffusionBackend,
        prompt: TokenSequence,
        config: DiffusionISConfig,
        base_sampling: DiffusionSamplingConfig,
        reward: DiffusionRewardFunction | None,
        rollout_backend: DiffusionBackend,
        rollout_sampling: DiffusionSamplingConfig,
        target_rollout_backend: DiffusionBackend,
        target_rollout_sampling: DiffusionSamplingConfig,
        apply_importance_correction: bool,
        reward_batch: DiffusionRewardBatchFunction | None,
    ) -> None:
        self.base_backend = base_backend
        self.prompt = prompt
        self.config = config
        self.base_sampling = base_sampling
        self.reward = reward
        self.rollout_backend = rollout_backend
        self.rollout_sampling = rollout_sampling
        self.target_rollout_backend = target_rollout_backend
        self.target_rollout_sampling = target_rollout_sampling
        self.apply_importance_correction = apply_importance_correction
        self.reward_batch = reward_batch
        self.stage_lengths = diffusion_decision_stage_lengths(
            prompt_length=len(prompt),
            total_length=config.total_length,
            decision_block_size=config.block_size,
            sampling=base_sampling,
        )
        self.same_rollout_policy = (
            rollout_backend.model_id == target_rollout_backend.model_id
            and rollout_sampling.policy_id == target_rollout_sampling.policy_id
        )
        self.needs_correction = _needs_trajectory_correction(
            rollout_backend=rollout_backend,
            rollout_sampling=rollout_sampling,
            target_backend=target_rollout_backend,
            target_sampling=target_rollout_sampling,
            apply_importance_correction=apply_importance_correction,
        )
        if self.needs_correction:
            if not rollout_sampling.has_exact_trajectory_density:
                raise ValueError(
                    "off-policy dLLM IS requires random remasking and positive temperature"
                )
            if not target_rollout_sampling.has_exact_trajectory_density:
                raise ValueError(
                    "the target dLLM policy must define an exact trajectory density"
                )
            if (
                rollout_sampling.block_length
                != target_rollout_sampling.block_length
                or rollout_sampling.steps_per_block
                != target_rollout_sampling.steps_per_block
            ):
                raise ValueError("proposal and target trajectory schedules must match")

    @property
    def initial_state(self) -> TokenSequence:
        return ()

    def is_terminal(self, state: TokenSequence) -> bool:
        return len(state) >= self.config.total_length

    def propose(
        self,
        state: TokenSequence,
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[DiffusionSample]:
        candidate_length = self.stage_lengths[step_index]
        requests = [
            DiffusionGenerationRequest(
                prefix=self.prompt + state,
                generation_length=candidate_length,
                sampling=self.base_sampling,
                seed=seeds.derive(
                    "dllm-is", step_index, "candidate", candidate_index
                ),
                request_id=f"dllm-is:step:{step_index}:candidate:{candidate_index}",
            )
            for candidate_index in range(self.config.candidate_count)
        ]
        samples = self.base_backend.sample_batch(requests)
        if len(samples) != self.config.candidate_count:
            raise RuntimeError("backend returned an invalid number of dLLM candidates")
        return samples

    def _weight_provider(self) -> MonteCarloRolloutWeightProvider[tuple[TokenSequence, str, str]]:
        if self.needs_correction:
            correction = "importance"
        elif self.same_rollout_policy:
            correction = "identity"
        else:
            correction = "none"
        return MonteCarloRolloutWeightProvider(
            reward_temperature=self.config.reward_temperature,
            correction=correction,
            log_ratio_clip=(
                self.config.importance_log_ratio_clip
                if correction == "importance"
                else None
            ),
        )

    def evaluate(
        self,
        state: TokenSequence,
        proposals: Sequence[DiffusionSample],
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[StepwiseCandidate[DiffusionConditionalCandidate]]:
        candidate_length = self.stage_lengths[step_index]
        remaining = self.config.total_length - len(state) - candidate_length
        observations: list[
            list[RolloutObservation[tuple[TokenSequence, str, str]]]
        ] = [[] for _ in proposals]

        if remaining == 0:
            continuations = [state + sample.token_ids for sample in proposals]
            rewards = _evaluate_rewards(
                prompt=self.prompt,
                continuations=continuations,
                reward=self.reward,
                reward_batch=self.reward_batch,
            )
            terminal_weights = MonteCarloRolloutWeightProvider[
                tuple[TokenSequence, str, str]
            ](
                reward_temperature=self.config.reward_temperature,
                correction="identity",
            )
            for candidate_index, reward_value in enumerate(rewards):
                observation = RolloutObservation(
                    reward=reward_value,
                    target_logprob=0.0,
                    proposal_logprob=0.0,
                    payload=(
                        (),
                        self.base_backend.model_id,
                        self.base_sampling.policy_id,
                    ),
                )
                observations[candidate_index].append(observation)
            provider = terminal_weights
        else:
            requests: list[DiffusionGenerationRequest] = []
            owners: list[int] = []
            for candidate_index, candidate in enumerate(proposals):
                rollout_prefix = self.prompt + state + candidate.token_ids
                for rollout_index in range(self.config.rollout_count):
                    requests.append(
                        DiffusionGenerationRequest(
                            prefix=rollout_prefix,
                            generation_length=remaining,
                            sampling=self.rollout_sampling,
                            seed=seeds.derive(
                                "dllm-is",
                                step_index,
                                "rollout",
                                candidate_index,
                                rollout_index,
                            ),
                            request_id=(
                                f"dllm-is:step:{step_index}:candidate:{candidate_index}:"
                                f"rollout:{rollout_index}"
                            ),
                        )
                    )
                    owners.append(candidate_index)
            samples = self.rollout_backend.sample_batch(requests)
            if len(samples) != len(requests):
                raise RuntimeError("backend returned an invalid number of dLLM rollouts")
            target_scores: list[float] | None = None
            if self.needs_correction:
                target_scores = self.target_rollout_backend.score_trajectories(
                    [
                        DiffusionTrajectoryScoreRequest(
                            sample, self.target_rollout_sampling
                        )
                        for sample in samples
                    ]
                )
                if len(target_scores) != len(samples):
                    raise RuntimeError(
                        "backend returned an invalid number of trajectory scores"
                    )
            continuations = [
                state + proposals[owner].token_ids + sample.token_ids
                for owner, sample in zip(owners, samples, strict=True)
            ]
            rewards = _evaluate_rewards(
                prompt=self.prompt,
                continuations=continuations,
                reward=self.reward,
                reward_batch=self.reward_batch,
            )
            for rollout_index, (owner, sample, reward_value) in enumerate(
                zip(owners, samples, rewards, strict=True)
            ):
                if self.needs_correction and sample.trajectory_logprob is None:
                    raise RuntimeError("proposal omitted an exact trajectory score")
                observations[owner].append(
                    RolloutObservation(
                        reward=reward_value,
                        target_logprob=(
                            float(target_scores[rollout_index])
                            if target_scores is not None
                            else None
                        ),
                        proposal_logprob=sample.trajectory_logprob,
                        payload=(sample.token_ids, sample.model_id, sample.policy_id),
                    )
                )
            provider = self._weight_provider()

        evaluated: list[StepwiseCandidate[DiffusionConditionalCandidate]] = []
        for proposal, candidate_observations in zip(
            proposals, observations, strict=True
        ):
            estimate = provider.estimate(candidate_observations)
            rollouts = tuple(
                DiffusionRolloutEvaluation(
                    token_ids=weighted.observation.payload[0],
                    reward=weighted.observation.reward,
                    proposal_trajectory_logprob=weighted.observation.proposal_logprob,
                    target_trajectory_logprob=weighted.observation.target_logprob,
                    raw_log_importance_ratio=weighted.raw_log_importance_ratio,
                    applied_log_importance_ratio=weighted.applied_log_importance_ratio,
                    log_weight=weighted.log_weight,
                    proposal_model_id=weighted.observation.payload[1],
                    proposal_policy_id=weighted.observation.payload[2],
                )
                for weighted in estimate.rollouts
            )
            candidate = DiffusionConditionalCandidate(
                token_ids=proposal.token_ids,
                rollouts=rollouts,
                log_weight=estimate.log_weight,
            )
            evaluated.append(StepwiseCandidate(candidate, estimate.log_weight))
        return tuple(evaluated)

    def advance(
        self,
        state: TokenSequence,
        selected: DiffusionConditionalCandidate,
        step_index: int,
    ) -> TokenSequence:
        del step_index
        return state + selected.token_ids


def run_conditional_diffusion_is(
    *,
    base_backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionISConfig,
    base_sampling: DiffusionSamplingConfig,
    reward: DiffusionRewardFunction | None = None,
    seed: int = 0,
    rollout_backend: DiffusionBackend | None = None,
    rollout_sampling: DiffusionSamplingConfig | None = None,
    target_rollout_backend: DiffusionBackend | None = None,
    target_rollout_sampling: DiffusionSamplingConfig | None = None,
    apply_importance_correction: bool = True,
    reward_batch: DiffusionRewardBatchFunction | None = None,
) -> DiffusionConditionalISResult:
    """Blockwise conditional IS using complete dLLM rollouts.

    Base candidates always come from ``base_backend``.  When rollout generation
    uses another model or transition policy, the optional correction is the
    likelihood ratio of the *same random-remasking trajectory* under the target
    and proposal kernels.
    """

    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    target_rollout_backend = target_rollout_backend or base_backend
    target_rollout_sampling = target_rollout_sampling or base_sampling
    seeds = SeedStream(seed)
    adapter = DiffusionStepwiseAdapter(
        base_backend=base_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        reward=reward,
        rollout_backend=rollout_backend,
        rollout_sampling=rollout_sampling,
        target_rollout_backend=target_rollout_backend,
        target_rollout_sampling=target_rollout_sampling,
        apply_importance_correction=apply_importance_correction,
        reward_batch=reward_batch,
    )
    generic = run_stepwise_generation(
        adapter,
        seeds,
        selection_namespace=("dllm-is",),
    )
    steps = tuple(
        DiffusionConditionalISStep(
            generated_length_before=len(step.state_before),
            candidates=tuple(candidate.value for candidate in step.candidates),
            probabilities=step.probabilities,
            selected_index=step.selected_index,
        )
        for step in generic.steps
    )
    return DiffusionConditionalISResult(
        prompt=prompt,
        token_ids=generic.final_state,
        steps=steps,
    )


__all__ = [
    "DiffusionConditionalCandidate",
    "DiffusionConditionalISResult",
    "DiffusionConditionalISStep",
    "DiffusionRewardBatchFunction",
    "DiffusionRewardFunction",
    "DiffusionRolloutEvaluation",
    "DiffusionSIRItem",
    "DiffusionSIRResult",
    "DiffusionStepwiseAdapter",
    "resample_diffusion_candidates",
    "run_conditional_diffusion_is",
]
