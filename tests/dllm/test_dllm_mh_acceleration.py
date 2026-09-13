from __future__ import annotations

from dataclasses import replace
from math import log

import numpy as np
import pytest

from inference_scaling.dllm.algorithms import (
    run_diffusion_replay_mixture_mh,
    run_diffusion_reward_mh,
    run_diffusion_reward_mh_delayed,
    run_progressive_diffusion_is,
    run_diffusion_smc_rollout_forest,
)
from inference_scaling.dllm.config import (
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionSamplingConfig,
)
from inference_scaling.dllm.types import DiffusionSample, DiffusionTraceStep
from inference_scaling.shared.config import SMCForestConfig


class CountingCoinBackend:
    def __init__(self, probability_one: float = 0.6) -> None:
        self.model_id = "coin"
        self.probability_one = probability_one
        self.batch_calls = 0

    def _logprob(self, token: int) -> float:
        return log(self.probability_one if token else 1 - self.probability_one)

    def sample_batch(self, requests):
        self.batch_calls += 1
        outputs = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            tokens = tuple(
                int(rng.random() < self.probability_one)
                for _ in range(request.generation_length)
            )
            trace = tuple(
                DiffusionTraceStep(
                    block_index=position,
                    step_index=0,
                    positions=(position,),
                    token_ids=(token,),
                    logprob=self._logprob(token),
                )
                for position, token in enumerate(tokens)
            )
            outputs.append(
                DiffusionSample(
                    prefix=request.prefix,
                    token_ids=tokens,
                    trace=trace,
                    trajectory_logprob=sum(self._logprob(token) for token in tokens),
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                )
            )
        return outputs

    def score_trajectories(self, requests):
        return [
            sum(self._logprob(token) for token in request.sample.token_ids)
            for request in requests
        ]


EXACT = DiffusionSamplingConfig(
    block_length=1,
    steps_per_block=1,
    temperature=1.0,
    remasking="random",
)
CONFIG = DiffusionMHConfig(total_length=2, updates=6, reward_temperature=1.0)


def _reward(_prompt, continuation):
    return float(sum(continuation))


def _zero_reward(_prompt, continuations):
    return [0.0 for _ in continuations]


def test_independence_mh_batch_prefetch_preserves_the_exact_chain():
    sequential_backend = CountingCoinBackend()
    batched_backend = CountingCoinBackend()
    sequential = run_diffusion_reward_mh(
        backend=sequential_backend,
        prompt=(9,),
        config=CONFIG,
        sampling=EXACT,
        reward=_reward,
        proposal_batch_size=1,
        seed=4,
    )
    batched = run_diffusion_reward_mh(
        backend=batched_backend,
        prompt=(9,),
        config=CONFIG,
        sampling=EXACT,
        reward=_reward,
        proposal_batch_size=None,
        seed=4,
    )

    assert sequential == batched
    assert sequential_backend.batch_calls == CONFIG.updates + 1
    assert batched_backend.batch_calls == 1


def test_delayed_acceptance_skips_some_exact_reward_evaluations():
    backend = CountingCoinBackend()
    result = run_diffusion_reward_mh_delayed(
        backend=backend,
        prompt=(9,),
        config=CONFIG,
        sampling=EXACT,
        reward=_reward,
        surrogate_reward=lambda _prompt, tokens: -100.0 * sum(tokens),
        seed=0,
    )

    assert result.exact_reward_evaluations < CONFIG.updates + 1
    assert result.surrogate_reward_evaluations == CONFIG.updates + 1
    assert all(
        step.exact_reward_evaluated == step.stage_one_accepted
        for step in result.steps
    )


def test_zero_history_weight_replay_mixture_reduces_to_base_independence_mh():
    backend = CountingCoinBackend()
    history = run_diffusion_reward_mh(
        backend=backend,
        prompt=(9,),
        config=DiffusionMHConfig(total_length=2, updates=1, reward_temperature=1.0),
        sampling=EXACT,
        reward_batch=_zero_reward,
        seed=2,
    ).initial
    result = run_diffusion_replay_mixture_mh(
        backend=backend,
        prompt=(9,),
        config=CONFIG,
        sampling=EXACT,
        history=(history,),
        history_probability=0.0,
        reward_batch=_zero_reward,
        seed=5,
    )

    assert result.history_draws == 0
    assert result.base_draws == CONFIG.updates + 1
    assert result.acceptance_rate == 1.0


def test_replay_mixture_rejects_a_cache_from_another_model():
    backend = CountingCoinBackend()
    history = run_diffusion_reward_mh(
        backend=backend,
        prompt=(9,),
        config=DiffusionMHConfig(total_length=2, updates=1, reward_temperature=1.0),
        sampling=EXACT,
        reward_batch=_zero_reward,
        seed=2,
    ).initial

    with pytest.raises(ValueError, match="match the prompt and exact policy"):
        run_diffusion_replay_mixture_mh(
            backend=backend,
            prompt=(9,),
            config=CONFIG,
            sampling=EXACT,
            history=(replace(history, model_id="another-model"),),
            history_probability=0.5,
            reward_batch=_zero_reward,
            seed=5,
        )


def test_progressive_is_separates_pilot_and_evaluation_rollouts():
    backend = CountingCoinBackend()
    result = run_progressive_diffusion_is(
        backend=backend,
        prompt=(9,),
        config=DiffusionISConfig(
            candidate_count=3,
            rollout_count=2,
            block_size=1,
            total_length=2,
            reward_temperature=1.0,
        ),
        sampling=EXACT,
        reward_batch=lambda _prompt, values: [float(sum(value)) for value in values],
        pilot_rollouts_per_candidate=2,
        evaluation_rollout_budget=6,
        seed=8,
    )

    first = result.steps[0]
    assert first.pilot_rollouts == 6
    assert sum(item.fresh_count for item in first.allocations) <= 6
    assert all(item.fresh_count >= 1 for item in first.allocations)
    assert result.steps[-1].pilot_rollouts == 0


def test_progressive_is_does_not_require_a_tractable_trajectory_density():
    backend = CountingCoinBackend()
    non_exact = replace(EXACT, temperature=0.0, remasking="low_confidence")

    result = run_progressive_diffusion_is(
        backend=backend,
        prompt=(9,),
        config=DiffusionISConfig(
            candidate_count=2,
            rollout_count=1,
            block_size=1,
            total_length=2,
        ),
        sampling=non_exact,
        reward_batch=lambda _prompt, values: [float(sum(value)) for value in values],
        pilot_rollouts_per_candidate=2,
        evaluation_rollout_budget=2,
        seed=8,
    )

    assert len(result.token_ids) == 2


def test_diffusion_smc_reuses_conditional_rollout_suffixes():
    common = dict(
        particle_count=3,
        branch_factor=2,
        rollout_count=2,
        block_size=1,
        total_length=3,
        reward_temperature=1.0,
    )
    fresh = run_diffusion_smc_rollout_forest(
        backend=CountingCoinBackend(),
        prompt=(9,),
        config=SMCForestConfig(**common, reuse_rollout_forest=False),
        sampling=EXACT,
        reward_batch=lambda _prompt, values: [float(sum(value)) for value in values],
        seed=17,
    )
    reused = run_diffusion_smc_rollout_forest(
        backend=CountingCoinBackend(),
        prompt=(9,),
        config=SMCForestConfig(**common, reuse_rollout_forest=True),
        sampling=EXACT,
        reward_batch=lambda _prompt, values: [float(sum(value)) for value in values],
        seed=17,
    )

    assert len(fresh.token_ids) == len(reused.token_ids) == 3
    assert reused.reused_rollouts > 0
    assert reused.fresh_rollouts < fresh.fresh_rollouts
