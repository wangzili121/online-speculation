from __future__ import annotations

from math import exp, log

import numpy as np
import pytest

from inference_scaling.dllm.config import DiffusionISConfig, DiffusionSamplingConfig
from inference_scaling.dllm.dynamic_is import (
    draw_defensive_diffusion_candidates,
    run_dynamic_diffusion_is,
)
from inference_scaling.dllm.types import DiffusionSample, DiffusionTraceStep


class CoinBackend:
    def __init__(self, model_id: str, probability_one: float) -> None:
        self.model_id = model_id
        self.probability_one = probability_one

    def _logprob(self, token: int) -> float:
        return log(self.probability_one if token else 1 - self.probability_one)

    def sample_batch(self, requests):
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


def _reward(_prompt, continuations):
    return [float(continuation[-1]) for continuation in continuations]


def test_defensive_candidate_draw_records_exact_mixture_ratio():
    target = CoinBackend("target", 0.8)
    auxiliary = CoinBackend("auxiliary", 0.2)
    draws = draw_defensive_diffusion_candidates(
        target_backend=target,
        auxiliary_backend=auxiliary,
        prefix=(9,),
        generation_length=1,
        count=16,
        sampling=EXACT,
        auxiliary_probability=0.4,
        seed=7,
        stage_index=0,
    )

    assert {draw.source for draw in draws} == {"target", "auxiliary"}
    for draw in draws:
        expected = 0.6 * exp(draw.target_logprob) + 0.4 * exp(
            draw.auxiliary_logprob
        )
        assert exp(draw.mixture_logprob) == pytest.approx(expected)
        assert draw.outer_log_ratio == pytest.approx(
            draw.target_logprob - draw.mixture_logprob
        )


@pytest.mark.parametrize(
    "arm",
    (
        "base_candidate_fixed",
        "trajectory_replay_aware_fixed",
        "trajectory_replay_aware_optimal",
    ),
)
def test_every_dynamic_arm_runs_through_the_same_diffusion_core(arm):
    target = CoinBackend("target", 0.8)
    auxiliary = CoinBackend("auxiliary", 0.3)
    result = run_dynamic_diffusion_is(
        arm=arm,
        target_backend=target,
        auxiliary_backend=auxiliary,
        prompt=(9,),
        config=DiffusionISConfig(
            candidate_count=4,
            rollout_count=1,
            block_size=1,
            total_length=2,
            reward_temperature=1.0,
        ),
        sampling=EXACT,
        reward_batch=_reward,
        history_rollouts=2,
        fresh_rollouts=1,
        truncation=2.0,
        design_rollouts=2,
        seed=13,
    )

    assert len(result.token_ids) == 2
    assert len(result.steps) == 2
    assert all(sum(step.selection.probabilities) == pytest.approx(1.0) for step in result.steps)
    assert all(item.fresh_count >= 1 for item in result.steps[0].allocations)
    assert all(item.estimated_cost == 0 for item in result.steps[-1].allocations)
    if arm == "trajectory_replay_aware_optimal":
        assert result.steps[0].design_rollouts == 16
        assert result.steps[0].evaluation_history_rollouts == 8
