from __future__ import annotations

from math import log

import numpy as np

from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.replay import (
    build_diffusion_replay_history,
    select_diffusion_candidates_with_replay,
)
from inference_scaling.dllm.types import (
    DiffusionSample,
    DiffusionTraceStep,
)


class CoinDiffusionBackend:
    def __init__(self, model_id: str, probability_one: float) -> None:
        self.model_id = model_id
        self.probability_one = probability_one
        self.sample_requests = 0
        self.score_requests = 0

    def _logprob(self, token: int) -> float:
        return log(self.probability_one if token == 1 else 1 - self.probability_one)

    def sample_batch(self, requests):
        self.sample_requests += len(requests)
        outputs = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            tokens = tuple(
                int(rng.random() < self.probability_one)
                for _ in range(request.generation_length)
            )
            trace = tuple(
                DiffusionTraceStep(
                    block_index=index,
                    step_index=0,
                    positions=(index,),
                    token_ids=(token,),
                    logprob=self._logprob(token),
                )
                for index, token in enumerate(tokens)
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
        self.score_requests += len(requests)
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


def _candidate(token: int) -> DiffusionSample:
    return DiffusionSample(
        prefix=(9,),
        token_ids=(token,),
        trace=(),
        trajectory_logprob=None,
        policy_id="candidate",
        model_id="target",
        request_id=f"candidate-{token}",
    )


def _reward(_prompt, continuations):
    return [float(continuation[-1]) for continuation in continuations]


def test_replay_history_precomputes_target_density_for_behavior_rollouts():
    target = CoinDiffusionBackend("target", 0.8)
    behavior = CoinDiffusionBackend("behavior", 0.3)
    candidates = (_candidate(0), _candidate(1))

    histories = build_diffusion_replay_history(
        target_backend=target,
        behavior_backend=behavior,
        prompt=(9,),
        generated_prefix=(),
        candidates=candidates,
        rollout_length=1,
        count_per_candidate=2,
        target_sampling=EXACT,
        behavior_sampling=EXACT,
        reward_batch=_reward,
        seed=4,
    )

    assert [len(history.records) for history in histories] == [2, 2]
    assert target.score_requests == 4
    assert behavior.sample_requests == 4
    assert all(
        record.target_trajectory_logprob
        != record.behavior_trajectory_logprob
        for history in histories
        for record in history.records
    )


def test_online_replay_uses_two_history_and_one_fresh_tail_per_candidate():
    target = CoinDiffusionBackend("target", 0.8)
    behavior = CoinDiffusionBackend("behavior", 0.3)
    candidates = (_candidate(0), _candidate(1))
    histories = build_diffusion_replay_history(
        target_backend=target,
        behavior_backend=behavior,
        prompt=(9,),
        generated_prefix=(),
        candidates=candidates,
        rollout_length=1,
        count_per_candidate=2,
        target_sampling=EXACT,
        behavior_sampling=EXACT,
        reward_batch=_reward,
        seed=4,
    )
    target_samples_before = target.sample_requests
    target_scores_before = target.score_requests
    behavior_samples_before = behavior.sample_requests
    behavior_scores_before = behavior.score_requests

    selection = select_diffusion_candidates_with_replay(
        target_backend=target,
        behavior_backend=behavior,
        prompt=(9,),
        generated_prefix=(),
        candidates=candidates,
        histories=histories,
        rollout_length=1,
        fresh_count=1,
        target_sampling=EXACT,
        behavior_sampling=EXACT,
        reward_batch=_reward,
        reward_temperature=1.0,
        truncation=2.0,
        seed=12,
    )

    assert [candidate.estimate.history_count for candidate in selection.candidates] == [2, 2]
    assert [candidate.estimate.fresh_count for candidate in selection.candidates] == [1, 1]
    assert target.sample_requests - target_samples_before == 2
    assert target.score_requests == target_scores_before
    assert behavior.sample_requests == behavior_samples_before
    assert behavior.score_requests - behavior_scores_before == 2
    assert sum(selection.probabilities) == 1.0


def test_fresh_only_arm_needs_no_behavior_model_or_rescoring():
    target = CoinDiffusionBackend("target", 0.8)
    candidates = (_candidate(0), _candidate(1))

    selection = select_diffusion_candidates_with_replay(
        target_backend=target,
        behavior_backend=None,
        prompt=(9,),
        generated_prefix=(),
        candidates=candidates,
        histories=None,
        rollout_length=1,
        fresh_count=3,
        target_sampling=EXACT,
        behavior_sampling=None,
        reward_batch=_reward,
        reward_temperature=1.0,
        truncation=2.0,
        seed=12,
    )

    assert [candidate.estimate.history_count for candidate in selection.candidates] == [0, 0]
    assert [candidate.estimate.fresh_count for candidate in selection.candidates] == [3, 3]
    assert target.sample_requests == 6
    assert target.score_requests == 0


def test_replay_accepts_per_candidate_fresh_budgets_and_outer_weights():
    target = CoinDiffusionBackend("target", 0.8)
    candidates = (_candidate(0), _candidate(1))

    selection = select_diffusion_candidates_with_replay(
        target_backend=target,
        behavior_backend=None,
        prompt=(9,),
        generated_prefix=(),
        candidates=candidates,
        histories=None,
        rollout_length=1,
        fresh_count=(1, 3),
        target_sampling=EXACT,
        behavior_sampling=None,
        reward_batch=_reward,
        reward_temperature=1.0,
        truncation=2.0,
        seed=12,
        candidate_log_ratios=(2.0, -2.0),
    )

    assert [candidate.estimate.fresh_count for candidate in selection.candidates] == [1, 3]
    assert [candidate.outer_log_ratio for candidate in selection.candidates] == [2.0, -2.0]
    assert target.sample_requests == 4
    assert selection.probabilities[0] > selection.probabilities[1]
