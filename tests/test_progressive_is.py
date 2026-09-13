from __future__ import annotations

from collections import Counter

from inference_scaling.experimental.arllm.progressive_is import (
    progressive_is_step,
    run_progressive_conditional_is,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import ProgressiveISConfig, SamplingConfig
from inference_scaling.shared.metrics import total_variation
from inference_scaling.shared.rng import SeedStream


class RecordingBackend:
    def __init__(self, backend):
        self.backend = backend
        self.request_ids: list[str] = []

    @property
    def model_id(self):
        return self.backend.model_id

    @property
    def parameter_count(self):
        return 1

    def sample_batch(self, requests):
        self.request_ids.extend(request.request_id for request in requests)
        return self.backend.sample_batch(requests)

    def score_batch(self, requests):
        return self.backend.score_batch(requests)


def test_pilot_rollouts_freeze_budget_but_do_not_enter_evaluation_estimate() -> None:
    backend = RecordingBackend(
        TabularAutoregressiveBackend({}, fallback=[0.6, 0.4], model_id="base")
    )
    step = progressive_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ProgressiveISConfig(
            candidate_count=2,
            pilot_rollouts_per_candidate=2,
            evaluation_cost_budget=2.0,
            minimum_evaluation_per_candidate=1,
            block_size=1,
            total_length=2,
        ),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(generated[-1]),
        seeds=SeedStream(3),
        step_index=0,
    )

    assert all(candidate.pilot.rollout_count == 2 for candidate in step.candidates)
    assert all(candidate.evaluation_count == 1 for candidate in step.candidates)
    assert all(len(candidate.evaluation_rollouts) == 1 for candidate in step.candidates)
    pilot_ids = {value for value in backend.request_ids if ":pilot:" in value}
    evaluation_ids = {value for value in backend.request_ids if ":evaluation:" in value}
    assert len(pilot_ids) == 4
    assert len(evaluation_ids) == 2
    assert pilot_ids.isdisjoint(evaluation_ids)


def test_progressive_constant_reward_preserves_base_distribution() -> None:
    base_probabilities = {0: 0.7, 1: 0.3}
    backend = TabularAutoregressiveBackend(
        {}, fallback=list(base_probabilities.values()), model_id="base"
    )
    counts: Counter[int] = Counter()
    trials = 600
    config = ProgressiveISConfig(
        candidate_count=12,
        pilot_rollouts_per_candidate=1,
        evaluation_cost_budget=12.0,
        minimum_evaluation_per_candidate=1,
        block_size=1,
        total_length=1,
    )
    for trial in range(trials):
        result = run_progressive_conditional_is(
            backend,
            (),
            config,
            lambda _prompt, _generated: 0.0,
            SeedStream(10_000 + trial),
            streaming_rewards=False,
        )
        counts[result.token_ids[0]] += 1
    empirical = {token: value / trials for token, value in counts.items()}
    assert total_variation(empirical, base_probabilities) < 0.04
