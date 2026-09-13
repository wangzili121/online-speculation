from __future__ import annotations

from math import log

import pytest

from inference_scaling.shared.importance import (
    MonteCarloRolloutWeightProvider,
    ProbabilityObservation,
    RolloutObservation,
    TruncatedReplayRolloutWeightProvider,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.smc import (
    normalize_smc_log_weights,
    partition_resampled_reservoirs,
    systematic_resample,
)
from inference_scaling.shared.stepwise import (
    StepwiseCandidate,
    normalize_log_weights,
    run_stepwise_generation,
)


class BinaryStepwiseBackend:
    initial_state: tuple[int, ...] = ()

    def is_terminal(self, state):
        return len(state) == 2

    def propose(self, state, step_index, seeds):
        del state, step_index, seeds
        return (0, 1)

    def evaluate(self, state, proposals, step_index, seeds):
        del step_index, seeds
        return tuple(
            StepwiseCandidate(state + (proposal,), float(proposal))
            for proposal in proposals
        )

    def advance(self, state, selected, step_index):
        del state, step_index
        return selected


def test_common_stepwise_driver_is_state_and_model_agnostic():
    first = run_stepwise_generation(
        BinaryStepwiseBackend(), SeedStream(7), selection_namespace=("test",)
    )
    second = run_stepwise_generation(
        BinaryStepwiseBackend(), SeedStream(7), selection_namespace=("test",)
    )

    assert first == second
    assert len(first.final_state) == 2
    assert len(first.steps) == 2
    assert all(len(step.candidates) == 2 for step in first.steps)
    assert all(sum(step.probabilities) == pytest.approx(1.0) for step in first.steps)


def test_log_weight_normalization_matches_softmax():
    probabilities = normalize_log_weights((0.0, log(3.0)))
    assert probabilities == pytest.approx((0.25, 0.75))


def test_rollout_provider_covers_identity_importance_and_uncorrected_modes():
    observation = RolloutObservation(
        reward=2.0,
        target_logprob=log(0.8),
        proposal_logprob=log(0.4),
    )
    corrected = MonteCarloRolloutWeightProvider(
        reward_temperature=2.0, correction="importance"
    ).weight(observation)
    identity = MonteCarloRolloutWeightProvider(
        reward_temperature=2.0, correction="identity"
    ).weight(RolloutObservation(reward=2.0))
    uncorrected = MonteCarloRolloutWeightProvider(
        reward_temperature=2.0, correction="none"
    ).weight(observation)

    assert corrected.log_weight == pytest.approx(1.0 + log(2.0))
    assert identity.raw_log_importance_ratio == 0.0
    assert identity.log_weight == pytest.approx(1.0)
    assert uncorrected.raw_log_importance_ratio is None
    assert uncorrected.log_weight == pytest.approx(1.0)


def test_replay_provider_exposes_history_and_fresh_terms():
    provider = TruncatedReplayRolloutWeightProvider(
        truncation=1.5, reward_temperature=1.0
    )
    estimate = provider.estimate(
        [ProbabilityObservation(log(0.8), log(0.5), 1.0)],
        [ProbabilityObservation(log(0.8), log(0.5), 1.0)],
    )

    assert len(estimate.history_log_terms) == 1
    assert len(estimate.fresh_log_terms) == 1
    assert estimate.log_weight > estimate.history_log_terms[0]


def test_common_smc_primitives_normalize_resample_and_split_without_copying():
    probabilities, ess = normalize_smc_log_weights((0.0, log(3.0)))
    assert probabilities == pytest.approx((0.25, 0.75))
    assert ess == pytest.approx(1.6)
    selected = systematic_resample(
        probabilities,
        4,
        SeedStream(3).generator("smc-test"),
    )
    assert len(selected) == 4

    buckets = partition_resampled_reservoirs(
        (("a", "b", "c", "d"), ("e",)),
        (0, 0, 1),
    )
    assert buckets == (("a", "c"), ("b", "d"), ("e",))
