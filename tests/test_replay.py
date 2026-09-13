from math import exp, log

import numpy as np
import pytest

from inference_scaling.arllm.algorithms.base_replay import (
    ProbabilityObservation,
    base_replay_step,
    corrected_replay_log_weight,
    run_base_replay,
)
from inference_scaling.arllm.algorithms.conditional_is import _sample_candidates
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import BaseReplayConfig, SamplingConfig
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    BrokeredReplayState,
    InMemoryReplayStore,
    ReplayKey,
    ReplayRecord,
    ReplaySampleRequest,
    sample_replay_record,
    sample_replay_records_brokered,
    validate_record_probabilities,
)
from inference_scaling.arllm.rollout_broker import AsyncRolloutBroker
from inference_scaling.shared.rng import SeedStream


class BatchCountingBackend:
    def __init__(self, backend):
        self.backend = backend
        self.sample_batch_sizes: list[int] = []
        self.score_samplings = []

    @property
    def model_id(self):
        return self.backend.model_id

    def sample_batch(self, requests):
        self.sample_batch_sizes.append(len(requests))
        return self.backend.sample_batch(requests)

    def score_batch(self, requests):
        self.score_samplings.extend(request.sampling for request in requests)
        return self.backend.score_batch(requests)


def test_truncated_history_and_fresh_tail_are_exact_in_expectation() -> None:
    p = np.asarray([0.8, 0.2])
    q = np.asarray([0.55, 0.45])
    rewards = np.asarray([0.0, 1.0])
    truncation = 1.25
    expected_estimator = 0.0
    for history_token in (0, 1):
        for fresh_token in (0, 1):
            log_weight, _, _ = corrected_replay_log_weight(
                [
                    ProbabilityObservation(
                        log(p[history_token]),
                        log(q[history_token]),
                        rewards[history_token],
                    )
                ],
                [
                    ProbabilityObservation(
                        log(p[fresh_token]), log(q[fresh_token]), rewards[fresh_token]
                    )
                ],
                truncation=truncation,
                reward_temperature=1.0,
            )
            expected_estimator += q[history_token] * p[fresh_token] * exp(log_weight)
    exact_weight = float(np.dot(p, np.exp(rewards)))
    assert expected_estimator == pytest.approx(exact_weight)


def test_evaluation_claims_are_metadata_only_disjoint_and_single_use() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.4])
    behavior = BehaviorPolicy.for_backend(
        backend, SamplingConfig(temperature=0.8), label="behavior"
    )
    key = ReplayKey((), (), (0,), "reward-v1")
    store = InMemoryReplayStore()
    for index in range(3):
        store.add_evaluation(
            sample_replay_record(
                behavior,
                key,
                1,
                lambda _prompt, generated: float(sum(generated)),
                seed=index,
                record_id=f"record-{index}",
            )
        )

    first, second = store.freeze_claims([key, key], 2)
    assert first.count == 2
    assert second.count == 1
    assert store.inventory(key) == {}
    assert store.reserved_count == 3
    first_records = store.reveal_and_consume(first)
    second_records = store.reveal_and_consume(second)
    assert set(record.record_id for record in first_records).isdisjoint(
        record.record_id for record in second_records
    )
    assert store.design_count == 3
    with pytest.raises(ValueError):
        store.reveal_and_consume(first)


def test_probability_validation_allows_small_per_token_roundoff_only() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.4])
    behavior = BehaviorPolicy.for_backend(backend, SamplingConfig(), label="behavior")
    registry = BehaviorRegistry([behavior])
    key = ReplayKey((), (), (0,), "reward-v1")
    exact = 2 * log(0.4)
    rounded = ReplayRecord(
        "rounded",
        key,
        (1, 1),
        0.0,
        behavior.behavior_id,
        exact + 3e-4,
    )
    corrupted = ReplayRecord(
        "corrupted",
        key,
        (1, 1),
        0.0,
        behavior.behavior_id,
        exact + 1e-2,
    )

    validate_record_probabilities([rounded], registry)
    with pytest.raises(ValueError, match="cannot be reproduced"):
        validate_record_probabilities([corrupted], registry)


def test_base_replay_uses_matching_history_and_moves_all_used_data_to_design() -> None:
    backend = TabularAutoregressiveBackend(
        {(): [1.0, 0.0], (0,): [0.7, 0.3]}, fallback=[0.5, 0.5]
    )
    behavior = BehaviorPolicy.for_backend(
        backend, SamplingConfig(temperature=0.6), label="old-policy"
    )
    registry = BehaviorRegistry([behavior])
    store = InMemoryReplayStore()
    key = ReplayKey((), (), (0,), "reward-v1")
    for index in range(2):
        store.add_evaluation(
            sample_replay_record(
                behavior,
                key,
                1,
                lambda _prompt, generated: float(generated[-1]),
                seed=100 + index,
                record_id=f"history-{index}",
            )
        )

    step = base_replay_step(
        base_backend=backend,
        registry=registry,
        store=store,
        prompt=(),
        generated_prefix=(),
        config=BaseReplayConfig(
            candidate_count=1,
            block_size=1,
            total_length=2,
            max_history_per_candidate=2,
            fresh_rollouts=2,
            truncation=1.5,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(generated[-1]),
        reward_version="reward-v1",
        seeds=SeedStream(7),
        step_index=0,
    )
    estimate = step.selected.estimate
    assert estimate.history_count == 2
    assert estimate.fresh_count == 2
    assert store.evaluation_count == 0
    assert store.design_count == 4
    assert estimate.behavior_counts == (("old-policy", 2),)


def test_no_history_reduces_to_fresh_base_completion_mean() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    store = InMemoryReplayStore()
    step = base_replay_step(
        base_backend=backend,
        registry=BehaviorRegistry(),
        store=store,
        prompt=(),
        generated_prefix=(),
        config=BaseReplayConfig(
            candidate_count=2,
            block_size=1,
            total_length=2,
            max_history_per_candidate=3,
            fresh_rollouts=3,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(sum(generated)),
        reward_version="reward-v1",
        seeds=SeedStream(19),
        step_index=0,
    )
    assert all(candidate.estimate.history_count == 0 for candidate in step.candidates)
    assert all(candidate.estimate.fresh_count == 3 for candidate in step.candidates)
    assert store.design_count == 6
    assert store.evaluation_count == 0


def test_post_selection_reserve_can_use_an_off_policy_behavior() -> None:
    backend = TabularAutoregressiveBackend(
        {
            (): [1.0, 0.0],
            (0,): [1.0, 0.0],
            (0, 0): [0.25, 0.75],
        },
        fallback=[0.5, 0.5],
    )
    reserve_policy = BehaviorPolicy.for_backend(
        backend, SamplingConfig(temperature=0.45), label="off-policy-reserve"
    )
    store = InMemoryReplayStore()
    result = run_base_replay(
        backend,
        BehaviorRegistry(),
        store,
        (),
        BaseReplayConfig(
            candidate_count=1,
            block_size=1,
            total_length=3,
            max_history_per_candidate=2,
            fresh_rollouts=1,
            reserve_rollouts=2,
        ),
        lambda _prompt, generated: float(generated[-1]),
        "reward-v1",
        SeedStream(23),
        reserve_policy=reserve_policy,
    )

    replayed = result.steps[1].selected.estimate
    assert result.reserve_records_written == 2
    assert replayed.behavior_counts == (("off-policy-reserve", 2),)
    assert replayed.history_count == 2
    assert store.evaluation_count == 0
    assert store.reserved_count == 0


def test_fresh_replay_is_generated_in_one_cross_candidate_batch() -> None:
    backend = BatchCountingBackend(
        TabularAutoregressiveBackend({}, fallback=[0.6, 0.4])
    )
    step = base_replay_step(
        base_backend=backend,
        registry=BehaviorRegistry(),
        store=InMemoryReplayStore(),
        prompt=(),
        generated_prefix=(),
        config=BaseReplayConfig(
            candidate_count=4,
            block_size=1,
            total_length=2,
            fresh_rollouts=3,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(sum(generated)),
        reward_version="reward-v1",
        seeds=SeedStream(31),
        step_index=0,
    )

    assert len(step.candidates) == 4
    assert backend.sample_batch_sizes == [4, 12]


def test_base_replay_can_reuse_frozen_candidate_draws() -> None:
    backend = BatchCountingBackend(
        TabularAutoregressiveBackend({}, fallback=[0.6, 0.4])
    )
    sampling = SamplingConfig()
    seeds = SeedStream(37)
    candidates = _sample_candidates(
        backend,
        (),
        4,
        1,
        sampling,
        seeds,
        0,
    )

    step = base_replay_step(
        base_backend=backend,
        registry=BehaviorRegistry(),
        store=InMemoryReplayStore(),
        prompt=(),
        generated_prefix=(),
        config=BaseReplayConfig(
            candidate_count=4,
            block_size=1,
            total_length=2,
            fresh_rollouts=3,
        ),
        base_sampling=sampling,
        reward=lambda _prompt, generated: float(sum(generated)),
        reward_version="reward-v1",
        seeds=seeds,
        step_index=0,
        candidate_samples=candidates,
    )

    assert [candidate.token_ids for candidate in step.candidates] == [
        candidate.token_ids for candidate in candidates
    ]
    assert backend.sample_batch_sizes == [4, 12]


def test_base_replay_rejects_a_mismatched_frozen_candidate() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.4])
    candidates = _sample_candidates(
        backend,
        (),
        1,
        1,
        SamplingConfig(),
        SeedStream(41),
        0,
    )

    with pytest.raises(ValueError, match="frozen candidate"):
        base_replay_step(
            base_backend=backend,
            registry=BehaviorRegistry(),
            store=InMemoryReplayStore(),
            prompt=(1,),
            generated_prefix=(),
            config=BaseReplayConfig(
                candidate_count=1,
                block_size=1,
                total_length=2,
                fresh_rollouts=1,
            ),
            base_sampling=SamplingConfig(),
            reward=lambda _prompt, generated: float(sum(generated)),
            reward_version="reward-v1",
            seeds=SeedStream(41),
            step_index=0,
            candidate_samples=candidates,
        )


def test_history_is_scored_under_the_configured_base_temperature() -> None:
    backend = BatchCountingBackend(
        TabularAutoregressiveBackend({(): [1.0, 0.0]}, fallback=[0.8, 0.2])
    )
    behavior = BehaviorPolicy.for_backend(
        backend, SamplingConfig(temperature=0.5), label="old-policy"
    )
    registry = BehaviorRegistry([behavior])
    store = InMemoryReplayStore()
    key = ReplayKey((), (), (0,), "reward-v1")
    store.add_evaluation(
        sample_replay_record(
            behavior,
            key,
            1,
            lambda _prompt, generated: float(generated[-1]),
            seed=5,
            record_id="history",
        )
    )
    base_sampling = SamplingConfig(temperature=0.7)

    base_replay_step(
        base_backend=backend,
        registry=registry,
        store=store,
        prompt=(),
        generated_prefix=(),
        config=BaseReplayConfig(
            candidate_count=1,
            block_size=1,
            total_length=2,
            max_history_per_candidate=1,
            fresh_rollouts=1,
        ),
        base_sampling=base_sampling,
        reward=lambda _prompt, generated: float(generated[-1]),
        reward_version="reward-v1",
        seeds=SeedStream(11),
        step_index=0,
    )

    assert base_sampling in backend.score_samplings
    assert None not in backend.score_samplings


def test_brokered_replay_only_exposes_completed_records_and_resumes_partials() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    policy = BehaviorPolicy.for_backend(backend, SamplingConfig(), label="base")
    key = ReplayKey((), (), (1,), "reward-v1")
    requests = (
        ReplaySampleRequest(key, 1, 10, "short"),
        ReplaySampleRequest(key, 4, 11, "long-a"),
        ReplaySampleRequest(key, 4, 12, "long-b"),
    )
    broker = AsyncRolloutBroker(backend, chunk_tokens=2)
    callbacks: list[str] = []

    first = sample_replay_records_brokered(
        policy,
        requests,
        lambda _prompt, generated: float(sum(generated)),
        broker,
        completion_target=1,
        on_record=lambda record: callbacks.append(record.record_id),
    )
    assert tuple(record.record_id for record in first.records) == ("short",)
    assert callbacks == ["short"]
    assert len(first.partial) == 2
    assert all(isinstance(item, BrokeredReplayState) for item in first.partial)
    assert all(len(item.rollout.token_ids) == 2 for item in first.partial)
    assert first.snapshot.partial_tokens_preserved == 4

    second = sample_replay_records_brokered(
        policy,
        first.partial,
        lambda _prompt, generated: float(sum(generated)),
        broker,
        on_record=lambda record: callbacks.append(record.record_id),
    )
    assert {record.record_id for record in second.records} == {"long-a", "long-b"}
    assert not second.partial
    assert set(callbacks) == {"short", "long-a", "long-b"}
    assert all(len(record.completion) == 4 for record in second.records)
    assert all(np.isfinite(record.behavior_logprob) for record in second.records)
    assert second.snapshot.initial_partial_tokens == 4
    assert second.snapshot.resumed_prefill_tokens == 4
