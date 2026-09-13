from collections import Counter
from math import exp, log

import pytest

from inference_scaling.experimental.arllm.dynamic_is import (
    CandidateProposal,
    DesignStatisticsContext,
    RolloutBudgetContext,
    VarianceCostEstimate,
    allocate_variance_cost_budget,
    dynamic_is_step,
    empirical_design_statistics,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import DynamicISConfig, SamplingConfig
from inference_scaling.shared.metrics import total_variation
from inference_scaling.shared.budget import allocate_fresh_rollout_budget
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    ReplayRecord,
)
from inference_scaling.shared.rng import SeedStream


class CountingBackend:
    def __init__(self, backend: TabularAutoregressiveBackend) -> None:
        self.backend = backend
        self.sample_batch_sizes: list[int] = []
        self.score_batch_sizes: list[int] = []

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def sample_batch(self, requests):
        self.sample_batch_sizes.append(len(requests))
        return self.backend.sample_batch(requests)

    def score_batch(self, requests):
        self.score_batch_sizes.append(
            sum(len(request.continuations) for request in requests)
        )
        return self.backend.score_batch(requests)


def test_variance_cost_allocation_matches_continuous_optimum() -> None:
    allocations = allocate_variance_cost_budget(
        outer_ratios=[2.0, 1.0],
        statistics=[
            VarianceCostEstimate(3.0, 1.0, history_cost=4.0, fresh_cost=1.0),
            VarianceCostEstimate(0.0, 2.0, history_cost=1.0, fresh_cost=4.0),
        ],
        history_capacities=[100, 0],
        history_groups=["first", "second"],
        group_capacities={"first": 100, "second": 0},
        rollout_budget=180.0,
        minimum_fresh=0,
    )
    assert allocations[0].continuous_history == pytest.approx(30.0)
    assert allocations[0].continuous_fresh == pytest.approx(20.0)
    assert allocations[1].continuous_history == 0
    assert allocations[1].continuous_fresh == pytest.approx(10.0)
    assert sum(item.estimated_cost for item in allocations) == pytest.approx(180.0)


def test_allocation_respects_shared_history_inventory_and_fresh_minimum() -> None:
    allocations = allocate_variance_cost_budget(
        outer_ratios=[1.0, 1.0],
        statistics=[VarianceCostEstimate(1.0, 1.0)] * 2,
        history_capacities=[4, 4],
        history_groups=["same-key", "same-key"],
        group_capacities={"same-key": 1},
        rollout_budget=6.0,
        minimum_fresh=1,
    )
    assert sum(item.history_count for item in allocations) <= 1
    assert all(item.fresh_count >= 1 for item in allocations)
    assert sum(item.estimated_cost for item in allocations) <= 6.0


def test_allocation_requires_a_capacity_for_every_history_group() -> None:
    with pytest.raises(ValueError, match="missing capacities"):
        allocate_variance_cost_budget(
            outer_ratios=[1.0],
            statistics=[VarianceCostEstimate(1.0, 1.0)],
            history_capacities=[1],
            history_groups=["missing"],
            group_capacities={},
            rollout_budget=2.0,
        )


def test_fresh_only_allocator_reuses_the_common_variance_cost_solver() -> None:
    allocations = allocate_fresh_rollout_budget(
        outer_ratios=(1.0, 1.0),
        standard_deviations=(1.0, 2.0),
        costs=(1.0, 1.0),
        rollout_budget=6.0,
        minimum_fresh=1,
    )

    assert sum(item.fresh_count for item in allocations) == 6
    assert allocations[1].fresh_count > allocations[0].fresh_count
    assert all(item.history_count == 0 for item in allocations)


def test_step_budget_can_be_frozen_from_candidate_metadata() -> None:
    base = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5], model_id="base")
    seen: list[RolloutBudgetContext] = []

    def budget(context: RolloutBudgetContext) -> float:
        seen.append(context)
        return 2.0

    step = dynamic_is_step(
        base_backend=base,
        registry=BehaviorRegistry(),
        store=InMemoryReplayStore(),
        prompt=(),
        generated_prefix=(),
        config=DynamicISConfig(
            candidate_count=2,
            block_size=1,
            total_length=2,
            max_history_per_candidate=0,
            rollout_budget=99.0,
            auxiliary_mixture=0.0,
            minimum_fresh_per_candidate=1,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, _generated: 0.0,
        reward_version="constant",
        seeds=SeedStream(19),
        step_index=0,
        rollout_budget_provider=budget,
    )

    assert len(seen) == 1
    assert seen[0].history_capacities == (0, 0)
    assert all(capacity == 0 for capacity in seen[0].group_capacities.values())
    assert seen[0].terminal == (False, False)
    assert sum(candidate.allocation.fresh_count for candidate in step.candidates) == 2


def test_design_preparation_runs_before_statistics_are_read() -> None:
    base = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5], model_id="base")
    prepared: list[ReplayKey] = []

    def prepare(contexts: tuple[DesignStatisticsContext, ...]) -> None:
        for index, context in enumerate(contexts):
            prepared.append(context.key)
            context.store.add_design(
                ReplayRecord(
                    f"prepared-{index}",
                    context.key,
                    (0,),
                    float(index),
                    "base",
                    log(0.5),
                )
            )

    observed_design_counts: list[int] = []

    def statistics(context: DesignStatisticsContext) -> VarianceCostEstimate:
        observed_design_counts.append(len(context.store.design_records(context.key)))
        return VarianceCostEstimate(0.0, 1.0)

    dynamic_is_step(
        base_backend=base,
        registry=BehaviorRegistry(),
        store=InMemoryReplayStore(),
        prompt=(),
        generated_prefix=(),
        config=DynamicISConfig(
            candidate_count=2,
            block_size=1,
            total_length=2,
            max_history_per_candidate=0,
            rollout_budget=2.0,
            auxiliary_mixture=0.0,
            minimum_fresh_per_candidate=1,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, _generated: 0.0,
        reward_version="constant",
        seeds=SeedStream(23),
        step_index=0,
        design_prepare=prepare,
        statistics_provider=statistics,
    )

    assert len(prepared) == 2
    assert len(observed_design_counts) == 2
    assert all(count >= 1 for count in observed_design_counts)


def test_empirical_statistics_read_only_the_design_pool() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    base_policy = BehaviorPolicy.for_backend(backend, SamplingConfig(), label="base")
    registry = BehaviorRegistry([base_policy])
    store = InMemoryReplayStore()
    key = ReplayKey((), (), (0,), "reward-v1")
    store.add_design(
        ReplayRecord("design-0", key, (0,), 0.0, "base", log(0.5))
    )
    store.add_design(
        ReplayRecord("design-1", key, (1,), 1.0, "base", log(0.5))
    )
    store.add_evaluation(
        ReplayRecord("hidden-evaluation", key, (0,), 100.0, "base", log(0.5))
    )
    estimate = empirical_design_statistics(
        DesignStatisticsContext(
            key=key,
            evaluation_inventory=(("base", 1),),
            store=store,
            registry=registry,
            base_policy=base_policy,
            reward_temperature=1.0,
            truncation=0.5,
        )
    )
    assert estimate.history_std == pytest.approx((exp(1.0) - 1.0) / 2**1.5)
    assert estimate.fresh_std == pytest.approx((exp(1.0) - 1.0) / 2**1.5)
    assert estimate.history_cost == 2.0
    assert estimate.fresh_cost == 2.0


def test_dynamic_candidate_logs_exact_defensive_mixture_probability() -> None:
    base = TabularAutoregressiveBackend({}, fallback=[0.75, 0.25], model_id="base")
    auxiliary_backend = TabularAutoregressiveBackend(
        {}, fallback=[0.1, 0.9], model_id="auxiliary"
    )
    proposal = CandidateProposal.for_backend(
        auxiliary_backend, SamplingConfig(), label="skewed"
    )
    mixture = 0.6
    step = dynamic_is_step(
        base_backend=base,
        registry=BehaviorRegistry(),
        store=InMemoryReplayStore(),
        prompt=(),
        generated_prefix=(),
        config=DynamicISConfig(
            candidate_count=40,
            block_size=1,
            total_length=1,
            rollout_budget=1.0,
            auxiliary_mixture=mixture,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, _generated: 0.0,
        reward_version="constant",
        seeds=SeedStream(11),
        step_index=0,
        auxiliary_proposal=proposal,
    )
    expected_q = (0.4 * 0.75 + 0.6 * 0.1, 0.4 * 0.25 + 0.6 * 0.9)
    for candidate in step.candidates:
        token = candidate.token_ids[0]
        assert candidate.draw.proposal_logprob == pytest.approx(log(expected_q[token]))
        assert exp(candidate.draw.outer_log_ratio) == pytest.approx(
            (0.75, 0.25)[token] / expected_q[token]
        )
        assert candidate.log_weight == pytest.approx(candidate.draw.outer_log_ratio)


def test_static_candidate_proposals_are_generated_and_scored_in_batches() -> None:
    base = CountingBackend(
        TabularAutoregressiveBackend({}, fallback=[0.75, 0.25], model_id="base")
    )
    auxiliary = CountingBackend(
        TabularAutoregressiveBackend({}, fallback=[0.1, 0.9], model_id="auxiliary")
    )
    candidate_count = 40
    step = dynamic_is_step(
        base_backend=base,
        registry=BehaviorRegistry(),
        store=InMemoryReplayStore(),
        prompt=(),
        generated_prefix=(),
        config=DynamicISConfig(
            candidate_count=candidate_count,
            block_size=1,
            total_length=1,
            rollout_budget=1.0,
            auxiliary_mixture=0.5,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, _generated: 0.0,
        reward_version="constant",
        seeds=SeedStream(92),
        step_index=0,
        auxiliary_proposal=CandidateProposal.for_backend(
            auxiliary, SamplingConfig(), label="static-auxiliary"
        ),
    )

    assert len(step.candidates) == candidate_count
    assert len(base.sample_batch_sizes) == 1
    assert len(auxiliary.sample_batch_sizes) == 1
    assert sum(base.sample_batch_sizes + auxiliary.sample_batch_sizes) == candidate_count
    assert base.score_batch_sizes == [candidate_count]
    assert auxiliary.score_batch_sizes == [candidate_count]


def test_outer_importance_resampling_recovers_base_candidate_distribution() -> None:
    base_probabilities = {0: 0.7, 1: 0.3}
    base = TabularAutoregressiveBackend(
        {}, fallback=list(base_probabilities.values()), model_id="base"
    )
    auxiliary = CandidateProposal.for_backend(
        TabularAutoregressiveBackend({}, fallback=[0.08, 0.92], model_id="auxiliary"),
        SamplingConfig(),
        label="skewed",
    )
    counts: Counter[int] = Counter()
    trials = 900
    for trial in range(trials):
        step = dynamic_is_step(
            base_backend=base,
            registry=BehaviorRegistry(),
            store=InMemoryReplayStore(),
            prompt=(),
            generated_prefix=(),
            config=DynamicISConfig(
                candidate_count=48,
                block_size=1,
                total_length=1,
                rollout_budget=1.0,
                auxiliary_mixture=0.75,
            ),
            base_sampling=SamplingConfig(),
            reward=lambda _prompt, _generated: 0.0,
            reward_version="constant",
            seeds=SeedStream(10_000 + trial),
            step_index=0,
            auxiliary_proposal=auxiliary,
        )
        counts[step.selected.token_ids[0]] += 1
    empirical = {token: count / trials for token, count in counts.items()}
    assert total_variation(empirical, base_probabilities) < 0.04
