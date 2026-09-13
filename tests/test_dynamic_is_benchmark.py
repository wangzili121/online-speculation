import pytest

from experiments.arllm.gsm8k_dynamic_is_benchmark import (
    FixedPerCandidateStatistics,
    MatchedProxyBudget,
    _fixed_history_targets,
)
from inference_scaling.experimental.arllm.dynamic_is import (
    DesignStatisticsContext,
    RolloutBudgetContext,
    allocate_variance_cost_budget,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
)


def test_fixed_targets_share_duplicate_candidate_inventory() -> None:
    shared = ReplayKey((), (), (7,), "reward")
    other = ReplayKey((), (), (8,), "reward")
    targets = _fixed_history_targets(
        keys=(shared, shared, other),
        terminal=(False, False, False),
        history_capacities=(2, 2, 0),
        group_capacities={shared: 2, other: 0},
        rollouts_per_candidate=3,
    )

    assert targets == (2, 0, 0)
    assert sum(targets[:2]) == 2


def test_matched_budget_fills_duplicate_candidates_with_fresh_rollouts() -> None:
    shared = ReplayKey((), (), (7,), "reward")
    other = ReplayKey((), (), (8,), "reward")
    context = RolloutBudgetContext(
        draws=(),
        keys=(shared, shared, other),
        terminal=(False, False, False),
        history_capacities=(2, 2, 0),
        group_capacities={shared: 2, other: 0},
    )
    budget = MatchedProxyBudget(
        history_cost=1.0,
        fresh_cost=1.32,
        rollouts_per_candidate=3,
    )

    # The first duplicate claims both one-use records; the second and third
    # candidates are topped up with fresh draws to preserve three each.
    assert budget(context) == pytest.approx(2 * 1.0 + 7 * 1.32)


def test_fixed_statistics_preserve_count_under_shared_inventory() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[1.0], model_id="base")
    sampling = SamplingConfig()
    policy = BehaviorPolicy.for_backend(backend, sampling, label="base")
    store = InMemoryReplayStore()
    registry = BehaviorRegistry([policy])
    key = ReplayKey((), (), (0,), "reward")
    contexts = tuple(
        DesignStatisticsContext(
            key=key,
            evaluation_inventory=(("history", 2),),
            store=store,
            registry=registry,
            base_policy=policy,
            reward_temperature=1.0,
            truncation=10.0,
        )
        for _ in range(2)
    )
    statistics_provider = FixedPerCandidateStatistics(
        proposal_backend=backend,
        proposal_sampling=sampling,
        mixture=0.0,
        history_cost=1.0,
        fresh_cost=1.32,
        max_history=2,
        rollouts_per_candidate=3,
    )
    statistics_provider.prepare(contexts)
    statistics = tuple(statistics_provider(context) for context in contexts)
    budget_provider = MatchedProxyBudget(1.0, 1.32, 3)
    budget = budget_provider(
        RolloutBudgetContext(
            draws=(),
            keys=(key, key),
            terminal=(False, False),
            history_capacities=(2, 2),
            group_capacities={key: 2},
        )
    )

    allocations = allocate_variance_cost_budget(
        outer_ratios=(1.0, 1.0),
        statistics=statistics,
        history_capacities=(2, 2),
        history_groups=(key, key),
        group_capacities={key: 2},
        rollout_budget=budget,
        minimum_fresh=(1, 1),
    )

    assert [(item.history_count, item.fresh_count) for item in allocations] == [
        (2, 1),
        (0, 3),
    ]
    assert all(item.history_count + item.fresh_count == 3 for item in allocations)
