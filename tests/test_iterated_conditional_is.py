from collections import Counter
from math import exp

import pytest

from inference_scaling.experimental.arllm.iterated_is import (
    iterated_conditional_is_step,
    run_iterated_conditional_is,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import (
    IteratedConditionalISConfig,
    SamplingConfig,
)
from inference_scaling.shared.metrics import total_variation
from inference_scaling.shared.rng import SeedStream


def _base_backend() -> TabularAutoregressiveBackend:
    return TabularAutoregressiveBackend(
        {
            (): [0.7, 0.3],
            (0,): [0.9, 0.1],
            (1,): [0.2, 0.8],
        },
        fallback=[0.5, 0.5],
        model_id="base",
    )


def _reward(_prompt, generated) -> float:
    return float(tuple(generated) == (1, 1))


def _first_token_target() -> dict[int, float]:
    first = (0.7, 0.3)
    continuation = ((0.9, 0.1), (0.2, 0.8))
    unnormalized = {
        candidate: first[candidate]
        * sum(
            continuation[candidate][suffix]
            * exp(_reward((), (candidate, suffix)))
            for suffix in (0, 1)
        )
        for candidate in (0, 1)
    }
    normalizer = sum(unnormalized.values())
    return {key: value / normalizer for key, value in unnormalized.items()}


@pytest.mark.parametrize("off_policy", [False, True])
def test_iterated_conditional_is_converges_with_a_finite_pool(off_policy) -> None:
    base = _base_backend()
    proposal = (
        TabularAutoregressiveBackend(
            {
                (0,): [0.35, 0.65],
                (1,): [0.75, 0.25],
            },
            fallback=[0.5, 0.5],
            model_id="proposal",
        )
        if off_policy
        else base
    )
    counts: Counter[int] = Counter()
    trials = 2500
    for trial in range(trials):
        step = iterated_conditional_is_step(
            base_backend=base,
            rollout_backend=proposal,
            prompt=(),
            generated_prefix=(),
            config=IteratedConditionalISConfig(
                pool_size=3,
                updates=14,
                rollout_count=1,
                block_size=1,
                total_length=2,
            ),
            base_sampling=SamplingConfig(),
            rollout_sampling=SamplingConfig(),
            reward=_reward,
            seeds=SeedStream(30_000 + trial),
            step_index=0,
        )
        counts[step.selected.token_ids[0]] += 1

    empirical = {token: count / trials for token, count in counts.items()}
    assert total_variation(empirical, _first_token_target()) < 0.04


def test_iterated_is_evaluates_only_distinct_fresh_states_and_reuses_current() -> None:
    class CountingBackend(TabularAutoregressiveBackend):
        def __init__(self):
            super().__init__({}, fallback=[0.6, 0.4])
            self.request_batches: list[tuple[str, ...]] = []

        def sample_batch(self, requests):
            self.request_batches.append(tuple(request.request_id for request in requests))
            return super().sample_batch(requests)

    backend = CountingBackend()
    config = IteratedConditionalISConfig(
        pool_size=3,
        updates=4,
        rollout_count=2,
        block_size=1,
        total_length=2,
    )
    step = iterated_conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=config,
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(),
        reward=lambda _prompt, sequence: float(sum(sequence)),
        seeds=SeedStream(71),
        step_index=0,
    )

    assert config.fresh_candidate_evaluations == 9
    assert config.pool_candidate_uses == 12
    assert len(step.evaluated_candidates) == 9
    assert len(step.transitions) == 4
    assert len(backend.request_batches) == 2
    assert len(backend.request_batches[0]) == 9
    assert len(backend.request_batches[1]) == 18
    for previous, current in zip(step.transitions, step.transitions[1:]):
        assert current.pool[0] is previous.selected


def test_iterated_conditional_is_respects_total_length() -> None:
    result = run_iterated_conditional_is(
        TabularAutoregressiveBackend({}, fallback=[0.5, 0.5]),
        (),
        IteratedConditionalISConfig(
            pool_size=2,
            updates=2,
            rollout_count=1,
            block_size=2,
            total_length=5,
        ),
        lambda _prompt, sequence: float(sum(sequence)),
        SeedStream(19),
    )
    assert len(result.token_ids) == 5
    assert [len(step.selected.token_ids) for step in result.steps] == [2, 2, 1]
    assert result.fresh_candidate_evaluations == 9
    assert result.reused_pool_entries == 6


def test_iterated_conditional_is_batches_a_fixed_pointwise_reward() -> None:
    calls: list[tuple[tuple[int, ...], ...]] = []

    def reward_batch(_prompt, sequences):
        batch = tuple(tuple(sequence) for sequence in sequences)
        calls.append(batch)
        return tuple(float(sum(sequence)) for sequence in batch)

    result = run_iterated_conditional_is(
        TabularAutoregressiveBackend({}, fallback=[0.5, 0.5]),
        (),
        IteratedConditionalISConfig(
            pool_size=2,
            updates=2,
            rollout_count=2,
            block_size=1,
            total_length=2,
        ),
        None,
        SeedStream(29),
        reward_batch=reward_batch,
    )

    assert len(result.steps) == 2
    assert len(calls) == 2
    assert len(calls[0]) == 6
    assert len(calls[1]) == 3


def test_iterated_conditional_is_configuration_rejects_degenerate_pool() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        IteratedConditionalISConfig(pool_size=1)
