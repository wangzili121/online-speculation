from itertools import product
from math import exp, prod

import pytest

from inference_scaling.arllm.algorithms.mh import run_reward_mh_chain
from inference_scaling.arllm.algorithms.mh_acceleration import (
    FrozenReplaySuffixProposal,
    run_reward_mh_chain_delayed,
    run_reward_mh_chain_prefetched,
    run_reward_mh_chain_replay_proposal,
    run_reward_mh_chains_replay_proposal,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import RewardMHConfig, SamplingConfig
from inference_scaling.shared.metrics import empirical_distribution, total_variation
from inference_scaling.shared.rng import SeedStream


def _reward_target(probabilities, *, length, temperature, reward):
    weights = {
        sequence: prod(probabilities[token] for token in sequence)
        * exp(reward((), sequence) / temperature)
        for sequence in product(range(len(probabilities)), repeat=length)
    }
    normalizer = sum(weights.values())
    return {sequence: weight / normalizer for sequence, weight in weights.items()}


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_one_step_prefetch_preserves_the_ordinary_mh_path_exactly(
    schedule: str,
) -> None:
    probabilities = (0.6, 0.3, 0.1)
    config = RewardMHConfig(
        total_length=5,
        block_size=2,
        steps_per_block=4,
        reward_temperature=0.7,
        suffix_schedule=schedule,
    )
    sampling = SamplingConfig(temperature=0.8)

    def reward(_, sequence):
        return float(sum(token == 2 for token in sequence))

    ordinary = run_reward_mh_chain(
        TabularAutoregressiveBackend({}, fallback=probabilities),
        (1,),
        config,
        sampling,
        reward,
        SeedStream(2026),
        chain_id=7,
    )
    prefetched = run_reward_mh_chain_prefetched(
        TabularAutoregressiveBackend({}, fallback=probabilities),
        (1,),
        config,
        sampling,
        reward,
        SeedStream(2026),
        chain_id=7,
    )

    assert prefetched.chain == ordinary
    assert prefetched.snapshot.used_proposals == config.updates
    assert prefetched.snapshot.prefetched_proposals == 2 * config.updates - 1
    assert prefetched.snapshot.unused_prefetched_proposals == config.updates - 1
    assert prefetched.snapshot.reward_evaluations == config.updates + 1


def test_delayed_acceptance_approaches_the_exact_reward_target() -> None:
    probabilities = (0.7, 0.3)
    temperature = 0.8
    config = RewardMHConfig(
        total_length=2,
        block_size=1,
        steps_per_block=20,
        reward_temperature=temperature,
    )

    def reward(_, sequence):
        return float(sequence == (1, 1))

    outputs = tuple(
        run_reward_mh_chain_delayed(
            TabularAutoregressiveBackend({}, fallback=probabilities),
            (),
            config,
            SamplingConfig(temperature=0.7),
            reward,
            lambda prompt, sequence: 0.6 * reward(prompt, sequence),
            SeedStream(91),
            chain_id=chain_id,
        )
        for chain_id in range(2500)
    )
    empirical = empirical_distribution(result.token_ids for result in outputs)
    target = _reward_target(
        probabilities,
        length=2,
        temperature=temperature,
        reward=reward,
    )
    assert total_variation(empirical, target) < 0.04


def test_delayed_acceptance_can_skip_exact_reward_calls() -> None:
    def reward(_, sequence):
        return float(sequence == (1,))

    config = RewardMHConfig(
        total_length=1,
        block_size=1,
        steps_per_block=120,
        reward_temperature=0.15,
    )
    result = run_reward_mh_chain_delayed(
        TabularAutoregressiveBackend({}, fallback=(0.5, 0.5)),
        (),
        config,
        SamplingConfig(),
        reward,
        reward,
        SeedStream(19),
    )
    assert result.exact_reward_evaluations < result.surrogate_reward_evaluations
    assert any(not step.exact_reward_evaluated for step in result.trace)
    assert all(
        step.stage_two_log_acceptance == pytest.approx(0.0)
        for step in result.trace
        if step.exact_reward_evaluated
    )


def test_replay_mixture_at_zero_reward_and_zero_history_weight_accepts_all() -> None:
    proposal = FrozenReplaySuffixProposal(
        TabularAutoregressiveBackend({}, fallback=(0.8, 0.2)),
        history_mixture=0.0,
    )
    config = RewardMHConfig(
        total_length=4,
        block_size=2,
        steps_per_block=8,
        reward_temperature=1.0,
    )
    result = run_reward_mh_chain_replay_proposal(
        proposal,
        (),
        config,
        lambda _prompt, _sequence: 0.0,
        SeedStream(4),
    )
    assert result.accepted == result.attempts
    assert all(step.log_acceptance == pytest.approx(0.0) for step in result.trace)


def test_frozen_replay_proposal_approaches_the_exact_reward_target() -> None:
    probabilities = (0.65, 0.35)
    temperature = 0.7
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    proposal = FrozenReplaySuffixProposal(backend, history_mixture=0.65)
    proposal.observe_sequences((), ((1, 1),) * 40 + ((1, 0),) * 10)

    def reward(_, sequence):
        return float(sequence == (1, 1))

    outputs = run_reward_mh_chains_replay_proposal(
        proposal,
        (),
        RewardMHConfig(
            total_length=2,
            block_size=1,
            steps_per_block=20,
            reward_temperature=temperature,
        ),
        reward,
        SeedStream(117),
        chains=2500,
    )
    empirical = empirical_distribution(result.token_ids for result in outputs)
    target = _reward_target(
        probabilities,
        length=2,
        temperature=temperature,
        reward=reward,
    )
    assert total_variation(empirical, target) < 0.04
    snapshot = proposal.snapshot()
    assert snapshot.base_draws > 0
    assert snapshot.history_draws > 0
    assert snapshot.logprob_queries > 0


def test_replay_history_is_frozen_before_the_chain_starts() -> None:
    proposal = FrozenReplaySuffixProposal(
        TabularAutoregressiveBackend({}, fallback=(0.5, 0.5)),
        history_mixture=0.5,
    )
    proposal.observe_sequence((), (1, 1))
    proposal.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        proposal.observe_sequence((), (0, 0))
