from itertools import product

import pytest

from inference_scaling.arllm.algorithms.mh import (
    run_mh_chain,
    run_mh_chains,
    run_mh_chains_batched,
    run_reward_mh_chains,
    suffix_length_probabilities,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import MHConfig, RewardMHConfig, SamplingConfig
from inference_scaling.shared.metrics import empirical_distribution, total_variation
from inference_scaling.shared.rng import SeedStream
from inference_scaling.arllm.types import SequenceSample


def _power_target(probabilities: tuple[float, ...], length: int, alpha: float):
    weights = {
        sequence: float(__import__("math").prod(probabilities[token] for token in sequence) ** alpha)
        for sequence in product(range(len(probabilities)), repeat=length)
    }
    normalizer = sum(weights.values())
    return {sequence: weight / normalizer for sequence, weight in weights.items()}


def test_explicit_iterations_preserve_full_length_kernel_and_seed_stream():
    from inference_scaling.arllm.algorithms.mh import run_reward_mh_chain
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    sampling = SamplingConfig(temperature=0.8)
    explicit = MHConfig(alpha=2, total_length=7, block_size=2, steps_per_block=10, iterations=3)
    full_stage = MHConfig(alpha=2, total_length=7, block_size=7, steps_per_block=3)
    left = run_mh_chain(backend, (), explicit, sampling, SeedStream(2))
    right = run_mh_chain(backend, (), full_stage, sampling, SeedStream(2))
    assert left == right
    assert left.attempts == 3
    assert {step.stage_length for step in left.trace} == {7}
    reward = lambda prompt, completion: float(sum(completion))
    left = run_reward_mh_chain(backend, (), RewardMHConfig(total_length=7, block_size=2, iterations=3), sampling, reward, SeedStream(2))
    right = run_reward_mh_chain(backend, (), RewardMHConfig(total_length=7, block_size=7, steps_per_block=3), sampling, reward, SeedStream(2))
    assert left == right
    assert left.attempts == 3


def test_reward_mh_observer_preserves_rng_and_reports_every_complete_state():
    from inference_scaling.arllm.algorithms.mh import run_reward_mh_chain
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    config = RewardMHConfig(total_length=4, block_size=4, iterations=3)
    reward = lambda prompt, completion: float(sum(completion))
    states = []
    plain = run_reward_mh_chain(backend, (), config, SamplingConfig(), reward, SeedStream(12))
    observed = run_reward_mh_chain(backend, (), config, SamplingConfig(), reward, SeedStream(12), on_state=states.append)
    assert observed == plain
    assert [state.attempts for state in states] == [0, 1, 2, 3]
    assert states[-1] == observed


def test_mh_returns_fixed_length_and_all_suffix_starts_are_reachable() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    result = run_mh_chain(
        backend,
        (),
        MHConfig(alpha=2, total_length=5, block_size=2, steps_per_block=40),
        SamplingConfig(temperature=0.8),
        SeedStream(11),
    )
    assert len(result.token_ids) == 5
    assert {step.stage_length for step in result.trace} == {2, 4, 5}
    final_cuts = {step.cut for step in result.trace if step.stage_length == 5}
    assert final_cuts == set(range(5))
    assert 0 <= result.acceptance_rate <= 1


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_suffix_length_schedules_have_normalized_full_support(schedule: str) -> None:
    probabilities = suffix_length_probabilities(16, schedule)
    assert len(probabilities) == 16
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in probabilities)


def test_nonuniform_schedules_reduce_the_expected_proposed_suffix_length() -> None:
    lengths = range(1, 17)

    def expectation(schedule: str) -> float:
        return sum(
            length * probability
            for length, probability in zip(
                lengths, suffix_length_probabilities(16, schedule), strict=True
            )
        )

    uniform = expectation("uniform")
    assert expectation("inverse_length") < uniform
    assert expectation("multiscale") < uniform


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_mh_empirical_output_approaches_enumerated_power_target(
    schedule: str,
) -> None:
    probabilities = (0.65, 0.35)
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    config = MHConfig(
        alpha=2,
        total_length=2,
        block_size=2,
        steps_per_block=20,
        chains=2500,
        suffix_schedule=schedule,
    )
    outputs = run_mh_chains(
        backend,
        (),
        config,
        SamplingConfig(temperature=0.7),
        SeedStream(2026),
    )
    empirical = empirical_distribution(result.token_ids for result in outputs)
    target = _power_target(probabilities, length=2, alpha=2)
    assert total_variation(empirical, target) < 0.045


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_batched_mh_preserves_independent_chain_random_streams_exactly(
    schedule: str,
) -> None:
    config = MHConfig(
        alpha=3,
        total_length=7,
        block_size=2,
        steps_per_block=4,
        suffix_schedule=schedule,
    )
    proposal = SamplingConfig(temperature=0.5)
    roots = (SeedStream(17), SeedStream(29), SeedStream(41))
    sequential = tuple(
        run_mh_chain(
            TabularAutoregressiveBackend({}, fallback=[0.6, 0.3, 0.1]),
            (2,),
            config,
            proposal,
            root,
        )
        for root in roots
    )
    batched = run_mh_chains_batched(
        TabularAutoregressiveBackend({}, fallback=[0.6, 0.3, 0.1]),
        (2,),
        config,
        proposal,
        roots,
    )
    assert batched == sequential


def test_base_proposal_at_alpha_one_accepts_every_move() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2])
    result = run_mh_chain(
        backend,
        (),
        MHConfig(alpha=1, total_length=4, block_size=4, steps_per_block=20),
        SamplingConfig(),
        SeedStream(3),
    )
    assert result.accepted == result.attempts
    assert all(step.log_acceptance == pytest.approx(0.0) for step in result.trace)


def test_mh_reuses_reference_scores_emitted_during_proposal_generation() -> None:
    class DualScoreBackend:
        model_id = "dual-score"

        def __init__(self) -> None:
            self.score_calls = 0

        def sample_batch(self, requests):
            return [
                SequenceSample(
                    prefix=request.prefix,
                    token_ids=(0,) * request.max_new_tokens,
                    token_logprobs=(-0.2,) * request.max_new_tokens,
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                    reference_token_logprobs=(-0.4,) * request.max_new_tokens,
                    reference_policy_id=SamplingConfig().policy_id,
                )
                for request in requests
            ]

        def score_batch(self, requests):
            self.score_calls += 1
            raise AssertionError("cached reference scores should avoid rescoring")

    backend = DualScoreBackend()
    result = run_mh_chain(
        backend,
        (),
        MHConfig(alpha=2, total_length=4, block_size=2, steps_per_block=3),
        SamplingConfig(temperature=0.5),
        SeedStream(7),
    )
    assert len(result.token_ids) == 4
    assert backend.score_calls == 0


def test_reward_mh_approaches_enumerated_base_times_weight_target() -> None:
    probabilities = (0.7, 0.3)
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    temperature = 0.8

    def reward(_, sequence):
        return float(sequence == (1, 1))

    outputs = run_reward_mh_chains(
        backend,
        (),
        RewardMHConfig(
            total_length=2,
            block_size=1,
            steps_per_block=25,
            reward_temperature=temperature,
        ),
        SamplingConfig(temperature=0.7),
        reward,
        SeedStream(91),
        chains=3000,
    )
    weights = {
        sequence: float(
            __import__("math").prod(probabilities[token] for token in sequence)
            * __import__("math").exp(reward((), sequence) / temperature)
        )
        for sequence in product(range(2), repeat=2)
    }
    normalizer = sum(weights.values())
    target = {sequence: weight / normalizer for sequence, weight in weights.items()}
    empirical = empirical_distribution(result.token_ids for result in outputs)
    assert total_variation(empirical, target) < 0.04


@pytest.mark.parametrize(
    "sampling",
    [SamplingConfig(top_k=1), SamplingConfig(top_p=0.9), SamplingConfig(eos_token_id=1)],
)
def test_mh_rejects_proposals_that_break_fixed_support_contract(sampling) -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2])
    with pytest.raises(ValueError):
        run_mh_chain(
            backend,
            (),
            MHConfig(total_length=2, block_size=2, steps_per_block=1),
            sampling,
            SeedStream(0),
        )
