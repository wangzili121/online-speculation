from collections import Counter
from math import exp

import pytest

from inference_scaling.arllm.algorithms.conditional_is import (
    conditional_is_step,
    run_conditional_is,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import ConditionalISConfig, SamplingConfig
from inference_scaling.shared.metrics import total_variation
from inference_scaling.shared.rng import SeedStream
from inference_scaling.experimental.shared.rqmc import (
    randomized_lattice_uniforms,
    scrambled_sobol_uniforms,
)
from inference_scaling.arllm.types import ScoreRequest


def _backend() -> TabularAutoregressiveBackend:
    return TabularAutoregressiveBackend(
        {
            (): [0.7, 0.3],
            (0,): [0.9, 0.1],
            (1,): [0.2, 0.8],
        },
        fallback=[0.5, 0.5],
    )


def _reward(_prompt, generated) -> float:
    return 1.0 if tuple(generated) == (1, 1) else 0.0


def _exact_first_token_target() -> dict[int, float]:
    base_first = (0.7, 0.3)
    completion = ((0.9, 0.1), (0.2, 0.8))
    weights = []
    for candidate in (0, 1):
        weight = sum(
            completion[candidate][token] * exp(_reward((), (candidate, token)))
            for token in (0, 1)
        )
        weights.append(weight)
    weights = [base_first[index] * weights[index] for index in (0, 1)]
    total = sum(weights)
    return {index: weights[index] / total for index in (0, 1)}


@pytest.mark.parametrize("off_policy", [False, True])
def test_first_candidate_approaches_exact_conditional_is_target(off_policy) -> None:
    backend = _backend()
    config = ConditionalISConfig(
        candidate_count=12,
        rollout_count=8,
        block_size=1,
        total_length=2,
        reward_temperature=1.0,
    )
    counts: Counter[int] = Counter()
    trials = 500
    for trial in range(trials):
        step = conditional_is_step(
            base_backend=backend,
            rollout_backend=backend,
            prompt=(),
            generated_prefix=(),
            config=config,
            base_sampling=SamplingConfig(),
            rollout_sampling=SamplingConfig(temperature=0.55 if off_policy else 1.0),
            reward=_reward,
            seeds=SeedStream(10_000 + trial),
            step_index=0,
        )
        counts[step.selected.token_ids[0]] += 1
    empirical = {token: count / trials for token, count in counts.items()}
    assert total_variation(empirical, _exact_first_token_target()) < 0.08


def test_off_policy_ratio_scores_only_rollout_suffix() -> None:
    backend = _backend()
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalISConfig(
            candidate_count=2, rollout_count=2, block_size=1, total_length=2
        ),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(temperature=0.5),
        reward=_reward,
        seeds=SeedStream(91),
        step_index=0,
    )
    for candidate in step.candidates:
        for rollout in candidate.rollouts:
            assert len(rollout.token_ids) == 1
            assert rollout.log_weight == pytest.approx(
                rollout.reward + rollout.base_logprob - rollout.proposal_logprob
            )


def test_temperature_scaled_base_policy_is_used_in_off_policy_ratio() -> None:
    backend = _backend()
    base_sampling = SamplingConfig(temperature=0.8)
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalISConfig(
            candidate_count=2, rollout_count=2, block_size=1, total_length=2
        ),
        base_sampling=base_sampling,
        rollout_sampling=SamplingConfig(temperature=0.5),
        reward=_reward,
        seeds=SeedStream(192),
        step_index=0,
    )
    for candidate in step.candidates:
        for rollout in candidate.rollouts:
            expected = backend.score_batch(
                [
                    ScoreRequest(
                        candidate.token_ids,
                        (rollout.token_ids,),
                        base_sampling,
                    )
                ]
            )[0]
            assert rollout.base_logprob == pytest.approx(sum(expected))


def test_optional_log_ratio_clipping_is_explicit_in_rollout_record() -> None:
    backend = _backend()
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalISConfig(
            candidate_count=4,
            rollout_count=4,
            block_size=1,
            total_length=2,
            importance_log_ratio_clip=0.05,
        ),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(temperature=0.25),
        reward=_reward,
        seeds=SeedStream(293),
        step_index=0,
    )
    rollouts = [
        rollout for candidate in step.candidates for rollout in candidate.rollouts
    ]
    assert all(abs(item.applied_log_importance_ratio) <= 0.05 for item in rollouts)
    assert any(
        item.raw_log_importance_ratio != item.applied_log_importance_ratio
        for item in rollouts
    )
    for item in rollouts:
        assert item.log_weight == pytest.approx(
            item.reward + item.applied_log_importance_ratio
        )


def test_uncorrected_off_policy_rollouts_skip_base_rescoring() -> None:
    class NoScoreBackend(TabularAutoregressiveBackend):
        def score_batch(self, requests):
            raise AssertionError("uncorrected proposal rollouts must not be rescored")

    base = NoScoreBackend({}, fallback=[0.6, 0.4], model_id="base")
    proposal = TabularAutoregressiveBackend(
        {}, fallback=[0.2, 0.8], model_id="proposal"
    )
    step = conditional_is_step(
        base_backend=base,
        rollout_backend=proposal,
        prompt=(),
        generated_prefix=(),
        config=ConditionalISConfig(
            candidate_count=3,
            rollout_count=2,
            block_size=1,
            total_length=2,
            apply_importance_correction=False,
        ),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(),
        reward=_reward,
        seeds=SeedStream(394),
        step_index=0,
    )
    rollouts = [
        rollout for candidate in step.candidates for rollout in candidate.rollouts
    ]
    assert all(item.base_logprob is None for item in rollouts)
    assert all(item.raw_log_importance_ratio is None for item in rollouts)
    assert all(item.applied_log_importance_ratio is None for item in rollouts)
    assert all(item.log_weight == pytest.approx(item.reward) for item in rollouts)


def test_uncorrected_rollouts_reject_irrelevant_ratio_clipping() -> None:
    with pytest.raises(ValueError, match="importance_log_ratio_clip requires"):
        ConditionalISConfig(
            importance_log_ratio_clip=1.0,
            apply_importance_correction=False,
        )


def test_rollout_budget_subtracts_candidate_block() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalISConfig(
            candidate_count=3, rollout_count=2, block_size=2, total_length=5
        ),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(sum(generated)),
        seeds=SeedStream(4),
        step_index=0,
    )
    assert all(len(candidate.token_ids) == 2 for candidate in step.candidates)
    assert all(
        len(rollout.token_ids) == 3
        for candidate in step.candidates
        for rollout in candidate.rollouts
    )


def test_conditional_is_never_exceeds_total_length() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    result = run_conditional_is(
        backend,
        (),
        ConditionalISConfig(
            candidate_count=2, rollout_count=2, block_size=2, total_length=5
        ),
        lambda _prompt, generated: float(sum(generated)),
        SeedStream(17),
    )
    assert len(result.token_ids) == 5
    assert [len(step.selected.token_ids) for step in result.steps] == [2, 2, 1]


@pytest.mark.parametrize(
    "candidate_sampling,rollout_sampling",
    [
        (SamplingConfig(top_p=0.9), SamplingConfig()),
        (SamplingConfig(top_k=1), SamplingConfig()),
        (SamplingConfig(), SamplingConfig(top_p=0.9)),
        (SamplingConfig(), SamplingConfig(top_k=1)),
    ],
)
def test_conditional_is_rejects_sampling_policies_that_break_the_weight_formula(
    candidate_sampling, rollout_sampling
) -> None:
    with pytest.raises(ValueError):
        run_conditional_is(
            _backend(),
            (),
            ConditionalISConfig(
                candidate_count=2, rollout_count=2, block_size=1, total_length=2
            ),
            _reward,
            SeedStream(1),
            base_sampling=candidate_sampling,
            rollout_sampling=rollout_sampling,
        )


def test_conditional_is_accepts_one_joint_batch_reward() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    seen: list[tuple[tuple[int, ...], ...]] = []

    def reward_batch(_prompt, generated):
        seen.append(tuple(generated))
        return tuple(float(tokens[-1] == 1) for tokens in generated)

    result = run_conditional_is(
        backend,
        (),
        ConditionalISConfig(
            candidate_count=2,
            rollout_count=2,
            block_size=1,
            total_length=2,
            reward_temperature=1.0,
        ),
        None,
        SeedStream(91),
        reward_batch=reward_batch,
    )

    assert len(result.token_ids) == 2
    assert seen
    assert len(seen[0]) == 4


def test_scrambled_sobol_rollouts_receive_one_point_set_per_candidate() -> None:
    class RecordingBackend(TabularAutoregressiveBackend):
        def __init__(self):
            super().__init__({}, fallback=(0.6, 0.4))
            self.generation_calls = []

        def sample_batch(self, requests):
            self.generation_calls.append(tuple(requests))
            return super().sample_batch(requests)

    backend = RecordingBackend()
    seed = 731
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalISConfig(
            candidate_count=2,
            rollout_count=4,
            block_size=1,
            total_length=4,
            rollout_design="scrambled_sobol",
        ),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(sum(generated)),
        seeds=SeedStream(seed),
        step_index=0,
    )

    assert len(step.candidates) == 2
    rollout_requests = backend.generation_calls[1]
    assert len(rollout_requests) == 8
    for candidate_index in range(2):
        expected = scrambled_sobol_uniforms(
            4,
            3,
            seed=SeedStream(seed).derive(
                "conditional_is",
                0,
                "candidate",
                candidate_index,
                "scrambled_sobol",
            ),
        )
        observed = tuple(
            request.uniforms
            for request in rollout_requests[
                candidate_index * 4 : (candidate_index + 1) * 4
            ]
        )
        assert observed == expected


def test_arithmetic_lattice_rollouts_receive_one_shifted_grid_per_candidate() -> None:
    class RecordingBackend(TabularAutoregressiveBackend):
        def __init__(self):
            super().__init__({}, fallback=(0.6, 0.4))
            self.generation_calls = []

        def sample_batch(self, requests):
            self.generation_calls.append(tuple(requests))
            return super().sample_batch(requests)

    backend = RecordingBackend()
    seed = 829
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalISConfig(
            candidate_count=2,
            rollout_count=4,
            block_size=1,
            total_length=4,
            rollout_design="arithmetic_lattice",
        ),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(sum(generated)),
        seeds=SeedStream(seed),
        step_index=0,
    )

    assert len(step.candidates) == 2
    rollout_requests = backend.generation_calls[1]
    assert len(rollout_requests) == 8
    for candidate_index in range(2):
        expected = randomized_lattice_uniforms(
            4,
            seed=SeedStream(seed).derive(
                "conditional_is",
                0,
                "candidate",
                candidate_index,
                "arithmetic_lattice",
            ),
        )
        observed = tuple(
            request.arithmetic_uniform
            for request in rollout_requests[
                candidate_index * 4 : (candidate_index + 1) * 4
            ]
        )
        assert observed == expected
        assert all(request.uniforms is None for request in rollout_requests)


def test_scrambled_sobol_rejects_batch_coupled_reward() -> None:
    with pytest.raises(ValueError, match="fixed pointwise reward"):
        run_conditional_is(
            _backend(),
            (),
            ConditionalISConfig(
                candidate_count=2,
                rollout_count=2,
                block_size=1,
                total_length=2,
                rollout_design="scrambled_sobol",
            ),
            None,
            SeedStream(19),
            reward_batch=lambda _prompt, generated: [0.0] * len(generated),
        )


def test_arithmetic_lattice_rejects_batch_coupled_reward() -> None:
    with pytest.raises(ValueError, match="fixed pointwise reward"):
        run_conditional_is(
            _backend(),
            (),
            ConditionalISConfig(
                candidate_count=2,
                rollout_count=2,
                block_size=1,
                total_length=2,
                rollout_design="arithmetic_lattice",
            ),
            None,
            SeedStream(20),
            reward_batch=lambda _prompt, generated: [0.0] * len(generated),
        )


def test_exact_bounded_early_stop_matches_full_algorithm_and_skips_rollouts() -> None:
    full_config = ConditionalISConfig(
        candidate_count=3,
        rollout_count=4,
        block_size=1,
        total_length=2,
    )
    early_config = ConditionalISConfig(
        candidate_count=3,
        rollout_count=4,
        block_size=1,
        total_length=2,
        exact_rollout_early_stop=True,
        rollout_log_weight_bounds=(0.0, 0.0),
        rollout_evaluation_batch_size=1,
    )
    for seed in range(20):
        full = run_conditional_is(
            _backend(),
            (),
            full_config,
            lambda _prompt, _generated: 0.0,
            SeedStream(seed),
        )
        early = run_conditional_is(
            _backend(),
            (),
            early_config,
            lambda _prompt, _generated: 0.0,
            SeedStream(seed),
        )

        assert early.token_ids == full.token_ids
        assert [step.selected_index for step in early.steps] == [
            step.selected_index for step in full.steps
        ]
        assert early.steps[0].rollout_evaluations_planned == 12
        assert early.steps[0].rollout_evaluations_performed == 3
        assert early.steps[0].rollout_evaluations_skipped == 9
        assert early.steps[0].selection_invariant_verified is True
        assert all(
            candidate.planned_rollout_count == 4
            and len(candidate.rollouts) == 1
            and candidate.log_weight_lower_bound == pytest.approx(0.0)
            and candidate.log_weight_upper_bound == pytest.approx(0.0)
            for candidate in early.steps[0].candidates
        )


def test_bounded_staged_evaluation_matches_full_algorithm_with_variable_weights() -> (
    None
):
    full_config = ConditionalISConfig(
        candidate_count=4,
        rollout_count=4,
        block_size=1,
        total_length=2,
    )
    early_config = ConditionalISConfig(
        candidate_count=4,
        rollout_count=4,
        block_size=1,
        total_length=2,
        exact_rollout_early_stop=True,
        rollout_log_weight_bounds=(0.0, 2.0),
        rollout_evaluation_batch_size=1,
    )
    reward = lambda _prompt, generated: float(sum(generated))
    for seed in range(50):
        full = run_conditional_is(
            _backend(),
            (),
            full_config,
            reward,
            SeedStream(10_000 + seed),
        )
        early = run_conditional_is(
            _backend(),
            (),
            early_config,
            reward,
            SeedStream(10_000 + seed),
        )
        assert early.token_ids == full.token_ids
        assert [step.selected_index for step in early.steps] == [
            step.selected_index for step in full.steps
        ]


def test_bounded_early_stop_rejects_invalid_weight_claim_and_batch_reward() -> None:
    config = ConditionalISConfig(
        candidate_count=2,
        rollout_count=2,
        block_size=1,
        total_length=2,
        exact_rollout_early_stop=True,
        rollout_log_weight_bounds=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="outside the declared bounds"):
        run_conditional_is(
            _backend(),
            (),
            config,
            lambda _prompt, _generated: 1.0,
            SeedStream(31),
        )
    with pytest.raises(ValueError, match="fixed pointwise reward"):
        run_conditional_is(
            _backend(),
            (),
            config,
            None,
            SeedStream(31),
            reward_batch=lambda _prompt, generated: [0.0] * len(generated),
        )


def test_bounded_staged_off_policy_weights_match_complete_evaluation() -> None:
    base = TabularAutoregressiveBackend({}, fallback=(0.8, 0.2), model_id="base")
    proposal = TabularAutoregressiveBackend(
        {}, fallback=(0.5, 0.5), model_id="proposal"
    )
    full_config = ConditionalISConfig(
        candidate_count=3,
        rollout_count=4,
        block_size=1,
        total_length=2,
    )
    early_config = ConditionalISConfig(
        candidate_count=3,
        rollout_count=4,
        block_size=1,
        total_length=2,
        exact_rollout_early_stop=True,
        rollout_log_weight_bounds=(-2.0, 2.0),
        rollout_evaluation_batch_size=2,
    )
    for seed in range(20):
        full = run_conditional_is(
            base,
            (),
            full_config,
            lambda _prompt, _generated: 0.0,
            SeedStream(20_000 + seed),
            rollout_backend=proposal,
        )
        staged = run_conditional_is(
            base,
            (),
            early_config,
            lambda _prompt, _generated: 0.0,
            SeedStream(20_000 + seed),
            rollout_backend=proposal,
        )
        assert staged.token_ids == full.token_ids
        assert [step.selected_index for step in staged.steps] == [
            step.selected_index for step in full.steps
        ]
