from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.algorithms import (
    resample_diffusion_candidates,
    run_conditional_diffusion_is,
    run_diffusion_reward_mh,
)
from inference_scaling.dllm.backends import LLaDATransformersBackend
from inference_scaling.dllm.config import (
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionSamplingConfig,
)
from inference_scaling.dllm.types import DiffusionSample


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, bias, name):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path=name, mask_token_id=3)

    def forward(self, token_ids):
        batch, length = token_ids.shape
        return SimpleNamespace(
            logits=self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        )


class TinyTokenizer:
    mask_token_id = 3


def _backend(bias, name):
    return LLaDATransformersBackend(TinyMaskedModel(bias, name), TinyTokenizer())


def test_conditional_is_applies_same_trajectory_off_policy_ratio():
    base = _backend((0.0, 0.5, 1.0, -2.0), "base")
    proposal = _backend((1.0, 0.0, 0.5, -2.0), "proposal")
    candidate_sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=1,
        temperature=0.0,
        remasking="low_confidence",
    )
    proposal_sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=1.4,
        remasking="random",
    )
    target_sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=0.9,
        remasking="random",
    )

    result = run_conditional_diffusion_is(
        base_backend=base,
        prompt=(0,),
        config=DiffusionISConfig(
            candidate_count=2,
            rollout_count=3,
            block_size=2,
            total_length=4,
            reward_temperature=1.0,
        ),
        base_sampling=candidate_sampling,
        rollout_backend=proposal,
        rollout_sampling=proposal_sampling,
        target_rollout_backend=base,
        target_rollout_sampling=target_sampling,
        reward=lambda _prompt, continuation: float(sum(token == 2 for token in continuation)),
        seed=5,
    )

    assert len(result.token_ids) == 4
    first_rollouts = [rollout for candidate in result.steps[0].candidates for rollout in candidate.rollouts]
    assert len(first_rollouts) == 6
    assert all(rollout.raw_log_importance_ratio is not None for rollout in first_rollouts)
    assert all(rollout.target_trajectory_logprob is not None for rollout in first_rollouts)


def test_conditional_is_decision_block_can_span_native_diffusion_blocks():
    base = _backend((0.0, 0.5, 1.0, -2.0), "base")
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=1.0,
        remasking="random",
    )

    result = run_conditional_diffusion_is(
        base_backend=base,
        prompt=(0,),
        config=DiffusionISConfig(
            candidate_count=2,
            rollout_count=1,
            block_size=4,
            total_length=8,
            reward_temperature=1.0,
        ),
        base_sampling=sampling,
        reward=lambda _prompt, continuation: float(sum(continuation)),
        seed=17,
    )

    assert len(result.steps) == 2
    assert all(len(candidate.token_ids) == 4 for step in result.steps for candidate in step.candidates)
    assert len(result.token_ids) == 8


def test_conditional_is_rejects_decision_block_that_splits_native_block():
    base = _backend((0.0, 0.5, 1.0, -2.0), "base")
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        temperature=1.0,
        remasking="random",
    )

    with pytest.raises(ValueError, match="divisible by block_length"):
        run_conditional_diffusion_is(
            base_backend=base,
            prompt=(0,),
            config=DiffusionISConfig(
                candidate_count=2,
                rollout_count=1,
                block_size=6,
                total_length=12,
                reward_temperature=1.0,
            ),
            base_sampling=sampling,
            reward=lambda _prompt, continuation: float(sum(continuation)),
            seed=17,
        )


def test_uncorrected_off_policy_dllm_rollouts_skip_target_scoring():
    class NoScoreBackend:
        def __init__(self, backend):
            self.backend = backend
            self.model_id = backend.model_id

        def sample_batch(self, requests):
            return self.backend.sample_batch(requests)

        def score_trajectories(self, requests):
            raise AssertionError("uncorrected dLLM rollouts must not be rescored")

    base = NoScoreBackend(_backend((0.0, 0.5, 1.0, -2.0), "base"))
    proposal = _backend((1.0, 0.0, 0.5, -2.0), "proposal")
    candidate_sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=1,
        temperature=0.0,
        remasking="low_confidence",
    )
    proposal_sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=1.0,
        remasking="random",
    )

    result = run_conditional_diffusion_is(
        base_backend=base,
        prompt=(0,),
        config=DiffusionISConfig(
            candidate_count=2,
            rollout_count=2,
            block_size=2,
            total_length=4,
        ),
        base_sampling=candidate_sampling,
        rollout_backend=proposal,
        rollout_sampling=proposal_sampling,
        apply_importance_correction=False,
        reward=lambda _prompt, continuation: float(sum(continuation)),
        seed=8,
    )

    first_rollouts = [
        rollout
        for candidate in result.steps[0].candidates
        for rollout in candidate.rollouts
    ]
    assert all(item.raw_log_importance_ratio is None for item in first_rollouts)
    assert all(item.applied_log_importance_ratio is None for item in first_rollouts)
    assert all(item.log_weight == pytest.approx(item.reward) for item in first_rollouts)


def _empty_sample(value: int, request_id: str) -> DiffusionSample:
    return DiffusionSample(
        prefix=(),
        token_ids=(value,),
        trace=(),
        trajectory_logprob=None,
        policy_id="uniform",
        model_id="coin",
        request_id=request_id,
    )


def test_reward_only_sir_does_not_require_a_dllm_likelihood():
    samples = [_empty_sample(0, "zero"), _empty_sample(1, "one")]
    counts = 0
    for seed in range(3000):
        result = resample_diffusion_candidates(
            samples=samples,
            rewards=(0.0, 1.0),
            reward_temperature=1.0,
            rng=np.random.default_rng(seed),
        )
        counts += result.selected.sample.token_ids[0]
    expected = np.e / (1.0 + np.e)
    assert counts / 3000 == pytest.approx(expected, abs=0.025)


class CoinBackend:
    model_id = "coin"

    def sample_batch(self, requests):
        return [
            _empty_sample(int(np.random.default_rng(request.seed).integers(0, 2)), request.request_id)
            for request in requests
        ]

    def score_trajectories(self, requests):
        raise AssertionError("reward-only MH must not score dLLM likelihoods")


def test_independence_mh_approaches_base_times_reward_target_without_scores():
    ones = 0
    runs = 2500
    for seed in range(runs):
        result = run_diffusion_reward_mh(
            backend=CoinBackend(),
            prompt=(),
            config=DiffusionMHConfig(total_length=1, updates=8, reward_temperature=1.0),
            sampling=DiffusionSamplingConfig(
                block_length=1,
                steps_per_block=1,
                temperature=0.0,
            ),
            reward=lambda _prompt, continuation: float(continuation[0]),
            seed=seed,
        )
        ones += result.final.token_ids[0]
    expected = np.e / (1.0 + np.e)
    assert ones / runs == pytest.approx(expected, abs=0.03)
