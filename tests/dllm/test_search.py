from __future__ import annotations

from math import log

import numpy as np
import pytest

from inference_scaling.dllm.algorithms import (
    run_diffusion_block_beam,
    run_diffusion_trajectory_power_mh,
)
from inference_scaling.dllm.config import (
    DiffusionBlockBeamConfig,
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
)
from inference_scaling.dllm.types import DiffusionSample, DiffusionTraceStep


class BinaryTrajectoryBackend:
    model_id = "binary"

    def sample_batch(self, requests):
        samples = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            token = int(rng.choice(2, p=(0.8, 0.2)))
            logprob = log((0.8, 0.2)[token])
            steps = tuple(
                DiffusionTraceStep(
                    block_index=index,
                    step_index=0,
                    positions=(index,),
                    token_ids=(token,),
                    logprob=logprob,
                )
                for index in range(request.generation_length)
            )
            samples.append(
                DiffusionSample(
                    prefix=request.prefix,
                    token_ids=(token,) * request.generation_length,
                    trace=steps,
                    trajectory_logprob=logprob * request.generation_length,
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                )
            )
        return samples

    def score_trajectories(self, requests):
        return [float(request.sample.trajectory_logprob) for request in requests]


EXACT = DiffusionSamplingConfig(
    block_length=1,
    steps_per_block=1,
    temperature=1.0,
    remasking="random",
)


def test_trajectory_power_mh_sharpens_the_exact_trajectory_distribution():
    zeroes = 0
    runs = 2000
    for seed in range(runs):
        result = run_diffusion_trajectory_power_mh(
            backend=BinaryTrajectoryBackend(),
            prompt=(),
            config=DiffusionPowerMHConfig(
                total_length=1,
                decision_block_size=1,
                updates_per_stage=8,
                alpha=2.0,
            ),
            sampling=EXACT,
            seed=seed,
        )
        zeroes += result.final.token_ids[0] == 0

    expected = 0.8**2 / (0.8**2 + 0.2**2)
    assert zeroes / runs == pytest.approx(expected, abs=0.025)


def test_block_beam_retains_width_and_accumulates_stage_probabilities():
    result = run_diffusion_block_beam(
        backend=BinaryTrajectoryBackend(),
        prompt=(9,),
        config=DiffusionBlockBeamConfig(
            total_length=3,
            decision_block_size=1,
            width=4,
            branching_factor=2,
        ),
        sampling=EXACT,
        seed=4,
    )

    assert [stage.proposals for stage in result.stages] == [4, 8, 8]
    assert len(result.beams) == 4
    assert len(result.best.token_ids) == 3
    assert result.best.trajectory_logprob == sum(
        float(sample.trajectory_logprob) for sample in result.best.samples
    )


def test_search_algorithms_reject_intractable_remasking_policy():
    inexact = DiffusionSamplingConfig(
        block_length=1,
        steps_per_block=1,
        temperature=0.0,
        remasking="low_confidence",
    )
    with pytest.raises(ValueError, match="exact diffusion policy"):
        run_diffusion_trajectory_power_mh(
            backend=BinaryTrajectoryBackend(),
            prompt=(),
            config=DiffusionPowerMHConfig(
                total_length=1,
                decision_block_size=1,
                updates_per_stage=2,
                alpha=2.0,
            ),
            sampling=inexact,
        )
    with pytest.raises(ValueError, match="exact diffusion policy"):
        run_diffusion_block_beam(
            backend=BinaryTrajectoryBackend(),
            prompt=(),
            config=DiffusionBlockBeamConfig(
                total_length=1,
                decision_block_size=1,
                width=2,
                branching_factor=2,
            ),
            sampling=inexact,
        )
