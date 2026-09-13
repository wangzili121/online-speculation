from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np
import pytest

from experiments.dllm.gsm8k_reproduction import (
    METHODS,
    _capped_generation_length,
    run_method,
)
from experiments.shared.paired_protocol import load_pairing
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionSample, DiffusionTraceStep
from inference_scaling.shared.evaluation import GSM8KProblem


class TinyExperimentBackend:
    def __init__(self, model_id: str, probability_one: float) -> None:
        self.model_id = model_id
        self.probability_one = probability_one

    def sample_batch(self, requests):
        outputs = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            tokens = tuple(
                int(rng.random() < self.probability_one)
                for _ in range(request.generation_length)
            )
            exact = request.sampling.has_exact_trajectory_density
            steps = []
            trajectory_logprob = 0.0
            for position, token in enumerate(tokens):
                probability = self.probability_one if token == 1 else 1 - self.probability_one
                value = log(probability) if exact else None
                trajectory_logprob += value or 0.0
                steps.append(
                    DiffusionTraceStep(
                        block_index=position,
                        step_index=0,
                        positions=(position,),
                        token_ids=(token,),
                        logprob=value,
                    )
                )
            outputs.append(
                DiffusionSample(
                    prefix=request.prefix,
                    token_ids=tokens,
                    trace=tuple(steps),
                    trajectory_logprob=trajectory_logprob if exact else None,
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                )
            )
        return outputs

    def score_trajectories(self, requests):
        values = []
        for request in requests:
            values.append(
                sum(
                    log(
                        self.probability_one
                        if token == 1
                        else 1 - self.probability_one
                    )
                    for token in request.sample.token_ids
                )
            )
        return values

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return f"calculation #### {token_ids[-1]}"


def _config():
    return {
        "generation": {
            "max_new_tokens": 2,
            "block_length": 1,
            "denoising_steps": 1,
            "temperature": 1.0,
            "remasking": "low_confidence",
        },
        "exact_policy": {
            "block_length": 1,
            "denoising_steps": 1,
            "temperature": 1.0,
            "remasking": "random",
        },
        "search": {"width": 2, "branching_factor": 2, "decision_block_size": 1},
        "best_of_n": {"samples": 2},
        "mh": {
            "alpha": 2.0,
            "updates": 3,
            "decision_block_size": 1,
            "updates_per_stage": 3,
            "reward_temperature": 0.5,
        },
        "conditional_is": {
            "candidate_count": 2,
            "rollout_count": 1,
            "decision_block_size": 1,
            "reward_temperature": 1.0,
            "importance_log_ratio_clip": 3.0,
        },
        "replay": {
            "history_rollouts": 2,
            "fresh_rollouts": 1,
            "truncation": 2.0,
        },
        "dynamic_is": {
            "auxiliary_probability": 0.5,
            "history_cost": 0.05,
            "fresh_cost": 1.0,
            "design_rollouts": 2,
        },
        "verifier": {
            "provider": "python",
            "name": "numeric_reference",
            "factory": (
                "inference_scaling.shared.evaluation.numeric:"
                "build_numeric_reference_verifier"
            ),
            "requires_reference": True,
        },
    }


@pytest.mark.parametrize("method", METHODS)
def test_every_declared_quality_method_executes_with_the_shared_protocol(method):
    base = TinyExperimentBackend("base", 0.7)
    proposal = TinyExperimentBackend("proposal", 0.4)
    problem = GSM8KProblem(3, "q", "#### 1", 1)

    tokens, diagnostics = run_method(
        method,
        base,
        problem,
        prompt=(7,),
        config=_config(),
        seed=41,
        proposal_backend=proposal,
    )

    assert len(tokens) == 2
    assert diagnostics


def test_reduced_layer_variants_separate_clipped_exact_and_uncorrected_weights():
    base = TinyExperimentBackend("base", 0.8)
    proposal = TinyExperimentBackend("proposal", 0.2)
    problem = GSM8KProblem(3, "q", "#### 1", 1)
    config = _config()

    _, clipped = run_method(
        "conditional_is_reduced_layer_proposal",
        base,
        problem,
        (7,),
        config,
        seed=9,
        proposal_backend=proposal,
    )
    _, unclipped = run_method(
        "conditional_is_reduced_layer_proposal_unclipped",
        base,
        problem,
        (7,),
        config,
        seed=9,
        proposal_backend=proposal,
    )
    _, uncorrected = run_method(
        "conditional_is_reduced_layer_proposal_uncorrected",
        base,
        problem,
        (7,),
        config,
        seed=9,
        proposal_backend=proposal,
    )

    assert clipped["importance_log_ratio_clip"] == 3.0
    assert unclipped["importance_log_ratio_clip"] is None
    assert unclipped["apply_importance_correction"] is True
    assert uncorrected["apply_importance_correction"] is False

    config["conditional_is"]["importance_log_ratio_clip"] = None
    _, configured_unclipped = run_method(
        "conditional_is_reduced_layer_proposal",
        base,
        problem,
        (7,),
        config,
        seed=9,
        proposal_backend=proposal,
    )
    assert configured_unclipped["importance_log_ratio_clip"] is None
    assert configured_unclipped["apply_importance_correction"] is True


def test_llada_generation_cap_retains_complete_diffusion_blocks():
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
    )

    assert _capped_generation_length(
        prompt_length=5, maximum=192, sampling=sampling
    ) == 192
    assert _capped_generation_length(
        prompt_length=5, maximum=190, sampling=sampling
    ) == 188


def test_every_configured_quality_and_passk_counterpart_is_executable():
    _, sections = load_pairing(Path("configs/gsm8k_llada_moe_3090.toml"))
    configured = {
        pair.dllm
        for section in ("main_pairs", "passk_pairs", "distribution_pairs")
        for pair in sections[section]
    }

    assert configured <= set(METHODS)
