import pytest

from inference_scaling.arllm.config import (
    BaseReplayConfig,
    ConditionalISConfig,
    DynamicISConfig,
    MHConfig,
    SamplingConfig,
)
from inference_scaling.arllm.types import GenerationRequest, SequenceSample


def test_sampling_config_identifies_actual_policy() -> None:
    config = SamplingConfig(temperature=0.7, top_p=0.9, top_k=20, eos_token_id=2)
    assert config.policy_id == "temperature=0.7;top_p=0.9;top_k=20;eos=2"


def test_policy_id_preserves_distinct_float_values() -> None:
    assert (
        SamplingConfig(temperature=1.0000001).policy_id
        != SamplingConfig(temperature=1.0000002).policy_id
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SamplingConfig(temperature=0),
        lambda: SamplingConfig(top_p=1.1),
        lambda: MHConfig(total_length=4, block_size=8),
        lambda: MHConfig(suffix_schedule="unknown"),
        lambda: BaseReplayConfig(fresh_rollouts=0),
        lambda: DynamicISConfig(auxiliary_mixture=1.0),
        lambda: SamplingConfig(temperature=float("nan")),
        lambda: SamplingConfig(top_p=float("inf")),
        lambda: ConditionalISConfig(reward_temperature=float("inf")),
        lambda: ConditionalISConfig(rollout_design="unknown"),
        lambda: ConditionalISConfig(exact_rollout_early_stop=True),
        lambda: ConditionalISConfig(
            rollout_log_weight_bounds=(0.0, 1.0),
        ),
        lambda: ConditionalISConfig(
            exact_rollout_early_stop=True,
            rollout_log_weight_bounds=(1.0, 0.0),
        ),
        lambda: ConditionalISConfig(
            rollout_design="scrambled_sobol",
            exact_rollout_early_stop=True,
            rollout_log_weight_bounds=(0.0, 1.0),
        ),
        lambda: DynamicISConfig(auxiliary_mixture=float("nan")),
    ],
)
def test_invalid_configs_fail_early(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_sampled_token_logprob_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        SequenceSample((), (1,), (float("nan"),), "policy", "model", "request")


@pytest.mark.parametrize(
    "uniforms",
    [(0.1,), (0.1, float("nan")), (0.1, 1.0), (0.1, -0.1)],
)
def test_generation_request_validates_explicit_uniforms(uniforms) -> None:
    with pytest.raises(ValueError, match="uniform"):
        GenerationRequest((), 2, SamplingConfig(), 1, "invalid", uniforms=uniforms)


def test_generation_request_validates_arithmetic_uniform() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        GenerationRequest(
            (),
            2,
            SamplingConfig(),
            1,
            "invalid",
            uniforms=(0.1, 0.2),
            arithmetic_uniform=0.3,
        )
    for value in (-0.1, 1.0, float("inf")):
        with pytest.raises(ValueError, match="arithmetic sampling uniform"):
            GenerationRequest(
                (),
                2,
                SamplingConfig(),
                1,
                "invalid",
                arithmetic_uniform=value,
            )
