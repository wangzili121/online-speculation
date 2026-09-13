from math import isinf

from inference_scaling.arllm.backends import AbsorbingEOSBackend, TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest


def test_absorbing_eos_pads_samples_and_scores_forced_tail() -> None:
    inner = TabularAutoregressiveBackend({}, fallback=(0.0, 1.0))
    backend = AbsorbingEOSBackend(inner, eos_token_id=1)
    request = GenerationRequest((), 4, SamplingConfig(), 3, "sample")
    sample = backend.sample_batch([request])[0]

    assert sample.token_ids == (1, 1, 1, 1)
    assert sample.token_logprobs == (0.0, 0.0, 0.0, 0.0)
    assert sample.policy_id == request.sampling.policy_id
    assert backend.score_batch([ScoreRequest((), ((1, 1, 1),), None)]) == [
        (0.0, 0.0, 0.0)
    ]
    invalid = backend.score_batch([ScoreRequest((), ((1, 0),), None)])[0]
    assert invalid[0] == 0.0
    assert isinf(invalid[1]) and invalid[1] < 0


def test_absorbing_eos_handles_terminal_prefix_without_model_call() -> None:
    inner = TabularAutoregressiveBackend({}, fallback=(1.0, 0.0))
    backend = AbsorbingEOSBackend(inner, eos_token_id=1)
    sample = backend.sample_batch(
        [GenerationRequest((0, 1), 3, SamplingConfig(), 5, "terminal")]
    )[0]
    assert sample.token_ids == (1, 1, 1)
    assert sample.token_logprobs == (0.0, 0.0, 0.0)


def test_prompt_eos_is_not_treated_as_generated_eos() -> None:
    inner = TabularAutoregressiveBackend({}, fallback=(0.0, 1.0))
    backend = AbsorbingEOSBackend(inner, eos_token_id=1, absorbing_after=2)
    sample = backend.sample_batch(
        [GenerationRequest((1, 0), 2, SamplingConfig(), 7, "prompt-eos")]
    )[0]
    assert sample.token_ids == (1, 1)
