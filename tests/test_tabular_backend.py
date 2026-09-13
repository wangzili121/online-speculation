import numpy as np

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest


def test_sample_logprob_matches_actual_truncated_policy() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.3, 0.1])
    sampling = SamplingConfig(temperature=1.0, top_k=2)
    request = GenerationRequest((), 4, sampling, 5, "sample")
    sample = backend.sample_batch([request])[0]

    assert 2 not in sample.token_ids
    rescored = backend.score_batch([ScoreRequest((), (sample.token_ids,), sampling)])[0]
    np.testing.assert_allclose(rescored, sample.token_logprobs)


def test_base_and_behavior_scores_are_distinct() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2])
    continuation = (1, 0)
    base = backend.score_batch([ScoreRequest((), (continuation,), None)])[0]
    behavior = backend.score_batch(
        [ScoreRequest((), (continuation,), SamplingConfig(temperature=0.5))]
    )[0]
    assert not np.allclose(base, behavior)


def test_explicit_uniforms_use_float64_inverse_cdf() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.2, 0.3, 0.5])
    request = GenerationRequest(
        (),
        6,
        SamplingConfig(),
        5,
        "explicit-uniforms",
        uniforms=(0.0, 0.1999, 0.2, 0.4999, 0.5, 0.9999),
    )

    sample = backend.sample_batch([request])[0]

    assert sample.token_ids == (0, 0, 1, 1, 2, 2)
    np.testing.assert_allclose(
        sample.token_logprobs,
        np.log([0.2, 0.2, 0.3, 0.3, 0.5, 0.5]),
    )


def test_arithmetic_uniform_is_rescaled_inside_each_selected_interval() -> None:
    backend = TabularAutoregressiveBackend(
        {
            (): [0.7, 0.3],
            (1,): [0.2, 0.8],
        },
        fallback=[0.5, 0.5],
    )
    request = GenerationRequest(
        (),
        2,
        SamplingConfig(),
        5,
        "arithmetic-uniform",
        arithmetic_uniform=0.75,
    )

    sample = backend.sample_batch([request])[0]

    # 0.75 first selects token 1 from [0.7, 1), then rescales to 1/6.
    # The next row is ordered by decreasing probability, so 1/6 selects token 1.
    assert sample.token_ids == (1, 1)
    np.testing.assert_allclose(sample.token_logprobs, np.log([0.3, 0.8]))


def test_arithmetic_sampling_has_the_autoregressive_sequence_distribution() -> None:
    backend = TabularAutoregressiveBackend(
        {
            (): [0.7, 0.3],
            (0,): [0.9, 0.1],
            (1,): [0.2, 0.8],
        }
    )
    grid = np.linspace(0.0, 1.0, 10_000, endpoint=False) + 0.5 / 10_000
    requests = [
        GenerationRequest(
            (),
            2,
            SamplingConfig(),
            index,
            f"arithmetic:{index}",
            arithmetic_uniform=float(value),
        )
        for index, value in enumerate(grid)
    ]

    samples = backend.sample_batch(requests)
    frequencies = {
        sequence: sum(sample.token_ids == sequence for sample in samples) / len(samples)
        for sequence in ((0, 0), (0, 1), (1, 0), (1, 1))
    }

    assert frequencies == {
        (0, 0): 0.63,
        (0, 1): 0.07,
        (1, 0): 0.06,
        (1, 1): 0.24,
    }
