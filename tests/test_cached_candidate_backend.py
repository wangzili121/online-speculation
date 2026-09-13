import pytest

from inference_scaling.arllm.backends import CachedCandidateBackend, TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest


def test_cached_candidate_backend_replays_subset_without_resampling() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.4], model_id="proposal")
    sampling = SamplingConfig()
    requests = [
        GenerationRequest((), 2, sampling, 11, "candidate-0"),
        GenerationRequest((), 2, sampling, 12, "candidate-1"),
    ]
    samples = backend.sample_batch(requests)
    replay = CachedCandidateBackend(backend, samples)

    assert replay.sample_batch([requests[1]]) == [samples[1]]
    continuation = samples[0].token_ids
    assert replay.score_batch([ScoreRequest((), (continuation,), sampling)]) == (
        backend.score_batch([ScoreRequest((), (continuation,), sampling)])
    )


def test_cached_candidate_backend_rejects_unknown_request() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.4], model_id="proposal")
    sampling = SamplingConfig()
    sample = backend.sample_batch(
        [GenerationRequest((), 1, sampling, 11, "candidate-0")]
    )[0]
    replay = CachedCandidateBackend(backend, [sample])

    with pytest.raises(KeyError, match="absent from the frozen cache"):
        replay.sample_batch(
            [GenerationRequest((), 1, sampling, 12, "missing-candidate")]
        )
