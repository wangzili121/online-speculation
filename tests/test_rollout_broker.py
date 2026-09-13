from __future__ import annotations

from collections import Counter
from itertools import product

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.metrics import total_variation
from inference_scaling.arllm.rollout_broker import AsyncRolloutBroker
from inference_scaling.arllm.types import GenerationRequest, SequenceSample


class _SegmentBackend:
    model_id = "segments"

    def sample_batch(self, requests):
        return [self._sample(request) for request in requests]

    def sample_batch_with_callback(self, requests, on_complete):
        samples = [self._sample(request) for request in requests]
        for index in reversed(range(len(samples))):
            on_complete(index, samples[index])
        return samples

    @staticmethod
    def _sample(request):
        root = request.request_id.split(":segment:")[0]
        eos = root == "fast" and request.request_id.endswith(":segment:0")
        tokens = (1,) if eos else (0,) * request.max_new_tokens
        return SequenceSample(
            prefix=request.prefix,
            token_ids=tokens,
            token_logprobs=(-0.1,) * len(tokens),
            policy_id=request.sampling.policy_id,
            model_id="segments",
            request_id=request.request_id,
            finish_reason="eos" if eos else "length",
        )

    def score_batch(self, requests):
        return [tuple(-0.1 for _ in continuation) for request in requests for continuation in request.continuations]


def test_broker_stops_at_target_and_preserves_overprovisioned_partial_tokens() -> None:
    sampling = SamplingConfig(eos_token_id=1)
    requests = [
        GenerationRequest((), 4, sampling, index, request_id)
        for index, request_id in enumerate(("slow-a", "fast", "slow-b"))
    ]
    broker = AsyncRolloutBroker(_SegmentBackend(), chunk_tokens=1)
    first = broker.run_until(requests, completion_target=1)

    assert [sample.request_id for sample in first.completed] == ["fast"]
    assert {state.request.request_id for state in first.partial} == {"slow-a", "slow-b"}
    assert all(len(state.token_ids) == 1 for state in first.partial)
    assert first.snapshot.partial_tokens_preserved == 2

    resumed = broker.run_until(first.partial)
    assert {sample.request_id for sample in resumed.completed} == {"slow-a", "slow-b"}
    assert all(len(sample.token_ids) == 4 for sample in resumed.completed)
    assert resumed.snapshot.initial_partial_tokens == 2
    assert resumed.snapshot.resumed_prefill_tokens > 0


def test_chunked_rollouts_preserve_the_requested_autoregressive_distribution() -> None:
    probabilities = (0.7, 0.3)
    backend = TabularAutoregressiveBackend({}, fallback=probabilities, model_id="base")
    broker = AsyncRolloutBroker(backend, chunk_tokens=1, max_batch_size=32)
    requests = [
        GenerationRequest((), 3, SamplingConfig(), 10_000 + index, f"sample-{index}")
        for index in range(4000)
    ]
    results = broker.run_until(requests).completed
    empirical_counts = Counter(sample.token_ids for sample in results)
    empirical = {sequence: count / len(results) for sequence, count in empirical_counts.items()}
    target = {
        sequence: probabilities[sequence[0]]
        * probabilities[sequence[1]]
        * probabilities[sequence[2]]
        for sequence in product(range(2), repeat=3)
    }
    assert total_variation(empirical, target) < 0.025


def test_chunked_rollout_preserves_explicit_uniform_stream() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.7, 0.3), model_id="base")
    broker = AsyncRolloutBroker(backend, chunk_tokens=1)
    request = GenerationRequest(
        (),
        3,
        SamplingConfig(),
        77,
        "explicit",
        uniforms=(0.1, 0.9, 0.2),
    )

    result = broker.run_until([request]).completed

    assert len(result) == 1
    assert result[0].token_ids == (0, 1, 0)
