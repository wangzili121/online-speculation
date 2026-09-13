import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from inference_scaling.arllm.backends import (
    ContinuousBatchingBackend,
    TabularAutoregressiveBackend,
)
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest


class RecordingBackend:
    def __init__(self) -> None:
        self.backend = TabularAutoregressiveBackend({}, fallback=[0.65, 0.35])
        self.lock = threading.Lock()
        self.sample_batch_sizes: list[int] = []
        self.score_batch_sizes: list[int] = []

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def sample_batch(self, requests):
        with self.lock:
            self.sample_batch_sizes.append(len(requests))
        return self.backend.sample_batch(requests)

    def score_batch(self, requests):
        with self.lock:
            self.score_batch_sizes.append(
                sum(len(request.continuations) for request in requests)
            )
        return self.backend.score_batch(requests)


def test_concurrent_sampling_is_coalesced_and_seed_stable() -> None:
    recording = RecordingBackend()
    requests = [
        GenerationRequest((), 3, SamplingConfig(), 100 + index, f"request-{index}")
        for index in range(12)
    ]
    expected = recording.backend.sample_batch(requests)
    with ContinuousBatchingBackend(
        recording,
        max_batch_size=12,
        max_batch_tokens=100,
        batch_wait_seconds=0.02,
    ) as batched:
        barrier = threading.Barrier(len(requests))

        def run(request):
            barrier.wait()
            return batched.sample_batch([request])[0]

        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            actual = list(executor.map(run, requests))
        snapshot = batched.snapshot()

    assert actual == expected
    assert max(recording.sample_batch_sizes) > 1
    assert snapshot.sample_requests == len(requests)
    assert snapshot.maximum_sample_batch > 1


def test_caller_sample_groups_are_not_split_to_fill_an_unrelated_batch() -> None:
    recording = RecordingBackend()
    request_groups = [
        [
            GenerationRequest(
                (group,), 3, SamplingConfig(), 100 * group + index, f"{group}-{index}"
            )
            for index in range(4)
        ]
        for group in range(2)
    ]
    expected = [recording.backend.sample_batch(group) for group in request_groups]
    with ContinuousBatchingBackend(
        recording,
        max_batch_size=6,
        max_batch_tokens=100,
        batch_wait_seconds=0.02,
    ) as batched:
        barrier = threading.Barrier(2)

        def run(group):
            barrier.wait()
            return batched.sample_batch(group)

        with ThreadPoolExecutor(max_workers=2) as executor:
            actual = list(executor.map(run, request_groups))

    assert actual == expected
    assert sorted(recording.sample_batch_sizes) == [4, 4]


def test_compatible_caller_groups_are_merged_without_losing_order() -> None:
    recording = RecordingBackend()
    request_groups = [
        [
            GenerationRequest(
                (group,), 3, SamplingConfig(), 100 * group + index, f"{group}-{index}"
            )
            for index in range(4)
        ]
        for group in range(2)
    ]
    expected = [recording.backend.sample_batch(group) for group in request_groups]
    with ContinuousBatchingBackend(
        recording,
        max_batch_size=8,
        max_batch_tokens=100,
        batch_wait_seconds=0.02,
    ) as batched:
        barrier = threading.Barrier(2)

        def run(group):
            barrier.wait()
            return batched.sample_batch(group)

        with ThreadPoolExecutor(max_workers=2) as executor:
            actual = list(executor.map(run, request_groups))

    assert actual == expected
    assert recording.sample_batch_sizes == [8]


def test_oversized_rollout_group_splits_on_repeated_prefix_boundaries() -> None:
    recording = RecordingBackend()
    requests = [
        GenerationRequest(
            (candidate,),
            3,
            SamplingConfig(),
            100 * candidate + rollout,
            f"{candidate}-{rollout}",
        )
        for candidate in range(15)
        for rollout in range(3)
    ]
    expected = recording.backend.sample_batch(requests)
    with ContinuousBatchingBackend(
        recording,
        max_batch_size=32,
        max_batch_tokens=1_000,
        batch_wait_seconds=0.0,
    ) as batched:
        actual = batched.sample_batch(requests)

    assert actual == expected
    assert recording.sample_batch_sizes == [30, 15]


def test_score_requests_are_flattened_and_split_without_reordering() -> None:
    recording = RecordingBackend()
    requests = [
        ScoreRequest((index % 2,), ((0,), (1, 0)), SamplingConfig())
        for index in range(6)
    ]
    expected = recording.backend.score_batch(requests)
    with ContinuousBatchingBackend(
        recording,
        max_batch_size=12,
        max_batch_tokens=100,
        batch_wait_seconds=0.01,
    ) as batched:
        actual = batched.score_batch(requests)
        snapshot = batched.snapshot()

    assert actual == expected
    assert snapshot.score_sequences == 12
    assert snapshot.maximum_score_batch == 12


def test_closed_batching_backend_rejects_new_work() -> None:
    batched = ContinuousBatchingBackend(RecordingBackend())
    batched.close()
    with pytest.raises(RuntimeError, match="closed"):
        batched.sample_batch(
            [GenerationRequest((), 1, SamplingConfig(), 1, "after-close")]
        )


def test_native_continuous_backend_is_not_serialized_by_dispatcher() -> None:
    class NativeBackend(RecordingBackend):
        supports_native_continuous_batching = True

    raw = NativeBackend()
    batched = ContinuousBatchingBackend(raw)
    requests = [
        GenerationRequest((), 1, SamplingConfig(), seed, f"request-{seed}")
        for seed in (1, 2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda item: batched.sample_batch([item]), requests))
    assert [group[0].request_id for group in outputs] == ["request-1", "request-2"]
    assert raw.sample_batch_sizes == [1, 1]
    assert batched.snapshot().sample_batches == 2
    batched.close()
