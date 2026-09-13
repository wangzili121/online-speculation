"""Cross-request continuous batching for synchronous algorithm callers."""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Literal

from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
)

_Kind = Literal["sample", "score"]
_STOP = object()


@dataclass(slots=True)
class _QueuedRequestGroup:
    kind: _Kind
    requests: tuple[GenerationRequest | ScoreRequest, ...]
    future: Future
    sequence_count: int
    token_cost: int
    batch_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class BatchingSnapshot:
    sample_batches: int
    score_batches: int
    sample_requests: int
    score_sequences: int
    maximum_sample_batch: int
    maximum_score_batch: int

    @property
    def average_sample_batch(self) -> float:
        return self.sample_requests / self.sample_batches if self.sample_batches else 0.0

    @property
    def average_score_batch(self) -> float:
        return self.score_sequences / self.score_batches if self.score_batches else 0.0


class ContinuousBatchingBackend:
    """Merge requests from concurrent prompts without changing their random streams.

    Algorithms keep using the synchronous backend protocol.  Independent prompt
    workers can share this wrapper; a background worker combines compatible caller
    groups that become ready within ``batch_wait_seconds`` and dispatches one backend
    call.  Keeping each caller group intact preserves repeated-prefix KV reuse.  An
    oversized generation group is split only at repeated-prefix run boundaries when
    possible.
    """

    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        max_batch_size: int = 32,
        max_batch_tokens: int = 4096,
        batch_wait_seconds: float = 0.002,
    ) -> None:
        if max_batch_size <= 0 or max_batch_tokens <= 0:
            raise ValueError("batch limits must be positive")
        if batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be non-negative")
        self._backend = backend
        self._native_passthrough = bool(
            getattr(backend, "supports_native_continuous_batching", False)
        )
        self._max_batch_size = int(max_batch_size)
        self._max_batch_tokens = int(max_batch_tokens)
        self._batch_wait_seconds = float(batch_wait_seconds)
        self._queue: queue.Queue[_QueuedRequestGroup | object] = queue.Queue()
        self._state_lock = threading.Lock()
        self._statistics_lock = threading.Lock()
        self._closed = False
        self._sample_batches = 0
        self._score_batches = 0
        self._sample_requests = 0
        self._score_sequences = 0
        self._maximum_sample_batch = 0
        self._maximum_score_batch = 0
        self._worker: threading.Thread | None = None
        if not self._native_passthrough:
            self._worker = threading.Thread(
                target=self._run,
                name=f"batching-backend:{backend.model_id}",
                daemon=True,
            )
            self._worker.start()

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @staticmethod
    def _generation_token_cost(request: GenerationRequest) -> int:
        return max(1, len(request.prefix) + request.max_new_tokens)

    @staticmethod
    def _score_token_cost(request: ScoreRequest) -> int:
        return max(
            1,
            sum(len(request.prefix) + len(continuation) for continuation in request.continuations),
        )

    def _submit(self, item: _QueuedRequestGroup) -> Future:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("continuous batching backend is closed")
            self._queue.put(item)
        return item.future

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        if not requests:
            return []
        if self._native_passthrough:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("continuous batching backend is closed")
            self._record_batch("sample", len(requests))
            return self._backend.sample_batch(requests)
        futures = [
            self._submit(
                _QueuedRequestGroup(
                    kind="sample",
                    requests=tuple(group),
                    future=Future(),
                    sequence_count=len(group),
                    token_cost=sum(self._generation_token_cost(item) for item in group),
                    batch_key=self._sample_batch_key(group),
                )
            )
            for group in self._sample_request_groups(requests)
        ]
        outputs: list[SequenceSample] = []
        for future in futures:
            outputs.extend(future.result())
        return outputs

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        if not requests:
            return []
        if self._native_passthrough:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("continuous batching backend is closed")
            sequence_count = sum(len(request.continuations) for request in requests)
            self._record_batch("score", sequence_count)
            return self._backend.score_batch(requests)
        groups = self._score_request_groups(requests)
        futures = [
            self._submit(
                _QueuedRequestGroup(
                    kind="score",
                    requests=tuple(group),
                    future=Future(),
                    sequence_count=sum(len(item.continuations) for item in group),
                    token_cost=sum(self._score_token_cost(item) for item in group),
                    batch_key=self._score_batch_key(group),
                )
            )
            for group in groups
        ]
        flattened: list[tuple[float, ...]] = []
        for future in futures:
            flattened.extend(future.result())
        return flattened

    @staticmethod
    def _sample_batch_key(
        requests: Sequence[GenerationRequest],
    ) -> tuple[object, ...]:
        first_sampling = requests[0].sampling
        sampling = (
            first_sampling
            if all(request.sampling == first_sampling for request in requests)
            else None
        )
        first_length = requests[0].max_new_tokens
        maximum_new_tokens = (
            first_length
            if all(request.max_new_tokens == first_length for request in requests)
            else None
        )
        repeat_counts = set(Counter(request.prefix for request in requests).values())
        uniform_prefix_repeats = repeat_counts.pop() if len(repeat_counts) == 1 else None
        return "sample", sampling, maximum_new_tokens, uniform_prefix_repeats

    @staticmethod
    def _score_batch_key(requests: Sequence[ScoreRequest]) -> tuple[object, ...]:
        first_sampling = requests[0].sampling
        sampling = (
            first_sampling
            if all(request.sampling == first_sampling for request in requests)
            else None
        )
        maximum_length = max(
            (
                len(request.prefix) + len(continuation)
                for request in requests
                for continuation in request.continuations
            ),
            default=0,
        )
        length_bucket = ((maximum_length + 63) // 64) * 64
        return "score", sampling, length_bucket

    def _within_limits(self, sequence_count: int, token_cost: int) -> bool:
        return (
            sequence_count <= self._max_batch_size
            and token_cost <= self._max_batch_tokens
        )

    def _sample_request_groups(
        self, requests: Sequence[GenerationRequest]
    ) -> list[tuple[GenerationRequest, ...]]:
        requests = tuple(requests)
        total_cost = sum(self._generation_token_cost(request) for request in requests)
        if self._within_limits(len(requests), total_cost):
            return [requests]

        runs: list[list[GenerationRequest]] = []
        run_keys: list[tuple[object, ...]] = []
        for request in requests:
            key = (request.sampling, request.prefix, request.max_new_tokens)
            if not runs or key != run_keys[-1]:
                runs.append([])
                run_keys.append(key)
            runs[-1].append(request)

        groups: list[tuple[GenerationRequest, ...]] = []
        current: list[GenerationRequest] = []
        current_cost = 0

        def flush() -> None:
            nonlocal current, current_cost
            if current:
                groups.append(tuple(current))
                current = []
                current_cost = 0

        for run in runs:
            remaining = list(run)
            while remaining:
                remaining_cost = sum(
                    self._generation_token_cost(request) for request in remaining
                )
                if self._within_limits(
                    len(current) + len(remaining), current_cost + remaining_cost
                ):
                    current.extend(remaining)
                    current_cost += remaining_cost
                    remaining.clear()
                    continue
                if current:
                    flush()
                    continue

                take = 0
                taken_cost = 0
                for request in remaining:
                    request_cost = self._generation_token_cost(request)
                    if take and not self._within_limits(
                        take + 1, taken_cost + request_cost
                    ):
                        break
                    take += 1
                    taken_cost += request_cost
                    if not self._within_limits(take, taken_cost):
                        break
                take = max(1, take)
                current.extend(remaining[:take])
                current_cost = sum(
                    self._generation_token_cost(request) for request in current
                )
                del remaining[:take]
                if remaining or not self._within_limits(len(current), current_cost):
                    flush()
        flush()
        return groups

    def _score_request_groups(
        self, requests: Sequence[ScoreRequest]
    ) -> list[tuple[ScoreRequest, ...]]:
        groups: list[tuple[ScoreRequest, ...]] = []
        current: list[ScoreRequest] = []
        sequence_count = 0
        token_cost = 0
        for request in requests:
            request_sequences = len(request.continuations)
            request_cost = self._score_token_cost(request)
            if current and not self._within_limits(
                sequence_count + request_sequences, token_cost + request_cost
            ):
                groups.append(tuple(current))
                current = []
                sequence_count = 0
                token_cost = 0
            current.append(request)
            sequence_count += request_sequences
            token_cost += request_cost
            if not self._within_limits(sequence_count, token_cost):
                groups.append(tuple(current))
                current = []
                sequence_count = 0
                token_cost = 0
        if current:
            groups.append(tuple(current))
        return groups

    def _fits(
        self,
        batch: list[_QueuedRequestGroup],
        candidate: _QueuedRequestGroup,
        sequence_count: int,
        token_cost: int,
    ) -> bool:
        if candidate.kind != batch[0].kind or candidate.batch_key != batch[0].batch_key:
            return False
        return (
            sequence_count + candidate.sequence_count <= self._max_batch_size
            and token_cost + candidate.token_cost <= self._max_batch_tokens
        )

    def _record_batch(self, kind: _Kind, sequence_count: int) -> None:
        with self._statistics_lock:
            if kind == "sample":
                self._sample_batches += 1
                self._sample_requests += sequence_count
                self._maximum_sample_batch = max(
                    self._maximum_sample_batch, sequence_count
                )
            else:
                self._score_batches += 1
                self._score_sequences += sequence_count
                self._maximum_score_batch = max(self._maximum_score_batch, sequence_count)

    def _dispatch(self, batch: list[_QueuedRequestGroup]) -> None:
        kind = batch[0].kind
        sequence_count = sum(item.sequence_count for item in batch)
        self._record_batch(kind, sequence_count)
        try:
            if kind == "sample":
                requests = [
                    request
                    for item in batch
                    for request in item.requests
                ]
                outputs = self._backend.sample_batch(requests)  # type: ignore[arg-type]
                if len(outputs) != sequence_count:
                    raise RuntimeError("underlying backend returned an invalid sample batch")
                offset = 0
                for item in batch:
                    end = offset + item.sequence_count
                    item.future.set_result(outputs[offset:end])
                    offset = end
                return

            requests = [
                request
                for item in batch
                for request in item.requests
            ]
            outputs = self._backend.score_batch(requests)  # type: ignore[arg-type]
            expected = sum(item.sequence_count for item in batch)
            if len(outputs) != expected:
                raise RuntimeError("underlying backend returned an invalid score batch")
            offset = 0
            for item in batch:
                end = offset + item.sequence_count
                item.future.set_result(outputs[offset:end])
                offset = end
        except BaseException as error:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(error)

    @staticmethod
    def _take_compatible_pending(
        pending: deque[_QueuedRequestGroup],
        batch: list[_QueuedRequestGroup],
        sequence_count: int,
        token_cost: int,
        fits,
    ) -> tuple[int, int]:
        retained: deque[_QueuedRequestGroup] = deque()
        while pending:
            candidate = pending.popleft()
            if fits(batch, candidate, sequence_count, token_cost):
                batch.append(candidate)
                sequence_count += candidate.sequence_count
                token_cost += candidate.token_cost
            else:
                retained.append(candidate)
        pending.extend(retained)
        return sequence_count, token_cost

    def _run(self) -> None:
        pending: deque[_QueuedRequestGroup] = deque()
        stopping = False
        while True:
            if pending:
                first = pending.popleft()
            else:
                queued = self._queue.get()
                if queued is _STOP:
                    break
                assert isinstance(queued, _QueuedRequestGroup)
                first = queued
            batch = [first]
            sequence_count = first.sequence_count
            token_cost = first.token_cost
            sequence_count, token_cost = self._take_compatible_pending(
                pending,
                batch,
                sequence_count,
                token_cost,
                self._fits,
            )
            deadline = time.monotonic() + self._batch_wait_seconds
            while sequence_count < self._max_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    queued = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if queued is _STOP:
                    stopping = True
                    break
                assert isinstance(queued, _QueuedRequestGroup)
                if self._fits(batch, queued, sequence_count, token_cost):
                    batch.append(queued)
                    sequence_count += queued.sequence_count
                    token_cost += queued.token_cost
                else:
                    pending.append(queued)
            self._dispatch(batch)
            if stopping and not pending:
                break

    def snapshot(self) -> BatchingSnapshot:
        with self._statistics_lock:
            return BatchingSnapshot(
                sample_batches=self._sample_batches,
                score_batches=self._score_batches,
                sample_requests=self._sample_requests,
                score_sequences=self._score_sequences,
                maximum_sample_batch=self._maximum_sample_batch,
                maximum_score_batch=self._maximum_score_batch,
            )

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            if self._native_passthrough:
                return
            self._queue.put(_STOP)
        assert self._worker is not None
        self._worker.join()

    def __enter__(self) -> "ContinuousBatchingBackend":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
