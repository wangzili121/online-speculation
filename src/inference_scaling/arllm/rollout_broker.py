"""Backend-independent rollout scheduling with resumable partial trajectories.

The broker operates on token sequences rather than engine-owned KV objects.  A
partial trajectory can therefore be resumed by Transformers or vLLM, and it
remains valid even when the serving engine has evicted its cache.  Every resumed
segment is sampled from the requested policy conditional on the tokens already
generated; the returned token log-probabilities are the probabilities that were
actually used for that segment.

Chunking bounds the work that can remain unfinished when an over-provisioned
batch reaches its completion target.  It is a portable scheduling primitive,
not a claim that old KV tensors remain valid after model weights change.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from inference_scaling.arllm.acceleration import sample_batch_with_callback
from inference_scaling.shared.rng import SeedStream
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    SequenceSample,
    TokenSequence,
)


@dataclass(frozen=True, slots=True)
class PartialRollout:
    """Serializable state for one interrupted autoregressive request."""

    request: GenerationRequest
    token_ids: TokenSequence = ()
    token_logprobs: tuple[float, ...] = ()
    reference_token_logprobs: tuple[float, ...] | None = None
    reference_policy_id: str | None = None
    policy_id: str | None = None
    model_id: str | None = None
    finish_reason: str = "partial"
    segments: int = 0
    priority: float = 0.0

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("partial rollout tokens and log-probabilities must align")
        if (self.reference_token_logprobs is None) != (
            self.reference_policy_id is None
        ):
            raise ValueError("partial reference probabilities require a policy id")
        if self.reference_token_logprobs is not None and len(
            self.reference_token_logprobs
        ) != len(self.token_ids):
            raise ValueError("partial reference probabilities must align with tokens")
        if len(self.token_ids) > self.request.max_new_tokens:
            raise ValueError("partial rollout exceeds its original token budget")
        if self.segments < 0:
            raise ValueError("partial rollout segment count must be non-negative")
        if self.complete and self.finish_reason == "partial":
            raise ValueError("a completed partial rollout needs a final finish reason")
        if not self.complete and self.finish_reason != "partial":
            raise ValueError("an unfinished rollout must use finish_reason='partial'")

    @classmethod
    def from_request(
        cls, request: GenerationRequest, *, priority: float = 0.0
    ) -> "PartialRollout":
        return cls(request=request, priority=float(priority))

    @property
    def remaining_tokens(self) -> int:
        return self.request.max_new_tokens - len(self.token_ids)

    @property
    def complete(self) -> bool:
        return self.finish_reason != "partial"

    def next_request(self, chunk_tokens: int) -> GenerationRequest:
        if self.complete or self.remaining_tokens <= 0:
            raise ValueError("completed rollout cannot be resumed")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        segment_length = min(int(chunk_tokens), self.remaining_tokens)
        seed = (
            self.request.seed
            if self.segments == 0
            else SeedStream(self.request.seed).derive(
                "partial-rollout", self.request.request_id, self.segments
            )
        )
        return GenerationRequest(
            prefix=self.request.prefix + self.token_ids,
            max_new_tokens=segment_length,
            sampling=self.request.sampling,
            seed=seed,
            request_id=f"{self.request.request_id}:segment:{self.segments}",
            uniforms=(
                self.request.uniforms[
                    len(self.token_ids) : len(self.token_ids) + segment_length
                ]
                if self.request.uniforms is not None
                else None
            ),
        )

    def append(
        self,
        segment_request: GenerationRequest,
        sample: SequenceSample,
    ) -> "PartialRollout":
        expected_prefix = self.request.prefix + self.token_ids
        if segment_request.prefix != expected_prefix or sample.prefix != expected_prefix:
            raise RuntimeError("partial rollout segment has the wrong prefix")
        if sample.request_id != segment_request.request_id:
            raise RuntimeError("partial rollout segment has the wrong request id")
        if sample.policy_id != self.request.sampling.policy_id:
            raise RuntimeError("partial rollout segment used the wrong policy")
        if len(sample.token_ids) > segment_request.max_new_tokens:
            raise RuntimeError("partial rollout segment exceeded its chunk budget")
        if not sample.token_ids and self.remaining_tokens:
            raise RuntimeError("partial rollout segment made no progress")
        if self.policy_id is not None and sample.policy_id != self.policy_id:
            raise RuntimeError("partial rollout changed policy between segments")
        if self.model_id is not None and sample.model_id != self.model_id:
            raise RuntimeError("partial rollout changed model between segments")
        if self.reference_policy_id is not None and (
            sample.reference_policy_id != self.reference_policy_id
        ):
            raise RuntimeError("partial rollout changed reference policy between segments")
        if self.reference_policy_id is None and self.segments and (
            sample.reference_policy_id is not None
        ):
            raise RuntimeError("partial rollout added reference scores after the first segment")
        if self.reference_policy_id is not None and sample.reference_token_logprobs is None:
            raise RuntimeError("partial rollout omitted previously available reference scores")

        tokens = self.token_ids + sample.token_ids
        token_logprobs = self.token_logprobs + sample.token_logprobs
        if sample.reference_token_logprobs is None:
            references = None
            reference_policy_id = None
        else:
            references = (self.reference_token_logprobs or ()) + sample.reference_token_logprobs
            reference_policy_id = sample.reference_policy_id
        finished = sample.finish_reason == "eos" or len(tokens) >= self.request.max_new_tokens
        finish_reason = sample.finish_reason if sample.finish_reason == "eos" else (
            "length" if finished else "partial"
        )
        return PartialRollout(
            request=self.request,
            token_ids=tokens,
            token_logprobs=token_logprobs,
            reference_token_logprobs=references,
            reference_policy_id=reference_policy_id,
            policy_id=sample.policy_id,
            model_id=sample.model_id,
            finish_reason=finish_reason,
            segments=self.segments + 1,
            priority=self.priority,
        )

    def to_sample(self) -> SequenceSample:
        if not self.complete:
            raise ValueError("unfinished rollout cannot be materialized as a sample")
        assert self.policy_id is not None and self.model_id is not None
        return SequenceSample(
            prefix=self.request.prefix,
            token_ids=self.token_ids,
            token_logprobs=self.token_logprobs,
            policy_id=self.policy_id,
            model_id=self.model_id,
            request_id=self.request.request_id,
            finish_reason=self.finish_reason,
            reference_token_logprobs=self.reference_token_logprobs,
            reference_policy_id=self.reference_policy_id,
        )


@dataclass(frozen=True, slots=True)
class RolloutBrokerSnapshot:
    requested_completion_target: int
    completed_rollouts: int
    partial_rollouts: int
    completion_overshoot: int
    physical_batches: int
    sampled_segments: int
    generated_tokens: int
    initial_partial_tokens: int
    partial_tokens_preserved: int
    resumed_prefill_tokens: int


@dataclass(frozen=True, slots=True)
class RolloutBrokerResult:
    completed: tuple[SequenceSample, ...]
    partial: tuple[PartialRollout, ...]
    snapshot: RolloutBrokerSnapshot


RolloutCompletionCallback = Callable[[SequenceSample], None]
RolloutSegmentCallback = Callable[[PartialRollout], None]


class AsyncRolloutBroker:
    """Run over-provisioned requests in bounded resumable chunks.

    Backends with a completion callback (notably persistent vLLM) retain their
    physical completion order.  Synchronous backends fall back to ordered batch
    completion.  Partial trajectories are always scheduled before untouched
    requests with the same explicit priority.
    """

    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        chunk_tokens: int = 16,
        max_batch_size: int | None = None,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.backend = backend
        self.chunk_tokens = int(chunk_tokens)
        self.max_batch_size = None if max_batch_size is None else int(max_batch_size)

    def run_until(
        self,
        rollouts: Sequence[PartialRollout | GenerationRequest],
        *,
        completion_target: int | None = None,
        on_complete: RolloutCompletionCallback | None = None,
        on_segment: RolloutSegmentCallback | None = None,
    ) -> RolloutBrokerResult:
        states = [
            item if isinstance(item, PartialRollout) else PartialRollout.from_request(item)
            for item in rollouts
        ]
        if not states:
            if completion_target not in (None, 0):
                raise ValueError("an empty broker run cannot complete requests")
            target = 0
        else:
            target = len(states) if completion_target is None else int(completion_target)
            if target <= 0 or target > len(states):
                raise ValueError("completion_target must lie within the submitted rollout count")
        request_ids = [state.request.request_id for state in states]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("rollout broker request ids must be unique")
        initially_complete = [state for state in states if state.complete]
        if initially_complete:
            raise ValueError("completed samples should not be resubmitted to the broker")

        initial_partial_tokens = sum(len(state.token_ids) for state in states)
        completed: list[SequenceSample] = []
        pending = list(states)
        physical_batches = 0
        sampled_segments = 0
        generated_tokens = 0
        resumed_prefill_tokens = 0

        while pending and len(completed) < target:
            pending.sort(
                key=lambda state: (
                    -state.priority,
                    -int(bool(state.token_ids)),
                    -len(state.token_ids),
                    state.request.request_id,
                )
            )
            limit = self.max_batch_size or len(pending)
            batch_states = pending[:limit]
            pending = pending[limit:]
            segment_requests = [
                state.next_request(self.chunk_tokens) for state in batch_states
            ]
            completion_order: list[int] = []
            order_lock = threading.Lock()

            def segment_completed(
                index: int,
                _sample: SequenceSample,
                order_lock=order_lock,
                completion_order=completion_order,
            ) -> None:
                with order_lock:
                    completion_order.append(index)

            samples = sample_batch_with_callback(
                self.backend, segment_requests, segment_completed
            )
            if len(samples) != len(batch_states):
                raise RuntimeError("rollout backend returned an invalid segment count")
            missing = [index for index in range(len(samples)) if index not in completion_order]
            completion_order.extend(missing)
            physical_batches += 1
            sampled_segments += len(samples)
            generated_tokens += sum(len(sample.token_ids) for sample in samples)
            resumed_prefill_tokens += sum(len(state.token_ids) for state in batch_states)

            updated: dict[int, PartialRollout] = {}
            for index in completion_order:
                if index in updated:
                    continue
                state = batch_states[index].append(segment_requests[index], samples[index])
                updated[index] = state
                if on_segment is not None:
                    on_segment(state)
                if state.complete:
                    sample = state.to_sample()
                    completed.append(sample)
                    if on_complete is not None:
                        on_complete(sample)
                else:
                    pending.append(state)

        partial_tokens = sum(len(state.token_ids) for state in pending)
        snapshot = RolloutBrokerSnapshot(
            requested_completion_target=target,
            completed_rollouts=len(completed),
            partial_rollouts=len(pending),
            completion_overshoot=max(0, len(completed) - target),
            physical_batches=physical_batches,
            sampled_segments=sampled_segments,
            generated_tokens=generated_tokens,
            initial_partial_tokens=initial_partial_tokens,
            partial_tokens_preserved=partial_tokens,
            resumed_prefill_tokens=resumed_prefill_tokens,
        )
        return RolloutBrokerResult(tuple(completed), tuple(pending), snapshot)
