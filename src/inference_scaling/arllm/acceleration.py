"""Distribution-preserving rollout acceleration primitives.

The objects in this module deliberately separate two kinds of reuse:

* statistical replay records are estimator inputs and must retain their sampling
  probabilities and lifecycle;
* draft-cache entries only predict likely tokens.  The base model verifies every
  draft, so entries may come from old, off-policy, or already-consumed rollouts.

Keeping those stores separate makes it possible to improve wall-clock efficiency
without silently reusing a statistical sample twice.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    SequenceSample,
    TokenSequence,
)
from inference_scaling.shared.verifier import TokenReward


@dataclass(frozen=True, slots=True)
class SpeculationTier:
    """Use at most ``draft_tokens`` when the active batch is at most ``max_batch``."""

    max_batch: int
    draft_tokens: int

    def __post_init__(self) -> None:
        if self.max_batch <= 0:
            raise ValueError("max_batch must be positive")
        if self.draft_tokens < 0:
            raise ValueError("draft_tokens must be non-negative")


@dataclass(frozen=True, slots=True)
class ActiveBatchSpeculationConfig:
    """Load-aware speculative schedule shared by Transformers and vLLM.

    The default is intentionally aggressive only in the long tail.  At high
    concurrency ordinary batching normally has better arithmetic intensity, so
    speculation is disabled instead of multiplying verification work.
    """

    tiers: tuple[SpeculationTier, ...] = (
        SpeculationTier(4, 8),
        SpeculationTier(16, 4),
        SpeculationTier(512, 0),
    )
    min_context_tokens: int = 2
    min_token_probability: float = 0.1
    tree_max_context_tokens: int = 24
    tree_max_contexts: int = 100_000
    vllm_max_cached_requests: int = 10_000
    stochastic_tree: bool = False

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("at least one speculation tier is required")
        previous = 0
        for tier in self.tiers:
            if tier.max_batch <= previous:
                raise ValueError("speculation tiers must have increasing max_batch values")
            previous = tier.max_batch
        if self.min_context_tokens <= 0:
            raise ValueError("min_context_tokens must be positive")
        if not 0 <= self.min_token_probability <= 1:
            raise ValueError("min_token_probability must lie in [0, 1]")
        if self.tree_max_context_tokens <= 0 or self.tree_max_contexts <= 0:
            raise ValueError("token-tree limits must be positive")
        if self.vllm_max_cached_requests < 0:
            raise ValueError("vllm_max_cached_requests must be non-negative")

    @property
    def maximum_draft_tokens(self) -> int:
        return max(tier.draft_tokens for tier in self.tiers)

    def draft_tokens(self, active_batch: int) -> int:
        if active_batch <= 0:
            raise ValueError("active_batch must be positive")
        for tier in self.tiers:
            if active_batch <= tier.max_batch:
                return tier.draft_tokens
        return self.tiers[-1].draft_tokens

    def vllm_batch_schedule(self) -> list[list[int]]:
        """Return vLLM's inclusive ``[start_bs, end_bs, K]`` table."""

        start = 1
        schedule: list[list[int]] = []
        for tier in self.tiers:
            schedule.append([start, tier.max_batch, tier.draft_tokens])
            start = tier.max_batch + 1
        return schedule

    def vllm_suffix_config(self, *, dynamic: bool = False) -> dict[str, Any]:
        """Build the native suffix-decoding engine configuration.

        Native suffix decoding verifies drafts with the target model.  The
        optional batch table is passed through only when requested because vLLM
        still labels dynamic schedules on non-EAGLE proposers as less mature.
        """

        if self.stochastic_tree:
            raise ValueError(
                "stochastic_tree uses an explicit empirical proposal and residual "
                "correction available only in the Transformers verifier; vLLM's "
                "native suffix proposer must be benchmarked as a separate arm"
            )
        if self.maximum_draft_tokens <= 0:
            raise ValueError("native suffix speculation needs at least one positive tier")
        result: dict[str, Any] = {
            "method": "custom_class" if dynamic else "suffix",
            "num_speculative_tokens": self.maximum_draft_tokens,
            "suffix_decoding_max_tree_depth": self.tree_max_context_tokens,
            "suffix_decoding_max_cached_requests": self.vllm_max_cached_requests,
            "suffix_decoding_max_spec_factor": 1.0,
            "suffix_decoding_min_token_prob": self.min_token_probability,
        }
        if dynamic:
            result["model"] = (
                "inference_scaling.arllm.vllm_suffix_proposer."
                "DynamicSuffixDecodingProposer"
            )
            result["num_speculative_tokens_per_batch_size"] = self.vllm_batch_schedule()
        return result


@dataclass(frozen=True, slots=True)
class DraftModelSpeculationConfig:
    """Exact target-model sampling assisted by a smaller draft model.

    Transformers currently supports native assisted generation only for one
    active request.  ``single_request_only`` therefore preserves ordinary
    target-model batching whenever more than one request is submitted.  Setting
    it to ``False`` is an explicit diagnostic arm that executes those requests
    one at a time.
    """

    draft_tokens: int = 4
    single_request_only: bool = True
    confidence_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.draft_tokens <= 0:
            raise ValueError("draft_tokens must be positive")
        if (
            self.confidence_threshold is not None
            and not 0 < self.confidence_threshold < 1
        ):
            raise ValueError("confidence_threshold must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class DraftProposal:
    token_ids: TokenSequence
    token_probabilities: tuple[float, ...]
    matched_context_tokens: int
    token_distributions: tuple[tuple[tuple[int, float], ...], ...] = ()

    @property
    def stochastic(self) -> bool:
        return bool(self.token_distributions)


@dataclass(frozen=True, slots=True)
class RolloutTokenTreeSnapshot:
    observed_sequences: int
    observed_tokens: int
    contexts: int
    evicted_contexts: int
    queries: int
    hits: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.queries if self.queries else 0.0

    @property
    def acceptance_rate(self) -> float:
        verified = self.accepted_tokens + self.rejected_tokens
        return self.accepted_tokens / verified if verified else 0.0


class RolloutTokenTree:
    """A bounded CPU suffix table built from any historical rollout source.

    It stores empirical next-token counts for suffix contexts.  No probability
    from this table is ever used as an IS weight: its output is a deterministic
    draft and must be verified by the base model.
    """

    def __init__(
        self,
        *,
        max_context_tokens: int = 24,
        max_contexts: int = 100_000,
        min_context_tokens: int = 2,
        min_token_probability: float = 0.1,
    ) -> None:
        if max_context_tokens <= 0 or max_contexts <= 0:
            raise ValueError("token-tree limits must be positive")
        if min_context_tokens <= 0 or min_context_tokens > max_context_tokens:
            raise ValueError("min_context_tokens must lie within the context limit")
        if not 0 <= min_token_probability <= 1:
            raise ValueError("min_token_probability must lie in [0, 1]")
        self.max_context_tokens = int(max_context_tokens)
        self.max_contexts = int(max_contexts)
        self.min_context_tokens = int(min_context_tokens)
        self.min_token_probability = float(min_token_probability)
        self._contexts: OrderedDict[TokenSequence, Counter[int]] = OrderedDict()
        self._lock = threading.RLock()
        self._observed_sequences = 0
        self._observed_tokens = 0
        self._evicted_contexts = 0
        self._queries = 0
        self._hits = 0
        self._proposed_tokens = 0
        self._accepted_tokens = 0
        self._rejected_tokens = 0

    @classmethod
    def from_config(cls, config: ActiveBatchSpeculationConfig) -> "RolloutTokenTree":
        return cls(
            max_context_tokens=config.tree_max_context_tokens,
            max_contexts=config.tree_max_contexts,
            min_context_tokens=config.min_context_tokens,
            min_token_probability=config.min_token_probability,
        )

    def observe(self, tokens: TokenSequence) -> None:
        values = tuple(int(token) for token in tokens)
        if len(values) <= self.min_context_tokens:
            return
        with self._lock:
            self._observed_sequences += 1
            self._observed_tokens += len(values)
            for position in range(self.min_context_tokens, len(values)):
                maximum = min(self.max_context_tokens, position)
                next_token = values[position]
                for length in range(self.min_context_tokens, maximum + 1):
                    context = values[position - length : position]
                    counts = self._contexts.get(context)
                    if counts is None:
                        counts = Counter()
                        self._contexts[context] = counts
                    else:
                        self._contexts.move_to_end(context)
                    counts[next_token] += 1
            while len(self._contexts) > self.max_contexts:
                self._contexts.popitem(last=False)
                self._evicted_contexts += 1

    def observe_sample(self, sample: SequenceSample) -> None:
        self.observe(sample.full_sequence)

    def observe_samples(self, samples: Iterable[SequenceSample]) -> None:
        for sample in samples:
            self.observe_sample(sample)

    def draft(
        self,
        prefix: TokenSequence,
        max_tokens: int,
        *,
        stochastic: bool = False,
        seed: int | None = None,
    ) -> DraftProposal:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if max_tokens == 0:
            return DraftProposal((), (), 0)
        if stochastic and seed is None:
            raise ValueError("stochastic token-tree drafts require an explicit seed")
        if seed is not None and seed < 0:
            raise ValueError("draft seed must be non-negative")
        rng = np.random.default_rng(seed) if stochastic else None
        working = list(int(token) for token in prefix)
        drafted: list[int] = []
        probabilities: list[float] = []
        distributions: list[tuple[tuple[int, float], ...]] = []
        initial_match = 0
        with self._lock:
            self._queries += 1
            for _ in range(max_tokens):
                counts: Counter[int] | None = None
                matched = 0
                maximum = min(self.max_context_tokens, len(working))
                for length in range(maximum, self.min_context_tokens - 1, -1):
                    context = tuple(working[-length:])
                    candidate = self._contexts.get(context)
                    if candidate:
                        counts = candidate
                        matched = length
                        self._contexts.move_to_end(context)
                        break
                if counts is None:
                    break
                total = sum(counts.values())
                distribution = tuple(
                    (int(token), float(count / total))
                    for token, count in sorted(counts.items())
                )
                maximum_probability = max(probability for _, probability in distribution)
                if maximum_probability < self.min_token_probability:
                    break
                if stochastic:
                    assert rng is not None
                    support = np.asarray([token for token, _ in distribution], dtype=np.int64)
                    masses = np.asarray(
                        [probability for _, probability in distribution], dtype=np.float64
                    )
                    token = int(rng.choice(support, p=masses))
                    probability = dict(distribution)[token]
                    distributions.append(distribution)
                else:
                    # Stable tie-breaking keeps deterministic drafts reproducible.
                    token, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
                    probability = count / total
                if not drafted:
                    initial_match = matched
                drafted.append(int(token))
                probabilities.append(float(probability))
                working.append(int(token))
            if drafted:
                self._hits += 1
                self._proposed_tokens += len(drafted)
        return DraftProposal(
            tuple(drafted),
            tuple(probabilities),
            initial_match,
            tuple(distributions),
        )

    def record_verification(self, *, proposed: int, accepted: int) -> None:
        if proposed < 0 or accepted < 0 or accepted > proposed:
            raise ValueError("verification counts must satisfy 0 <= accepted <= proposed")
        with self._lock:
            self._accepted_tokens += int(accepted)
            self._rejected_tokens += int(proposed - accepted)

    def snapshot(self) -> RolloutTokenTreeSnapshot:
        with self._lock:
            return RolloutTokenTreeSnapshot(
                observed_sequences=self._observed_sequences,
                observed_tokens=self._observed_tokens,
                contexts=len(self._contexts),
                evicted_contexts=self._evicted_contexts,
                queries=self._queries,
                hits=self._hits,
                proposed_tokens=self._proposed_tokens,
                accepted_tokens=self._accepted_tokens,
                rejected_tokens=self._rejected_tokens,
            )


SampleCompletionCallback = Callable[[int, SequenceSample], None]


def sample_batch_with_callback(
    backend: AutoregressiveBackend,
    requests: Sequence[GenerationRequest],
    on_complete: SampleCompletionCallback,
) -> list[SequenceSample]:
    """Use a backend's streaming hook, with an ordered batch fallback."""

    callback = getattr(backend, "sample_batch_with_callback", None)
    if callable(callback):
        return list(callback(requests, on_complete))
    samples = backend.sample_batch(requests)
    for index, sample in enumerate(samples):
        on_complete(index, sample)
    return samples


@dataclass(frozen=True, slots=True)
class StreamingRewardSnapshot:
    generation_seconds: float
    reward_tail_seconds: float
    submitted_rewards: int


class StreamingRewardEvaluator:
    """Start CPU reward work as soon as each GPU sequence finishes."""

    def __init__(self, *, workers: int = 4) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=int(workers),
            thread_name_prefix="inference-scaling-reward",
        )
        self._closed = False

    def sample_and_score(
        self,
        backend: AutoregressiveBackend,
        requests: Sequence[GenerationRequest],
        reward_inputs: Sequence[tuple[TokenSequence, TokenSequence]],
        reward: TokenReward,
        on_generation_complete: Callable[[Sequence[SequenceSample]], None] | None = None,
    ) -> tuple[list[SequenceSample], tuple[float, ...], StreamingRewardSnapshot]:
        if self._closed:
            raise RuntimeError("streaming reward evaluator is closed")
        if len(requests) != len(reward_inputs):
            raise ValueError("reward_inputs must have one entry per request")
        futures: list[Future[float] | None] = [None] * len(requests)
        lock = threading.Lock()

        def completed(index: int, sample: SequenceSample) -> None:
            prompt, generated_prefix = reward_inputs[index]
            with lock:
                if futures[index] is not None:
                    raise RuntimeError("backend completed one request more than once")
                futures[index] = self._executor.submit(
                    reward,
                    prompt,
                    generated_prefix + sample.token_ids,
                )

        started = time.perf_counter()
        samples = sample_batch_with_callback(backend, requests, completed)
        generation_finished = time.perf_counter()
        if on_generation_complete is not None:
            on_generation_complete(samples)
        if len(samples) != len(requests) or any(future is None for future in futures):
            raise RuntimeError("backend returned an incomplete streaming batch")
        rewards = tuple(float(future.result()) for future in futures if future is not None)
        finished = time.perf_counter()
        if any(not isfinite(value) for value in rewards):
            raise ValueError("reward must be finite")
        return samples, rewards, StreamingRewardSnapshot(
            generation_seconds=generation_finished - started,
            reward_tail_seconds=finished - generation_finished,
            submitted_rewards=len(rewards),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "StreamingRewardEvaluator":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class RunAheadSnapshot:
    submitted_requests: int
    completed_requests: int
    completed_tokens: int
    failed_requests: int
    queued_requests: int
    critical_calls: int
    critical_wait_seconds: float


class LowPriorityRunAheadBackend:
    """Populate a draft tree only in idle gaps, in bounded preemptible chunks.

    Run-ahead outputs are never returned as evaluation samples.  They merely add
    token paths to ``tree``.  A newly arriving foreground call waits for at most
    one configured background chunk and is then served before the next chunk.
    """

    def __init__(
        self,
        backend: AutoregressiveBackend,
        tree: RolloutTokenTree | None,
        *,
        chunk_tokens: int = 32,
        queue_limit: int = 1_024,
        outputs_already_observed: bool = False,
    ) -> None:
        if chunk_tokens <= 0 or queue_limit <= 0:
            raise ValueError("run-ahead limits must be positive")
        if tree is None and not outputs_already_observed:
            raise ValueError(
                "a draft tree is required unless the backend observes its own outputs"
            )
        self.backend = backend
        self.tree = tree
        self._outputs_already_observed = bool(outputs_already_observed)
        self.chunk_tokens = int(chunk_tokens)
        self._queue: queue.Queue[GenerationRequest | None] = queue.Queue(queue_limit)
        self._condition = threading.Condition()
        self._foreground_waiters = 0
        self._foreground_active = 0
        self._background_active = False
        self._closed = False
        self._submitted = 0
        self._completed = 0
        self._completed_tokens = 0
        self._failed = 0
        self._critical_calls = 0
        self._critical_wait_seconds = 0.0
        self._worker = threading.Thread(
            target=self._run,
            name="inference-scaling-runahead",
            daemon=True,
        )
        self._worker.start()

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def submit_run_ahead(self, requests: Sequence[GenerationRequest]) -> int:
        accepted = 0
        with self._condition:
            if self._closed:
                raise RuntimeError("run-ahead backend is closed")
        for request in requests:
            try:
                self._queue.put_nowait(request)
            except queue.Full:
                break
            accepted += 1
        with self._condition:
            self._submitted += accepted
            self._condition.notify_all()
        return accepted

    def _enter_foreground(self) -> None:
        started = time.perf_counter()
        with self._condition:
            self._foreground_waiters += 1
            try:
                while self._background_active:
                    self._condition.wait()
                self._foreground_active += 1
            finally:
                self._foreground_waiters -= 1
            self._critical_calls += 1
            self._critical_wait_seconds += time.perf_counter() - started

    def _leave_foreground(self) -> None:
        with self._condition:
            self._foreground_active -= 1
            self._condition.notify_all()

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        self._enter_foreground()
        try:
            return self.backend.sample_batch(requests)
        finally:
            self._leave_foreground()

    def sample_batch_with_callback(
        self,
        requests: Sequence[GenerationRequest],
        on_complete: SampleCompletionCallback,
    ) -> list[SequenceSample]:
        self._enter_foreground()
        try:
            return sample_batch_with_callback(self.backend, requests, on_complete)
        finally:
            self._leave_foreground()

    def score_batch(self, requests):
        self._enter_foreground()
        try:
            return self.backend.score_batch(requests)
        finally:
            self._leave_foreground()

    @staticmethod
    def _continuation_seed(seed: int, generated: int) -> int:
        state = np.random.SeedSequence([int(seed), int(generated)]).generate_state(
            1, dtype=np.uint64
        )[0]
        return int(state % np.uint64(2**63 - 1))

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                self._queue.task_done()
                return
            with self._condition:
                while (
                    not self._closed
                    and (self._foreground_active or self._foreground_waiters)
                ):
                    self._condition.wait()
                if self._closed:
                    self._queue.task_done()
                    return
                self._background_active = True
            try:
                length = min(self.chunk_tokens, request.max_new_tokens)
                chunk_request = GenerationRequest(
                    prefix=request.prefix,
                    max_new_tokens=length,
                    sampling=request.sampling,
                    seed=request.seed,
                    request_id=request.request_id,
                    uniforms=(
                        request.uniforms[:length]
                        if request.uniforms is not None
                        else None
                    ),
                )
                sample = self.backend.sample_batch([chunk_request])[0]
                if not self._outputs_already_observed:
                    assert self.tree is not None
                    self.tree.observe_sample(sample)
                with self._condition:
                    self._completed_tokens += len(sample.token_ids)
                eos = request.sampling.eos_token_id
                stopped = eos is not None and eos in sample.token_ids
                remaining = request.max_new_tokens - len(sample.token_ids)
                if remaining > 0 and not stopped and sample.token_ids:
                    continuation = GenerationRequest(
                        prefix=request.prefix + sample.token_ids,
                        max_new_tokens=remaining,
                        sampling=request.sampling,
                        seed=self._continuation_seed(request.seed, len(sample.token_ids)),
                        request_id=f"{request.request_id}:continued:{len(sample.token_ids)}",
                        uniforms=(
                            request.uniforms[len(sample.token_ids) :]
                            if request.uniforms is not None
                            else None
                        ),
                    )
                    with self._condition:
                        may_continue = not self._closed
                    if may_continue:
                        try:
                            self._queue.put_nowait(continuation)
                        except queue.Full:
                            with self._condition:
                                self._completed += 1
                    else:
                        with self._condition:
                            self._completed += 1
                else:
                    with self._condition:
                        self._completed += 1
            except BaseException:
                with self._condition:
                    self._failed += 1
            finally:
                with self._condition:
                    self._background_active = False
                    self._condition.notify_all()
                self._queue.task_done()

    def wait_for_run_ahead(self) -> None:
        self._queue.join()

    def snapshot(self) -> RunAheadSnapshot:
        with self._condition:
            return RunAheadSnapshot(
                submitted_requests=self._submitted,
                completed_requests=self._completed,
                completed_tokens=self._completed_tokens,
                failed_requests=self._failed,
                queued_requests=self._queue.qsize() + int(self._background_active),
                critical_calls=self._critical_calls,
                critical_wait_seconds=self._critical_wait_seconds,
            )

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        # The sentinel is bounded and only inserted after existing work.  Clear
        # pending draft-only work so shutdown never waits for a large idle queue.
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
                if pending is None:
                    break
        self._queue.put(None)
        self._worker.join()

    def __enter__(self) -> "LowPriorityRunAheadBackend":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
