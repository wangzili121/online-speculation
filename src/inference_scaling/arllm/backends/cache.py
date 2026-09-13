"""Exact score caching keyed by the complete stochastic policy."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

_ScoreKey = tuple[SamplingConfig | None, TokenSequence, TokenSequence]


@dataclass(frozen=True, slots=True)
class ScoreCacheSnapshot:
    entries: int
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ScoreCachingBackend:
    """Cache deterministic continuation scores without caching random generations."""

    def __init__(self, backend: AutoregressiveBackend, *, maximum_entries: int = 100_000):
        if maximum_entries <= 0:
            raise ValueError("maximum_entries must be positive")
        self._backend = backend
        self._maximum_entries = int(maximum_entries)
        self._cache: OrderedDict[_ScoreKey, tuple[float, ...]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        return self._backend.sample_batch(requests)

    @staticmethod
    def _key(request: ScoreRequest, continuation: TokenSequence) -> _ScoreKey:
        return request.sampling, request.prefix, continuation

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        ordered_keys: list[_ScoreKey] = []
        results: dict[_ScoreKey, tuple[float, ...]] = {}
        missing: OrderedDict[_ScoreKey, None] = OrderedDict()
        with self._lock:
            for request in requests:
                for continuation in request.continuations:
                    key = self._key(request, continuation)
                    ordered_keys.append(key)
                    cached = self._cache.get(key)
                    if cached is None:
                        self._misses += 1
                        missing.setdefault(key, None)
                    else:
                        self._hits += 1
                        self._cache.move_to_end(key)
                        results[key] = cached

        if missing:
            grouped: OrderedDict[
                tuple[SamplingConfig | None, TokenSequence], list[TokenSequence]
            ] = OrderedDict()
            for sampling, prefix, continuation in missing:
                grouped.setdefault((sampling, prefix), []).append(continuation)
            score_requests = [
                ScoreRequest(prefix, tuple(continuations), sampling)
                for (sampling, prefix), continuations in grouped.items()
            ]
            score_keys = [
                (sampling, prefix, continuation)
                for (sampling, prefix), continuations in grouped.items()
                for continuation in continuations
            ]
            scored = self._backend.score_batch(score_requests)
            if len(scored) != len(score_keys):
                raise RuntimeError("underlying backend returned an invalid score batch")
            with self._lock:
                for key, value in zip(score_keys, scored, strict=True):
                    if len(value) != len(key[2]):
                        raise RuntimeError("underlying backend returned an invalid score shape")
                    existing = self._cache.get(key)
                    if existing is None:
                        self._cache[key] = value
                        while len(self._cache) > self._maximum_entries:
                            self._cache.popitem(last=False)
                            self._evictions += 1
                        results[key] = value
                    else:
                        self._cache.move_to_end(key)
                        results[key] = existing
        return [results[key] for key in ordered_keys]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def snapshot(self) -> ScoreCacheSnapshot:
        with self._lock:
            return ScoreCacheSnapshot(
                entries=len(self._cache),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
