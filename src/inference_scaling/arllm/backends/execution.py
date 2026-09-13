"""Attach model metadata to a probability-preserving execution dispatcher."""

from __future__ import annotations

from typing import Any


class ExecutionBackend:
    def __init__(self, executor: Any, backend: Any) -> None:
        self.executor = executor
        self.backend = backend

    @property
    def model_id(self):
        return self.executor.model_id

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    @property
    def parameter_count(self):
        return self.backend.parameter_count

    def sample_batch(self, requests):
        return self.executor.sample_batch(requests)

    def score_batch(self, requests):
        return self.executor.score_batch(requests)

    def score_statistics_batch(self, requests, *, confidence_top_k=None):
        return self.backend.score_statistics_batch(requests, confidence_top_k=confidence_top_k)

    def encode(self, text, *, add_special_tokens=True):
        return self.backend.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens, *, skip_special_tokens=True):
        if skip_special_tokens:
            return self.backend.decode(tokens)
        return self.backend.decode(tokens, skip_special_tokens=False)

    def direct_generate(self, *args, **kwargs):
        return self.backend.direct_generate(*args, **kwargs)

    def snapshot(self):
        return self.backend.snapshot()


def execution_model(executor: Any) -> Any:
    """Find model metadata below the repository's cache/batching wrappers."""
    seen = set()
    while id(executor) not in seen:
        seen.add(id(executor))
        if hasattr(executor, "tokenizer"):
            return executor
        executor = getattr(executor, "backend", getattr(executor, "_backend", None))
        if executor is None:
            break
    raise ValueError("execution wrapper does not expose model/tokenizer metadata")
