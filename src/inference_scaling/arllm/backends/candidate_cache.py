"""Explicit replay of candidate samples with exact delegated scoring."""

from __future__ import annotations

from collections.abc import Sequence

from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
)


class CachedCandidateBackend:
    """Replay frozen candidate draws while delegating exact policy scoring.

    This is an algorithm-level candidate store, not a transparent random-generation
    cache.  A request can only retrieve the sample frozen under the same request id,
    prefix, policy, model, and maximum length.
    """

    def __init__(
        self,
        scoring_backend: AutoregressiveBackend,
        samples: Sequence[SequenceSample],
    ) -> None:
        self.scoring_backend = scoring_backend
        self.samples = {sample.request_id: sample for sample in samples}
        if len(self.samples) != len(samples):
            raise ValueError("cached candidate request ids must be unique")

    @property
    def model_id(self) -> str:
        return self.scoring_backend.model_id

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]:
        replayed: list[SequenceSample] = []
        for request in requests:
            try:
                sample = self.samples[request.request_id]
            except KeyError as error:
                raise KeyError(
                    f"candidate request {request.request_id!r} is absent from the frozen cache"
                ) from error
            if (
                sample.prefix != request.prefix
                or sample.policy_id != request.sampling.policy_id
                or sample.model_id != self.model_id
                or len(sample.token_ids) > request.max_new_tokens
            ):
                raise RuntimeError("cached candidate does not match its frozen request")
            replayed.append(sample)
        return replayed

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        return self.scoring_backend.score_batch(requests)
