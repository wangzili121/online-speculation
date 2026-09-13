"""Core request and backend contracts.

Algorithms depend on these contracts rather than on Transformers or a particular
inference server.  A backend must return probabilities under the *actual* sampling
policy, not merely unprocessed model logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence, runtime_checkable

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prefix: TokenSequence
    max_new_tokens: int
    sampling: SamplingConfig
    seed: int
    request_id: str
    uniforms: tuple[float, ...] | None = None
    arithmetic_uniform: float | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.uniforms is not None:
            if len(self.uniforms) != self.max_new_tokens:
                raise ValueError("explicit sampling uniforms must match max_new_tokens")
            if any(
                not isfinite(value) or not 0.0 <= value < 1.0 for value in self.uniforms
            ):
                raise ValueError("sampling uniforms must be finite values in [0, 1)")
        if self.arithmetic_uniform is not None:
            if self.uniforms is not None:
                raise ValueError(
                    "token uniforms and an arithmetic uniform are mutually exclusive"
                )
            if (
                not isfinite(self.arithmetic_uniform)
                or not 0.0 <= self.arithmetic_uniform < 1.0
            ):
                raise ValueError(
                    "the arithmetic sampling uniform must be finite and in [0, 1)"
                )


@dataclass(frozen=True, slots=True)
class SequenceSample:
    prefix: TokenSequence
    token_ids: TokenSequence
    token_logprobs: tuple[float, ...]
    policy_id: str
    model_id: str
    request_id: str
    finish_reason: str = "length"
    reference_token_logprobs: tuple[float, ...] | None = None
    reference_policy_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError(
                "each sampled token must have one actual-policy log-probability"
            )
        if any(not isfinite(value) for value in self.token_logprobs):
            raise ValueError("actual-policy token log-probabilities must be finite")
        if (self.reference_token_logprobs is None) != (
            self.reference_policy_id is None
        ):
            raise ValueError(
                "reference token probabilities and their policy id must be provided together"
            )
        if self.reference_token_logprobs is not None and len(self.token_ids) != len(
            self.reference_token_logprobs
        ):
            raise ValueError(
                "each sampled token must have one reference-policy log-probability"
            )

    @property
    def logprob(self) -> float:
        return float(sum(self.token_logprobs))

    @property
    def full_sequence(self) -> TokenSequence:
        return self.prefix + self.token_ids


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    prefix: TokenSequence
    continuations: tuple[TokenSequence, ...]
    sampling: SamplingConfig | None = None


@runtime_checkable
class AutoregressiveBackend(Protocol):
    """Minimal interface required by MH, conditional IS, and replay correction."""

    @property
    def model_id(self) -> str: ...

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]: ...

    def score_batch(
        self, requests: Sequence[ScoreRequest]
    ) -> list[tuple[float, ...]]: ...
