"""Sampling scope and output assembly shared by AR inference algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from inference_scaling.arllm.backends.stopping import StoppedSequenceBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import thinking_format_from_backend, output_settings_from_config
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.output import OutputParser, full_sequence_segments
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class SamplingScope:
    scope: Literal["full", "thinking"] = "full"
    thinking_format: OutputParser | None = None
    generation_chunk_size: int = 256
    requested_scope: Literal["full", "thinking"] | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"full", "thinking"}:
            raise ValueError("sampling scope must be full or thinking")
        if self.requested_scope is None:
            object.__setattr__(self, "requested_scope", self.scope)
        format_ = self.thinking_format
        if self.scope == "thinking":
            reason = None
            if getattr(format_, "mode", None) == "disabled":
                reason = "disabled"
            elif format_ is None or not getattr(format_, "supports_early_stop", True):
                reason = (
                    "structured_format_requires_full_sequence" if getattr(format_, "kind", None) in {"json", "xml"}
                    else "unrecognized_format"
                )
            if reason is not None:
                object.__setattr__(self, "scope", "full")
                object.__setattr__(self, "fallback_reason", reason)
        if self.generation_chunk_size <= 0:
            raise ValueError("generation_chunk_size must be positive")

    @classmethod
    def from_config(cls, backend: Any, config: Mapping[str, Any], *, active: bool = True):
        options = output_settings_from_config(config)
        scope = options.get("sampling_scope", "full") if active else "full"
        if scope not in {"full", "thinking"}:
            raise ValueError("sampling scope must be full or thinking")
        return cls(
            scope="thinking" if scope == "thinking" else "full",
            thinking_format=thinking_format_from_backend(backend, options),
            generation_chunk_size=int(options.get("generation_chunk_size", 256)),
        )

    def full_fallback(self, reason: str):
        return replace(self, scope="full", fallback_reason=reason)

    def for_prompt(self, prompt: TokenSequence):
        callback = getattr(self.thinking_format, "prompt_mode", None)
        if self.scope == "thinking" and callback is not None:
            if callback(prompt) == "disabled":
                return self.full_fallback("disabled")
        return self

    def wrap(self, backend: Any, prompt: TokenSequence):
        if self.scope == "full":
            return backend
        assert self.thinking_format is not None
        eos = getattr(backend.tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("thinking sampling requires an EOS padding token")
        return StoppedSequenceBackend(
            backend, stop_token_sequences=(), thinking_parser=self.thinking_format, thinking_prompt=prompt,
            eos_token_id=eos, protected_prefix_length=len(prompt),
            generation_chunk_size=self.generation_chunk_size,
        )

    def finish(
        self, backend: Any, prompt: TokenSequence, tokens: TokenSequence, *,
        max_new_tokens: int, sampling: SamplingConfig, seed: int,
    ) -> tuple[TokenSequence, dict[str, Any]]:
        sequence = tuple(tokens)
        final_generated = 0
        if self.scope == "thinking":
            assert self.thinking_format is not None
            segments = self.thinking_format.split(prompt, sequence, eos_token_id=backend.tokenizer.eos_token_id)
            if segments.has_complete_thinking and segments.boundary_end is not None:
                sequence = sequence[:segments.boundary_end]
                remaining = max_new_tokens - len(sequence)
                if remaining > 0:
                    sample = backend.sample_batch([GenerationRequest(
                        prefix=prompt + sequence, max_new_tokens=remaining, sampling=sampling,
                        seed=seed, request_id="scope:final-content",
                    )])[0]
                    final_generated = len(sample.token_ids)
                    sequence += sample.token_ids
        information = self.describe_output(backend, prompt, sequence)
        effective_scope, reason = self.scope, self.fallback_reason
        if self.scope == "thinking" and information["thinking_status"] != "complete":
            effective_scope, reason = "full", information["thinking_status"]
        elif self.requested_scope == "thinking" and information.get("thinking_format_name") in {"json", "xml"}:
            effective_scope, reason = "full", "structured_format_requires_full_sequence"
        information["requested_sampling_scope"] = self.requested_scope
        information["sampling_scope"] = effective_scope
        information["sampling_fallback_reason"] = reason
        information["sampling_policy_scope"] = (
            "thinking_with_full_sequence_fallback" if self.scope == "thinking" else "full"
        )
        information["final_content_generated_tokens"] = final_generated
        information["generation_budget_exhausted"] = (
            len(sequence) >= max_new_tokens and not information["ended_by_eos"]
        )
        return sequence, information

    def describe_output(self, backend: Any, prompt: TokenSequence, tokens: TokenSequence) -> dict[str, Any]:
        eos = getattr(backend.tokenizer, "eos_token_id", None)
        if self.thinking_format is None:
            segments = full_sequence_segments(tokens, eos_token_id=eos, reason="unrecognized_format")
        else:
            segments = self.thinking_format.split(prompt, tuple(tokens), eos_token_id=eos)
            if not segments.has_complete_thinking:
                segments = full_sequence_segments(tokens, eos_token_id=eos, reason=segments.status)
        return {
            **segments.describe(),
            "thinking_token_ids": segments.thinking_token_ids,
            "thinking_text": segments.thinking_text if segments.thinking_text is not None else backend.decode(segments.thinking_token_ids),
            "content_token_ids": segments.content_token_ids,
            "content_text": segments.content_text if segments.content_text is not None else backend.decode(segments.content_token_ids),
            "thinking_format": self.thinking_format.describe() if self.thinking_format is not None else None,
        }


__all__ = ["SamplingScope"]
