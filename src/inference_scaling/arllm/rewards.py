"""Autoregressive rewards derived from model probabilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from typing import Any, Literal, Sequence

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, ScoreRequest
from inference_scaling.shared.types import TokenSequence
from inference_scaling.shared.consilience import confidence_windows
from inference_scaling.shared.output import OutputParser, ThinkingFormat, ThinkingParser
from inference_scaling.arllm.output import thinking_format_from_backend


@dataclass(frozen=True, slots=True)
class SequenceLogProbabilityReward:
    """Return a scaled full-sequence ``log p(completion | prompt)``.

    This is a model-derived reward, not an external verifier.  The backend must
    support exact scoring under ``sampling``.  With scale ``c`` and reward
    temperature ``tau``, reward reweighting targets
    ``p(completion | prompt) ** (1 + c / tau)``.
    """

    backend: AutoregressiveBackend
    sampling: SamplingConfig | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.scale):
            raise ValueError("log-probability reward scale must be finite")

    def __call__(self, prompt: TokenSequence, completion: TokenSequence) -> float:
        return self.batch(prompt, (completion,))[0]

    def batch(
        self,
        prompt: TokenSequence,
        completions: Sequence[TokenSequence],
    ) -> tuple[float, ...]:
        if not completions:
            return ()
        scored = self.backend.score_batch(
            [ScoreRequest(prompt, tuple(map(tuple, completions)), self.sampling)]
        )
        if len(scored) != len(completions):
            raise RuntimeError("backend returned an invalid log-probability score batch")
        if any(
            len(token_scores) != len(completion)
            for token_scores, completion in zip(scored, completions, strict=True)
        ):
            raise RuntimeError("backend returned an invalid token score shape")
        return tuple(self.scale * float(sum(token_scores)) for token_scores in scored)

    def describe(self) -> dict[str, object]:
        return {
            "source": "model_sequence_log_probability",
            "model_id": self.backend.model_id,
            "policy_id": self.sampling.policy_id if self.sampling is not None else None,
            "scale": self.scale,
        }


@dataclass(frozen=True, slots=True)
class ConsilienceReward:
    """Verifier-free confidence-trajectory reward for one generated sequence.

    At token ``t``, confidence is the negative mean log-probability of the
    model's top-``K`` next-token candidates.  The sequence reward is the final
    window mean minus ``initial_penalty`` times the initial window mean.  The
    first ``skip_fraction`` of tokens are omitted from the initial window.

    The result is pointwise: it never depends on the other candidates in a
    batch.  It can therefore be used unchanged by Best-of-N, conditional IS,
    and replay-based methods.
    """

    backend: AutoregressiveBackend
    sampling: SamplingConfig | None = None
    top_k: int = 5
    window_fraction: float = 0.2
    window_tokens: int | None = None
    skip_fraction: float = 0.05
    initial_penalty: float = 3.0
    scale: float = 1.0
    reasoning_end_token_ids: TokenSequence | None = None
    thinking_format: OutputParser | None = None
    scope: Literal["thinking", "full"] = "thinking"
    _scope_counts: Counter[tuple[str, str | None]] = field(default_factory=Counter, init=False, repr=False, compare=False)
    _scope_lock: Any = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("Consilience top_k must be positive")
        self._trajectory_score((0.0,))
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Consilience scale must be finite and positive")
        if self.reasoning_end_token_ids is not None and not self.reasoning_end_token_ids:
            raise ValueError("reasoning_end_token_ids must be nonempty when provided")
        if self.scope not in {"thinking", "full"}:
            raise ValueError("Consilience scope must be thinking or full")
        if self.thinking_format is not None and self.reasoning_end_token_ids is not None:
            raise ValueError("provide thinking_format or reasoning_end_token_ids, not both")
        if self.scope == "thinking" and self.thinking_format is None:
            resolved = (
                ThinkingFormat(end_token_ids=tuple(self.reasoning_end_token_ids))
                if self.reasoning_end_token_ids is not None
                else thinking_format_from_backend(self.backend)
            )
            object.__setattr__(self, "thinking_format", resolved)
        if isinstance(self.thinking_format, ThinkingFormat):
            object.__setattr__(self, "thinking_format", ThinkingParser((self.thinking_format,)))

    def __call__(self, prompt: TokenSequence, completion: TokenSequence) -> float:
        return self.batch(prompt, (completion,))[0]

    def _score_input(self, prompt: TokenSequence, completion: TokenSequence):
        prefix, tokens = tuple(prompt), tuple(completion)
        eos = getattr(getattr(self.backend, "tokenizer", None), "eos_token_id", None)
        if eos is not None and eos in tokens:
            tokens = tokens[:tokens.index(eos) + 1]
        used, reason = "full", None
        if self.scope == "thinking":
            assert self.thinking_format is not None
            segments = self.thinking_format.split(prefix, tokens, eos_token_id=eos)
            if segments.has_complete_thinking:
                assert segments.thinking_start is not None
                prefix += tokens[:segments.thinking_start]
                tokens = segments.thinking_token_ids
                used = "thinking"
            else:
                reason = segments.status
        return prefix, tokens, {
            "requested_reward_scope": self.scope, "reward_scope": used,
            "reward_mode": "consilience_" + used,
            "reward_fallback_reason": reason,
        }

    def describe_completion(self, prompt: TokenSequence, completion: TokenSequence) -> dict[str, object]:
        """Report the deterministic scope decision without another model forward."""
        return self._score_input(prompt, completion)[2]

    def scope_statistics(self) -> dict[str, object]:
        with self._scope_lock:
            counts = dict(self._scope_counts)
        return {
            "evaluated_sequences": sum(counts.values()),
            "thinking_sequences": sum(count for (mode, _), count in counts.items() if mode == "thinking"),
            "full_sequences": sum(count for (mode, _), count in counts.items() if mode == "full"),
            "fallback_reasons": {reason: count for (_, reason), count in counts.items() if reason is not None},
        }

    def _trajectory_score(self, values: Sequence[float]) -> float:
        windows = confidence_windows(
            values,
            window_fraction=self.window_fraction,
            window_tokens=self.window_tokens,
            skip_fraction=self.skip_fraction,
            initial_penalty=self.initial_penalty,
        )
        score = self.scale * windows.score
        if not isfinite(score):
            raise ValueError("Consilience scaled score must be finite")
        return score

    def batch(
        self,
        prompt: TokenSequence,
        completions: Sequence[TokenSequence],
    ) -> tuple[float, ...]:
        if not completions:
            return ()
        # The empty full sequence has neutral score; generated EOS is retained
        # as one genuine token, while absorbing padding after it is excluded.
        rewards = [0.0] * len(completions)
        grouped: dict[TokenSequence, list[tuple[int, TokenSequence]]] = {}
        modes = []
        for index, completion in enumerate(completions):
            prefix, tokens, decision = self._score_input(prompt, tuple(completion))
            modes.append((decision["reward_scope"], decision["reward_fallback_reason"]))
            if tokens:
                grouped.setdefault(prefix, []).append((index, tokens))
        if not grouped:
            with self._scope_lock:
                self._scope_counts.update(modes)
            return tuple(rewards)
        callback: Any = getattr(self.backend, "score_statistics_batch", None)
        if callback is None:
            raise ValueError(
                "Consilience reward requires a backend with top-K score statistics"
            )
        requests = [
            ScoreRequest(prefix, tuple(tokens for _, tokens in group), self.sampling)
            for prefix, group in grouped.items()
        ]
        ordered = [item for group in grouped.values() for item in group]
        statistics = callback(
            requests,
            confidence_top_k=self.top_k,
        )
        if len(statistics) != len(ordered):
            raise RuntimeError("backend returned an invalid Consilience score batch")
        for (index, sequence), item in zip(ordered, statistics, strict=True):
            values = tuple(float(value) for value in item.token_topk_confidences)
            if len(values) != len(sequence):
                raise RuntimeError("backend returned an invalid Consilience trajectory")
            rewards[index] = self._trajectory_score(values)
        with self._scope_lock:
            self._scope_counts.update(modes)
        return tuple(rewards)

    def describe(self) -> dict[str, object]:
        return {
            "source": "model_consilience",
            "model_id": self.backend.model_id,
            "policy_id": self.sampling.policy_id if self.sampling is not None else None,
            "top_k": self.top_k,
            "window_fraction": self.window_fraction,
            "window_tokens": self.window_tokens,
            "skip_fraction": self.skip_fraction,
            "initial_penalty": self.initial_penalty,
            "scale": self.scale,
            "reasoning_end_token_ids": self.reasoning_end_token_ids,
            "scope": self.scope,
            "fallback": "full_sequence",
            "thinking_format": (
                self.thinking_format.describe() if self.thinking_format is not None else None
            ),
        }


__all__ = ["ConsilienceReward", "SequenceLogProbabilityReward"]
