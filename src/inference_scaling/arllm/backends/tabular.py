"""An exact finite backend used for distributional tests.

This backend is intentionally small enough that target distributions can be
enumerated.  It also implements temperature, top-k, and nucleus truncation so
tests exercise actual behavior-policy probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import (
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have a positive finite sum")
    return values / total


class TabularAutoregressiveBackend:
    def __init__(
        self,
        probabilities: Mapping[TokenSequence, Sequence[float]],
        *,
        fallback: Sequence[float] | None = None,
        model_id: str = "tabular",
    ) -> None:
        if not probabilities and fallback is None:
            raise ValueError(
                "at least one transition or a fallback distribution is required"
            )
        raw = {
            tuple(prefix): np.asarray(row, dtype=np.float64)
            for prefix, row in probabilities.items()
        }
        first = next(iter(raw.values()), np.asarray(fallback, dtype=np.float64))
        assert first is not None
        self._vocab_size = int(first.shape[0])
        self._probabilities: dict[TokenSequence, np.ndarray] = {}
        for prefix, row in raw.items():
            self._probabilities[prefix] = self._validate_row(row)
        self._fallback = (
            None
            if fallback is None
            else self._validate_row(np.asarray(fallback, dtype=np.float64))
        )
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _validate_row(self, row: np.ndarray) -> np.ndarray:
        if row.ndim != 1 or row.shape[0] != self._vocab_size:
            raise ValueError(
                "all transition rows must have the same one-dimensional vocabulary"
            )
        if np.any(row < 0) or np.any(~np.isfinite(row)):
            raise ValueError("transition probabilities must be finite and non-negative")
        return _normalize(row.copy())

    def _base_probabilities(self, prefix: TokenSequence) -> np.ndarray:
        if prefix in self._probabilities:
            return self._probabilities[prefix]
        if self._fallback is not None:
            return self._fallback
        raise KeyError(f"no transition probabilities for prefix {prefix!r}")

    def probabilities(
        self, prefix: TokenSequence, sampling: SamplingConfig | None = None
    ) -> np.ndarray:
        base = self._base_probabilities(prefix)
        if sampling is None:
            return base.copy()

        positive = base > 0
        scaled = np.zeros_like(base)
        scaled[positive] = np.exp(np.log(base[positive]) / sampling.temperature)

        if sampling.top_k is not None and sampling.top_k < self._vocab_size:
            keep = np.argpartition(scaled, -sampling.top_k)[-sampling.top_k :]
            mask = np.zeros(self._vocab_size, dtype=bool)
            mask[keep] = True
            scaled[~mask] = 0

        scaled = _normalize(scaled)
        if sampling.top_p < 1:
            order = np.argsort(-scaled, kind="stable")
            cumulative = np.cumsum(scaled[order])
            count = int(np.searchsorted(cumulative, sampling.top_p, side="left")) + 1
            keep = order[:count]
            mask = np.zeros(self._vocab_size, dtype=bool)
            mask[keep] = True
            scaled[~mask] = 0
            scaled = _normalize(scaled)
        return scaled

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]:
        outputs: list[SequenceSample] = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            arithmetic_uniform = request.arithmetic_uniform
            context = list(request.prefix)
            tokens: list[int] = []
            logprobs: list[float] = []
            finish_reason = "length"
            for step in range(request.max_new_tokens):
                probs = self.probabilities(tuple(context), request.sampling)
                if arithmetic_uniform is not None:
                    order = np.argsort(-probs, kind="stable")
                    ordered = probs[order]
                    cumulative = np.cumsum(ordered, dtype=np.float64)
                    cumulative[-1] = 1.0
                    rank = int(
                        np.searchsorted(
                            cumulative,
                            arithmetic_uniform,
                            side="right",
                        )
                    )
                    rank = min(rank, self._vocab_size - 1)
                    token = int(order[rank])
                    lower = 0.0 if rank == 0 else float(cumulative[rank - 1])
                    probability = float(ordered[rank])
                    if probability <= 0:
                        raise RuntimeError(
                            "arithmetic sampling selected a zero-probability token"
                        )
                    arithmetic_uniform = min(
                        max((arithmetic_uniform - lower) / probability, 0.0),
                        float(np.nextafter(1.0, 0.0)),
                    )
                elif request.uniforms is None:
                    token = int(rng.choice(self._vocab_size, p=probs))
                else:
                    token = int(
                        np.searchsorted(
                            np.cumsum(probs, dtype=np.float64),
                            request.uniforms[step],
                            side="right",
                        )
                    )
                    token = min(token, self._vocab_size - 1)
                tokens.append(token)
                logprobs.append(float(np.log(probs[token])))
                context.append(token)
                if request.sampling.eos_token_id == token:
                    finish_reason = "eos"
                    break
            outputs.append(
                SequenceSample(
                    prefix=request.prefix,
                    token_ids=tuple(tokens),
                    token_logprobs=tuple(logprobs),
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                    finish_reason=finish_reason,
                )
            )
        return outputs

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        outputs: list[tuple[float, ...]] = []
        for request in requests:
            for continuation in request.continuations:
                context = list(request.prefix)
                token_logprobs: list[float] = []
                for token in continuation:
                    probs = self.probabilities(tuple(context), request.sampling)
                    probability = float(probs[token])
                    token_logprobs.append(
                        float("-inf")
                        if probability == 0
                        else float(np.log(probability))
                    )
                    context.append(token)
                outputs.append(tuple(token_logprobs))
        return outputs
