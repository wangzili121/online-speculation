"""An absorbing fixed-length view stopped at token-sequence boundaries.

The stop marker retains its model probability. Positions after it are forced
EOS padding with probability one. The same sample/score contract is usable by
conditional IS, replay and fixed-length MH, independently of the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest, ScoreRequest, SequenceSample
from inference_scaling.shared.output import find_token_sequence, OutputParser
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


class StoppedSequenceBackend:
    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        stop_token_sequences: Sequence[TokenSequence],
        eos_token_id: int,
        protected_prefix_length: int,
        generation_chunk_size: int | None = 256,
        thinking_parser: OutputParser | None = None,
        thinking_prompt: TokenSequence = (),
    ) -> None:
        stops = tuple(tuple(marker) for marker in stop_token_sequences)
        if any(not marker or any(token < 0 for token in marker) for marker in stops):
            raise ValueError("stop_token_sequences must contain nonempty token markers")
        if eos_token_id < 0 or protected_prefix_length < 0 or (
            generation_chunk_size is not None and generation_chunk_size <= 0
        ):
            raise ValueError("invalid EOS, protected prefix length or generation chunk size")
        self.backend = backend
        self.stop_token_sequences = stops
        self.eos_token_id = eos_token_id
        self.protected_prefix_length = protected_prefix_length
        self.generation_chunk_size = generation_chunk_size
        self.thinking_parser = thinking_parser
        self.thinking_prompt = tuple(thinking_prompt)
        if thinking_parser is not None and len(thinking_prompt) != protected_prefix_length:
            raise ValueError("thinking_prompt must match the protected prefix length")
        self.model_id = (
            f"{backend.model_id}|stops={stops};eos={eos_token_id};"
            f"after={protected_prefix_length};chunk={generation_chunk_size}"
        )
        if thinking_parser is not None:
            self.model_id += f";thinking={thinking_parser.describe()}"

    @property
    def tokenizer(self):
        return getattr(self.backend, "tokenizer")

    @property
    def parameter_count(self):
        return getattr(self.backend, "parameter_count")

    def encode(self, text: str, *, add_special_tokens: bool = True):
        return getattr(self.backend, "encode")(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: TokenSequence, *, skip_special_tokens: bool = True):
        return getattr(self.backend, "decode")(tokens, skip_special_tokens=skip_special_tokens)

    def snapshot(self):
        return getattr(self.backend, "snapshot")()

    def score_statistics_batch(self, requests, *, confidence_top_k=None):
        # Statistic spans are chosen explicitly by the reward. score_batch also
        # implements the stopped measure; these remain original model statistics.
        return getattr(self.backend, "score_statistics_batch")(requests, confidence_top_k=confidence_top_k)

    def _stop_end(self, generated: TokenSequence) -> int | None:
        ends = []
        if self.thinking_parser is not None:
            callback = getattr(self.thinking_parser, "stop_boundary", None)
            if callback is not None:
                boundary = callback(self.thinking_prompt, generated, eos_token_id=self.eos_token_id)
            else:
                segments = self.thinking_parser.split(self.thinking_prompt, generated, eos_token_id=self.eos_token_id)
                boundary = segments.boundary_end if segments.has_complete_thinking else None
            if boundary is not None:
                ends.append(boundary)
        for marker in (*self.stop_token_sequences, (self.eos_token_id,)):
            index = find_token_sequence(generated, marker)
            if index is not None:
                ends.append(index + len(marker))
        return min(ends) if ends else None

    def _generated_prefix(self, prefix: TokenSequence) -> TokenSequence:
        if len(prefix) < self.protected_prefix_length:
            raise ValueError("prefix is shorter than the protected prompt")
        generated = prefix[self.protected_prefix_length :]
        end = self._stop_end(generated)
        if end is not None and any(token != self.eos_token_id for token in generated[end:]):
            raise ValueError("a stopped prefix contains non-padding tokens after its boundary")
        return generated

    def _inner_policy(self, sampling: SamplingConfig | None) -> SamplingConfig:
        source = sampling or SamplingConfig()
        if source.eos_token_id not in (None, self.eos_token_id):
            raise ValueError("sampling uses a different EOS token")
        return replace(source, eos_token_id=self.eos_token_id)

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        tokens: list[list[int]] = [[] for _ in requests]
        logs: list[list[float]] = [[] for _ in requests]
        references: list[list[float] | None] = [[] for _ in requests]
        done = [self._stop_end(self._generated_prefix(request.prefix)) is not None for request in requests]
        expected_reference = SamplingConfig(eos_token_id=self.eos_token_id).policy_id
        while True:
            pending: list[GenerationRequest] = []
            indices: list[int] = []
            for index, request in enumerate(requests):
                offset = len(tokens[index])
                if done[index] or offset >= request.max_new_tokens:
                    continue
                remaining = request.max_new_tokens - offset
                # Arithmetic coding uses one interval throughout a request.
                # Preserve that request intact rather than resetting its interval.
                chunk = remaining if request.arithmetic_uniform is not None or self.generation_chunk_size is None else min(
                    remaining, self.generation_chunk_size
                )
                seed = request.seed if offset == 0 else SeedStream(request.seed).derive("stop-chunk", offset)
                pending.append(GenerationRequest(
                    prefix=request.prefix + tuple(tokens[index]), max_new_tokens=chunk,
                    sampling=self._inner_policy(request.sampling), seed=seed,
                    request_id=f"{request.request_id}:stop-chunk:{offset}",
                    uniforms=None if request.uniforms is None else request.uniforms[offset : offset + chunk],
                    arithmetic_uniform=request.arithmetic_uniform,
                ))
                indices.append(index)
            if not pending:
                break
            sampled = self.backend.sample_batch(pending)
            if len(sampled) != len(pending):
                raise RuntimeError("wrapped backend returned an invalid sample count")
            for index, inner_request, sample in zip(indices, pending, sampled, strict=True):
                if (
                    sample.prefix != inner_request.prefix
                    or sample.policy_id != inner_request.sampling.policy_id
                    or sample.model_id != self.backend.model_id
                    or not 0 < len(sample.token_ids) <= inner_request.max_new_tokens
                ):
                    raise RuntimeError("wrapped backend violated the generation contract")
                generated_prefix = self._generated_prefix(inner_request.prefix)
                end = self._stop_end(generated_prefix + sample.token_ids)
                keep = len(sample.token_ids) if end is None else end - len(generated_prefix)
                tokens[index].extend(sample.token_ids[:keep])
                logs[index].extend(sample.token_logprobs[:keep])
                if sample.reference_policy_id != expected_reference:
                    references[index] = None
                else:
                    reference = references[index]
                    if reference is not None:
                        assert sample.reference_token_logprobs is not None
                        reference.extend(sample.reference_token_logprobs[:keep])
                if end is not None:
                    done[index] = True
                elif len(sample.token_ids) != inner_request.max_new_tokens:
                    raise RuntimeError("generation ended without EOS or a configured stop marker")
        outputs = []
        for index, request in enumerate(requests):
            missing = request.max_new_tokens - len(tokens[index])
            reference = references[index]
            outputs.append(SequenceSample(
                prefix=request.prefix,
                token_ids=tuple(tokens[index]) + (self.eos_token_id,) * missing,
                token_logprobs=tuple(logs[index]) + (0.0,) * missing,
                model_id=self.model_id, policy_id=request.sampling.policy_id,
                request_id=request.request_id, finish_reason="stop" if done[index] else "length",
                reference_token_logprobs=None if reference is None else tuple(reference) + (0.0,) * missing,
                reference_policy_id=None if reference is None else SamplingConfig(
                    eos_token_id=request.sampling.eos_token_id
                ).policy_id,
            ))
        return outputs

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        pending: list[ScoreRequest] = []
        metadata: list[tuple[int, TokenSequence]] = []
        results: list[tuple[float, ...] | None] = []
        for request in requests:
            generated = self._generated_prefix(request.prefix)
            terminal = self._stop_end(generated) is not None
            for continuation in request.continuations:
                index = len(results)
                results.append(None)
                if terminal or not continuation:
                    results[index] = tuple(
                        0.0 if token == self.eos_token_id else float("-inf") for token in continuation
                    )
                    continue
                end = self._stop_end(generated + continuation)
                keep = len(continuation) if end is None else end - len(generated)
                pending.append(ScoreRequest(
                    request.prefix, (continuation[:keep],), self._inner_policy(request.sampling)
                ))
                metadata.append((index, continuation[keep:]))
        scored = self.backend.score_batch(pending) if pending else []
        if len(scored) != len(pending):
            raise RuntimeError("wrapped backend returned an invalid score count")
        for (index, tail), request, scores in zip(metadata, pending, scored, strict=True):
            if len(scores) != len(request.continuations[0]):
                raise RuntimeError("wrapped backend returned an invalid score shape")
            results[index] = tuple(scores) + tuple(
                0.0 if token == self.eos_token_id else float("-inf") for token in tail
            )
        if any(value is None for value in results):
            raise RuntimeError("incomplete stopped-sequence score batch")
        return [value for value in results if value is not None]


__all__ = ["StoppedSequenceBackend"]
