"""Batched Transformers backend for LLaDA-style masked diffusion models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import lgamma
import re
from threading import Lock, RLock
from time import perf_counter
from typing import Any, Literal

from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import (
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTraceStep,
    DiffusionTrajectoryScoreRequest,
)


@dataclass(frozen=True, slots=True)
class LLaDABackendSnapshot:
    sample_requests: int
    score_requests: int
    forward_calls: int
    model_sequences: int
    model_token_slots: int
    generated_tokens: int
    elapsed_seconds: float
    total_parameters: int
    active_parameters: int
    resident_parameters: int
    sample_forward_calls: int
    score_forward_calls: int
    sample_model_sequences: int
    score_model_sequences: int
    sample_model_token_slots: int
    score_model_token_slots: int
    sample_elapsed_seconds: float
    score_elapsed_seconds: float

    @property
    def estimated_active_flops(self) -> float:
        return 2.0 * self.active_parameters * self.model_token_slots

    @property
    def estimated_total_parameter_flops(self) -> float:
        return 2.0 * self.total_parameters * self.model_token_slots


class LLaDATransformersBackend:
    """Execute blockwise masked diffusion and record committed trajectories.

    The implementation follows the public LLaDA sampler: all masked positions
    are predicted in parallel and each reverse step commits a fixed number of
    positions.  ``low_confidence`` commits the most confident predictions.
    ``random`` commits a uniform subset independent of sampled token values;
    ``sequential`` commits the leftmost still-masked positions.  The latter two
    policies have tractable trajectory probabilities and support off-policy IS.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str | None = None,
        mask_token_id: int | None = None,
        active_parameters: int | None = None,
        layer_limit: int | None = None,
        max_batch_size: int | None = None,
        execution_lock: RLock | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised without GPU extras
            raise RuntimeError("LLaDA backend requires PyTorch") from exc

        self.model = model.eval()
        self.tokenizer = tokenizer
        self._torch = torch
        configured_name = getattr(getattr(model, "config", None), "_name_or_path", None)
        self._model_id = model_id or configured_name or model.__class__.__name__
        self._mask_token_id = self._infer_mask_token_id(mask_token_id)
        self._device = self._infer_device()
        self._layer_limit = layer_limit
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when provided")
        self._max_batch_size = max_batch_size
        self._execution_lock = execution_lock or RLock()
        self._layer_parent, self._layer_attribute, self._full_layers = (
            self._resolve_decoder_layers()
        )
        if self._layer_limit is not None:
            if self._full_layers is None:
                raise ValueError("layer_limit requires a model with decoder layers")
            if not 0 < self._layer_limit <= len(self._full_layers):
                raise ValueError("layer_limit must lie within the decoder layer count")
        self._resident_parameters = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
        inferred_total, inferred_active = self._effective_parameter_counts()
        self._total_parameters = inferred_total
        self._active_parameters = int(
            active_parameters if active_parameters is not None else inferred_active
        )
        if self._active_parameters <= 0 or self._active_parameters > self._total_parameters:
            raise ValueError("active_parameters must lie in (0, total_parameters]")
        self._lock = Lock()
        self._sample_requests = 0
        self._score_requests = 0
        self._forward_calls = 0
        self._model_sequences = 0
        self._model_token_slots = 0
        self._generated_tokens = 0
        self._elapsed_seconds = 0.0
        self._sample_forward_calls = 0
        self._score_forward_calls = 0
        self._sample_model_sequences = 0
        self._score_model_sequences = 0
        self._sample_model_token_slots = 0
        self._score_model_token_slots = 0
        self._sample_elapsed_seconds = 0.0
        self._score_elapsed_seconds = 0.0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        mask_token_id: int | None = None,
        active_parameters: int | None = None,
        max_batch_size: int | None = None,
        **model_kwargs: Any,
    ) -> "LLaDATransformersBackend":
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError("install the dllm optional dependency set") from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
            **model_kwargs,
        ).to(device)
        return cls(
            model,
            tokenizer,
            model_id=model_name_or_path,
            mask_token_id=mask_token_id,
            active_parameters=active_parameters,
            max_batch_size=max_batch_size,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    def with_prefix_layers(self, layer_count: int) -> "LLaDATransformersBackend":
        """Create a lower-compute early-exit backend that shares resident weights."""

        return type(self)(
            self.model,
            self.tokenizer,
            model_id=f"{self.model_id}#prefix-layers={layer_count}",
            mask_token_id=self.mask_token_id,
            layer_limit=layer_count,
            max_batch_size=self._max_batch_size,
            execution_lock=self._execution_lock,
        )

    def _request_chunks(self, group: Sequence[Any]) -> Sequence[Sequence[Any]]:
        chunk_size = self._max_batch_size or len(group)
        return tuple(
            group[start : start + chunk_size]
            for start in range(0, len(group), chunk_size)
        )

    def encode_chat(
        self,
        user_text: str,
        *,
        system_text: str = "You are a helpful AI assistant.",
    ) -> tuple[int, ...]:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        else:
            encoded = self.tokenizer(user_text, add_special_tokens=True)["input_ids"]
        return tuple(int(token_id) for token_id in encoded)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return str(
            self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
        )

    def snapshot(self) -> LLaDABackendSnapshot:
        with self._lock:
            return LLaDABackendSnapshot(
                sample_requests=self._sample_requests,
                score_requests=self._score_requests,
                forward_calls=self._forward_calls,
                model_sequences=self._model_sequences,
                model_token_slots=self._model_token_slots,
                generated_tokens=self._generated_tokens,
                elapsed_seconds=self._elapsed_seconds,
                total_parameters=self._total_parameters,
                active_parameters=self._active_parameters,
                resident_parameters=self._resident_parameters,
                sample_forward_calls=self._sample_forward_calls,
                score_forward_calls=self._score_forward_calls,
                sample_model_sequences=self._sample_model_sequences,
                score_model_sequences=self._score_model_sequences,
                sample_model_token_slots=self._sample_model_token_slots,
                score_model_token_slots=self._score_model_token_slots,
                sample_elapsed_seconds=self._sample_elapsed_seconds,
                score_elapsed_seconds=self._score_elapsed_seconds,
            )

    def sample_batch(
        self, requests: Sequence[DiffusionGenerationRequest]
    ) -> list[DiffusionSample]:
        if not requests:
            return []
        started = perf_counter()
        groups: dict[
            tuple[int, int, DiffusionSamplingConfig], list[tuple[int, DiffusionGenerationRequest]]
        ] = defaultdict(list)
        for index, request in enumerate(requests):
            groups[(len(request.prefix), request.generation_length, request.sampling)].append(
                (index, request)
            )
        outputs: list[DiffusionSample | None] = [None] * len(requests)
        with self._torch.inference_mode():
            for group in groups.values():
                for chunk in self._request_chunks(group):
                    indices = [item[0] for item in chunk]
                    samples = self._sample_group([item[1] for item in chunk])
                    for index, sample in zip(indices, samples, strict=True):
                        outputs[index] = sample
        elapsed = perf_counter() - started
        with self._lock:
            self._sample_requests += len(requests)
            self._generated_tokens += sum(request.generation_length for request in requests)
            self._elapsed_seconds += elapsed
            self._sample_elapsed_seconds += elapsed
        if any(output is None for output in outputs):
            raise RuntimeError("internal dLLM batch reordering failure")
        return [output for output in outputs if output is not None]

    def score_trajectories(
        self, requests: Sequence[DiffusionTrajectoryScoreRequest]
    ) -> list[float]:
        if not requests:
            return []
        started = perf_counter()
        groups: dict[
            tuple[int, int, DiffusionSamplingConfig],
            list[tuple[int, DiffusionTrajectoryScoreRequest]],
        ] = defaultdict(list)
        for index, request in enumerate(requests):
            sample = request.sample
            groups[(len(sample.prefix), len(sample.token_ids), request.sampling)].append(
                (index, request)
            )
        outputs: list[float | None] = [None] * len(requests)
        with self._torch.inference_mode():
            for group in groups.values():
                for chunk in self._request_chunks(group):
                    indices = [item[0] for item in chunk]
                    scores = self._score_group([item[1] for item in chunk])
                    for index, score in zip(indices, scores, strict=True):
                        outputs[index] = score
        elapsed = perf_counter() - started
        with self._lock:
            self._score_requests += len(requests)
            self._elapsed_seconds += elapsed
            self._score_elapsed_seconds += elapsed
        if any(output is None for output in outputs):
            raise RuntimeError("internal dLLM score reordering failure")
        return [float(output) for output in outputs if output is not None]

    def _infer_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._torch.device("cpu")

    def _infer_mask_token_id(self, explicit: int | None) -> int:
        candidates = (
            explicit,
            getattr(self.tokenizer, "mask_token_id", None),
            getattr(self.tokenizer, "gmask_token_id", None),
            getattr(getattr(self.model, "config", None), "mask_token_id", None),
        )
        for candidate in candidates:
            if isinstance(candidate, int) and candidate >= 0:
                return candidate
        converter = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(converter):
            candidate = converter("[MASK]")
            if isinstance(candidate, int) and candidate >= 0:
                return candidate
        raise ValueError("mask_token_id is absent from the model and tokenizer; provide it explicitly")

    def _resolve_decoder_layers(self) -> tuple[Any | None, str | None, Any | None]:
        candidates = (
            self.model,
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "model", None), "model", None),
        )
        for parent in candidates:
            if parent is not None and hasattr(parent, "layers"):
                layers = getattr(parent, "layers")
                if hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
                    return parent, "layers", layers
        return None, None, None

    def _effective_parameter_counts(self) -> tuple[int, int]:
        config = getattr(self.model, "config", None)
        expert_count = int(getattr(config, "num_experts", 0) or 0)
        active_experts = int(getattr(config, "num_experts_per_tok", 0) or 0)
        expert_fraction = (
            active_experts / expert_count
            if 0 < active_experts <= expert_count
            else 1.0
        )
        total = 0
        active = 0.0
        layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")
        for name, parameter in self.model.named_parameters():
            match = layer_pattern.search(name)
            if (
                self._layer_limit is not None
                and match is not None
                and int(match.group(1)) >= self._layer_limit
            ):
                continue
            count = int(parameter.numel())
            total += count
            active += count * expert_fraction if ".experts." in name else count
        return total, int(round(active))

    @contextmanager
    def _selected_decoder_layers(self):
        with self._execution_lock:
            if self._layer_limit is None:
                yield
                return
            assert self._layer_parent is not None
            assert self._layer_attribute is not None
            assert self._full_layers is not None
            current = getattr(self._layer_parent, self._layer_attribute)
            if current is not self._full_layers:
                raise RuntimeError("shared LLaDA decoder layers were modified concurrently")
            limited = self._torch.nn.ModuleList(
                list(self._full_layers[: self._layer_limit])
            )
            setattr(self._layer_parent, self._layer_attribute, limited)
            try:
                yield
            finally:
                setattr(self._layer_parent, self._layer_attribute, self._full_layers)

    def _resolve_mask_token_id(self, sampling: DiffusionSamplingConfig) -> int:
        return sampling.mask_token_id if sampling.mask_token_id is not None else self._mask_token_id

    def _record_forward(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        phase: Literal["sample", "score"],
    ) -> None:
        token_slots = batch_size * sequence_length
        with self._lock:
            self._forward_calls += 1
            self._model_sequences += batch_size
            self._model_token_slots += token_slots
            if phase == "sample":
                self._sample_forward_calls += 1
                self._sample_model_sequences += batch_size
                self._sample_model_token_slots += token_slots
            else:
                self._score_forward_calls += 1
                self._score_model_sequences += batch_size
                self._score_model_token_slots += token_slots

    def _model_logits(
        self,
        tokens: Any,
        *,
        prompt_length: int,
        sampling: DiffusionSamplingConfig,
        mask_token_id: int,
        phase: Literal["sample", "score"],
    ) -> Any:
        with self._selected_decoder_layers():
            if sampling.cfg_scale > 0:
                unconditional = tokens.clone()
                unconditional[:, :prompt_length] = mask_token_id
                model_input = self._torch.cat((tokens, unconditional), dim=0)
                output = self.model(model_input)
                self._record_forward(
                    model_input.shape[0], model_input.shape[1], phase=phase
                )
                logits, unconditional_logits = self._torch.chunk(output.logits, 2, dim=0)
                logits = unconditional_logits + (sampling.cfg_scale + 1.0) * (
                    logits - unconditional_logits
                )
            else:
                output = self.model(tokens)
                self._record_forward(tokens.shape[0], tokens.shape[1], phase=phase)
                logits = output.logits
        # A committed mask would leave the state unchanged and violate the fixed
        # transfer schedule.  The reverse policy is therefore normalized over
        # ordinary vocabulary tokens only.
        logits = logits.clone()
        logits[..., mask_token_id] = -self._torch.inf
        return self._filter_logits(logits, sampling)

    def _filter_logits(self, logits: Any, sampling: DiffusionSamplingConfig) -> Any:
        """Apply the normalized top-k/top-p policy used for sampling and scoring."""

        if sampling.top_k > 0:
            kept = min(sampling.top_k, logits.shape[-1])
            threshold = self._torch.topk(logits, kept, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, -self._torch.inf)
        if sampling.top_p < 1:
            sorted_logits, sorted_indices = self._torch.sort(logits, descending=True, dim=-1)
            cumulative = self._torch.cumsum(
                self._torch.softmax(sorted_logits.float(), dim=-1), dim=-1
            )
            sorted_remove = cumulative > sampling.top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = self._torch.zeros_like(sorted_remove).scatter(
                -1, sorted_indices, sorted_remove
            )
            logits = logits.masked_fill(remove, -self._torch.inf)
        return logits

    def _draw_tokens(
        self,
        logits: Any,
        *,
        temperature: float,
        generators: Sequence[Any],
    ) -> tuple[Any, Any | None]:
        if temperature == 0:
            return self._torch.argmax(logits, dim=-1), None
        sampled_rows = []
        logprob_rows = []
        for row, generator in zip(logits, generators, strict=True):
            scaled = row.float() / temperature
            exponential = self._torch.empty_like(scaled).exponential_(1.0, generator=generator)
            sampled = self._torch.argmax(scaled - self._torch.log(exponential), dim=-1)
            log_normalizer = self._torch.logsumexp(scaled, dim=-1)
            sampled_logprob = self._torch.gather(
                scaled, dim=-1, index=sampled.unsqueeze(-1)
            ).squeeze(-1) - log_normalizer
            sampled_rows.append(sampled)
            logprob_rows.append(sampled_logprob)
        return self._torch.stack(sampled_rows), self._torch.stack(logprob_rows)

    @staticmethod
    def _subset_logprob(available: int, selected: int) -> float:
        if not 0 <= selected <= available:
            raise ValueError("invalid subset size")
        return -(lgamma(available + 1) - lgamma(selected + 1) - lgamma(available - selected + 1))

    @staticmethod
    def _transfer_schedule(block_length: int, steps: int) -> tuple[int, ...]:
        quotient, remainder = divmod(block_length, steps)
        return tuple(quotient + (index < remainder) for index in range(steps))

    def _sample_group(
        self, requests: Sequence[DiffusionGenerationRequest]
    ) -> list[DiffusionSample]:
        first = requests[0]
        prompt_length = len(first.prefix)
        generation_length = first.generation_length
        sampling = first.sampling
        if sampling.remasking == "low_confidence_dynamic":
            raise ValueError("LLaDA backend does not implement dynamic-threshold remasking")
        mask_token_id = self._resolve_mask_token_id(sampling)
        batch_size = len(requests)
        tokens = self._torch.full(
            (batch_size, prompt_length + generation_length),
            mask_token_id,
            dtype=self._torch.long,
            device=self._device,
        )
        for row, request in enumerate(requests):
            tokens[row, :prompt_length] = self._torch.tensor(
                request.prefix, dtype=self._torch.long, device=self._device
            )
        generators = [
            self._torch.Generator(device=self._device).manual_seed(request.seed)
            for request in requests
        ]
        traces: list[list[DiffusionTraceStep]] = [[] for _ in requests]
        trajectory_logprobs = [0.0 for _ in requests]
        exact = sampling.has_exact_trajectory_density
        schedule = self._transfer_schedule(sampling.block_length, sampling.steps_per_block)
        block_count = generation_length // sampling.block_length

        for block_index in range(block_count):
            relative_start = block_index * sampling.block_length
            relative_end = relative_start + sampling.block_length
            absolute_start = prompt_length + relative_start
            absolute_end = prompt_length + relative_end
            for step_index, transfer_count in enumerate(schedule):
                logits = self._model_logits(
                    tokens,
                    prompt_length=prompt_length,
                    sampling=sampling,
                    mask_token_id=mask_token_id,
                    phase="sample",
                )
                sampled_tokens, sampled_logprobs = self._draw_tokens(
                    logits, temperature=sampling.temperature, generators=generators
                )
                if sampling.remasking in {"low_confidence", "low_confidence_static"}:
                    chosen_logits = self._torch.gather(
                        logits.float(), dim=-1, index=sampled_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    confidence = chosen_logits - self._torch.logsumexp(
                        logits.float(), dim=-1
                    )
                else:
                    confidence = None

                for row, generator in enumerate(generators):
                    masked_absolute = self._torch.nonzero(
                        tokens[row, absolute_start:absolute_end] == mask_token_id,
                        as_tuple=False,
                    ).flatten() + absolute_start
                    available = int(masked_absolute.numel())
                    if transfer_count > available:
                        raise RuntimeError("reverse-diffusion schedule over-committed a block")
                    if sampling.remasking == "random":
                        priorities = self._torch.rand(
                            available, device=self._device, generator=generator
                        )
                    elif sampling.remasking == "sequential":
                        priorities = -self._torch.arange(
                            available, device=self._device, dtype=self._torch.float32
                        )
                    else:
                        assert confidence is not None
                        priorities = confidence[row, masked_absolute]
                    selected_local = self._torch.topk(
                        priorities, k=transfer_count, sorted=False
                    ).indices
                    selected_absolute = masked_absolute[selected_local]
                    selected_relative = selected_absolute - prompt_length
                    order = self._torch.argsort(selected_relative)
                    selected_absolute = selected_absolute[order]
                    selected_relative = selected_relative[order]
                    selected_tokens = sampled_tokens[row, selected_absolute]
                    tokens[row, selected_absolute] = selected_tokens

                    step_logprob: float | None = None
                    if exact:
                        assert sampled_logprobs is not None
                        subset_logprob = (
                            self._subset_logprob(available, transfer_count)
                            if sampling.remasking == "random"
                            else 0.0
                        )
                        step_logprob = subset_logprob + float(
                            sampled_logprobs[row, selected_absolute].sum().item()
                        )
                        trajectory_logprobs[row] += step_logprob
                    traces[row].append(
                        DiffusionTraceStep(
                            block_index=block_index,
                            step_index=step_index,
                            positions=tuple(int(value) for value in selected_relative.tolist()),
                            token_ids=tuple(int(value) for value in selected_tokens.tolist()),
                            logprob=step_logprob,
                        )
                    )

        samples: list[DiffusionSample] = []
        for row, request in enumerate(requests):
            continuation = tuple(int(value) for value in tokens[row, prompt_length:].tolist())
            samples.append(
                DiffusionSample(
                    prefix=request.prefix,
                    token_ids=continuation,
                    trace=tuple(traces[row]),
                    trajectory_logprob=trajectory_logprobs[row] if exact else None,
                    policy_id=sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                )
            )
        return samples

    def _score_group(
        self, requests: Sequence[DiffusionTrajectoryScoreRequest]
    ) -> list[float]:
        first = requests[0]
        prompt_length = len(first.sample.prefix)
        generation_length = len(first.sample.token_ids)
        sampling = first.sampling
        if not sampling.has_exact_trajectory_density:
            raise ValueError(
                "trajectory scoring requires random or sequential remasking and positive temperature"
            )
        sampling.validate_generation_length(
            generation_length,
            prefix_length=prompt_length,
        )
        mask_token_id = self._resolve_mask_token_id(sampling)
        expected_steps = sampling.total_steps(generation_length)
        for request in requests:
            if len(request.sample.trace) != expected_steps:
                raise ValueError("sample trace does not match the target transition schedule")

        tokens = self._torch.full(
            (len(requests), prompt_length + generation_length),
            mask_token_id,
            dtype=self._torch.long,
            device=self._device,
        )
        for row, request in enumerate(requests):
            tokens[row, :prompt_length] = self._torch.tensor(
                request.sample.prefix, dtype=self._torch.long, device=self._device
            )
        totals = [0.0 for _ in requests]

        for trace_index in range(expected_steps):
            logits = self._model_logits(
                tokens,
                prompt_length=prompt_length,
                sampling=sampling,
                mask_token_id=mask_token_id,
                phase="score",
            ).float()
            scaled = logits / sampling.temperature
            log_normalizers = self._torch.logsumexp(scaled, dim=-1)
            for row, request in enumerate(requests):
                step = request.sample.trace[trace_index]
                expected_block = trace_index // sampling.steps_per_block
                expected_step = trace_index % sampling.steps_per_block
                if step.block_index != expected_block or step.step_index != expected_step:
                    raise ValueError("sample trace steps are out of schedule order")
                relative_start = expected_block * sampling.block_length
                relative_end = relative_start + sampling.block_length
                absolute_start = prompt_length + relative_start
                absolute_end = prompt_length + relative_end
                available = int(
                    (tokens[row, absolute_start:absolute_end] == mask_token_id).sum().item()
                )
                expected_transfer = self._transfer_schedule(
                    sampling.block_length, sampling.steps_per_block
                )[expected_step]
                if len(step.positions) != expected_transfer:
                    raise ValueError("sample trace committed the wrong number of positions")
                if any(not relative_start <= position < relative_end for position in step.positions):
                    raise ValueError("sample trace committed a position outside its block")
                absolute_positions = self._torch.tensor(
                    [prompt_length + position for position in step.positions],
                    dtype=self._torch.long,
                    device=self._device,
                )
                if not bool(
                    self._torch.all(tokens[row, absolute_positions] == mask_token_id).item()
                ):
                    raise ValueError("sample trace commits an already visible position")
                selected_tokens = self._torch.tensor(
                    step.token_ids, dtype=self._torch.long, device=self._device
                )
                selected_logits = scaled[row, absolute_positions, selected_tokens]
                selected_logprobs = selected_logits - log_normalizers[
                    row, absolute_positions
                ]
                subset_logprob = (
                    self._subset_logprob(available, len(step.positions))
                    if sampling.remasking == "random"
                    else 0.0
                )
                totals[row] += subset_logprob + float(selected_logprobs.sum().item())
                tokens[row, absolute_positions] = selected_tokens

        for row, request in enumerate(requests):
            final = tuple(int(value) for value in tokens[row, prompt_length:].tolist())
            if final != request.sample.token_ids:
                raise ValueError("sample trace does not reconstruct its final continuation")
        return totals


__all__ = ["LLaDABackendSnapshot", "LLaDATransformersBackend"]
