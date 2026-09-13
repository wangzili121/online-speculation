"""Exact Transformers speculative decoding with a separate draft model."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from dataclasses import dataclass
from types import MethodType, TracebackType
from typing import Any

from inference_scaling.arllm.acceleration import (
    DraftModelSpeculationConfig,
    SampleCompletionCallback,
)
from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, SequenceSample
from inference_scaling.shared.compute import dense_forward_flops


_NATIVE_GENERATION_LOCK = threading.RLock()
_CACHE_ALIGNED_CANDIDATE_GENERATOR: type[Any] | None = None


def _discard_stale_assistant_cache(cache: Any, input_length: int) -> int:
    """Remove rejected draft positions left beyond the accepted prefix."""

    get_seq_length = getattr(cache, "get_seq_length", None)
    crop = getattr(cache, "crop", None)
    if not callable(get_seq_length) or not callable(crop):
        return 0
    stale = max(int(get_seq_length()) - int(input_length), 0)
    if stale:
        crop(-stale)
    return stale


def _cache_aligned_candidate_generator() -> type[Any]:
    global _CACHE_ALIGNED_CANDIDATE_GENERATOR
    if _CACHE_ALIGNED_CANDIDATE_GENERATOR is not None:
        return _CACHE_ALIGNED_CANDIDATE_GENERATOR
    from transformers.generation.candidate_generator import AssistedCandidateGenerator

    class CacheAlignedAssistedCandidateGenerator(AssistedCandidateGenerator):
        """Crop rejected assistant positions before the library's one-token crop."""

        def _update_past_and_masks(
            self,
            input_ids: Any,
            remove_from_pkv: int = 0,
            num_added_tokens: int = 1,
        ) -> bool:
            cache = self.assistant_kwargs.get("past_key_values")
            if cache is not None:
                _discard_stale_assistant_cache(cache, int(input_ids.shape[-1]))
            return super()._update_past_and_masks(
                input_ids,
                remove_from_pkv=remove_from_pkv,
                num_added_tokens=num_added_tokens,
            )

    _CACHE_ALIGNED_CANDIDATE_GENERATOR = CacheAlignedAssistedCandidateGenerator
    return CacheAlignedAssistedCandidateGenerator


@contextmanager
def _aligned_candidate_generator(target_model: Any, draft_model: Any):
    """Install a request-local generator that keeps the assistant KV cache aligned."""

    original = getattr(target_model, "_get_candidate_generator", None)
    if not callable(original):
        yield
        return
    candidate_class = _cache_aligned_candidate_generator()
    had_instance_override = "_get_candidate_generator" in vars(target_model)
    prior_instance_value = vars(target_model).get("_get_candidate_generator")

    def build(
        _target: Any,
        generation_config: Any,
        input_ids: Any,
        inputs_tensor: Any,
        logits_processor: Any,
        model_kwargs: dict[str, Any],
        assistant_model: Any = None,
        target_tokenizer: Any = None,
        assistant_tokenizer: Any = None,
    ) -> Any:
        if (
            assistant_model is draft_model
            and target_tokenizer is None
            and assistant_tokenizer is None
        ):
            return candidate_class(
                input_ids=input_ids,
                assistant_model=assistant_model,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
            )
        return original(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            assistant_model=assistant_model,
            target_tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )

    target_model._get_candidate_generator = MethodType(build, target_model)
    try:
        yield
    finally:
        if had_instance_override:
            target_model._get_candidate_generator = prior_instance_value
        else:
            delattr(target_model, "_get_candidate_generator")


@dataclass(frozen=True, slots=True)
class DraftModelSpeculationSnapshot:
    sample_calls: int
    score_calls: int
    sampled_sequences: int
    generated_tokens: int
    prefill_tokens: int
    shared_prefill_tokens_saved: int
    scored_tokens: int
    generation_forward_token_slots: int
    score_forward_token_slots: int
    estimated_dense_forward_flops: int
    speculative_requests: int
    speculative_hits: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    speculative_verification_forward_token_slots: int
    draft_model_prefill_tokens: int
    draft_model_forward_token_slots: int
    draft_model_estimated_dense_forward_flops: int
    total_estimated_dense_forward_flops: int
    verification_rounds: int
    ordinary_batched_requests: int

    @property
    def draft_acceptance_rate(self) -> float:
        return (
            self.draft_tokens_accepted / self.draft_tokens_proposed
            if self.draft_tokens_proposed
            else 0.0
        )


@dataclass(slots=True)
class _ForwardCounter:
    calls: int = 0
    token_slots: int = 0
    proposed_tokens: int = 0

    def hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            self.token_slots += int(input_ids.numel())
        else:
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is not None:
                self.token_slots += int(inputs_embeds.shape[0] * inputs_embeds.shape[1])
        self.calls += 1
        logits_to_keep = kwargs.get("logits_to_keep")
        if isinstance(logits_to_keep, int):
            self.proposed_tokens += max(logits_to_keep - 1, 0)


class _ForwardAccounting:
    def __init__(self, model: Any) -> None:
        self.counter = _ForwardCounter()
        self._handle = model.register_forward_pre_hook(
            self.counter.hook,
            with_kwargs=True,
        )

    def __enter__(self) -> _ForwardCounter:
        return self.counter

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._handle.remove()


class DraftModelSpeculativeBackend:
    """Preserve a target policy while a smaller model proposes token blocks.

    The implementation delegates acceptance and residual correction to
    Transformers' speculative-sampling kernel.  The target model verifies every
    proposed block.  Consequently the returned sequence follows the target
    model's requested sampling policy; the draft model changes execution cost,
    not the output distribution.

    Native assisted generation is restricted to batch size one in Transformers.
    The default therefore applies it only to single-request tails and retains the
    target backend's ordinary batched decoder for larger active batches.
    """

    def __init__(
        self,
        target: TransformersBackend,
        draft: TransformersBackend,
        *,
        config: DraftModelSpeculationConfig | None = None,
    ) -> None:
        if target is draft or target.model is draft.model:
            raise ValueError("target and draft models must be distinct")
        if target.tokenizer.get_vocab() != draft.tokenizer.get_vocab():
            raise ValueError(
                "target and draft tokenizers must have identical vocabularies"
            )
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(target.tokenizer, name, None) != getattr(
                draft.tokenizer, name, None
            ):
                raise ValueError(f"target and draft tokenizers disagree on {name}")
        self.target = target
        self.draft = draft
        self.config = config or DraftModelSpeculationConfig()
        self.tokenizer = target.tokenizer
        self.device = target.device
        self._statistics_lock = threading.Lock()
        self._sample_calls = 0
        self._sampled_sequences = 0
        self._generated_tokens = 0
        self._native_target_prefill_tokens = 0
        self._native_draft_prefill_tokens = 0
        self._native_target_slots = 0
        self._native_draft_slots = 0
        self._speculative_requests = 0
        self._speculative_hits = 0
        self._draft_tokens_proposed = 0
        self._draft_tokens_accepted = 0
        self._verification_rounds = 0
        self._ordinary_batched_requests = 0

    @property
    def model_id(self) -> str:
        return self.target.model_id

    @property
    def draft_model_id(self) -> str:
        return self.draft.model_id

    @property
    def parameter_count(self) -> int:
        return self.target.parameter_count

    @property
    def draft_parameter_count(self) -> int:
        return self.draft.parameter_count

    @staticmethod
    def _cuda_rng_devices(*devices: Any) -> list[int]:
        torch_module = __import__("torch")
        indices: set[int] = set()
        for device in devices:
            resolved = torch_module.device(device)
            if resolved.type != "cuda":
                continue
            indices.add(
                int(resolved.index)
                if resolved.index is not None
                else int(torch_module.cuda.current_device())
            )
        return sorted(indices)

    def _sample_one(self, request: GenerationRequest) -> SequenceSample:
        if request.uniforms is not None or request.arithmetic_uniform is not None:
            raise ValueError(
                "native draft-model speculation cannot consume explicit token uniforms "
                "or arithmetic uniforms"
            )
        torch_module = __import__("torch")
        prefix = self.target._model_prefix(request.prefix)
        input_ids = torch_module.tensor(
            [prefix], dtype=torch_module.long, device=self.target.device
        )
        attention_mask = torch_module.ones_like(input_ids)
        assistant_config = self.draft.model.generation_config
        saved = {
            "num_assistant_tokens": getattr(
                assistant_config, "num_assistant_tokens", None
            ),
            "num_assistant_tokens_schedule": getattr(
                assistant_config, "num_assistant_tokens_schedule", None
            ),
            "assistant_confidence_threshold": getattr(
                assistant_config, "assistant_confidence_threshold", None
            ),
        }
        generated: Any
        target_counter: _ForwardCounter
        draft_counter: _ForwardCounter
        rng_devices = self._cuda_rng_devices(self.target.device, self.draft.device)
        with _NATIVE_GENERATION_LOCK, self.target._model_lock, self.draft._model_lock:
            assistant_config.num_assistant_tokens = self.config.draft_tokens
            assistant_config.num_assistant_tokens_schedule = "constant"
            if self.config.confidence_threshold is not None:
                assistant_config.assistant_confidence_threshold = (
                    self.config.confidence_threshold
                )
            try:
                with (
                    torch_module.random.fork_rng(devices=rng_devices),
                    _ForwardAccounting(self.target.model) as target_counter,
                    _ForwardAccounting(self.draft.model) as draft_counter,
                    _aligned_candidate_generator(self.target.model, self.draft.model),
                    torch_module.inference_mode(),
                ):
                    torch_module.manual_seed(request.seed)
                    generated = self.target.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        assistant_model=self.draft.model,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=True,
                        temperature=request.sampling.temperature,
                        top_p=request.sampling.top_p,
                        top_k=(
                            0
                            if request.sampling.top_k is None
                            else request.sampling.top_k
                        ),
                        repetition_penalty=1.0,
                        no_repeat_ngram_size=0,
                        num_beams=1,
                        num_return_sequences=1,
                        eos_token_id=request.sampling.eos_token_id,
                        pad_token_id=self.target.pad_token_id,
                        bos_token_id=self.target.bos_token_id,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_logits=True,
                    )
            finally:
                for name, value in saved.items():
                    setattr(assistant_config, name, value)

        token_ids = tuple(
            int(token)
            for token in generated.sequences[0, input_ids.shape[1] :].tolist()
        )
        raw_logits = getattr(generated, "logits", None)
        if raw_logits is None or len(raw_logits) != len(token_ids):
            raise RuntimeError(
                "Transformers assisted generation omitted per-token target logits"
            )
        reference_sampling = SamplingConfig(eos_token_id=request.sampling.eos_token_id)
        token_logprobs: list[float] = []
        reference_logprobs: list[float] = []
        for token, logits in zip(token_ids, raw_logits, strict=True):
            row = logits[0]
            policy = self.target._policy_log_probs(row, request.sampling)
            reference = (
                policy
                if request.sampling == reference_sampling
                else self.target._policy_log_probs(row, reference_sampling)
            )
            token_logprobs.append(float(policy[token].detach().cpu()))
            reference_logprobs.append(float(reference[token].detach().cpu()))

        proposed = target_counter.proposed_tokens
        rounds = target_counter.calls
        accepted = max(len(token_ids) - rounds, 0)
        if proposed and accepted > proposed:
            raise RuntimeError(
                "Transformers reported more accepted than proposed draft tokens"
            )
        finish_reason = (
            "eos"
            if token_ids
            and request.sampling.eos_token_id is not None
            and token_ids[-1] == request.sampling.eos_token_id
            else "length"
        )
        with self._statistics_lock:
            self._native_target_prefill_tokens += len(prefix)
            self._native_draft_prefill_tokens += len(prefix)
            self._native_target_slots += target_counter.token_slots
            self._native_draft_slots += draft_counter.token_slots
            self._speculative_requests += 1
            self._speculative_hits += int(proposed > 0)
            self._draft_tokens_proposed += proposed
            self._draft_tokens_accepted += accepted
            self._verification_rounds += rounds
        return SequenceSample(
            prefix=request.prefix,
            token_ids=token_ids,
            token_logprobs=tuple(token_logprobs),
            policy_id=request.sampling.policy_id,
            model_id=self.target.model_id,
            request_id=request.request_id,
            finish_reason=finish_reason,
            reference_token_logprobs=tuple(reference_logprobs),
            reference_policy_id=reference_sampling.policy_id,
        )

    def _sample_batch(
        self,
        requests: list[GenerationRequest] | tuple[GenerationRequest, ...],
        on_complete: SampleCompletionCallback | None,
    ) -> list[SequenceSample]:
        if not requests:
            return []
        if self.config.single_request_only and len(requests) > 1:
            outputs = self.target.sample_batch(requests)
            with self._statistics_lock:
                self._ordinary_batched_requests += len(requests)
        else:
            outputs = []
            for index, request in enumerate(requests):
                sample = self._sample_one(request)
                outputs.append(sample)
                if on_complete is not None:
                    on_complete(index, sample)
        if (
            on_complete is not None
            and self.config.single_request_only
            and len(requests) > 1
        ):
            for index, sample in enumerate(outputs):
                on_complete(index, sample)
        with self._statistics_lock:
            self._sample_calls += 1
            self._sampled_sequences += len(outputs)
            self._generated_tokens += sum(len(output.token_ids) for output in outputs)
        return outputs

    def sample_batch(self, requests: Any) -> list[SequenceSample]:
        return self._sample_batch(tuple(requests), None)

    def sample_batch_with_callback(
        self,
        requests: Any,
        on_complete: SampleCompletionCallback,
    ) -> list[SequenceSample]:
        return self._sample_batch(tuple(requests), on_complete)

    def score_batch(self, requests: Any) -> list[tuple[float, ...]]:
        return self.target.score_batch(requests)

    def snapshot(self) -> DraftModelSpeculationSnapshot:
        target = self.target.snapshot()
        with self._statistics_lock:
            target_generation_slots = (
                target.generation_forward_token_slots + self._native_target_slots
            )
            target_flops = target.estimated_dense_forward_flops + dense_forward_flops(
                self.target.parameter_count,
                self._native_target_slots,
            )
            draft_flops = self.draft.snapshot().estimated_dense_forward_flops
            draft_flops += dense_forward_flops(
                self.draft.parameter_count,
                self._native_draft_slots,
            )
            return DraftModelSpeculationSnapshot(
                sample_calls=self._sample_calls,
                score_calls=target.score_calls,
                sampled_sequences=self._sampled_sequences,
                generated_tokens=self._generated_tokens,
                prefill_tokens=target.prefill_tokens
                + self._native_target_prefill_tokens,
                shared_prefill_tokens_saved=target.shared_prefill_tokens_saved,
                scored_tokens=target.scored_tokens,
                generation_forward_token_slots=target_generation_slots,
                score_forward_token_slots=target.score_forward_token_slots,
                estimated_dense_forward_flops=target_flops,
                speculative_requests=self._speculative_requests,
                speculative_hits=self._speculative_hits,
                draft_tokens_proposed=self._draft_tokens_proposed,
                draft_tokens_accepted=self._draft_tokens_accepted,
                speculative_verification_forward_token_slots=self._native_target_slots,
                draft_model_prefill_tokens=self._native_draft_prefill_tokens,
                draft_model_forward_token_slots=self._native_draft_slots,
                draft_model_estimated_dense_forward_flops=draft_flops,
                total_estimated_dense_forward_flops=target_flops + draft_flops,
                verification_rounds=self._verification_rounds,
                ordinary_batched_requests=self._ordinary_batched_requests,
            )

    def encode(self, text: str, *, add_special_tokens: bool = True):
        return self.target.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: Any, *, skip_special_tokens: bool = True) -> str:
        return self.target.decode(tokens, skip_special_tokens=skip_special_tokens)

    def direct_generate(self, *args: Any, **kwargs: Any):
        return self.target.direct_generate(*args, **kwargs)


__all__ = ["DraftModelSpeculationSnapshot", "DraftModelSpeculativeBackend"]
