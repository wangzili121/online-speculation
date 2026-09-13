"""Batched causal-LM backend with exact policy probabilities and KV decoding."""

from __future__ import annotations

import inspect
import math
import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from inference_scaling.arllm.acceleration import (
    ActiveBatchSpeculationConfig,
    DraftProposal,
    RolloutTokenTree,
    RolloutTokenTreeSnapshot,
    SampleCompletionCallback,
)
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.compute import dense_forward_flops
from inference_scaling.shared.model_loading import model_identity
from inference_scaling.arllm.backends.causal_scoring import iter_causal_logits, prefill_causal_model
from inference_scaling.shared.rng import SeedStream
from inference_scaling.arllm.types import (
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

try:
    import torch
except ImportError:  # pragma: no cover - exercised in dependency-free installations
    torch = None  # type: ignore[assignment]


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "TransformersBackend requires the optional GPU dependencies; "
            "install the project's gpu extra first"
        )
    return torch


@dataclass(frozen=True, slots=True)
class TransformersBackendSnapshot:
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
    speculative_requests: int = 0
    speculative_hits: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    speculative_verification_forward_token_slots: int = 0


@dataclass(frozen=True, slots=True)
class SequenceScoreStatistics:
    """Reference-policy statistics for one scored continuation."""

    token_logprobs: tuple[float, ...]
    mean_logprob: float
    mean_negative_entropy: float
    mean_self_certainty: float
    token_topk_confidences: tuple[float, ...] = ()
    confidence_top_k: int | None = None


class TransformersBackend:
    """Manual batched decoding with request-local, scheduling-independent RNG."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str | None = None,
        device: str | Any | None = None,
        max_score_batch_size: int = 8,
        score_chunk_size: int = 256,
        draft_tree: RolloutTokenTree | None = None,
        speculation: ActiveBatchSpeculationConfig | None = None,
    ) -> None:
        torch_module = _require_torch()
        self.model = model
        self.tokenizer = tokenizer
        self._model_id = model_id or str(
            getattr(
                getattr(model, "config", None), "_name_or_path", "transformers-model"
            )
        )
        inferred_device = getattr(model, "device", None)
        self.device = torch_module.device(
            device
            or inferred_device
            or ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_token_id)
        if max_score_batch_size <= 0:
            raise ValueError("max_score_batch_size must be positive")
        self.max_score_batch_size = int(max_score_batch_size)
        if score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")
        self.score_chunk_size = int(score_chunk_size)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.bos_token_id = None if bos_token_id is None else int(bos_token_id)
        self.model.eval()
        inspected_model = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
        forward_parameters = inspect.signature(inspected_model.forward).parameters
        self._supports_logits_to_keep = "logits_to_keep" in forward_parameters
        self._model_lock = threading.RLock()
        self.cache_growth_tokens = 0
        self._statistics_lock = threading.Lock()
        self._sample_calls = 0
        self._score_calls = 0
        self._sampled_sequences = 0
        self._generated_tokens = 0
        self._prefill_tokens = 0
        self._shared_prefill_tokens_saved = 0
        self._scored_tokens = 0
        self._generation_forward_token_slots = 0
        self._score_forward_token_slots = 0
        self._estimated_dense_forward_flops = 0
        count_parameters = getattr(model, "num_parameters", None)
        self._parameter_count = (
            int(count_parameters()) if callable(count_parameters)
            else sum(parameter.numel() for parameter in model.parameters())
        )
        self._speculation = speculation
        self._draft_tree = (
            draft_tree
            if draft_tree is not None
            else (
                RolloutTokenTree.from_config(speculation)
                if speculation is not None
                else None
            )
        )
        if self._draft_tree is not None and self._speculation is None:
            raise ValueError("draft_tree requires an active-batch speculation config")
        self._speculative_requests = 0
        self._speculative_hits = 0
        self._draft_tokens_proposed = 0
        self._draft_tokens_accepted = 0
        self._speculative_verification_forward_token_slots = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        adapter_name_or_path: str | None = None,
        device: str = "cuda",
        dtype: str = "float32",
        cache_dir: str | None = None,
        revision: str | None = None,
        tokenizer_name_or_path: str | None = None,
        tokenizer_revision: str | None = None,
        adapter_revision: str | None = None,
        device_map: str | dict[str, Any] | None = None,
        attn_implementation: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        max_score_batch_size: int = 8,
        score_chunk_size: int = 256,
        draft_tree: RolloutTokenTree | None = None,
        speculation: ActiveBatchSpeculationConfig | None = None,
    ) -> "TransformersBackend":
        torch_module = _require_torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise ModuleNotFoundError(
                "TransformersBackend.from_pretrained requires transformers"
            ) from error
        try:
            torch_dtype = "auto" if dtype == "auto" else getattr(torch_module, dtype)
        except AttributeError as error:
            raise ValueError(f"unknown torch dtype {dtype!r}") from error
        if dtype not in {"auto", "float32", "float16", "bfloat16", "float64"}:
            raise ValueError(f"unsupported floating-point dtype {dtype!r}")
        if torch_dtype != torch_module.float32:
            warnings.warn(
                "Reduced-precision logits can depend noticeably on batch shape. "
                "Use dtype='float32' when importance weights must match later rescoring.",
                RuntimeWarning,
                stacklevel=2,
            )
        common = {
            "cache_dir": cache_dir, "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        extra_model, extra_tokenizer = dict(model_kwargs or {}), dict(tokenizer_kwargs or {})
        reserved = {"dtype", "torch_dtype", "cache_dir", "revision", "local_files_only",
                    "trust_remote_code", "device_map", "attn_implementation", "token"}
        collision = reserved.intersection(extra_model.keys() | extra_tokenizer.keys())
        if collision:
            raise ValueError("loading kwargs duplicate explicit options or contain credentials: " + ", ".join(sorted(collision)))
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path,
            revision=tokenizer_revision if tokenizer_revision is not None else (revision if tokenizer_name_or_path is None else None),
            **common, **extra_tokenizer,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_load_kwargs = {**common, "revision": revision, **extra_model}
        if device_map is not None:
            model_load_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            model_load_kwargs["attn_implementation"] = attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch_dtype,
                **model_load_kwargs,
            )
        except TypeError as error:
            if "unexpected keyword argument 'dtype'" not in str(error):
                raise
            # Transformers 4.x names this argument ``torch_dtype``; 5.x uses
            # ``dtype``.  Supporting both keeps the AR backend usable from the
            # dLLM-compatible test environment without changing model values.
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                **model_load_kwargs,
            )
        if adapter_name_or_path is not None:
            try:
                from peft import PeftModel
            except ImportError as error:  # pragma: no cover - optional training extra
                raise ModuleNotFoundError(
                    "Loading a GRPO adapter requires the project's training extra"
                ) from error
            model = PeftModel.from_pretrained(
                model,
                adapter_name_or_path,
                local_files_only=local_files_only,
                revision=adapter_revision,
                cache_dir=cache_dir,
            )
        if device_map is None and not getattr(model, "is_quantized", False):
            model.to(torch_module.device(device))
        input_device = device
        if device_map is not None:
            input_device = model.get_input_embeddings().weight.device
            if str(input_device) == "meta":
                raise ValueError("input embeddings require a concrete device for manual decoding")
        return cls(
            model,
            tokenizer,
            model_id=model_identity(
                model_name_or_path, adapter_name_or_path, revision=revision,
                adapter_revision=adapter_revision, tokenizer=tokenizer_name_or_path,
                tokenizer_revision=tokenizer_revision,
            ),
            device=input_device,
            max_score_batch_size=max_score_batch_size,
            score_chunk_size=score_chunk_size,
            draft_tree=draft_tree,
            speculation=speculation,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def close(self) -> None:
        """Release model and draft references after all dispatchers have stopped."""
        with self._model_lock:
            self.model = None
            self._draft_tree = None

    @property
    def parameter_count(self) -> int:
        """Number of model parameters used by the dominant-matmul FLOP estimate."""

        return self._parameter_count

    def _dense_forward_flops(self, token_slots: int) -> int:
        """Return the conventional ``2 * parameters * tokens`` estimate.

        This deliberately estimates the dominant dense matrix multiplications.
        Attention's sequence-length term, elementwise operations, sampling, and
        host work are reported as exclusions rather than hidden in a wall-clock
        proxy.
        """

        return dense_forward_flops(self._parameter_count, token_slots)

    def _model_prefix(self, prefix: TokenSequence) -> TokenSequence:
        if prefix:
            return prefix
        if self.bos_token_id is None:
            raise ValueError("an empty prefix requires a tokenizer bos_token_id")
        return (self.bos_token_id,)

    @staticmethod
    def _position_ids(attention_mask):
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    @staticmethod
    def _policy_log_probs(logits, sampling: SamplingConfig | None):
        torch_module = _require_torch()
        policy = sampling or SamplingConfig()
        transformed = logits.to(dtype=torch_module.float32) / policy.temperature
        vocabulary_size = transformed.shape[-1]
        if policy.top_k is not None and policy.top_k < vocabulary_size:
            threshold = torch_module.topk(transformed, policy.top_k, dim=-1).values[
                ..., -1, None
            ]
            transformed = transformed.masked_fill(
                transformed < threshold, float("-inf")
            )
        if policy.top_p < 1:
            sorted_logits, sorted_indices = torch_module.sort(
                transformed, descending=True, dim=-1
            )
            sorted_probabilities = torch_module.softmax(sorted_logits, dim=-1)
            remove = sorted_probabilities.cumsum(dim=-1) > policy.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove_original = torch_module.zeros_like(remove).scatter(
                -1, sorted_indices, remove
            )
            transformed = transformed.masked_fill(remove_original, float("-inf"))
        return torch_module.log_softmax(transformed, dim=-1)

    def _padded_inputs(self, prefixes: Sequence[TokenSequence]):
        torch_module = _require_torch()
        maximum = max(len(prefix) for prefix in prefixes)
        input_ids = torch_module.full(
            (len(prefixes), maximum),
            self.pad_token_id,
            dtype=torch_module.long,
            device=self.device,
        )
        attention_mask = torch_module.zeros_like(input_ids)
        for index, prefix in enumerate(prefixes):
            values = torch_module.tensor(
                prefix, dtype=torch_module.long, device=self.device
            )
            input_ids[index, maximum - len(prefix) :] = values
            attention_mask[index, maximum - len(prefix) :] = 1
        return input_ids, attention_mask

    @staticmethod
    def _repeat_cache(cache, repeats: int):
        if cache is None:
            return None
        repeat_method = getattr(cache, "batch_repeat_interleave", None)
        if callable(repeat_method):
            repeat_method(repeats)
            return cache
        if isinstance(cache, (tuple, list)):
            repeated_layers = []
            for layer in cache:
                if not isinstance(layer, (tuple, list)):
                    return None
                repeated_layers.append(
                    tuple(value.repeat_interleave(repeats, dim=0) for value in layer)
                )
            return tuple(repeated_layers)
        return None

    @staticmethod
    def _crop_cache(cache, maximum_length: int):
        if cache is None:
            return None
        crop = getattr(cache, "crop", None)
        if callable(crop):
            parameters = inspect.signature(crop).parameters
            if "tokens_to_remove" in parameters:
                get_seq_length = getattr(cache, "get_seq_length", None)
                if not callable(get_seq_length):
                    return None
                tokens_to_remove = max(
                    int(get_seq_length()) - int(maximum_length),
                    0,
                )
                if tokens_to_remove:
                    crop(-tokens_to_remove)
            else:
                crop(int(maximum_length))
            return cache
        if isinstance(cache, (tuple, list)):
            cropped_layers = []
            for layer in cache:
                if not isinstance(layer, (tuple, list)):
                    return None
                cropped_layers.append(
                    tuple(value[..., :maximum_length, :] for value in layer)
                )
            return tuple(cropped_layers)
        return None

    @staticmethod
    def _sample_from_log_probs(log_probs, uniform: float):
        """Inverse-CDF sample one row with a float64 cumulative sum."""

        torch_module = _require_torch()
        probabilities = log_probs.exp()
        cumulative = probabilities.to(dtype=torch_module.float64).cumsum(dim=-1)
        value = torch_module.tensor(
            float(uniform), dtype=torch_module.float64, device=log_probs.device
        )
        token = (cumulative < value).sum(dim=-1)
        token = token.clamp_max(probabilities.shape[-1] - 1)
        selected = log_probs.gather(-1, token[..., None]).squeeze(-1)
        return token, selected

    @staticmethod
    def _sample_from_probabilities(probabilities, uniform: float):
        """Inverse-CDF sample from an already normalized probability row."""

        torch_module = _require_torch()
        cumulative = probabilities.to(dtype=torch_module.float64).cumsum(dim=-1)
        value = torch_module.tensor(
            float(uniform), dtype=torch_module.float64, device=probabilities.device
        )
        token = (cumulative < value).sum(dim=-1)
        return token.clamp_max(probabilities.shape[-1] - 1)

    def _sequence_sample(
        self,
        request: GenerationRequest,
        tokens: Sequence[int],
        token_logprobs: Sequence[float],
        reference_logprobs: Sequence[float],
        finish_reason: str,
    ) -> SequenceSample:
        reference_sampling = SamplingConfig(eos_token_id=request.sampling.eos_token_id)
        return SequenceSample(
            prefix=request.prefix,
            token_ids=tuple(int(token) for token in tokens),
            token_logprobs=tuple(float(value) for value in token_logprobs),
            policy_id=request.sampling.policy_id,
            model_id=self.model_id,
            request_id=request.request_id,
            finish_reason=finish_reason,
            reference_token_logprobs=tuple(
                float(value) for value in reference_logprobs
            ),
            reference_policy_id=reference_sampling.policy_id,
        )

    def _verify_draft(
        self,
        request: GenerationRequest,
        proposal: DraftProposal,
        uniforms: np.ndarray,
        acceptance_uniforms: np.ndarray | None = None,
    ) -> tuple[list[int], list[float], list[float], str, int, Any | None, int]:
        """Verify a deterministic or stochastic draft against the exact policy.

        Deterministic drafts use target-sample equality.  Stochastic drafts use
        standard speculative sampling: accept with ``min(1, p(x) / q(x))`` and,
        on rejection, sample from normalized ``(p-q)_+``.  Both paths preserve
        the requested target policy exactly.
        """

        torch_module = _require_torch()
        draft = proposal.token_ids[: request.max_new_tokens]
        if not draft:
            return [], [], [], "length", 0, None, 0
        stochastic = proposal.stochastic
        if stochastic:
            if len(proposal.token_distributions) < len(draft):
                raise RuntimeError("stochastic draft omitted proposal distributions")
            if acceptance_uniforms is None or len(acceptance_uniforms) < len(draft):
                raise RuntimeError("stochastic draft omitted acceptance random numbers")
        prefix = self._model_prefix(request.prefix)
        sequence = prefix + draft
        input_ids, attention_mask = self._padded_inputs([sequence])
        logits_to_keep = len(draft) + 1
        with self._model_lock, torch_module.inference_mode():
            outputs = self._prefill_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=self._position_ids(attention_mask),
                use_cache=True,
                return_dict=True,
                logits_to_keep=logits_to_keep,
            )
        first_predictor = len(prefix) - 1
        logits_start = len(sequence) - outputs.logits.shape[1]
        predictor_rows = (
            torch_module.arange(
                first_predictor,
                first_predictor + len(draft) + 1,
                device=self.device,
            )
            - logits_start
        )
        if (
            int(predictor_rows.min()) < 0
            or int(predictor_rows.max()) >= outputs.logits.shape[1]
        ):
            raise RuntimeError(
                "draft verification omitted a required predictor position"
            )
        token_logits = outputs.logits[0].index_select(0, predictor_rows)
        policy_log_probs = self._policy_log_probs(token_logits, request.sampling)
        reference_sampling = SamplingConfig(eos_token_id=request.sampling.eos_token_id)
        reference_log_probs = (
            policy_log_probs
            if request.sampling == reference_sampling
            else self._policy_log_probs(token_logits, reference_sampling)
        )

        tokens: list[int] = []
        token_logprobs: list[float] = []
        reference_values: list[float] = []
        accepted = 0
        consumed = 0
        finish_reason = "length"
        for position, draft_token in enumerate(draft):
            matches_draft = False
            if stochastic:
                distribution = proposal.token_distributions[position]
                proposal_probability = dict(distribution).get(int(draft_token), 0.0)
                if proposal_probability <= 0:
                    raise RuntimeError("drafted token has zero proposal probability")
                target_probability = float(
                    policy_log_probs[position, int(draft_token)].exp().detach().cpu()
                )
                threshold = min(1.0, target_probability / proposal_probability)
                if float(acceptance_uniforms[position]) < threshold:
                    sampled_token = int(draft_token)
                    selected = policy_log_probs[position, sampled_token]
                    matches_draft = True
                else:
                    residual = policy_log_probs[position].exp().clone()
                    for token, probability in distribution:
                        residual[int(token)] -= float(probability)
                    residual.clamp_min_(0.0)
                    total = residual.sum()
                    if (
                        not bool(torch_module.isfinite(total))
                        or float(total.detach().cpu()) <= 0
                    ):
                        raise RuntimeError(
                            "rejected stochastic draft has no residual mass"
                        )
                    residual /= total
                    sampled = self._sample_from_probabilities(
                        residual, uniforms[consumed]
                    )
                    sampled_token = int(sampled.detach().cpu())
                    selected = policy_log_probs[position, sampled_token]
            else:
                sampled, selected = self._sample_from_log_probs(
                    policy_log_probs[position], uniforms[consumed]
                )
                sampled_token = int(sampled.detach().cpu())
                matches_draft = sampled_token == int(draft_token)
            reference_selected = reference_log_probs[position, sampled_token]
            tokens.append(sampled_token)
            token_logprobs.append(float(selected.detach().cpu()))
            reference_values.append(float(reference_selected.detach().cpu()))
            consumed += 1
            if matches_draft:
                accepted += 1
            if (
                request.sampling.eos_token_id is not None
                and sampled_token == request.sampling.eos_token_id
            ):
                finish_reason = "eos"
                break
            if not matches_draft:
                break
        else:
            # When every draft token is accepted, the last verification logit
            # supplies the standard speculative-decoding bonus token.
            if len(tokens) < request.max_new_tokens:
                sampled, selected = self._sample_from_log_probs(
                    policy_log_probs[len(draft)], uniforms[consumed]
                )
                sampled_token = int(sampled.detach().cpu())
                tokens.append(sampled_token)
                token_logprobs.append(float(selected.detach().cpu()))
                reference_values.append(
                    float(reference_log_probs[len(draft), sampled_token].detach().cpu())
                )
                consumed += 1
                if (
                    request.sampling.eos_token_id is not None
                    and sampled_token == request.sampling.eos_token_id
                ):
                    finish_reason = "eos"

        self._draft_tree.record_verification(proposed=len(draft), accepted=accepted)
        reusable_cache = None
        cached_continuation_tokens = 0
        if finish_reason != "eos" and consumed < request.max_new_tokens:
            # The cache contains the complete hypothetical draft path.  Retain
            # only the matched draft prefix; the mismatch/bonus token has been
            # sampled by the base model but has not yet been inserted.
            cached_continuation_tokens = accepted
            reusable_cache = self._crop_cache(
                getattr(outputs, "past_key_values", None),
                len(prefix) + cached_continuation_tokens,
            )
        slots = int(input_ids.numel())
        with self._statistics_lock:
            self._speculative_hits += 1
            self._draft_tokens_proposed += len(draft)
            self._draft_tokens_accepted += accepted
            self._speculative_verification_forward_token_slots += slots
            self._prefill_tokens += len(prefix)
            self._generation_forward_token_slots += slots
            self._estimated_dense_forward_flops += self._dense_forward_flops(slots)
        return (
            tokens,
            token_logprobs,
            reference_values,
            finish_reason,
            consumed,
            reusable_cache,
            cached_continuation_tokens,
        )

    def _continue_verified_cache(
        self,
        request: GenerationRequest,
        *,
        cache: Any,
        cached_continuation_tokens: int,
        tokens: list[int],
        token_logprobs: list[float],
        reference_logprobs: list[float],
        uniforms: np.ndarray,
        consumed: int,
    ) -> str:
        """Continue after a rejected draft without recomputing its prefix."""

        torch_module = _require_torch()
        if not tokens or consumed >= request.max_new_tokens:
            return "length"
        prefix_length = len(self._model_prefix(request.prefix))
        cached_length = prefix_length + cached_continuation_tokens
        attention_mask = torch_module.ones(
            (1, cached_length + 1), dtype=torch_module.long, device=self.device
        )
        current = torch_module.tensor(
            [[tokens[-1]]], dtype=torch_module.long, device=self.device
        )
        generation_slots = 0
        reference_sampling = SamplingConfig(eos_token_id=request.sampling.eos_token_id)
        finish_reason = "length"
        with self._model_lock, torch_module.inference_mode():
            outputs = self.model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=torch_module.tensor(
                    [[cached_length]], dtype=torch_module.long, device=self.device
                ),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                **({"logits_to_keep": 1} if self._supports_logits_to_keep else {}),
            )
            generation_slots += 1
            logits = outputs.logits[:, -1, :]
            cache = getattr(outputs, "past_key_values", None)
            while consumed < request.max_new_tokens:
                policy = self._policy_log_probs(logits[0], request.sampling)
                reference = (
                    policy
                    if request.sampling == reference_sampling
                    else self._policy_log_probs(logits[0], reference_sampling)
                )
                sampled, selected = self._sample_from_log_probs(
                    policy, uniforms[consumed]
                )
                token = int(sampled.detach().cpu())
                tokens.append(token)
                token_logprobs.append(float(selected.detach().cpu()))
                reference_logprobs.append(float(reference[token].detach().cpu()))
                consumed += 1
                if (
                    request.sampling.eos_token_id is not None
                    and token == request.sampling.eos_token_id
                ):
                    finish_reason = "eos"
                    break
                if consumed >= request.max_new_tokens:
                    break
                position = attention_mask.shape[1]
                attention_mask = torch_module.cat(
                    [
                        attention_mask,
                        torch_module.ones(
                            (1, 1), dtype=attention_mask.dtype, device=self.device
                        ),
                    ],
                    dim=-1,
                )
                outputs = self.model(
                    input_ids=torch_module.tensor(
                        [[token]], dtype=torch_module.long, device=self.device
                    ),
                    attention_mask=attention_mask,
                    position_ids=torch_module.tensor(
                        [[position]], dtype=torch_module.long, device=self.device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    **({"logits_to_keep": 1} if self._supports_logits_to_keep else {}),
                )
                generation_slots += 1
                logits = outputs.logits[:, -1, :]
                cache = getattr(outputs, "past_key_values", None)
        with self._statistics_lock:
            self._generation_forward_token_slots += generation_slots
            self._estimated_dense_forward_flops += self._dense_forward_flops(
                generation_slots
            )
        return finish_reason

    def _sample_same_policy(
        self,
        indexed_requests: Sequence[tuple[int, GenerationRequest]],
        *,
        uniform_streams: Mapping[int, np.ndarray] | None = None,
        uniform_offsets: Mapping[int, int] | None = None,
        on_complete: SampleCompletionCallback | None = None,
    ) -> list[tuple[int, SequenceSample]]:
        torch_module = _require_torch()
        requests = [request for _, request in indexed_requests]
        sampling = requests[0].sampling
        prefixes = [self._model_prefix(request.prefix) for request in requests]
        prefix_positions: OrderedDict[TokenSequence, list[int]] = OrderedDict()
        for position, prefix in enumerate(prefixes):
            prefix_positions.setdefault(prefix, []).append(position)
        repeat_counts = {len(positions) for positions in prefix_positions.values()}
        reusable_prefixes: list[TokenSequence] | None = None
        prefix_repeat_count = 1
        if len(prefix_positions) < len(prefixes) and len(repeat_counts) == 1:
            prefix_repeat_count = repeat_counts.pop()
            if prefix_repeat_count > 1:
                row_order = [
                    position
                    for positions in prefix_positions.values()
                    for position in positions
                ]
                indexed_requests = [
                    indexed_requests[position] for position in row_order
                ]
                requests = [request for _, request in indexed_requests]
                prefixes = [self._model_prefix(request.prefix) for request in requests]
                reusable_prefixes = list(prefix_positions)
        uniforms = [
            (
                uniform_streams[original_index]
                if uniform_streams is not None and original_index in uniform_streams
                else np.asarray(request.uniforms, dtype=np.float64)
                if request.uniforms is not None
                else np.zeros(request.max_new_tokens, dtype=np.float64)
                if request.arithmetic_uniform is not None
                else np.random.default_rng(request.seed).random(request.max_new_tokens)
            )
            for original_index, request in indexed_requests
        ]
        offsets = [
            (
                int(uniform_offsets.get(original_index, 0))
                if uniform_offsets is not None
                else 0
            )
            for original_index, _ in indexed_requests
        ]
        arithmetic_mask = torch_module.tensor(
            [request.arithmetic_uniform is not None for request in requests],
            dtype=torch_module.bool,
            device=self.device,
        )
        arithmetic_uniforms = torch_module.tensor(
            [
                0.0
                if request.arithmetic_uniform is None
                else request.arithmetic_uniform
                for request in requests
            ],
            dtype=torch_module.float64,
            device=self.device,
        )
        token_lists: list[list[int]] = [[] for _ in requests]
        logprob_lists: list[list[float]] = [[] for _ in requests]
        reference_logprob_lists: list[list[float]] = [[] for _ in requests]
        reference_sampling = SamplingConfig(eos_token_id=sampling.eos_token_id)
        active = torch_module.ones(
            len(requests), dtype=torch_module.bool, device=self.device
        )
        finish_reasons = ["length"] * len(requests)
        callback_completed: set[int] = set()
        maximum_new_tokens = max(request.max_new_tokens for request in requests)
        prefill_tokens = sum(len(prefix) for prefix in prefixes)
        shared_prefill_tokens_saved = 0
        generation_forward_token_slots = len(requests) * max(
            len(prefix) for prefix in prefixes
        )

        with self._model_lock, torch_module.inference_mode():
            cache = None
            if reusable_prefixes is not None:
                unique_input_ids, unique_attention_mask = self._padded_inputs(
                    reusable_prefixes
                )
                unique_outputs = self._prefill_model(
                    input_ids=unique_input_ids,
                    attention_mask=unique_attention_mask,
                    position_ids=self._position_ids(unique_attention_mask),
                    use_cache=True,
                    return_dict=True,
                    **({"logits_to_keep": 1} if self._supports_logits_to_keep else {}),
                )
                cache = self._repeat_cache(
                    getattr(unique_outputs, "past_key_values", None),
                    prefix_repeat_count,
                )
                if cache is not None:
                    attention_mask = unique_attention_mask.repeat_interleave(
                        prefix_repeat_count, dim=0
                    )
                    logits = unique_outputs.logits[:, -1, :].repeat_interleave(
                        prefix_repeat_count, dim=0
                    )
                    prefill_tokens = sum(len(prefix) for prefix in reusable_prefixes)
                    shared_prefill_tokens_saved = sum(
                        (prefix_repeat_count - 1) * len(prefix)
                        for prefix in reusable_prefixes
                    )
                    generation_forward_token_slots = int(unique_input_ids.numel())
            if cache is None:
                input_ids, attention_mask = self._padded_inputs(prefixes)
                outputs = self._prefill_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=self._position_ids(attention_mask),
                    use_cache=True,
                    return_dict=True,
                    **({"logits_to_keep": 1} if self._supports_logits_to_keep else {}),
                )
                logits = outputs.logits[:, -1, :]
                cache = getattr(outputs, "past_key_values", None)
            for step in range(maximum_new_tokens):
                permitted = torch_module.tensor(
                    [step < request.max_new_tokens for request in requests],
                    dtype=torch_module.bool,
                    device=self.device,
                )
                step_active = active & permitted
                if not bool(step_active.any()):
                    break
                log_probs = self._policy_log_probs(logits, sampling)
                reference_log_probs = (
                    log_probs
                    if sampling == reference_sampling
                    else self._policy_log_probs(logits, reference_sampling)
                )
                probabilities = log_probs.exp()
                random_values = torch_module.tensor(
                    [
                        uniforms[index][offsets[index] + step]
                        if offsets[index] + step < len(uniforms[index])
                        else 0.0
                        for index in range(len(requests))
                    ],
                    dtype=torch_module.float64,
                    device=self.device,
                )
                # Inverse-CDF sampling is especially sensitive to accumulated
                # roundoff over a language model's large vocabulary.  A
                # float32 CDF can move a fixed request-local uniform across a
                # token boundary when otherwise equivalent requests are
                # decoded in different batch shapes.  Accumulating the same
                # policy probabilities in float64 preserves the categorical
                # policy while making request-local seeds robust to scheduling.
                probabilities_64 = probabilities.to(dtype=torch_module.float64)
                cumulative = probabilities_64.cumsum(dim=-1)
                cumulative[:, -1] = 1.0
                sampled_tokens = (cumulative < random_values[:, None]).sum(dim=-1)
                sampled_tokens = sampled_tokens.clamp_max(probabilities.shape[-1] - 1)
                if bool(arithmetic_mask.any()):
                    ordered_probabilities, order = torch_module.sort(
                        probabilities_64,
                        dim=-1,
                        descending=True,
                        stable=True,
                    )
                    ordered_cumulative = ordered_probabilities.cumsum(dim=-1)
                    ordered_cumulative[:, -1] = 1.0
                    arithmetic_ranks = (
                        ordered_cumulative < arithmetic_uniforms[:, None]
                    ).sum(dim=-1)
                    arithmetic_ranks = arithmetic_ranks.clamp_max(
                        probabilities.shape[-1] - 1
                    )
                    arithmetic_tokens = order.gather(
                        -1, arithmetic_ranks[:, None]
                    ).squeeze(-1)
                    sampled_tokens = torch_module.where(
                        arithmetic_mask,
                        arithmetic_tokens,
                        sampled_tokens,
                    )
                    arithmetic_probabilities = ordered_probabilities.gather(
                        -1, arithmetic_ranks[:, None]
                    ).squeeze(-1)
                    padded_cumulative = torch_module.cat(
                        [
                            torch_module.zeros(
                                (len(requests), 1),
                                dtype=torch_module.float64,
                                device=self.device,
                            ),
                            ordered_cumulative,
                        ],
                        dim=-1,
                    )
                    arithmetic_lower = padded_cumulative.gather(
                        -1, arithmetic_ranks[:, None]
                    ).squeeze(-1)
                    arithmetic_active = arithmetic_mask & step_active
                    if bool((arithmetic_probabilities[arithmetic_active] <= 0).any()):
                        raise RuntimeError(
                            "arithmetic sampling selected a zero-probability token"
                        )
                    updated_arithmetic_uniforms = (
                        arithmetic_uniforms - arithmetic_lower
                    ) / arithmetic_probabilities.clamp_min(
                        torch_module.finfo(torch_module.float64).tiny
                    )
                    updated_arithmetic_uniforms = updated_arithmetic_uniforms.clamp(
                        min=0.0,
                        max=float(np.nextafter(1.0, 0.0)),
                    )
                    arithmetic_uniforms = torch_module.where(
                        arithmetic_active,
                        updated_arithmetic_uniforms,
                        arithmetic_uniforms,
                    )
                sampled_logprobs = log_probs.gather(
                    -1, sampled_tokens[:, None]
                ).squeeze(-1)
                sampled_reference_logprobs = reference_log_probs.gather(
                    -1, sampled_tokens[:, None]
                ).squeeze(-1)

                sampled_cpu = sampled_tokens.detach().cpu().tolist()
                logprobs_cpu = sampled_logprobs.detach().cpu().tolist()
                reference_logprobs_cpu = (
                    sampled_reference_logprobs.detach().cpu().tolist()
                )
                for index, is_active in enumerate(step_active.detach().cpu().tolist()):
                    if not is_active:
                        continue
                    token = int(sampled_cpu[index])
                    token_lists[index].append(token)
                    logprob_lists[index].append(float(logprobs_cpu[index]))
                    reference_logprob_lists[index].append(
                        float(reference_logprobs_cpu[index])
                    )
                    if (
                        sampling.eos_token_id is not None
                        and token == sampling.eos_token_id
                    ):
                        finish_reasons[index] = "eos"
                    if on_complete is not None and (
                        finish_reasons[index] == "eos"
                        or step + 1 >= requests[index].max_new_tokens
                    ):
                        original_index, request = indexed_requests[index]
                        on_complete(
                            original_index,
                            self._sequence_sample(
                                request,
                                token_lists[index],
                                logprob_lists[index],
                                reference_logprob_lists[index],
                                finish_reasons[index],
                            ),
                        )
                        callback_completed.add(original_index)

                eos_finished = torch_module.tensor(
                    [
                        sampling.eos_token_id is not None
                        and int(sampled_cpu[index]) == sampling.eos_token_id
                        for index in range(len(requests))
                    ],
                    dtype=torch_module.bool,
                    device=self.device,
                )
                active_after = step_active & ~eos_finished
                active_after &= torch_module.tensor(
                    [step + 1 < request.max_new_tokens for request in requests],
                    dtype=torch_module.bool,
                    device=self.device,
                )
                if not bool(active_after.any()):
                    break
                next_tokens = torch_module.where(
                    step_active,
                    sampled_tokens,
                    torch_module.full_like(sampled_tokens, self.pad_token_id),
                )
                next_positions = attention_mask.sum(dim=-1, dtype=torch_module.long)
                attention_mask = torch_module.cat(
                    [
                        attention_mask,
                        step_active.to(dtype=attention_mask.dtype)[:, None],
                    ],
                    dim=-1,
                )
                outputs = self.model(
                    input_ids=next_tokens[:, None],
                    attention_mask=attention_mask,
                    position_ids=next_positions[:, None],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    **({"logits_to_keep": 1} if self._supports_logits_to_keep else {}),
                )
                generation_forward_token_slots += len(requests)
                logits = outputs.logits[:, -1, :]
                cache = getattr(outputs, "past_key_values", None)
                active = active_after

        results: list[tuple[int, SequenceSample]] = []
        for (
            (original_index, request),
            tokens,
            token_logprobs,
            reference_token_logprobs,
            finish_reason,
        ) in zip(
            indexed_requests,
            token_lists,
            logprob_lists,
            reference_logprob_lists,
            finish_reasons,
            strict=True,
        ):
            results.append(
                (
                    original_index,
                    self._sequence_sample(
                        request,
                        tokens,
                        token_logprobs,
                        reference_token_logprobs,
                        finish_reason,
                    ),
                )
            )
            if on_complete is not None and original_index not in callback_completed:
                on_complete(original_index, results[-1][1])
        with self._statistics_lock:
            self._prefill_tokens += prefill_tokens
            self._shared_prefill_tokens_saved += shared_prefill_tokens_saved
            self._generation_forward_token_slots += generation_forward_token_slots
            self._estimated_dense_forward_flops += self._dense_forward_flops(
                generation_forward_token_slots
            )
        return results

    def _sample_with_verified_draft(
        self,
        original_index: int,
        request: GenerationRequest,
        proposal: DraftProposal,
        on_complete: SampleCompletionCallback | None,
    ) -> SequenceSample:
        uniforms = (
            np.asarray(request.uniforms, dtype=np.float64)
            if request.uniforms is not None
            else np.random.default_rng(request.seed).random(request.max_new_tokens)
        )
        acceptance_uniforms = None
        if proposal.stochastic:
            acceptance_uniforms = np.random.default_rng(
                SeedStream(request.seed).derive(
                    "stochastic-draft-verification", request.request_id
                )
            ).random(len(proposal.token_ids))
        (
            tokens,
            logprobs,
            reference_logprobs,
            finish_reason,
            consumed,
            reusable_cache,
            cached_continuation_tokens,
        ) = self._verify_draft(
            request, proposal, uniforms, acceptance_uniforms=acceptance_uniforms
        )
        if finish_reason != "eos" and consumed < request.max_new_tokens:
            if reusable_cache is not None:
                finish_reason = self._continue_verified_cache(
                    request,
                    cache=reusable_cache,
                    cached_continuation_tokens=cached_continuation_tokens,
                    tokens=tokens,
                    token_logprobs=logprobs,
                    reference_logprobs=reference_logprobs,
                    uniforms=uniforms,
                    consumed=consumed,
                )
            else:
                tail_request = GenerationRequest(
                    prefix=request.prefix + tuple(tokens),
                    max_new_tokens=request.max_new_tokens - consumed,
                    sampling=request.sampling,
                    seed=request.seed,
                    request_id=f"{request.request_id}:verified-tail",
                )
                tail = self._sample_same_policy(
                    [(original_index, tail_request)],
                    uniform_streams={original_index: uniforms},
                    uniform_offsets={original_index: consumed},
                )[0][1]
                tokens.extend(tail.token_ids)
                logprobs.extend(tail.token_logprobs)
                if tail.reference_token_logprobs is None:
                    raise RuntimeError(
                        "Transformers tail omitted reference log-probabilities"
                    )
                reference_logprobs.extend(tail.reference_token_logprobs)
                finish_reason = tail.finish_reason
        sample = self._sequence_sample(
            request,
            tokens,
            logprobs,
            reference_logprobs,
            finish_reason,
        )
        if on_complete is not None:
            on_complete(original_index, sample)
        return sample

    def _sample_batch(
        self,
        requests: Sequence[GenerationRequest],
        on_complete: SampleCompletionCallback | None,
    ) -> list[SequenceSample]:
        if not requests:
            return []
        proposals: dict[int, DraftProposal] = {}
        if self._draft_tree is not None and self._speculation is not None:
            draft_tokens = self._speculation.draft_tokens(len(requests))
            if draft_tokens > 0:
                for index, request in enumerate(requests):
                    if request.arithmetic_uniform is not None:
                        continue
                    proposal = self._draft_tree.draft(
                        self._model_prefix(request.prefix),
                        min(draft_tokens, request.max_new_tokens),
                        stochastic=self._speculation.stochastic_tree,
                        seed=(
                            SeedStream(request.seed).derive(
                                "stochastic-draft", request.request_id
                            )
                            if self._speculation.stochastic_tree
                            else None
                        ),
                    )
                    if proposal.token_ids:
                        proposals[index] = proposal
            with self._statistics_lock:
                self._speculative_requests += len(requests)

        grouped: OrderedDict[SamplingConfig, list[tuple[int, GenerationRequest]]] = (
            OrderedDict()
        )
        for index, request in enumerate(requests):
            if index not in proposals:
                grouped.setdefault(request.sampling, []).append((index, request))
        indexed_outputs: list[tuple[int, SequenceSample]] = []
        for group in grouped.values():
            indexed_outputs.extend(
                self._sample_same_policy(group, on_complete=on_complete)
            )
        for index, proposal in proposals.items():
            indexed_outputs.append(
                (
                    index,
                    self._sample_with_verified_draft(
                        index, requests[index], proposal, on_complete
                    ),
                )
            )
        indexed_outputs.sort(key=lambda item: item[0])
        outputs = [sample for _, sample in indexed_outputs]
        if self._draft_tree is not None:
            self._draft_tree.observe_samples(outputs)
        with self._statistics_lock:
            self._sample_calls += 1
            self._sampled_sequences += len(outputs)
            self._generated_tokens += sum(len(output.token_ids) for output in outputs)
        return outputs

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]:
        return self._sample_batch(requests, None)

    def sample_batch_with_callback(
        self,
        requests: Sequence[GenerationRequest],
        on_complete: SampleCompletionCallback,
    ) -> list[SequenceSample]:
        """Invoke ``on_complete`` as each row reaches EOS or its token limit."""

        return self._sample_batch(requests, on_complete)

    def observe_draft_samples(self, samples: Iterable[SequenceSample]) -> None:
        """Add arbitrary historical/off-policy samples as draft-only material."""

        if self._draft_tree is not None:
            self._draft_tree.observe_samples(samples)

    def observe_draft_sequences(self, sequences: Iterable[TokenSequence]) -> None:
        if self._draft_tree is not None:
            for sequence in sequences:
                self._draft_tree.observe(sequence)

    def draft_cache_snapshot(self) -> RolloutTokenTreeSnapshot | None:
        return None if self._draft_tree is None else self._draft_tree.snapshot()

    def configure_cache_growth(self, block_tokens: int = 0) -> None:
        """Select lossless, bounded-slack KV storage for subsequent decoding calls."""
        if not isinstance(block_tokens, int) or block_tokens < 0:
            raise ValueError("cache growth block must be a non-negative integer")
        if block_tokens:
            from inference_scaling.arllm.backends.growing_cache import validate_cache_growth_support
            validate_cache_growth_support()
        with self._model_lock:
            self.cache_growth_tokens = block_tokens

    def _prefill_model(self, *, input_ids, attention_mask, position_ids, use_cache=True,
                       return_dict=True, logits_to_keep=1):
        output = prefill_causal_model(
            self.model, input_ids, attention_mask, position_ids,
            chunk_size=self.score_chunk_size, logits_to_keep=logits_to_keep,
            supports_logits_to_keep=self._supports_logits_to_keep,
        )
        if self.cache_growth_tokens:
            from inference_scaling.arllm.backends.growing_cache import block_allocated_cache
            output.past_key_values = block_allocated_cache(output.past_key_values, self.cache_growth_tokens)
        return output

    def _stream_score(
        self, request: ScoreRequest, continuation: TokenSequence, *,
        statistics: bool = False, confidence_top_k: int | None = None,
    ) -> tuple[SequenceScoreStatistics, int]:
        torch_module = _require_torch()
        logs: list[float] = []
        confidence: list[float] = []
        entropies: list[float] = []
        certainties: list[float] = []
        forwarded = 0
        effective_top_k = None

        def count(amount: int) -> None:
            nonlocal forwarded
            forwarded += amount

        with self._model_lock, torch_module.inference_mode():
            chunks = iter_causal_logits(
                self.model, self._model_prefix(request.prefix), continuation,
                device=self.device, chunk_size=self.score_chunk_size,
                supports_logits_to_keep=self._supports_logits_to_keep, on_forward=count,
            )
            try:
                for offset, logits in chunks:
                    log_probs = self._policy_log_probs(logits, request.sampling)
                    targets = torch_module.tensor(
                        continuation[offset:offset + logits.shape[0]],
                        dtype=torch_module.long, device=log_probs.device,
                    )
                    selected = log_probs.gather(-1, targets[:, None]).squeeze(-1)
                    logs.extend(float(v) for v in selected.cpu().tolist())
                    if statistics:
                        entropy = torch_module.special.xlogy(log_probs.exp(), log_probs.exp()).sum(-1)
                        certainty = -(math.log(log_probs.shape[-1]) + log_probs).mean(-1)
                        entropies.extend(float(v) for v in entropy.cpu().tolist())
                        certainties.extend(float(v) for v in certainty.cpu().tolist())
                        if confidence_top_k is not None:
                            effective_top_k = min(confidence_top_k, log_probs.shape[-1])
                            values = -log_probs.topk(effective_top_k, dim=-1).values.mean(-1)
                            confidence.extend(float(v) for v in values.cpu().tolist())
                            del values
                        del entropy, certainty
                    del logits, log_probs, selected, targets
            finally:
                chunks.close()
        if len(logs) != len(continuation):
            raise RuntimeError("incomplete chunked sequence score")
        return SequenceScoreStatistics(
            token_logprobs=tuple(logs), mean_logprob=sum(logs) / len(logs),
            mean_negative_entropy=math.fsum(entropies) / len(logs) if statistics else 0.0,
            mean_self_certainty=math.fsum(certainties) / len(logs) if statistics else 0.0,
            token_topk_confidences=tuple(confidence), confidence_top_k=effective_top_k,
        ), forwarded

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        torch_module = _require_torch()
        flattened: list[tuple[ScoreRequest, TokenSequence]] = [
            (request, continuation)
            for request in requests
            for continuation in request.continuations
        ]
        results: list[tuple[float, ...]] = [()] * len(flattened)
        nonempty: list[tuple[int, ScoreRequest, TokenSequence, TokenSequence]] = []
        score_forward_token_slots = 0
        for index, (request, continuation) in enumerate(flattened):
            if continuation and len(self._model_prefix(request.prefix)) + len(continuation) > self.score_chunk_size:
                scored, forwarded = self._stream_score(request, continuation)
                results[index] = scored.token_logprobs
                score_forward_token_slots += forwarded
            elif continuation:
                nonempty.append(
                    (index, request, continuation, self._model_prefix(request.prefix))
                )
        if nonempty:
            for start in range(0, len(nonempty), self.max_score_batch_size):
                chunk = nonempty[start : start + self.max_score_batch_size]
                sequences = [
                    prefix + continuation for _, _, continuation, prefix in chunk
                ]
                input_ids, attention_mask = self._padded_inputs(sequences)
                score_forward_token_slots += int(input_ids.numel())
                logits_to_keep = (
                    max(len(continuation) for _, _, continuation, _ in chunk) + 1
                )
                with self._model_lock, torch_module.inference_mode():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=self._position_ids(attention_mask),
                        use_cache=False,
                        return_dict=True,
                        **(
                            {"logits_to_keep": logits_to_keep}
                            if self._supports_logits_to_keep
                            else {}
                        ),
                    )
                padded_length = input_ids.shape[1]
                logits_start = padded_length - outputs.logits.shape[1]
                for row, (flat_index, request, continuation, prefix) in enumerate(
                    chunk
                ):
                    padding = padded_length - len(prefix) - len(continuation)
                    predictor_positions = (
                        torch_module.arange(
                            padding + len(prefix) - 1,
                            padding + len(prefix) + len(continuation) - 1,
                            device=outputs.logits.device,
                        )
                        - logits_start
                    )
                    if int(predictor_positions.min()) < 0:
                        raise RuntimeError(
                            "logits_to_keep omitted a required score position"
                        )
                    token_logits = outputs.logits[row].index_select(
                        0, predictor_positions
                    )
                    log_probs = self._policy_log_probs(token_logits, request.sampling)
                    targets = torch_module.tensor(
                        continuation, dtype=torch_module.long, device=log_probs.device
                    )
                    selected = log_probs.gather(-1, targets[:, None]).squeeze(-1)
                    results[flat_index] = tuple(
                        float(value) for value in selected.cpu().tolist()
                    )
        with self._statistics_lock:
            self._score_calls += 1
            self._scored_tokens += sum(
                len(continuation) for _, continuation in flattened
            )
            self._score_forward_token_slots += score_forward_token_slots
            self._estimated_dense_forward_flops += self._dense_forward_flops(
                score_forward_token_slots
            )
        return results

    def score_statistics_batch(
        self,
        requests: Sequence[ScoreRequest],
        *,
        confidence_top_k: int | None = None,
    ) -> list[SequenceScoreStatistics]:
        """Score continuations and return model-distribution statistics.

        Entropy and self-certainty require a finite log-probability for every
        vocabulary item, so this diagnostic deliberately accepts only a
        full-support temperature policy.  The forward passes are included in
        the same token-slot and FLOP counters as ordinary sequence scoring.
        """

        torch_module = _require_torch()
        if confidence_top_k is not None and confidence_top_k <= 0:
            raise ValueError("confidence_top_k must be positive")
        flattened: list[tuple[ScoreRequest, TokenSequence]] = [
            (request, continuation)
            for request in requests
            for continuation in request.continuations
        ]
        if any(not continuation for _, continuation in flattened):
            raise ValueError("confidence rewards require nonempty continuations")
        for request, _ in flattened:
            policy = request.sampling or SamplingConfig()
            if policy.top_p < 1 or policy.top_k is not None:
                raise ValueError(
                    "entropy and self-certainty require a full-support policy"
                )

        results: list[SequenceScoreStatistics | None] = [None] * len(flattened)
        score_forward_token_slots = 0
        indexed = [
            (index, request, continuation, self._model_prefix(request.prefix))
            for index, (request, continuation) in enumerate(flattened)
        ]
        short_indexed = []
        for item in indexed:
            index, request, continuation, prefix = item
            if len(prefix) + len(continuation) > self.score_chunk_size:
                results[index], forwarded = self._stream_score(
                    request, continuation, statistics=True, confidence_top_k=confidence_top_k,
                )
                score_forward_token_slots += forwarded
            else:
                short_indexed.append(item)
        indexed = short_indexed
        for start in range(0, len(indexed), self.max_score_batch_size):
            chunk = indexed[start : start + self.max_score_batch_size]
            sequences = [prefix + continuation for _, _, continuation, prefix in chunk]
            input_ids, attention_mask = self._padded_inputs(sequences)
            score_forward_token_slots += int(input_ids.numel())
            logits_to_keep = (
                max(len(continuation) for _, _, continuation, _ in chunk) + 1
            )
            with self._model_lock, torch_module.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=self._position_ids(attention_mask),
                    use_cache=False,
                    return_dict=True,
                    **(
                        {"logits_to_keep": logits_to_keep}
                        if self._supports_logits_to_keep
                        else {}
                    ),
                )
            padded_length = input_ids.shape[1]
            logits_start = padded_length - outputs.logits.shape[1]
            for row, (flat_index, request, continuation, prefix) in enumerate(chunk):
                padding = padded_length - len(prefix) - len(continuation)
                predictor_positions = (
                    torch_module.arange(
                        padding + len(prefix) - 1,
                        padding + len(prefix) + len(continuation) - 1,
                        device=outputs.logits.device,
                    )
                    - logits_start
                )
                if int(predictor_positions.min()) < 0:
                    raise RuntimeError(
                        "logits_to_keep omitted a required score position"
                    )
                token_logits = outputs.logits[row].index_select(0, predictor_positions)
                log_probs = self._policy_log_probs(token_logits, request.sampling)
                probabilities = log_probs.exp()
                targets = torch_module.tensor(
                    continuation, dtype=torch_module.long, device=log_probs.device
                )
                selected = log_probs.gather(-1, targets[:, None]).squeeze(-1)
                negative_entropy = (probabilities * log_probs).sum(dim=-1)
                vocabulary_size = log_probs.shape[-1]
                self_certainty = -(math.log(vocabulary_size) + log_probs).mean(dim=-1)
                if confidence_top_k is None:
                    token_topk_confidences: tuple[float, ...] = ()
                    effective_top_k = None
                else:
                    effective_top_k = min(confidence_top_k, vocabulary_size)
                    top_logprobs = torch_module.topk(
                        log_probs,
                        effective_top_k,
                        dim=-1,
                    ).values
                    topk_confidence = -top_logprobs.mean(dim=-1)
                    token_topk_confidences = tuple(
                        float(value) for value in topk_confidence.cpu().tolist()
                    )
                token_logprobs = tuple(
                    float(value) for value in selected.cpu().tolist()
                )
                results[flat_index] = SequenceScoreStatistics(
                    token_logprobs=token_logprobs,
                    mean_logprob=float(selected.mean().cpu()),
                    mean_negative_entropy=float(negative_entropy.mean().cpu()),
                    mean_self_certainty=float(self_certainty.mean().cpu()),
                    token_topk_confidences=token_topk_confidences,
                    confidence_top_k=effective_top_k,
                )

        with self._statistics_lock:
            self._score_calls += 1
            self._scored_tokens += sum(
                len(continuation) for _, continuation in flattened
            )
            self._score_forward_token_slots += score_forward_token_slots
            self._estimated_dense_forward_flops += self._dense_forward_flops(
                score_forward_token_slots
            )
        if any(result is None for result in results):
            raise RuntimeError("backend returned an incomplete confidence-score batch")
        return [result for result in results if result is not None]

    def snapshot(self) -> TransformersBackendSnapshot:
        with self._statistics_lock:
            return TransformersBackendSnapshot(
                sample_calls=self._sample_calls,
                score_calls=self._score_calls,
                sampled_sequences=self._sampled_sequences,
                generated_tokens=self._generated_tokens,
                prefill_tokens=self._prefill_tokens,
                shared_prefill_tokens_saved=self._shared_prefill_tokens_saved,
                scored_tokens=self._scored_tokens,
                generation_forward_token_slots=self._generation_forward_token_slots,
                score_forward_token_slots=self._score_forward_token_slots,
                estimated_dense_forward_flops=self._estimated_dense_forward_flops,
                speculative_requests=self._speculative_requests,
                speculative_hits=self._speculative_hits,
                draft_tokens_proposed=self._draft_tokens_proposed,
                draft_tokens_accepted=self._draft_tokens_accepted,
                speculative_verification_forward_token_slots=(
                    self._speculative_verification_forward_token_slots
                ),
            )

    def encode(self, text: str, *, add_special_tokens: bool = True) -> TokenSequence:
        return tuple(
            int(token)
            for token in self.tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            )
        )

    def decode(self, tokens: TokenSequence, *, skip_special_tokens: bool = True) -> str:
        return str(
            self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        )

    def direct_generate(
        self,
        prefix: TokenSequence,
        *,
        max_new_tokens: int,
        num_beams: int = 1,
    ) -> TokenSequence:
        """Greedy/beam baseline using Transformers' native generation path."""

        if max_new_tokens <= 0 or num_beams <= 0:
            raise ValueError("generation length and beam count must be positive")
        torch_module = _require_torch()
        input_ids = torch_module.tensor(
            [prefix], dtype=torch_module.long, device=self.device
        )
        attention_mask = torch_module.ones_like(input_ids)
        with self._model_lock, torch_module.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return tuple(int(token) for token in output[0, input_ids.shape[1] :].tolist())
