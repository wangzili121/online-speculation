"""Configuration-driven construction and cleanup of experiment backends."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inference_scaling.arllm.acceleration import (
    ActiveBatchSpeculationConfig,
    SpeculationTier,
)
from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.backends.vllm_backend import AsyncVLLMBackend, VLLMBackend
from inference_scaling.shared.model_loading import (
    model_loading_options, model_role, resolve_checkpoint_path,
)

BACKEND_CHOICES = ("transformers", "vllm", "vllm-sync")

_VLLM_SETTINGS = {
    "asynchronous",
    "data_parallel_size",
    "download_dir",
    "dtype",
    "enable_prefix_caching",
    "enforce_eager",
    "engine_kwargs",
    "exact_scoring_backend",
    "exact_scoring_device",
    "exact_scoring_dtype",
    "gpu_memory_utilization",
    "max_lora_rank",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "mh_fused_logprobs",
    "parameter_count",
    "quantization",
    "revision",
    "seed",
    "tensor_parallel_size",
    "trust_remote_code",
}
_MODEL_ROLES = ("base", "proposal", "rl")
_EXPLICIT_ENGINE_SETTINGS = {
    "data_parallel_size",
    "download_dir",
    "dtype",
    "enable_lora",
    "enable_prefix_caching",
    "enforce_eager",
    "generation_config",
    "gpu_memory_utilization",
    "logprobs_mode",
    "max_lora_rank",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "model",
    "tokenizer",
    "tokenizer_revision",
    "quantization",
    "revision",
    "seed",
    "tensor_parallel_size",
    "trust_remote_code",
    "worker_cls",
    "async_scheduling",
}


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return dict(value)


def _speculation_from_config(
    config: Mapping[str, Any],
) -> tuple[ActiveBatchSpeculationConfig | None, bool]:
    acceleration = _mapping(config.get("acceleration"), name="acceleration")
    table = _mapping(acceleration.get("speculation"), name="acceleration.speculation")
    if not table or not bool(table.pop("enabled", False)):
        return None, False
    dynamic_vllm = bool(table.pop("dynamic_vllm", False))
    raw_tiers = table.pop("tiers", None)
    tiers = None
    if raw_tiers is not None:
        if not isinstance(raw_tiers, (list, tuple)):
            raise TypeError("acceleration.speculation.tiers must be an array")
        try:
            tiers = tuple(
                SpeculationTier(int(item[0]), int(item[1])) for item in raw_tiers
            )
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(
                "each speculation tier must be [maximum_active_batch, draft_tokens]"
            ) from error
    aliases = {
        "min_context_tokens": "min_context_tokens",
        "min_token_probability": "min_token_probability",
        "tree_max_context_tokens": "tree_max_context_tokens",
        "tree_max_contexts": "tree_max_contexts",
        "vllm_max_cached_requests": "vllm_max_cached_requests",
        "stochastic_tree": "stochastic_tree",
    }
    unknown = sorted(set(table) - set(aliases))
    if unknown:
        raise ValueError("unknown speculation settings: " + ", ".join(unknown))
    kwargs = {aliases[name]: value for name, value in table.items()}
    if tiers is not None:
        kwargs["tiers"] = tiers
    return ActiveBatchSpeculationConfig(**kwargs), dynamic_vllm


def _infer_role(
    path: str,
    config: Mapping[str, Any],
    *,
    adapter_base: str | None,
) -> str:
    return model_role(path, config, adapter_base=adapter_base)


def _vllm_settings(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    table = _mapping(config.get("vllm"), name="vllm")
    settings = {
        key: value
        for key, value in table.items()
        if key not in _MODEL_ROLES
    }
    role_settings = _mapping(table.get(role), name=f"vllm.{role}")
    common_engine = _mapping(settings.pop("engine_kwargs", None), name="vllm.engine_kwargs")
    role_engine = _mapping(
        role_settings.pop("engine_kwargs", None),
        name=f"vllm.{role}.engine_kwargs",
    )
    settings.update(role_settings)
    settings["engine_kwargs"] = {**common_engine, **role_engine}
    unknown = sorted(set(settings) - _VLLM_SETTINGS)
    if unknown:
        raise ValueError("unknown vLLM settings: " + ", ".join(unknown))
    return settings


def configured_backend(config: Mapping[str, Any]) -> str:
    runtime = _mapping(config.get("runtime"), name="runtime")
    backend = str(runtime.get("backend", "transformers"))
    if backend not in BACKEND_CHOICES:
        raise ValueError(
            f"unknown runtime backend {backend!r}; expected one of {BACKEND_CHOICES}"
        )
    return backend


def set_backend_override(config: dict[str, Any], backend: str | None) -> None:
    """Apply a CLI backend override before fingerprinting an experiment."""

    if backend is None:
        return
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"unknown runtime backend {backend!r}")
    config.setdefault("runtime", {})["backend"] = backend


def _transformers_backend(
    model_name_or_path: str,
    adapter_name_or_path: str | None,
    runtime: Mapping[str, Any],
    *,
    device: str | None = None,
    dtype: str | None = None,
    speculation: ActiveBatchSpeculationConfig | None = None,
    loading: Mapping[str, Any] | None = None,
) -> TransformersBackend:
    kwargs: dict[str, Any] = {}
    if speculation is not None:
        kwargs["speculation"] = speculation
    kwargs.update(loading or {})
    kwargs.setdefault("local_files_only", True)
    kwargs.setdefault("trust_remote_code", bool(runtime.get("trust_remote_code", False)))
    kwargs["score_chunk_size"] = int(runtime.get("score_chunk_size", 256))
    return TransformersBackend.from_pretrained(
        model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        device=device or str(runtime.get("device", "cuda")),
        dtype=dtype or str(runtime.get("dtype", "float32")),
        max_score_batch_size=int(runtime.get("max_score_batch_size", 8)),
        **kwargs,
    )


def load_backend_from_config(
    path: str,
    config: Mapping[str, Any],
    *,
    adapter_base: str | None = None,
    role: str | None = None,
) -> Any:
    """Load one model using the selected runtime without changing its policy.

    ``vllm`` selects the persistent asynchronous engine, while ``vllm-sync``
    selects the offline ``LLM`` frontend.  Per-role tables such as
    ``[vllm.proposal]`` override common vLLM settings, which is useful when a
    base and proposal engine share one GPU.
    """

    runtime = _mapping(config.get("runtime"), name="runtime")
    speculation, dynamic_vllm_speculation = _speculation_from_config(config)
    backend_kind = configured_backend(config)
    model_name_or_path = adapter_base or path
    adapter_name_or_path = path if adapter_base is not None else None
    role = role or _infer_role(path, config, adapter_base=adapter_base)
    loading = model_loading_options(config, role)
    resolved = config.get("_resolved_models", {}).get(role, {})
    if resolved.get("source") == path:
        model_name_or_path = resolved["model"]
        adapter_name_or_path = resolved.get("adapter")
        if "tokenizer" in resolved:
            loading["tokenizer_name_or_path"] = resolved["tokenizer"]
    if backend_kind == "transformers":
        return _transformers_backend(
            model_name_or_path,
            adapter_name_or_path,
            runtime,
            speculation=speculation,
            loading=loading,
        )

    settings = _vllm_settings(config, role)
    # Resolve once so the engine and exact scorer share the same immutable files.
    for option, vllm_option in (("revision", "revision"), ("cache_dir", "download_dir"),
                                ("trust_remote_code", "trust_remote_code")):
        if vllm_option in settings:
            if option in loading and loading[option] != settings[vllm_option] and option == "revision":
                raise ValueError("vLLM and model_loading revision settings must agree")
            loading[option] = settings.pop(vllm_option)
    if "/" in model_name_or_path and not Path(model_name_or_path).is_dir():
        model_name_or_path = str(resolve_checkpoint_path(
            model_name_or_path, revision=loading.get("revision"),
            cache_dir=loading.get("cache_dir"), local_files_only=loading["local_files_only"],
        ))
    if adapter_name_or_path is not None:
        adapter_name_or_path = str(resolve_checkpoint_path(
            adapter_name_or_path, revision=loading.get("adapter_revision"),
            cache_dir=loading.get("cache_dir"), local_files_only=loading["local_files_only"],
        ))
    unsupported = set(loading) & {"device_map", "attn_implementation", "model_kwargs"}
    if unsupported:
        raise ValueError("use vllm settings for engine-specific options: " + ", ".join(sorted(unsupported)))
    exact_kind = str(settings.pop("exact_scoring_backend", "none"))
    exact_backend = None
    exact_options: dict[str, Any] = {}
    if exact_kind == "transformers":
        exact_options = {
            "device": str(settings.pop("exact_scoring_device", runtime.get("device", "cuda"))),
            "dtype": str(settings.pop("exact_scoring_dtype", runtime.get("dtype", "float32"))),
        }
    elif exact_kind != "none":
        raise ValueError("vllm.exact_scoring_backend must be 'none' or 'transformers'")
    elif "exact_scoring_device" in settings or "exact_scoring_dtype" in settings:
        raise ValueError(
            "exact_scoring_device/dtype require exact_scoring_backend='transformers'"
        )

    asynchronous = bool(settings.pop("asynchronous", True))
    if backend_kind == "vllm-sync":
        asynchronous = False
    mh_fused_logprobs = bool(settings.pop("mh_fused_logprobs", False))
    if mh_fused_logprobs and asynchronous:
        raise ValueError(
            "vllm.mh_fused_logprobs requires runtime.backend='vllm-sync'"
        )
    if mh_fused_logprobs and speculation is not None:
        raise ValueError(
            "vllm.mh_fused_logprobs cannot be combined with speculative decoding"
        )
    engine_kwargs = _mapping(
        settings.pop("engine_kwargs", None),
        name="vllm.engine_kwargs",
    )
    collisions = sorted(_EXPLICIT_ENGINE_SETTINGS.intersection(engine_kwargs))
    if collisions:
        raise ValueError(
            "vLLM engine_kwargs duplicate explicit settings: "
            + ", ".join(collisions)
        )
    beam_width = int(_mapping(config.get("beam"), name="beam").get("num_beams", 1))
    required_logprobs = max(20, 2 * beam_width)
    engine_kwargs["max_logprobs"] = max(
        required_logprobs,
        int(engine_kwargs.get("max_logprobs", required_logprobs)),
    )
    loader = AsyncVLLMBackend if asynchronous else VLLMBackend
    acceleration_kwargs: dict[str, Any] = {}
    if not asynchronous:
        acceleration_kwargs["enable_mh_fused_logprobs"] = mh_fused_logprobs
    if speculation is not None:
        acceleration_kwargs.update({
            "speculation": speculation,
            "dynamic_speculation": dynamic_vllm_speculation,
        })
    try:
        if exact_kind == "transformers":
            exact_backend = _transformers_backend(model_name_or_path, adapter_name_or_path, runtime, loading=loading, **exact_options)
        return loader.from_pretrained(
            model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dtype=str(settings.pop("dtype", runtime.get("dtype", "bfloat16"))),
            tensor_parallel_size=int(settings.pop("tensor_parallel_size", 1)),
            data_parallel_size=int(settings.pop("data_parallel_size", 1)),
            gpu_memory_utilization=float(settings.pop("gpu_memory_utilization", 0.9)),
            max_model_len=settings.pop("max_model_len", None),
            max_num_seqs=settings.pop("max_num_seqs", None),
            max_num_batched_tokens=settings.pop("max_num_batched_tokens", None),
            quantization=settings.pop("quantization", None),
            enforce_eager=bool(settings.pop("enforce_eager", False)),
            trust_remote_code=loading["trust_remote_code"],
            revision=loading.get("revision"),
            download_dir=loading.get("cache_dir"),
            tokenizer_name_or_path=loading.get("tokenizer_name_or_path"),
            tokenizer_revision=loading.get("tokenizer_revision"),
            adapter_revision=loading.get("adapter_revision"),
            tokenizer_kwargs=loading.get("tokenizer_kwargs"),
            local_files_only=loading["local_files_only"],
            seed=int(settings.pop("seed", config.get("run", {}).get("seed", 0))),
            parameter_count=settings.pop("parameter_count", None),
            scoring_backend=exact_backend,
            enable_prefix_caching=bool(settings.pop("enable_prefix_caching", True)),
            max_lora_rank=int(settings.pop("max_lora_rank", 16)),
            engine_kwargs=engine_kwargs,
            **acceleration_kwargs,
        )
    except BaseException:
        close_backend(exact_backend)
        raise


def close_backend(backend: Any | None) -> None:
    """Close vLLM processes/threads when present; safe for other backends."""

    if backend is None:
        return
    try:
        callback = getattr(backend, "close", None)
        if callback is not None:
            callback()
    finally:
        scoring_backend = getattr(backend, "scoring_backend", None)
        if scoring_backend is not None and scoring_backend is not backend:
            close_backend(scoring_backend)
