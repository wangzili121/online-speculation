"""Model-aware ``custom_class`` entry point for the DSV4 Orthrus head."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .head_contract import (
    FORMAT_NAME,
    NUM_SPECULATIVE_TOKENS,
    read_json,
    validate_manifest_header,
)

if TYPE_CHECKING:
    from .dsv4_proposer import AscendOrthrusProposer


_RUNTIME_API_VERSION = 1


def _orthrus_options(vllm_config) -> dict[str, Any]:
    additional = getattr(vllm_config, "additional_config", None) or {}
    custom = additional.get("custom_proposer_config")
    if not isinstance(custom, dict):
        raise ValueError(
            "model-aware Orthrus requires additional_config.custom_proposer_config"
        )
    options = custom.get("orthrus")
    if not isinstance(options, dict):
        raise ValueError(
            "model-aware Orthrus requires "
            "additional_config.custom_proposer_config.orthrus"
        )
    unknown = sorted(set(options) - {"head_model"})
    if unknown:
        raise ValueError(f"unsupported Orthrus proposer options: {unknown}")
    return options


def _head_path(vllm_config) -> Path:
    value = _orthrus_options(vllm_config).get("head_model")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "additional_config.custom_proposer_config.orthrus.head_model is required"
        )
    path = Path(value).expanduser().resolve()
    manifest = read_json(path / "orthrus_head_manifest.json")
    validate_manifest_header(manifest)
    return path


def _model_config_kwargs(target, head: Path) -> dict[str, Any]:
    """Build a draft ModelConfig without invoking speculative method inference."""

    values = {
        "model": str(head),
        "runner": "draft",
        "tokenizer": getattr(target, "tokenizer", None),
        "tokenizer_mode": getattr(target, "tokenizer_mode", "auto"),
        "trust_remote_code": False,
        "allowed_local_media_path": getattr(target, "allowed_local_media_path", ""),
        "allowed_media_domains": getattr(target, "allowed_media_domains", None),
        "dtype": getattr(target, "dtype", "auto"),
        "seed": getattr(target, "seed", 0),
        "tokenizer_revision": getattr(target, "tokenizer_revision", None),
        "max_model_len": getattr(target, "max_model_len", None),
        "spec_target_max_model_len": getattr(target, "max_model_len", None),
        "quantization": getattr(target, "quantization", None),
        "enforce_eager": True,
        "max_logprobs": getattr(target, "max_logprobs", 20),
        "config_format": getattr(target, "config_format", "auto"),
        "hf_overrides": {},
    }
    from vllm.config import ModelConfig

    parameters = inspect.signature(ModelConfig).parameters
    return {name: value for name, value in values.items() if name in parameters}


def build_orthrus_runtime_config(vllm_config):
    """Create the internal head config while preserving the public method.

    ``SpeculativeConfig.model`` is the Python class path for ``custom_class``.
    The head path therefore lives in
    ``additional_config.custom_proposer_config.orthrus.head_model``.
    Constructing the head ``ModelConfig`` here also avoids vLLM's DFlash and
    DSpark architecture rewriting, which does not apply to an Orthrus head.
    """

    from vllm.config import ModelConfig

    public_spec = vllm_config.speculative_config
    if public_spec is None or public_spec.method != "custom_class":
        raise ValueError("Orthrus custom proposer requires method=custom_class")
    if public_spec.num_speculative_tokens != NUM_SPECULATIVE_TOKENS:
        raise ValueError("DSV4 Orthrus requires exactly 31 speculative tokens")
    draft_sample_method = getattr(public_spec, "draft_sample_method", "greedy")
    if draft_sample_method not in {"greedy", "probabilistic"}:
        raise ValueError("DSV4 Orthrus draft sampling must be greedy or probabilistic")

    head = _head_path(vllm_config)
    target = vllm_config.model_config
    draft_model_config = ModelConfig(**_model_config_kwargs(target, head))
    architectures = set(getattr(draft_model_config, "architectures", ()) or ())
    if "OrthrusDSV4DraftModel" not in architectures:
        raise ValueError(
            f"{head} does not declare the OrthrusDSV4DraftModel architecture"
        )
    if (
        getattr(draft_model_config.hf_config, "orthrus_head_format", None)
        != FORMAT_NAME
    ):
        raise ValueError(f"{head} is not an {FORMAT_NAME} artifact")

    internal_spec = copy.copy(public_spec)
    internal_spec.method = "dflash"
    internal_spec.model = str(head)
    internal_spec.parallel_drafting = True
    internal_spec.enforce_eager = True
    internal_spec.draft_tensor_parallel_size = (
        vllm_config.parallel_config.tensor_parallel_size
    )
    internal_spec.draft_model_config = draft_model_config
    internal_spec.draft_parallel_config = vllm_config.parallel_config
    internal_spec.draft_load_config = getattr(vllm_config, "load_config", None)

    runtime_config = copy.copy(vllm_config)
    runtime_config.speculative_config = internal_spec
    return runtime_config


class OrthrusDSV4CustomProposer:
    """Expose the DSV4 model proposer through vLLM's ``custom_class`` API."""

    npu_model_runner_api_version = _RUNTIME_API_VERSION
    uses_hybrid_cache_groups = True

    def __init__(self, vllm_config) -> None:
        self.vllm_config = vllm_config
        # Fail before NPU allocation when the artifact contract is invalid.
        _head_path(vllm_config)
        self._backend: AscendOrthrusProposer | None = None

    def bind_runtime(self, runtime) -> None:
        if runtime.api_version != self.npu_model_runner_api_version:
            raise ValueError(
                "incompatible vLLM-Ascend model-aware proposer API: "
                f"expected {self.npu_model_runner_api_version}, "
                f"got {runtime.api_version}"
            )
        from .dsv4_proposer import AscendOrthrusProposer

        config = build_orthrus_runtime_config(runtime.vllm_config)
        self._backend = AscendOrthrusProposer(
            config, runtime.device, runner=runtime.runner
        )

    def propose(self, *args, **kwargs):
        """Satisfy vLLM's token-only custom proposer validation.

        vLLM-Ascend dispatches model-aware proposers through ``_propose`` after
        binding the NPU runtime. Reaching this method means that the required
        vLLM-Ascend extension is not installed.
        """
        del args, kwargs
        raise RuntimeError(
            "Orthrus requires the vLLM-Ascend model-aware custom proposer extension"
        )

    def _require_backend(self) -> AscendOrthrusProposer:
        if self._backend is None:
            raise RuntimeError("Orthrus proposer has not been bound to an NPU runtime")
        return self._backend

    def load_model(self, target_model) -> None:
        self._require_backend().load_model(target_model)

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes) -> None:
        self._require_backend().initialize_attn_backend(
            kv_cache_config, kernel_block_sizes
        )

    def prepare_next_token_ids_cpu(self, *args, **kwargs):
        return self._require_backend().prepare_next_token_ids_cpu(*args, **kwargs)

    def prepare_next_token_ids_padded(self, *args, **kwargs):
        return self._require_backend().prepare_next_token_ids_padded(*args, **kwargs)

    def prepare_inputs(self, *args, **kwargs):
        return self._require_backend().prepare_inputs(*args, **kwargs)

    def prepare_inputs_padded(self, *args, **kwargs):
        return self._require_backend().prepare_inputs_padded(*args, **kwargs)

    def _propose(self, *args, **kwargs):
        return self._require_backend()._propose(*args, **kwargs)

    def take_last_draft_probs(self):
        return self._require_backend().take_last_draft_probs()

    def dummy_run(self, *args, **kwargs) -> None:
        self._require_backend().dummy_run(*args, **kwargs)

    @property
    def dynamic_spec(self):
        return self._require_backend().dynamic_spec
