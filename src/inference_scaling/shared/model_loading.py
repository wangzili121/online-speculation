"""Model identity and loading options, independent of algorithms and datasets."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

MODEL_ROLES = ("base", "proposal", "rl")
_OPTIONS = frozenset({
    "revision", "tokenizer_name_or_path", "tokenizer_revision", "adapter_revision",
    "cache_dir", "local_files_only", "trust_remote_code", "device_map",
    "attn_implementation", "model_kwargs", "tokenizer_kwargs",
})


def model_role(path: str, config: Mapping[str, Any], *, adapter_base: str | None = None) -> str:
    for role in MODEL_ROLES:
        configured = config.get("models", {}).get(role)
        if configured is not None and (
            str(configured) == path or Path(str(configured)).resolve() == Path(path).resolve()
        ):
            return role
    return "rl" if adapter_base is not None else "base"


def model_loading_options(config: Mapping[str, Any], role: str = "base") -> dict[str, Any]:
    """Merge common and per-role options while preserving legacy configs."""
    if role not in MODEL_ROLES:
        raise ValueError(f"unknown model role: {role}")
    table = config.get("model_loading", {})
    if not isinstance(table, Mapping) or not isinstance(table.get(role, {}), Mapping):
        raise TypeError("model_loading and its role settings must be tables")
    common = {key: value for key, value in table.items() if key not in MODEL_ROLES}
    specific = dict(table.get(role, {}))
    result = {**common, **specific}
    unknown = set(result) - _OPTIONS
    if unknown:
        raise ValueError("unknown model_loading settings: " + ", ".join(sorted(unknown)))
    for name in ("model_kwargs", "tokenizer_kwargs"):
        left, right = common.get(name, {}), specific.get(name, {})
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise TypeError(f"model_loading.{name} must be a table")
        if left or right:
            result[name] = {**left, **right}
    runtime = config.get("runtime", {})
    for name, default in (("local_files_only", True), ("trust_remote_code", False)):
        result.setdefault(name, runtime.get(name, default))
        if not isinstance(result[name], bool):
            raise TypeError(f"model_loading.{name} must be boolean")
    models = config.get("models", {})
    revision_key = "base_revision" if role == "rl" and models.get("rl_kind") == "peft_adapter" else f"{role}_revision"
    # Legacy local training labels (e.g. configs/...toml) are metadata, not Hub refs.
    if "revision" not in result and revision_key in models and (role != "rl" or models.get("rl_kind") != "peft_adapter"):
        result["revision"] = str(models[revision_key])
    if role == "rl" and models.get("rl_kind") == "peft_adapter" and "revision" not in result:
        result["revision"] = model_loading_options(config, "base").get("revision")
    if str(runtime.get("backend", "transformers")).startswith("vllm"):
        engine = config.get("vllm", {})
        role_engine = engine.get(role, {})
        for option, engine_option in (("revision", "revision"), ("cache_dir", "download_dir"),
                                      ("trust_remote_code", "trust_remote_code")):
            if engine_option in role_engine or engine_option in engine:
                value = role_engine.get(engine_option, engine.get(engine_option))
                if option == "revision" and result.get(option) is not None and result[option] != value:
                    raise ValueError("vLLM and model_loading revision settings must agree")
                result[option] = value
    return result


def resolve_checkpoint_path(
    name_or_path: str, *, revision: str | None = None,
    cache_dir: str | None = None, local_files_only: bool = True,
) -> Path:
    """Resolve a local checkpoint or a Hub snapshot; offline by default.

    This function reads/downloads files only when explicitly called by a runtime,
    never during configuration parsing or CLI dry runs.
    """
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or name_or_path.startswith(("./", "../", ".\\", "..\\")):
        raise FileNotFoundError(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        name_or_path, revision=revision, cache_dir=cache_dir,
        local_files_only=local_files_only,
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.gguf", "*.ot", "original/*"],
    ))


def model_identity(model: str, adapter: str | None = None, *, revision: str | None = None,
                   adapter_revision: str | None = None, tokenizer: str | None = None,
                   tokenizer_revision: str | None = None) -> str:
    """Use the same identity for generation, rescoring, and replay caches."""
    value = model if adapter is None else f"{model}+adapter:{adapter}"
    for name, setting in (("revision", revision), ("adapter_revision", adapter_revision),
                          ("tokenizer", tokenizer), ("tokenizer_revision", tokenizer_revision)):
        if setting is not None:
            value += f";{name}={setting}"
    return value


def checkpoint_weight_files(path: Path) -> tuple[Path, ...]:
    """Resolve the standard safetensors/PyTorch weight layout without loading it."""
    directory = path.resolve()
    for single, index in (("model.safetensors", "model.safetensors.index.json"),
                          ("pytorch_model.bin", "pytorch_model.bin.index.json")):
        if (directory / single).is_file():
            return (directory / single,)
        if not (directory / index).is_file():
            continue
        value = json.loads((directory / index).read_text(encoding="utf-8"))
        mapping = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid checkpoint weight_map: {directory / index}")
        if any(not isinstance(name, str) for name in mapping.values()):
            raise ValueError("checkpoint shard names must be strings")
        files = []
        for name in sorted(set(mapping.values())):
            shard = Path(name)
            # Validate names, not resolved symlinks: Hub snapshots link to blobs.
            if shard.is_absolute() or shard.drive or ".." in shard.parts or "\\" in name:
                raise ValueError(f"checkpoint shard must be relative to its directory: {name}")
            target = directory / name
            if not target.is_file():
                raise FileNotFoundError(target)
            files.append(target)
        return tuple(files)
    raise FileNotFoundError(f"no supported checkpoint weights or shard index in {directory}")
