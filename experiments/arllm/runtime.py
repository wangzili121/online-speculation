"""Shared artifact validation for real-model AR experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.shared.artifacts import (
    adapter_hashes,
    checkpoint_weight_hashes,
    weight_manifest_digest,
    checkpoint_metadata_hashes,
    implementation_hashes,
)

from inference_scaling.shared.model_loading import model_loading_options, resolve_checkpoint_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HASH_CACHE = REPOSITORY_ROOT / ".cache" / "artifact_hashes"


def source_hashes(entrypoints: Iterable[str | Path]) -> dict[str, str]:
    return implementation_hashes(REPOSITORY_ROOT, entrypoints=entrypoints)


def set_rl_adapter_override(config: dict[str, Any], adapter: Path | None) -> None:
    """Point an AR evaluation config at an adapter produced by the same suite."""

    if adapter is None:
        return
    models = config["models"]
    models["rl"] = str(adapter)
    models["rl_source"] = "local GRPO adapter from the current reproduction suite"
    models["rl_revision"] = "suite-output"
    models["rl_kind"] = "peft_adapter"


def validate_model_artifacts(
    config: dict[str, Any], roles: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Resolve/hash checkpoints and pin subsequent loads to those exact files."""
    models = config["models"]
    requested = set(roles)
    unknown = requested - {"base", "proposal", "rl"}
    if unknown:
        raise ValueError(f"unknown AR model roles: {sorted(unknown)}")
    weights: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    shards: dict[str, Any] = {}
    resolved = config.setdefault("_resolved_models", {})
    for role in sorted(requested):
        options = model_loading_options(config, role)
        is_adapter = role == "rl" and models.get("rl_kind") == "peft_adapter"
        source = str(models.get("rl_base", models.get("base"))) if is_adapter else str(models[role])
        directory = resolve_checkpoint_path(
            source, revision=options.get("revision"), cache_dir=options.get("cache_dir"),
            local_files_only=options["local_files_only"],
        )
        files = checkpoint_weight_hashes(directory, cache_directory=HASH_CACHE)
        digest = weight_manifest_digest(files)
        expected = models.get(f"{role}_weight_sha256") if not is_adapter else None
        if expected is not None and digest != expected:
            raise ValueError(f"artifact hash mismatch for {role}: expected {expected}, got {digest}")
        base_key = f"{role}_base" if is_adapter else role
        weights[base_key], shards[base_key] = digest, files
        metadata[base_key] = checkpoint_metadata_hashes(directory)
        resolved[role] = {"source": str(models[role]), "model": str(directory)}
        if is_adapter:
            adapter = resolve_checkpoint_path(
                str(models[role]), revision=options.get("adapter_revision"),
                cache_dir=options.get("cache_dir"), local_files_only=options["local_files_only"],
            )
            manifest = adapter_hashes(adapter)
            adapters[role] = manifest
            weights["rl_adapter"] = weight_manifest_digest({
                name: value for name, value in manifest.items() if name.startswith("adapter_model.")
            })
            resolved[role]["adapter"] = str(adapter)
        if options.get("tokenizer_name_or_path"):
            tokenizer = resolve_checkpoint_path(
                options["tokenizer_name_or_path"], revision=options.get("tokenizer_revision"),
                cache_dir=options.get("cache_dir"), local_files_only=options["local_files_only"],
            )
            metadata[f"{role}_tokenizer"] = checkpoint_metadata_hashes(tokenizer)
            resolved[role]["tokenizer"] = str(tokenizer)
    return {"weight_sha256": weights, "metadata_sha256": metadata,
            "adapter_sha256": adapters, "shard_sha256": shards}


def model_metadata(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    models = config["models"]
    options = model_loading_options(config, role)
    metadata: dict[str, Any] = {
        "role": role, "name_or_path": str(models[role]),
        "source": str(models.get(f"{role}_source", models[role])),
        "revision": options.get("revision"),
    }
    if f"{role}_weight_sha256" in models:
        metadata["weight_sha256"] = models[f"{role}_weight_sha256"]
    if role == "rl":
        metadata["kind"] = models.get("rl_kind", "full_model")
        if "rl_base" in models:
            metadata["base_path"] = models["rl_base"]
    if role in config.get("_resolved_models", {}):
        metadata["resolved"] = config["_resolved_models"][role]
    return metadata


__all__ = [
    "REPOSITORY_ROOT",
    "model_metadata",
    "set_rl_adapter_override",
    "source_hashes",
    "validate_model_artifacts",
]
