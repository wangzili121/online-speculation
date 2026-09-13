"""Shared validation and device checks for real-model dLLM experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.shared.artifacts import (
    adapter_hashes,
    cached_file_sha256,
    checkpoint_metadata_hashes,
    dataclass_snapshot_delta,
    directory_hashes,
    file_sha256,
    implementation_hashes,
    json_fingerprint,
)
from inference_scaling.dllm.config import DiffusionSamplingConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HASH_CACHE = REPOSITORY_ROOT / ".cache" / "artifact_hashes"


def sampling_from_section(section: dict[str, Any]) -> DiffusionSamplingConfig:
    return DiffusionSamplingConfig(
        block_length=int(section["block_length"]),
        steps_per_block=int(section.get("denoising_steps", section.get("steps_per_block"))),
        temperature=float(section.get("temperature", 0.0)),
        top_k=int(section.get("top_k", 0)),
        top_p=float(section.get("top_p", 1.0)),
        cfg_scale=float(section.get("cfg_scale", 0.0)),
        remasking=str(section.get("remasking", "low_confidence")),
        confidence_threshold=float(section.get("confidence_threshold", 0.85)),
        mask_token_id=(
            int(section["mask_token_id"])
            if section.get("mask_token_id") is not None
            else None
        ),
    )


def capped_generation_length(
    *,
    prompt_length: int,
    maximum: int,
    sampling: DiffusionSamplingConfig,
) -> int:
    del prompt_length
    length = maximum - maximum % sampling.block_length
    if length <= 0:
        raise ValueError("generation budget is too small to complete a diffusion block")
    return length


def llada_snapshot_delta(before: Any, after: Any) -> dict[str, Any]:
    result = dataclass_snapshot_delta(
        before,
        after,
        constant_fields={"total_parameters", "active_parameters"},
    )
    result["estimated_active_flops"] = (
        2 * result["active_parameters"] * result["model_token_slots"]
    )
    result["estimated_sample_active_flops"] = (
        2 * result["active_parameters"] * result["sample_model_token_slots"]
    )
    result["estimated_score_active_flops"] = (
        2 * result["active_parameters"] * result["score_model_token_slots"]
    )
    return result


def empty_llada_compute() -> dict[str, int | float]:
    return {
        "sample_requests": 0,
        "score_requests": 0,
        "forward_calls": 0,
        "model_sequences": 0,
        "model_token_slots": 0,
        "generated_tokens": 0,
        "elapsed_seconds": 0.0,
        "total_parameters": 0,
        "active_parameters": 0,
        "resident_parameters": 0,
        "sample_forward_calls": 0,
        "score_forward_calls": 0,
        "sample_model_sequences": 0,
        "score_model_sequences": 0,
        "sample_model_token_slots": 0,
        "score_model_token_slots": 0,
        "sample_elapsed_seconds": 0.0,
        "score_elapsed_seconds": 0.0,
        "estimated_active_flops": 0,
        "estimated_sample_active_flops": 0,
        "estimated_score_active_flops": 0,
    }


def validate_runtime_device(config: dict[str, Any]) -> str:
    device = str(config["runtime"]["device"])
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def validate_llada_weights(config: dict[str, Any]) -> dict[str, str]:
    model_dir = Path(str(config["model"]["path"]))
    names = tuple(str(value) for value in config["model"]["weight_files"])
    hashes = tuple(str(value) for value in config["model"]["weight_sha256"])
    sizes = tuple(int(value) for value in config["model"]["weight_bytes"])
    if not (len(names) == len(hashes) == len(sizes)):
        raise ValueError("LLaDA weight manifest columns have different lengths")
    actual: dict[str, str] = {}
    for name, expected_hash, expected_size in zip(names, hashes, sizes, strict=True):
        weight = model_dir / name
        if not weight.is_file():
            raise FileNotFoundError(
                f"pinned LLaDA weight is absent: {weight}; "
                "run experiments/dllm/download_llada.py"
            )
        if weight.stat().st_size != expected_size:
            raise ValueError(f"LLaDA weight size does not match the manifest: {weight}")
        actual_hash = cached_file_sha256(
            weight,
            cache_directory=HASH_CACHE,
            expected=expected_hash,
        )
        actual[name] = actual_hash
    return actual


__all__ = [
    "adapter_hashes",
    "capped_generation_length",
    "checkpoint_metadata_hashes",
    "directory_hashes",
    "empty_llada_compute",
    "file_sha256",
    "implementation_hashes",
    "json_fingerprint",
    "llada_snapshot_delta",
    "sampling_from_section",
    "validate_llada_weights",
    "validate_runtime_device",
]
