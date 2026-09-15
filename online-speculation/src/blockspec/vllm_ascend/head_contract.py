"""Versioned contract for a DeepSeek-V4 Orthrus diffusion head.

This module deliberately has no torch/vLLM dependency. Export, preflight and
offline packaging must remain usable on a machine without an NPU runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FORMAT_NAME = "orthrus-dsv4-head-v2"
FORMAT_VERSION = 2
HISTORY_BACKEND = "shared_target_hybrid"
BASE_MODEL_ID = "Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"
BLOCK_SIZE = 32
NUM_SPECULATIVE_TOKENS = BLOCK_SIZE - 1

_TENSOR_NAME = re.compile(r"^(?:model\.)?layers\.(\d+)\.(?:attn|self_attn)\.(.+)$")
_BASE_ROOTS = (
    "attn_sink",
    "wq_a.",
    "q_norm.",
    "q_norm_without_weight.",
    "wq_b.",
    "wkv.",
    "kv_norm.",
    "wo_a.",
    "wo_b.",
)
_C4_ROOTS = ("indexer.wq_b.", "indexer.weights_proj.")
_REQUIRED_BASE = (
    "attn_sink",
    "wq_a.weight",
    "q_norm.weight",
    "wq_b.weight",
    "wkv.weight",
    "kv_norm.weight",
    "wo_a.weight",
    "wo_b.weight",
)
_REQUIRED_C4 = ("indexer.wq_b.weight", "indexer.weights_proj.weight")
GLOBAL_QUANT_KEYS = frozenset(
    {"fa_quant_type", "indexer_quant_type", "kv_cache_type"}
)


class ContractError(ValueError):
    """Raised when an artifact violates the v2 head contract."""


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise ContractError(f"missing {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON in {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.JSONEncoder(indent=2, ensure_ascii=True, sort_keys=True).encode(value)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_weight_index(directory: Path) -> Path:
    preferred = directory / "quant_model_weights.safetensors.index.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(directory.glob("*.safetensors.index.json"))
    if len(candidates) != 1:
        raise ContractError(
            f"expected one safetensors index in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def normalize_tensor_name(name: str) -> tuple[int, str] | None:
    match = _TENSOR_NAME.match(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def canonical_tensor_name(layer: int, suffix: str) -> str:
    return f"layers.{layer}.attn.{suffix}"


def runtime_tensor_name(name: str) -> str:
    normalized = normalize_tensor_name(name)
    if normalized is None:
        raise ContractError(f"not an Orthrus attention tensor: {name}")
    layer, suffix = normalized
    return f"orthrus.layers.{layer}.self_attn.{suffix}"


def is_head_tensor(name: str, compress_ratios: list[int]) -> bool:
    normalized = normalize_tensor_name(name)
    if normalized is None:
        return False
    layer, suffix = normalized
    if layer >= len(compress_ratios):
        return False
    if suffix == "attn_sink" or suffix.startswith(_BASE_ROOTS[1:]):
        return True
    return compress_ratios[layer] == 4 and suffix.startswith(_C4_ROOTS)


def select_weight_map(
    weight_map: dict[str, str], compress_ratios: list[int]
) -> dict[int, list[tuple[str, str]]]:
    selected: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for name, shard in weight_map.items():
        if not is_head_tensor(name, compress_ratios):
            continue
        normalized = normalize_tensor_name(name)
        assert normalized is not None
        layer, suffix = normalized
        selected[layer].append((canonical_tensor_name(layer, suffix), shard))
    for values in selected.values():
        values.sort()
    return dict(selected)


def required_suffixes(compress_ratio: int) -> tuple[str, ...]:
    return _REQUIRED_BASE + (_REQUIRED_C4 if compress_ratio == 4 else ())


def validate_selected_layers(
    selected: dict[int, list[tuple[str, str]]], compress_ratios: list[int]
) -> None:
    for layer, ratio in enumerate(compress_ratios):
        suffixes = {
            normalized[1]
            for name, _ in selected.get(layer, [])
            if (normalized := normalize_tensor_name(name)) is not None
        }
        missing = [suffix for suffix in required_suffixes(ratio) if suffix not in suffixes]
        if missing:
            raise ContractError(
                f"head layer {layer} (C{ratio or 1}) is missing: {', '.join(missing)}"
            )


def read_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ContractError(f"truncated safetensors header in {path.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            header_bytes = stream.read(header_size)
            if len(header_bytes) != header_size:
                raise ContractError(f"truncated safetensors header in {path.name}")
        header = json.loads(header_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid safetensors header in {path}: {error}") from error
    if not isinstance(header, dict):
        raise ContractError(f"invalid safetensors header object in {path}")
    return header


def tensor_schema(
    directory: Path, weight_map: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        by_shard[shard].add(name)

    schema: list[dict[str, Any]] = []
    total_size = 0
    for shard, expected in sorted(by_shard.items()):
        path = directory / shard
        if not path.is_file():
            raise ContractError(f"missing safetensors shard {path}")
        header = read_safetensors_header(path)
        actual = set(header) - {"__metadata__"}
        if actual != expected:
            raise ContractError(
                f"safetensors index/header mismatch in {shard}: "
                f"missing={sorted(expected - actual)[:4]}, "
                f"extra={sorted(actual - expected)[:4]}"
            )
        end = 0
        for name in sorted(expected):
            item = header[name]
            start, tensor_end = item["data_offsets"]
            if start < 0 or tensor_end < start:
                raise ContractError(f"invalid data offsets for {name} in {shard}")
            nbytes = tensor_end - start
            total_size += nbytes
            end = max(end, tensor_end)
            schema.append(
                {
                    "name": name,
                    "dtype": item["dtype"],
                    "shape": item["shape"],
                    "nbytes": nbytes,
                    "shard": shard,
                }
            )
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
        expected_size = 8 + header_size + end
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ContractError(
                f"truncated or oversized safetensors shard {shard}: "
                f"expected {expected_size} bytes, got {actual_size}"
            )
    return schema, total_size


def shard_manifest(directory: Path, shards: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": shard,
            "size": (directory / shard).stat().st_size,
            "sha256": sha256_file(directory / shard),
        }
        for shard in sorted(set(shards))
    ]


def target_identity(model: Path, model_id: str | None = None) -> dict[str, Any]:
    config_path = model / "config.json"
    index_path = find_weight_index(model)
    config = read_json(config_path)
    return {
        "id": model_id or config.get("_name_or_path") or model.name,
        "config_file": config_path.name,
        "config_sha256": sha256_file(config_path),
        "weight_index_file": index_path.name,
        "weight_index_sha256": sha256_file(index_path),
    }


def validate_manifest_header(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != FORMAT_NAME:
        if manifest.get("format") == 1 or manifest.get("format_version") == 1:
            raise ContractError("SWA-only Orthrus head v1 is rejected; export a v2 hybrid head")
        raise ContractError(f"unsupported Orthrus head format: {manifest.get('format')!r}")
    if int(manifest.get("format_version", 0)) != FORMAT_VERSION:
        raise ContractError("unsupported Orthrus head format_version")
    if manifest.get("history_backend") != HISTORY_BACKEND:
        raise ContractError("head must use history_backend=shared_target_hybrid")
    if int(manifest.get("block_size", 0)) != BLOCK_SIZE:
        raise ContractError("Orthrus DSV4 requires block_size=32")
    if int(manifest.get("num_speculative_tokens", -1)) != NUM_SPECULATIVE_TOKENS:
        raise ContractError("Orthrus DSV4 requires num_speculative_tokens=31")
