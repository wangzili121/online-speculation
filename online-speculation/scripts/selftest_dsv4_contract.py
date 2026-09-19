#!/usr/bin/env python3
"""Run the DSV4 head-contract gates without torch, vLLM, or an NPU."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.vllm_ascend.head_contract import (  # noqa: E402
    BASE_MODEL_ID,
    BLOCK_SIZE,
    FORMAT_NAME,
    FORMAT_VERSION,
    HISTORY_BACKEND,
    NUM_SPECULATIVE_TOKENS,
    runtime_tensor_name,
    sha256_file,
    shard_manifest,
    target_identity,
    tensor_schema,
    write_json,
)
from blockspec.reporting import dumps  # noqa: E402
from blockspec.vllm_ascend.artifacts import encode_private_json  # noqa: E402


BASE_SUFFIXES = (
    "attn_sink",
    "wq_a.weight",
    "q_norm.weight",
    "wq_b.weight",
    "wq_b.weight_scale",
    "wq_b.weight_offset",
    "wkv.weight",
    "kv_norm.weight",
    "wo_a.weight",
    "wo_b.weight",
    "wo_b.weight_scale",
    "wo_b.weight_offset",
)
C4_SUFFIXES = (
    "indexer.wq_b.weight",
    "indexer.wq_b.weight_scale",
    "indexer.wq_b.weight_offset",
    "indexer.weights_proj.weight",
)


def _write_safetensors(path: Path, tensors: dict[str, str]) -> None:
    header = {}
    offset = 0
    for name, dtype in sorted(tensors.items()):
        size = 1 if dtype == "I8" else 4
        header[name] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = encode_private_json(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _tensor_dtype(suffix: str) -> str:
    if suffix in ("wq_b.weight", "wo_b.weight", "indexer.wq_b.weight"):
        return "I8"
    return "F32"


def _quant_mode(suffix: str) -> str | None:
    if not suffix.endswith(".weight"):
        return None
    if suffix in ("wq_b.weight", "wo_b.weight", "indexer.wq_b.weight"):
        return "W8A8_DYNAMIC"
    return "FLOAT"


def _run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != expected:
        raise RuntimeError(
            f"command returned {result.returncode}, expected {expected}: "
            f"{' '.join(command)}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orthrus-contract-") as temporary:
        base = Path(temporary)
        target = base / "target"
        head = base / "head"
        target.mkdir()
        head.mkdir()
        ratios = [0, 0] + [4, 128] * 20 + [4]
        config = {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "hidden_size": 16,
            "num_attention_heads": 64,
            "vocab_size": 129280,
            "compress_ratios": ratios,
            "sliding_window": 128,
            "dspark_noise_token_id": 128799,
        }
        write_json(target / "config.json", config)

        target_tensors = {}
        head_tensors = {}
        target_quant = {}
        head_quant = {}
        for layer, ratio in enumerate(ratios):
            suffixes = BASE_SUFFIXES + (C4_SUFFIXES if ratio == 4 else ())
            for suffix in suffixes:
                source = f"model.layers.{layer}.self_attn.{suffix}"
                canonical = f"layers.{layer}.attn.{suffix}"
                dtype = _tensor_dtype(suffix)
                target_tensors[source] = dtype
                head_tensors[canonical] = dtype
                mode = _quant_mode(suffix)
                if mode is not None:
                    target_quant[source] = mode
                    head_quant[runtime_tensor_name(canonical)] = mode
            compressor = f"model.layers.{layer}.self_attn.compressor.wkv.weight"
            target_tensors[compressor] = "F32"
            target_quant[compressor] = "FLOAT"

        target_shard = "quant_model_weights-00001-of-00001.safetensors"
        _write_safetensors(target / target_shard, target_tensors)
        target_size = sum(1 if dtype == "I8" else 4 for dtype in target_tensors.values())
        write_json(
            target / "quant_model_weights.safetensors.index.json",
            {
                "metadata": {"total_size": target_size},
                "weight_map": {name: target_shard for name in target_tensors},
            },
        )
        write_json(target / "quant_model_description.json", target_quant)

        dry_run = _run(
            [
                sys.executable,
                str(ROOT / "scripts/export_dsv4_orthrus_head.py"),
                str(target),
                str(base / "unused"),
                "--dry-run",
            ]
        )
        if "validated 43 layers" not in dry_run.stdout:
            raise RuntimeError("export dry-run did not validate all layers")

        head_config = dict(config)
        head_config.update(
            {
                "architectures": ["OrthrusDSV4DraftModel"],
                "orthrus_head_format": FORMAT_NAME,
                "orthrus_head_format_version": FORMAT_VERSION,
                "orthrus_block_size": BLOCK_SIZE,
                "orthrus_shared_backbone": True,
                "mask_token_id": config["dspark_noise_token_id"],
                "pard_token": config["dspark_noise_token_id"],
                "parallel_drafting": True,
                "sample_from_anchor": False,
            }
        )
        write_json(head / "config.json", head_config)
        head_shard = "orthrus-head-00001-of-00001.safetensors"
        _write_safetensors(head / head_shard, head_tensors)
        head_index = {
            "metadata": {
                "total_size": sum(
                    1 if dtype == "I8" else 4 for dtype in head_tensors.values()
                )
            },
            "weight_map": {name: head_shard for name in head_tensors},
        }
        write_json(head / "quant_model_weights.safetensors.index.json", head_index)
        write_json(head / "quant_model_description.json", head_quant)
        schema, total_size = tensor_schema(head, head_index["weight_map"])
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "artifact_role": "initialized-plumbing",
            "precision": "w8a8",
            "base_model": target_identity(target, BASE_MODEL_ID),
            "block_size": BLOCK_SIZE,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "mask_token_id": config["dspark_noise_token_id"],
            "num_hidden_layers": len(ratios),
            "compress_ratios": ratios,
            "history_backend": HISTORY_BACKEND,
            "tensor_count": len(schema),
            "total_size": total_size,
            "tensor_schema": schema,
            "shards": shard_manifest(head, [head_shard]),
            "quantization": {
                "format": "modelslim-ascend-w8a8",
                "source_description_sha256": sha256_file(
                    target / "quant_model_description.json"
                ),
                "head_calibration_required": True,
                "independently_calibrated": False,
                "scale_origin": "target-copy-initialization",
                "shares_target_scale_storage": False,
            },
        }
        write_json(head / "orthrus_head_manifest.json", manifest)

        preflight = [
            sys.executable,
            str(ROOT / "scripts/preflight_dsv4_orthrus.py"),
            "--model",
            str(target),
            "--head",
            str(head),
            "--tensor-parallel-size",
            "8",
            "--production",
        ]
        checked = _run(preflight)
        if json.loads(checked.stdout)["status"] != "ok":
            raise RuntimeError("production preflight did not return ok")
        trained = _run(preflight + ["--require-trained"], expected=1)
        if "trained diffusion head" not in trained.stderr:
            raise RuntimeError("initialized head bypassed the trained-head gate")

        quant_path = head / "quant_model_description.json"
        broken_quant = dict(head_quant)
        broken_quant["orthrus.layers.0.self_attn.wq_b.weight"] = "FLOAT"
        write_json(quant_path, broken_quant)
        wrong_quant = _run(preflight, expected=1)
        if "projection quantization mode mismatch" not in wrong_quant.stderr:
            raise RuntimeError("incorrect projection mode bypassed the quantization gate")
        write_json(quant_path, head_quant)

        shard_path = head / head_shard
        shard_path.write_bytes(shard_path.read_bytes()[:-1])
        corrupted = _run(preflight, expected=1)
        if "truncated or oversized" not in corrupted.stderr:
            raise RuntimeError("corrupted shard bypassed integrity validation")

    print(
        dumps(
            {
                "status": "ok",
                "layers": 43,
                "history": {"swa": 2, "c4": 21, "c128": 20},
                "export_dry_run": True,
                "production_preflight": True,
                "trained_gate": True,
                "quantization_gate": True,
                "corruption_gate": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
