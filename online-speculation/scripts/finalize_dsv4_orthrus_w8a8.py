#!/usr/bin/env python3
"""Seal independently calibrated ModelSlim output as a deployable v2 head."""

from __future__ import annotations

import argparse
import sys
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
    ContractError,
    find_weight_index,
    read_json,
    select_weight_map,
    sha256_file,
    shard_manifest,
    target_identity,
    tensor_schema,
    validate_manifest_header,
    validate_selected_layers,
    write_json,
)


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bf16-head", required=True, type=Path)
    parser.add_argument("--quantized-head", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--calibration-data", required=True, type=Path)
    parser.add_argument(
        "--base-model-id",
        default=BASE_MODEL_ID,
    )
    args = parser.parse_args()

    model = args.model.resolve()
    bf16_head = args.bf16_head.resolve()
    head = args.quantized_head.resolve()
    calibration_path = args.calibration_manifest.resolve()
    calibration_data = args.calibration_data.resolve()
    try:
        target_config = read_json(model / "config.json")
        source_manifest = read_json(bf16_head / "orthrus_head_manifest.json")
        validate_manifest_header(source_manifest)
        source_index = read_json(find_weight_index(bf16_head))
        head_config = read_json(head / "config.json")
        index_path = find_weight_index(head)
        index = read_json(index_path)
        quant_path = head / "quant_model_description.json"
        read_json(quant_path)
        calibration = read_json(calibration_path)
    except ContractError as error:
        fail(str(error))

    if source_manifest.get("precision") != "bf16":
        fail("--bf16-head must be the unquantized training exchange artifact")
    if source_manifest.get("artifact_role") != "trained":
        fail("--bf16-head must be marked artifact_role=trained")
    required_calibration = (
        "tool",
        "tool_version",
        "dataset_sha256",
        "bf16_head_manifest_sha256",
        "command",
    )
    missing = [key for key in required_calibration if not calibration.get(key)]
    if missing:
        fail(f"calibration manifest is incomplete: {missing}")
    source_manifest_path = bf16_head / "orthrus_head_manifest.json"
    if calibration["bf16_head_manifest_sha256"] != sha256_file(source_manifest_path):
        fail("calibration manifest does not identify the supplied BF16 head")
    if not calibration_data.is_file():
        fail(f"calibration dataset does not exist: {calibration_data}")
    if calibration["dataset_sha256"] != sha256_file(calibration_data):
        fail("calibration dataset hash does not match the supplied data")

    layers = int(target_config.get("num_hidden_layers", 0))
    ratios = [int(value) for value in target_config.get("compress_ratios", [])[:layers]]
    if layers <= 0 or len(ratios) != layers:
        fail("target must contain one compress_ratio per layer")
    if list(source_manifest.get("compress_ratios", [])) != ratios:
        fail("BF16 head and target compress_ratios differ")
    actual_target = target_identity(model, source_manifest.get("base_model", {}).get("id"))
    for key in ("config_file", "config_sha256", "weight_index_file", "weight_index_sha256"):
        if source_manifest.get("base_model", {}).get(key) != actual_target.get(key):
            fail(f"BF16 head base model {key} mismatch")
    try:
        source_schema, source_size = tensor_schema(
            bf16_head, source_index.get("weight_map", {})
        )
    except ContractError as error:
        fail(str(error))
    if source_schema != source_manifest.get("tensor_schema"):
        fail("BF16 head tensor schema differs from its manifest")
    if source_size != int(source_manifest.get("total_size", -1)):
        fail("BF16 head total_size differs from its manifest")
    source_shards = {item["name"]: item for item in source_manifest.get("shards", [])}
    if set(source_shards) != set(source_index.get("weight_map", {}).values()):
        fail("BF16 head shard set differs from its manifest")
    for shard, item in source_shards.items():
        path = bf16_head / shard
        if path.stat().st_size != int(item.get("size", -1)):
            fail(f"BF16 head shard size mismatch: {shard}")
        if sha256_file(path) != item.get("sha256"):
            fail(f"BF16 head shard SHA256 mismatch: {shard}")
    if list(head_config.get("compress_ratios", [])[:layers]) != ratios:
        fail("quantized head and target compress_ratios differ")
    selected = select_weight_map(index.get("weight_map", {}), ratios)
    try:
        validate_selected_layers(selected, ratios)
    except ContractError as error:
        fail(str(error))
    selected_names = {
        name for entries in selected.values() for name, _ in entries
    }
    indexed_names = set(index.get("weight_map", {}))
    if selected_names != indexed_names:
        fail("quantized artifact contains non-head or non-canonical tensors")

    try:
        schema, total_size = tensor_schema(head, index["weight_map"])
    except ContractError as error:
        fail(str(error))
    index.setdefault("metadata", {})["total_size"] = total_size
    write_json(index_path, index)

    sealed_calibration = head / "orthrus_head_calibration.json"
    write_json(sealed_calibration, calibration)
    mask_token_id = int(head_config.get("mask_token_id", -1))
    if mask_token_id < 0:
        fail("quantized head config lacks mask_token_id")
    head_config.update(
        {
            "architectures": ["OrthrusDSV4DraftModel"],
            "pard_token": mask_token_id,
            "orthrus_head_format": FORMAT_NAME,
            "orthrus_head_format_version": FORMAT_VERSION,
            "orthrus_block_size": BLOCK_SIZE,
            "orthrus_shared_backbone": True,
            "parallel_drafting": True,
            "sample_from_anchor": False,
        }
    )
    write_json(head / "config.json", head_config)
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "artifact_role": "trained",
        "precision": "w8a8",
        "base_model": target_identity(model, args.base_model_id),
        "block_size": BLOCK_SIZE,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "mask_token_id": mask_token_id,
        "num_hidden_layers": layers,
        "compress_ratios": ratios,
        "history_backend": HISTORY_BACKEND,
        "tensor_count": len(schema),
        "total_size": total_size,
        "tensor_schema": schema,
        "shards": shard_manifest(head, index["weight_map"].values()),
        "quantization": {
            "format": "modelslim-ascend-w8a8",
            "source_description_sha256": sha256_file(quant_path),
            "head_calibration_required": False,
            "independently_calibrated": True,
            "scale_origin": "head-calibration",
            "shares_target_scale_storage": False,
            "calibration_manifest_sha256": sha256_file(sealed_calibration),
            "bf16_head_manifest_sha256": sha256_file(source_manifest_path),
        },
    }
    write_json(head / "orthrus_head_manifest.json", manifest)
    print(head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
