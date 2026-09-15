#!/usr/bin/env python3
"""Strict offline preflight for a DSV4 target and Orthrus v2 head."""

from __future__ import annotations

import argparse
import json
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
    runtime_tensor_name,
    select_weight_map,
    sha256_file,
    target_identity,
    tensor_schema,
    validate_manifest_header,
    validate_selected_layers,
)


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--require-trained", action="store_true")
    parser.add_argument("--allow-bf16", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    head = args.head.resolve()
    try:
        target_config = read_json(model / "config.json")
        head_config = read_json(head / "config.json")
        manifest = read_json(head / "orthrus_head_manifest.json")
        validate_manifest_header(manifest)
        head_index_path = find_weight_index(head)
        head_index = read_json(head_index_path)
    except ContractError as error:
        fail(str(error))

    if target_config.get("model_type") != "deepseek_v4":
        fail("target model_type must be deepseek_v4")
    if "DeepseekV4ForCausalLM" not in target_config.get("architectures", []):
        fail("target architecture must include DeepseekV4ForCausalLM")
    layers = int(target_config.get("num_hidden_layers", 0))
    ratios = [int(value) for value in target_config.get("compress_ratios", [])[:layers]]
    if layers <= 0 or len(ratios) != layers:
        fail("target must have one compress_ratio per hidden layer")
    if not any(ratio == 4 for ratio in ratios) or not any(ratio == 128 for ratio in ratios):
        fail("target is not the expected SWA/C4/C128 hybrid DSV4 architecture")
    if args.production and layers != 43:
        fail(f"production DSV4-Flash must have 43 layers, got {layers}")

    if head_config.get("model_type") != "deepseek_v4":
        fail("head model_type must be deepseek_v4")
    if head_config.get("architectures") != ["OrthrusDSV4DraftModel"]:
        fail("head architecture must be OrthrusDSV4DraftModel")
    if head_config.get("orthrus_head_format") != FORMAT_NAME:
        fail("head config is not orthrus-dsv4-head-v2")
    if int(head_config.get("orthrus_head_format_version", 0)) != FORMAT_VERSION:
        fail("head config format version must be 2")
    if not head_config.get("orthrus_shared_backbone"):
        fail("head lacks orthrus_shared_backbone marker")
    if head_config.get("parallel_drafting") is not True:
        fail("Orthrus v2 requires parallel_drafting=true")
    if head_config.get("sample_from_anchor") is not False:
        fail("Orthrus v2 requires sample_from_anchor=false")
    if int(head_config.get("orthrus_block_size", 0)) != BLOCK_SIZE:
        fail("head config must use K=32")
    if int(head_config.get("mask_token_id", -1)) < 0:
        fail("head config must define mask_token_id")
    if int(head_config.get("pard_token", -1)) != int(head_config["mask_token_id"]):
        fail("head config pard_token must equal mask_token_id for vLLM parallel drafting")
    if int(head_config["mask_token_id"]) != int(manifest.get("mask_token_id", -1)):
        fail("head config/manifest mask_token_id mismatch")
    target_mask = target_config.get(
        "mask_token_id", target_config.get("dspark_noise_token_id")
    )
    if target_mask is not None and int(head_config["mask_token_id"]) != int(target_mask):
        fail("target/head mask_token_id mismatch")
    if int(head_config.get("vocab_size", 0)) != int(target_config.get("vocab_size", -1)):
        fail("target/head vocabulary size mismatch")
    if int(head_config.get("num_hidden_layers", 0)) != layers:
        fail("target/head layer count mismatch")
    if list(head_config.get("compress_ratios", [])[:layers]) != ratios:
        fail("target/head compress_ratios mismatch")
    if list(manifest.get("compress_ratios", [])) != ratios:
        fail("target/manifest compress_ratios mismatch")
    if int(manifest.get("num_hidden_layers", 0)) != layers:
        fail("target/manifest layer count mismatch")

    try:
        actual_target = target_identity(model, manifest.get("base_model", {}).get("id"))
    except ContractError as error:
        fail(str(error))
    expected_target = manifest.get("base_model", {})
    if args.production and expected_target.get("id") != BASE_MODEL_ID:
        fail(
            f"production head base_model.id must be {BASE_MODEL_ID!r}, "
            f"got {expected_target.get('id')!r}"
        )
    for key in ("config_file", "config_sha256", "weight_index_file", "weight_index_sha256"):
        if actual_target.get(key) != expected_target.get(key):
            fail(f"base model {key} mismatch")

    num_heads = int(target_config.get("num_attention_heads", 0))
    tp = args.tensor_parallel_size
    if tp <= 0 or num_heads % tp:
        fail(f"num_attention_heads={num_heads} is not divisible by TP={tp}")
    if args.production and tp != 8:
        fail("production Eco-Tech DSV4-Flash W8A8 deployment requires TP=8")
    if args.require_trained and manifest.get("artifact_role") != "trained":
        fail("production acceptance requires a trained diffusion head")

    try:
        schema, tensor_bytes = tensor_schema(head, head_index.get("weight_map", {}))
        selected = select_weight_map(head_index.get("weight_map", {}), ratios)
        validate_selected_layers(selected, ratios)
    except ContractError as error:
        fail(str(error))
    selected_names = {name for entries in selected.values() for name, _ in entries}
    if selected_names != set(head_index.get("weight_map", {})):
        fail("head index contains non-head or non-canonical tensors")
    if schema != manifest.get("tensor_schema"):
        fail("head tensor schema differs from the signed manifest")
    if len(schema) != int(manifest.get("tensor_count", -1)):
        fail("head tensor_count mismatch")
    indexed_bytes = int(head_index.get("metadata", {}).get("total_size", 0))
    if tensor_bytes != indexed_bytes or tensor_bytes != int(manifest.get("total_size", -1)):
        fail("head total_size mismatch")
    expected_shards = {item["name"]: item for item in manifest.get("shards", [])}
    actual_shards = set(head_index.get("weight_map", {}).values())
    if set(expected_shards) != actual_shards:
        fail("head shard set differs from manifest")
    for shard in sorted(actual_shards):
        path = head / shard
        item = expected_shards[shard]
        if path.stat().st_size != int(item.get("size", -1)):
            fail(f"head shard size mismatch: {shard}")
        if sha256_file(path) != item.get("sha256"):
            fail(f"head shard SHA256 mismatch: {shard}")

    precision = manifest.get("precision")
    quant = manifest.get("quantization", {})
    if quant.get("shares_target_scale_storage") is not False:
        fail("head manifest must state shares_target_scale_storage=false")
    if precision == "bf16":
        if not args.allow_bf16:
            fail("BF16 is an exchange format; calibrate the head to W8A8 for deployment")
        if quant.get("head_calibration_required") is not True:
            fail("BF16 head must require independent calibration")
    elif precision == "w8a8":
        quant_path = head / "quant_model_description.json"
        try:
            head_quant = read_json(quant_path)
        except ContractError as error:
            fail(str(error))
        schema_names = {item["name"] for item in schema}
        schema_by_name = {item["name"]: item for item in schema}
        missing_quant = []
        missing_scales = []
        for item in schema:
            name = item["name"]
            if not name.endswith(".weight"):
                continue
            parent = name.removesuffix(".weight")
            if parent.endswith(("q_norm", "kv_norm", "indexer.weights_proj")):
                continue
            runtime_name = runtime_tensor_name(name)
            if runtime_name not in head_quant:
                missing_quant.append(runtime_name)
                continue
            if str(head_quant[runtime_name]).startswith("W8A8"):
                for suffix in (".weight_scale", ".weight_offset"):
                    required = parent + suffix
                    if required not in schema_names:
                        missing_scales.append(required)
        if missing_quant:
            fail(f"head quant metadata is missing {missing_quant[:8]}")
        if missing_scales:
            fail(f"W8A8 head tensors are missing scales/offsets: {missing_scales[:8]}")
        required_w8a8 = []
        for layer, ratio in enumerate(ratios):
            # Eco-Tech's ModelSlim artifact intentionally leaves the split O
            # projections in floating point. Q/KV and the C4 indexer query are
            # the projections that must use the Ascend W8A8 path.
            roots = ["wq_a", "wq_b", "wkv"]
            if ratio == 4:
                roots.append("indexer.wq_b")
            required_w8a8.extend(
                f"layers.{layer}.attn.{root}.weight" for root in roots
            )
        wrong_w8a8 = []
        wrong_weight_dtype = []
        for name in required_w8a8:
            runtime_name = runtime_tensor_name(name)
            if not str(head_quant.get(runtime_name, "")).startswith("W8A8"):
                wrong_w8a8.append(runtime_name)
            elif schema_by_name.get(name, {}).get("dtype") != "I8":
                wrong_weight_dtype.append(name)
        if wrong_w8a8:
            fail(f"required Orthrus projections are not W8A8: {wrong_w8a8[:8]}")
        if wrong_weight_dtype:
            fail(f"W8A8 projection weights are not I8: {wrong_weight_dtype[:8]}")
        wrong_output_quant = []
        wrong_output_dtype = []
        for layer in range(len(ratios)):
            for root in ("wo_a", "wo_b"):
                name = f"layers.{layer}.attn.{root}.weight"
                runtime_name = runtime_tensor_name(name)
                mode = str(head_quant.get(runtime_name, ""))
                dtype = schema_by_name.get(name, {}).get("dtype")
                if mode.startswith("W8A8"):
                    if dtype != "I8":
                        wrong_output_dtype.append(name)
                elif mode == "FLOAT":
                    if dtype not in {"BF16", "F16", "F32"}:
                        wrong_output_dtype.append(name)
                else:
                    wrong_output_quant.append(runtime_name)
        if wrong_output_quant:
            fail(
                "Orthrus output projections have unsupported quantization: "
                f"{wrong_output_quant[:8]}"
            )
        if wrong_output_dtype:
            fail(f"Orthrus output projection dtype mismatch: {wrong_output_dtype[:8]}")
        source_sha = quant.get("source_description_sha256")
        target_quant = model / "quant_model_description.json"
        if not target_quant.is_file():
            fail("W8A8 target is missing quant_model_description.json")
        if quant.get("scale_origin") == "target-copy-initialization":
            if source_sha != sha256_file(target_quant):
                fail("initialized head quant source does not match target description")
        elif quant.get("scale_origin") == "head-calibration":
            if not quant.get("independently_calibrated"):
                fail("trained W8A8 head is not marked independently calibrated")
            if quant.get("source_description_sha256") != sha256_file(quant_path):
                fail("trained W8A8 quant description hash mismatch")
            calibration_path = head / "orthrus_head_calibration.json"
            if not calibration_path.is_file():
                fail("trained W8A8 head lacks sealed calibration manifest")
            if quant.get("calibration_manifest_sha256") != sha256_file(calibration_path):
                fail("trained W8A8 calibration manifest hash mismatch")
            calibration = read_json(calibration_path)
            required_calibration = (
                "tool",
                "tool_version",
                "dataset_sha256",
                "bf16_head_manifest_sha256",
                "command",
            )
            missing_calibration = [
                key for key in required_calibration if not calibration.get(key)
            ]
            if missing_calibration:
                fail(f"trained W8A8 calibration provenance is incomplete: {missing_calibration}")
            if quant.get("bf16_head_manifest_sha256") != calibration.get(
                "bf16_head_manifest_sha256"
            ):
                fail("trained W8A8 BF16 source hash mismatch")
            if not quant.get("calibration_manifest_sha256"):
                fail("trained W8A8 head lacks calibration provenance")
        else:
            fail(f"unsupported W8A8 scale_origin: {quant.get('scale_origin')!r}")
        if args.require_trained and not quant.get("independently_calibrated"):
            fail("production acceptance requires independently calibrated head scales")
    else:
        fail(f"unsupported head precision: {precision!r}")

    result = {
        "status": "ok",
        "target": str(model),
        "head": str(head),
        "format": FORMAT_NAME,
        "artifact_role": manifest.get("artifact_role"),
        "precision": precision,
        "layers": layers,
        "compress_ratios": {"swa": sum(r <= 1 for r in ratios), "c4": ratios.count(4), "c128": ratios.count(128)},
        "tensor_parallel_size": tp,
        "head_size_gib": round(tensor_bytes / 2**30, 3),
        "history_backend": HISTORY_BACKEND,
        "block_size": BLOCK_SIZE,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    }
    print(json.JSONEncoder(indent=2, sort_keys=True).encode(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
