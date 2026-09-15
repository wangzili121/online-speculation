#!/usr/bin/env python3
"""Export an initialized or trained ``orthrus-dsv4-head-v2`` artifact."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.vllm_ascend.head_contract import (  # noqa: E402
    BASE_MODEL_ID,
    BLOCK_SIZE,
    FORMAT_NAME,
    FORMAT_VERSION,
    GLOBAL_QUANT_KEYS,
    HISTORY_BACKEND,
    NUM_SPECULATIVE_TOKENS,
    ContractError,
    find_weight_index,
    is_head_tensor,
    read_json,
    read_safetensors_header,
    canonical_tensor_name,
    normalize_tensor_name,
    runtime_tensor_name,
    select_weight_map,
    sha256_file,
    shard_manifest,
    target_identity,
    tensor_schema,
    validate_selected_layers,
    write_json,
)


def _source_name_map(
    weight_map: dict[str, str], ratios: list[int]
) -> dict[str, tuple[str, str]]:
    sources: dict[str, tuple[str, str]] = {}
    for source_name, shard in weight_map.items():
        if not is_head_tensor(source_name, ratios):
            continue
        normalized = normalize_tensor_name(source_name)
        assert normalized is not None
        canonical = canonical_tensor_name(*normalized)
        if canonical in sources:
            raise ContractError(f"duplicate source tensors normalize to {canonical}")
        sources[canonical] = (source_name, shard)
    return sources


def _source_tensor_bytes(
    model: Path,
    selected: dict[int, list[tuple[str, str]]],
    sources: dict[str, tuple[str, str]],
) -> int:
    names_by_shard: dict[str, set[str]] = defaultdict(set)
    for entries in selected.values():
        for name, shard in entries:
            names_by_shard[shard].add(name)
    total = 0
    for shard, canonical_names in names_by_shard.items():
        header = read_safetensors_header(model / shard)
        for canonical in canonical_names:
            source_name, source_shard = sources[canonical]
            if source_shard != shard or source_name not in header:
                raise ContractError(f"index/header mismatch for source tensor {source_name}")
            start, end = header[source_name]["data_offsets"]
            total += end - start
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument(
        "--artifact-role",
        choices=("initialized-plumbing", "trained"),
        default="initialized-plumbing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the selected tensor size without writing",
    )
    args = parser.parse_args()

    model = args.model.resolve()
    output = args.output.resolve()
    try:
        config = read_json(model / "config.json")
        index_path = find_weight_index(model)
        index = read_json(index_path)
    except ContractError as error:
        raise SystemExit(str(error)) from error

    if config.get("model_type") != "deepseek_v4":
        raise SystemExit(f"expected model_type=deepseek_v4, got {config.get('model_type')!r}")
    layers = int(config.get("num_hidden_layers", 0))
    ratios = [int(value) for value in config.get("compress_ratios", [])[:layers]]
    if layers <= 0 or len(ratios) != layers:
        raise SystemExit("config must contain one compress_ratio per hidden layer")
    if any(value not in (0, 1, 4, 128) for value in ratios):
        raise SystemExit(f"unsupported DSV4 compress_ratios: {ratios}")
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = config.get("mask_token_id", config.get("dspark_noise_token_id"))
    if mask_token_id is None or int(mask_token_id) < 0:
        raise SystemExit("mask token is required; pass --mask-token-id")
    mask_token_id = int(mask_token_id)

    weight_map = index.get("weight_map", {})
    selected = select_weight_map(weight_map, ratios)
    try:
        validate_selected_layers(selected, ratios)
        sources = _source_name_map(weight_map, ratios)
    except ContractError as error:
        raise SystemExit(str(error)) from error
    tensor_count = sum(map(len, selected.values()))
    selected_bytes = _source_tensor_bytes(model, selected, sources)
    print(
        f"validated {layers} layers ({ratios.count(0) + ratios.count(1)} SWA, "
        f"{ratios.count(4)} C4, {ratios.count(128)} C128), {tensor_count} tensors; "
        f"head payload is {selected_bytes / 2**30:.2f} GiB"
    )
    if args.dry_run:
        return 0

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise SystemExit("export requires safetensors and torch") from error

    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    out_weight_map: dict[str, str] = {}
    for layer in range(layers):
        tensors = {}
        by_shard: dict[str, list[str]] = defaultdict(list)
        for canonical, shard in selected[layer]:
            by_shard[shard].append(canonical)
        for shard, canonical_names in by_shard.items():
            with safe_open(model / shard, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for canonical in canonical_names:
                    source_name, source_shard = sources[canonical]
                    if source_shard != shard or source_name not in available:
                        raise SystemExit(
                            f"index/header mismatch for source tensor {source_name}"
                        )
                    tensors[canonical] = handle.get_tensor(source_name)
        filename = f"orthrus-head-{layer + 1:05d}-of-{layers:05d}.safetensors"
        save_file(tensors, output / filename, metadata={"format": "pt"})
        out_weight_map.update({name: filename for name in tensors})
        print(f"wrote layer {layer:02d}: {len(tensors)} tensors")

    head_config = dict(config)
    head_config.update(
        {
            "architectures": ["OrthrusDSV4DraftModel"],
            "compress_ratios": ratios,
            "mask_token_id": mask_token_id,
            # vLLM 0.26 names the parallel-drafting placeholder ``pard_token``.
            # Keep the semantic contract in mask_token_id and emit the runtime
            # alias with the exact same value.
            "pard_token": mask_token_id,
            "orthrus_block_size": BLOCK_SIZE,
            "orthrus_head_format": FORMAT_NAME,
            "orthrus_head_format_version": FORMAT_VERSION,
            "orthrus_shared_backbone": True,
            "parallel_drafting": True,
            "sample_from_anchor": False,
        }
    )
    write_json(output / "config.json", head_config)

    quant_path = model / "quant_model_description.json"
    precision = "bf16"
    quant_sha = None
    if quant_path.is_file():
        precision = "w8a8"
        source_quant = read_json(quant_path)
        selected_names = set(out_weight_map)
        head_quant = {}
        for name, value in source_quant.items():
            if name in GLOBAL_QUANT_KEYS:
                continue
            normalized = normalize_tensor_name(name)
            if normalized is None:
                continue
            canonical = canonical_tensor_name(*normalized)
            if canonical in selected_names:
                head_quant[runtime_tensor_name(canonical)] = value
        for key in GLOBAL_QUANT_KEYS:
            if key in source_quant:
                head_quant[key] = source_quant[key]
        write_json(output / "quant_model_description.json", head_quant)
        quant_sha = sha256_file(quant_path)

    index_name = "quant_model_weights.safetensors.index.json"
    out_index = {"metadata": {"total_size": selected_bytes}, "weight_map": out_weight_map}
    write_json(output / index_name, out_index)
    schema, total_size = tensor_schema(output, out_weight_map)
    if total_size != selected_bytes:
        raise SystemExit(
            f"exported tensor bytes changed: selected={selected_bytes}, output={total_size}"
        )
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "artifact_role": args.artifact_role,
        "precision": precision,
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
        "shards": shard_manifest(output, out_weight_map.values()),
        "quantization": {
            "format": "modelslim-ascend-w8a8" if precision == "w8a8" else "bf16-exchange",
            "source_description_sha256": quant_sha,
            "head_calibration_required": True,
            "independently_calibrated": False,
            "scale_origin": (
                "target-copy-initialization" if precision == "w8a8" else "unquantized"
            ),
            "shares_target_scale_storage": False,
        },
    }
    write_json(output / "orthrus_head_manifest.json", manifest)
    print(f"head size: {total_size / 2**30:.2f} GiB in {layers} shards")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
