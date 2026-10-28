#!/usr/bin/env python3
"""Create a linked, layer-truncated DSV4 model for small-card integration tests."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


LAYER_NAME = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
SPEC_PREFIXES = ("mtp.", "model.mtp.")
GENERATED_FILES = {
    "config.json",
    "quant_model_description.json",
    "quant_model_weights.safetensors.index.json",
    "smoke_model_manifest.json",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def keep_weight(name: str, layers: int) -> bool:
    match = LAYER_NAME.match(name)
    if match is not None:
        return int(match.group(1)) < layers
    return not name.startswith(SPEC_PREFIXES)


def link(source: Path, output: Path) -> None:
    output.symlink_to(os.path.relpath(source, output.parent))


def selected_bytes(model: Path, weight_map: dict[str, str]) -> int:
    from blockspec.vllm_ascend.head_contract import read_safetensors_header

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    total = 0
    for shard, names in by_shard.items():
        header = read_safetensors_header(model / shard)
        for name in names:
            if name not in header:
                raise SystemExit(f"weight index/header mismatch: {name} in {shard}")
            start, end = header[name]["data_offsets"]
            total += end - start
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="rewrite filtered safetensors shards for loaders that scan shard contents",
    )
    args = parser.parse_args()

    model = args.model.resolve()
    output = args.output.resolve()
    config = read_json(model / "config.json")
    source_layers = int(config.get("num_hidden_layers", 0))
    layers = args.layers
    if layers <= 0 or layers > source_layers:
        raise SystemExit(f"--layers must be in [1, {source_layers}]")
    ratios = [int(value) for value in config.get("compress_ratios", [])[:layers]]
    if len(ratios) != layers:
        raise SystemExit("source model has no compress_ratio for every selected layer")
    if not {4, 128}.issubset(ratios):
        raise SystemExit("smoke model must retain at least one C4 and one C128 layer")

    index_path = model / "quant_model_weights.safetensors.index.json"
    source_index = read_json(index_path)
    source_map = source_index.get("weight_map", {})
    weight_map = {
        name: shard for name, shard in source_map.items() if keep_weight(name, layers)
    }
    if not weight_map:
        raise SystemExit("no weights selected")
    shards = sorted(set(weight_map.values()))

    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.materialize:
        from safetensors import safe_open
        from safetensors.torch import save_file

        names_by_shard: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            names_by_shard.setdefault(shard, []).append(name)
        for index, shard in enumerate(shards, 1):
            with safe_open(model / shard, framework="pt", device="cpu") as handle:
                tensors = {
                    name: handle.get_tensor(name)
                    for name in names_by_shard[shard]
                }
            save_file(tensors, output / shard, metadata={"format": "pt"})
            print(f"materialized shard {index}/{len(shards)}: {shard}")
    else:
        for shard in shards:
            link(model / shard, output / shard)
    for path in model.iterdir():
        if not path.is_file() or path.name in GENERATED_FILES or path.suffix == ".safetensors":
            continue
        link(path, output / path.name)

    smoke_config = dict(config)
    smoke_config["num_hidden_layers"] = layers
    smoke_config["compress_ratios"] = ratios
    smoke_config["dspark_target_layer_ids"] = list(
        range(max(0, layers - 3), layers)
    )
    write_json(output / "config.json", smoke_config)

    total_size = selected_bytes(model, weight_map)
    write_json(
        output / "quant_model_weights.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )

    quant_path = model / "quant_model_description.json"
    if quant_path.is_file():
        source_quant = read_json(quant_path)
        quant = {
            name: value
            for name, value in source_quant.items()
            if name in weight_map
            or (name not in source_map and not name.startswith(SPEC_PREFIXES))
        }
        write_json(output / "quant_model_description.json", quant)

    write_json(
        output / "smoke_model_manifest.json",
        {
            "purpose": "non-production Orthrus DSV4 integration smoke",
            "source": str(model),
            "source_layers": source_layers,
            "selected_layers": layers,
            "compress_ratios": ratios,
            "weight_tensors": len(weight_map),
            "referenced_shards": len(shards),
            "selected_tensor_bytes": total_size,
            "materialized": args.materialize,
        },
    )
    storage = "materialized" if args.materialize else "linked"
    print(
        f"created {layers}-layer DSV4 smoke model with ratios {ratios}; "
        f"{total_size / 2**30:.2f} GiB selected in {len(shards)} {storage} shards"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
