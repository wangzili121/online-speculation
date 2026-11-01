#!/usr/bin/env python3
"""Materialize the first GLM-Flash layers at original width for NPU integration tests."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil


LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.")


def keep_weight(name, layers):
    match = LAYER.match(name)
    if match:
        return int(match[1]) < layers
    return not any(part == "mtp" for part in name.split("."))


def truncated_config(source, layers):
    config = copy.deepcopy(source)
    text = config["text_config"]
    if text.get("model_type") != "glm5_next_text" or not 4 <= layers <= text["num_hidden_layers"]:
        raise ValueError("GLM-Flash prefix of at least four layers required")
    for name in ("layer_types", "indexer_types", "mlp_layer_types"):
        text[name] = text[name][:layers]
    text["num_hidden_layers"] = layers
    text["num_nextn_predict_layers"] = 0
    for name in ("kda_layers", "full_attn_layers"):
        text["linear_attn_config"][name] = [i for i in text["linear_attn_config"][name] if i < layers]
    if set(text["layer_types"]) != {"linear_attention", "deepseek_sparse_attention"}:
        raise ValueError("retain both KDA and MLA")
    return config


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=5)
    args = parser.parse_args()
    source, output = args.model.resolve(), args.output.resolve()
    config = truncated_config(json.loads((source / "config.json").read_text()), args.layers)
    index_name = "quant_model_weights.safetensors.index.json"
    source_index = json.loads((source / index_name).read_text())
    weight_map = {name: shard for name, shard in source_index["weight_map"].items()
                  if keep_weight(name, args.layers)}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite a nonempty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    from safetensors import safe_open
    from safetensors.torch import save_file

    grouped = {}
    for name, shard in weight_map.items():
        grouped.setdefault(shard, []).append(name)
    total = 0
    for number, (shard, names) in enumerate(sorted(grouped.items()), 1):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in names}
            total += sum(value.numel() * value.element_size() for value in tensors.values())
            save_file(tensors, output / shard, metadata={"format": "pt"})
        print(f"materialized {number}/{len(grouped)}: {shard}", flush=True)
    generated = {"config.json", index_name, "quant_model_description.json", "smoke_model_manifest.json"}
    shards = set(source_index["weight_map"].values())
    for path in source.iterdir():
        if path.is_file() and path.name not in generated | shards:
            # 辅助文件复制到独立目录，容器不需要再挂载完整源模型。
            shutil.copy2(path, output / path.name)
    description = json.loads((source / "quant_model_description.json").read_text())
    write_json(output / "quant_model_description.json",
               {name: value for name, value in description.items() if keep_weight(name, args.layers)})
    write_json(output / "config.json", config)
    write_json(output / index_name, dict(metadata=dict(total_size=total), weight_map=weight_map))
    write_json(output / "smoke_model_manifest.json", dict(
        purpose="non-production layer-truncated real-weight GLM integration test",
        source=str(source), selected_layers=args.layers, selected_tensor_bytes=total,
        source_index_sha256=hashlib.sha256((source / index_name).read_bytes()).hexdigest(),
        status="complete"))
    print(json.dumps(dict(output=str(output), selected_gib=total / 2**30)), flush=True)


if __name__ == "__main__":
    main()
