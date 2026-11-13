#!/usr/bin/env python3
"""Create a loadable BF16 Orthrus-V4.1 head artifact.

The initialized-plumbing artifact is deliberately not a useful drafter: it
exists to validate model construction, TP sharding and the complete inference
lifecycle before trained diffusion weights arrive.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file

from blockspec.vllm_ascend.dsv41_contract import (
    HEAD_FORMAT,
    V41Topology,
    head_manifest,
    head_shapes,
)
from blockspec.vllm_ascend.head_contract import read_json, write_json


def _head_config(base: dict, mask_token_id: int) -> dict:
    config = dict(base)
    text = dict(config["text_config"])
    config["text_config"] = text
    config["architectures"] = ["OrthrusDSV41DraftModel"]
    config["orthrus_head_format"] = HEAD_FORMAT
    config["mask_token_id"] = mask_token_id
    config["has_own_embed_tokens"] = False
    config["has_own_lm_head"] = False
    dflash = dict(config.get("dflash_config") or {})
    dflash.update(mask_token_id=mask_token_id, causal=False, sample_from_anchor=False)
    config["dflash_config"] = dflash
    # The exchange artifact is BF16 even if the target is W4A8/W8A8.  Draft
    # quantization needs independent calibration and is a later deployment step.
    config.pop("quantization_config", None)
    text.pop("quantization_config", None)
    config["torch_dtype"] = text["torch_dtype"] = "bfloat16"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-token-id", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    base = read_json(args.model / "config.json")
    topology = V41Topology.from_config(base)
    shapes = head_shapes(base, topology)
    manifest = head_manifest(
        args.model,
        mask_token_id=args.mask_token_id,
        artifact_role="initialized-plumbing",
    )
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    write_json(temporary / "config.json", _head_config(base, args.mask_token_id))
    write_json(temporary / "manifest.json", manifest)

    weight_map: dict[str, str] = {}
    total_size = 0
    for layer in range(len(topology.layers)):
        prefix = f"layers.{layer}.attn."
        layer_shapes = {name: shape for name, shape in shapes.items() if name.startswith(prefix)}
        tensors = {}
        for name, shape in layer_shapes.items():
            dtype = torch.float32 if name.endswith("attn_sink") else torch.bfloat16
            value = torch.zeros(shape, dtype=dtype)
            if name.endswith(("q_norm.weight", "kv_norm.weight")):
                value.fill_(1)
            tensors[name] = value
            total_size += value.numel() * value.element_size()
        shard = f"model-{layer + 1:05d}-of-{len(topology.layers):05d}.safetensors"
        save_file(tensors, temporary / shard)
        weight_map.update(dict.fromkeys(tensors, shard))
    write_json(
        temporary / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    temporary.rename(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
