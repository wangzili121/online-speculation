#!/usr/bin/env python3
"""Package initialized or offline-trained weights for the Orthrus V4.1 loader."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from blockspec.vllm_ascend.dsv41_contract import (
    BLOCK_SIZE,
    HEAD_FORMAT,
    V41Topology,
    head_manifest,
    head_shapes,
    validate_head,
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
    # vLLM 0.32 applies its generic DeepSeek-MTP compatibility override to
    # method=dflash and reads this top-level field as ``n_predict``.  Orthrus
    # emits all 31 candidates in one parallel model call.  The nested text
    # config keeps the target checkpoint's three DSpark stages unchanged.
    config["num_nextn_predict_layers"] = BLOCK_SIZE - 1
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
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--from-target", action="store_true", help="initialize from the W4A8 target attention")
    inputs.add_argument(
        "--from-trained-head", type=Path, metavar="SAFETENSORS",
        help="pack offline-trained BF16/FP32 tensors in canonical layers.N.attn.* layout",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    base = read_json(args.model / "config.json")
    topology = V41Topology.from_config(base)
    shapes = head_shapes(base, topology)
    manifest = head_manifest(
        args.model,
        mask_token_id=args.mask_token_id,
        artifact_role="trained" if args.from_trained_head else "initialized-plumbing",
    )
    source = None
    if args.from_target:
        from blockspec.vllm_ascend.dsv41_last_layer import W4A8LastLayerWeights

        source = W4A8LastLayerWeights(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        trained = None
        if args.from_trained_head:
            trained = stack.enter_context(safe_open(args.from_trained_head, framework="pt", device="cpu"))
            if set(trained.keys()) != shapes.keys():
                raise ValueError("trained V4.1 head tensor names do not match the topology")

        temporary = Path(stack.enter_context(TemporaryDirectory(
            prefix=f".{args.output.name}.", dir=args.output.parent,
        )))
        write_json(temporary / "config.json", _head_config(base, args.mask_token_id))
        write_json(temporary / "manifest.json", manifest)
        weight_map: dict[str, str] = {}
        total_size = 0
        for layer in range(len(topology.layers)):
            prefix = f"layers.{layer}.attn."
            tensors = {}
            for name, shape in shapes.items():
                if not name.startswith(prefix):
                    continue
                dtype = torch.float32 if name.endswith(".attn_sink") else torch.bfloat16
                if trained is not None:
                    value = trained.get_tensor(name)
                    if value.dtype not in (torch.bfloat16, torch.float32):
                        raise ValueError(f"expected BF16 or FP32 for {name}, got {value.dtype}")
                    value = value.to(dtype)
                elif source is None:
                    value = torch.zeros(shape, dtype=dtype)
                    if name.endswith(("q_norm.weight", "kv_norm.weight")):
                        value.fill_(1)
                elif name.endswith(".attn_sink"):
                    value = source.load([name])[name].to(dtype)
                else:
                    value = source.linear_weight(name, "cpu").to(dtype)
                if tuple(value.shape) != shape or not value.isfinite().all():
                    raise ValueError(f"wrong shape or nonfinite values for {name}: {tuple(value.shape)}")
                tensors[name] = value
                total_size += value.numel() * value.element_size()
            shard = f"model-{layer + 1:05d}-of-{len(topology.layers):05d}.safetensors"
            save_file(tensors, temporary / shard)
            weight_map.update(dict.fromkeys(tensors, shard))
        write_json(
            temporary / "model.safetensors.index.json",
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
        validate_head(args.model, temporary, mask_token_id=args.mask_token_id)
        temporary.rename(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
