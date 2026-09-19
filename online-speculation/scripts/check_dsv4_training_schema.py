#!/usr/bin/env python3
"""Compare the BF16 trainer's meta head schema with an existing v2 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from blockspec.dsv4_training import OrthrusDSV4TrainingModel
from blockspec.reporting import dumps
from blockspec.vllm_ascend.head_contract import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--mask-token-id", type=int)
    args = parser.parse_args()
    from transformers import AutoConfig, DeepseekV4ForCausalLM

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = getattr(config, "dspark_noise_token_id", None)
    if mask_token_id is None:
        raise SystemExit("mask token unavailable; pass --mask-token-id")
    with torch.device("meta"):
        target = DeepseekV4ForCausalLM(config)
        model = OrthrusDSV4TrainingModel(
            target, block_size=32, mask_token_id=int(mask_token_id)
        )
    expected = {
        f"layers.{layer}.attn.{name}": list(value.shape)
        for layer, state in enumerate(model.export_state())
        for name, value in state.items()
    }
    manifest = read_json(args.head / "orthrus_head_manifest.json")
    actual = {entry["name"]: entry["shape"] for entry in manifest["tensor_schema"]}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    quant_auxiliary = sorted(
        name
        for name in extra
        if (
            (name.endswith(".weight_scale") and name.removesuffix("_scale") in expected)
            or (name.endswith(".weight_offset") and name.removesuffix("_offset") in expected)
        )
    )
    unsupported_extra = sorted(set(extra) - set(quant_auxiliary))
    mismatched = sorted(
        name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
    )
    result = {
        "layers": len(model.head),
        "compress_ratios": model.compress_ratios,
        "expected_tensors": len(expected),
        "artifact_tensors": len(actual),
        "missing": missing,
        "quant_auxiliary_tensors": len(quant_auxiliary),
        "unsupported_extra": unsupported_extra,
        "shape_mismatches": {
            name: {"trainer": expected[name], "artifact": actual[name]}
            for name in mismatched
        },
        "pass": not missing and not unsupported_extra and not mismatched,
    }
    print(dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
