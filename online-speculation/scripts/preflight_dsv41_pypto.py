#!/usr/bin/env python3
"""Check a DSV4.1 checkpoint against the pinned PyPTO A3 loader without using NPU."""

import argparse
import json
from pathlib import Path


DEFAULT_MODEL = Path("/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8")


def check_model_config(model):
    config = json.loads((model / "config.json").read_text())
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "ascend":
        raise ValueError("PyPTO A3 expects an Ascend W8A8_DYNAMIC checkpoint")
    if quant.get("model_quant_type") == "W8A8_DYNAMIC":
        return
    description = model / "quant_model_description.json"
    if description.is_file():
        exported = json.loads(description.read_text()).get("model_quant_type")
        if exported == "W8A8_DYNAMIC":
            raise ValueError(
                "This directory declares W8A8_DYNAMIC only in quant_model_description.json. "
                "The paired PyPTO loader expects an A3 export with W8A8_DYNAMIC in config.json, "
                "INT8 Q/KV projections and FP32 Engram scales. A config-only change cannot "
                "convert the underlying weights."
            )
    raise ValueError("config.json does not declare the W8A8_DYNAMIC layout required by PyPTO A3")


def validate_engram(checkpoint):
    table_bytes = 0
    for layer in (1, 14):
        prefix = f"layers.{layer}.engram.embed"
        values = checkpoint.tensor_metadata(prefix + ".weight")
        scales = checkpoint.tensor_metadata(prefix + ".scale")
        shape = values.get("shape", [])
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != 256 or values.get("dtype") != "I8":
            raise ValueError(f"{prefix}.weight must be an INT8 table with 256 columns")
        if scales.get("shape") != [shape[0], 8] or scales.get("dtype") != "F32":
            raise ValueError(
                f"{prefix}.scale expected FP32 with eight scales per row, got "
                f"{scales.get('dtype')} {scales.get('shape')}"
            )
        table_bytes += shape[0] * (256 + 8 * 4)
    rotation = checkpoint.directory / "optional/quarot.safetensors"
    if not rotation.is_file():
        raise ValueError(f"Engram rotation file is missing: {rotation}")
    return table_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--lib", type=Path, help="Paired PyPTO Lib checkout. Defaults to PYPTO_LIB_ROOT.")
    args = parser.parse_args()
    try:
        check_model_config(args.model)
        from pypto_serving_a3.source import KernelSource
        from pypto_serving_a3.weights import A3Checkpoint

        source = KernelSource.resolve(args.lib)
        checkpoint = A3Checkpoint(args.model, source)
        result = {
            "model": str(checkpoint.directory),
            "lib_commit": source.commit,
            "engram_host_table_bytes": validate_engram(checkpoint),
            **checkpoint.validate(),
        }
        if args.head is not None:
            from blockspec.vllm_ascend.dsv41_contract import validate_head

            manifest = json.loads((args.head / "manifest.json").read_text())
            validated = validate_head(args.model, args.head, mask_token_id=manifest["mask_token_id"])
            result["head_artifact_role"] = validated["artifact_role"]
            result["head"] = str(args.head.resolve())
    except (ImportError, KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"PyPTO checkpoint preflight failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
