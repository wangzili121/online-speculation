#!/usr/bin/env python3
"""Check a DSV4.1 checkpoint against the pinned PyPTO A3 loader without using NPU."""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


DEFAULT_MODEL = Path("/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8")


def check_model_config(model):
    config = json.loads((model / "config.json").read_text())
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "ascend":
        raise ValueError("PyPTO A3 expects an Ascend W8A8_DYNAMIC checkpoint")
    description = model / "quant_model_description.json"
    exported = json.loads(description.read_text()) if description.is_file() else {}
    declared = quant.get("model_quant_type")
    if declared not in (None, "W8A8_DYNAMIC") or exported.get("model_quant_type") not in (None, "W8A8_DYNAMIC"):
        raise ValueError("Checkpoint quantization metadata does not describe W8A8_DYNAMIC")
    if declared is None and exported.get("model_quant_type") != "W8A8_DYNAMIC":
        raise ValueError("Checkpoint has no W8A8_DYNAMIC declaration")
    return config


def inspect_checkpoint(model):
    """Read actual tensor headers before applying the paired loader's contract."""
    config = check_model_config(model)
    index_path = model / "quant_model_weights.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    headers, payload_sizes, issues, tensors = {}, {}, [], {}
    text = config["text_config"]

    def header(name):
        filename = index.get(name)
        if filename is None:
            return {}
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Invalid checkpoint shard name for {name}")
        path = (model / filename).resolve()
        if not path.is_relative_to(model.resolve()):
            raise ValueError(f"Checkpoint shard escapes model directory: {filename}")
        if filename not in headers:
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Truncated safetensors shard: {filename}")
                size, = struct.unpack("<Q", prefix)
                if not 0 < size <= min(64 * 1024 * 1024, path.stat().st_size - 8):
                    raise ValueError(f"Invalid safetensors header: {filename}")
                headers[filename] = json.loads(stream.read(size))
                payload_sizes[filename] = path.stat().st_size - 8 - size
        value = headers[filename].get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Indexed tensor is absent from {filename}: {name}")
        begin, end = value["data_offsets"]
        if not 0 <= begin <= end <= payload_sizes[filename]:
            raise ValueError(f"Truncated tensor payload: {name}")
        return value

    def expect(name, shape, dtype):
        value = header(name)
        actual = {key: value.get(key) for key in ("shape", "dtype")}
        tensors[name] = actual
        if actual != {"shape": list(shape), "dtype": dtype}:
            issues.append(dict(tensor=name, actual=actual, expected=dict(shape=list(shape), dtype=dtype)))

    for layer in range(text["num_hidden_layers"]):
        for projection, rows, cols in (
            ("wq_a", text["q_lora_rank"], text["hidden_size"]),
            ("wq_b", text["num_attention_heads"] * text["head_dim"], text["q_lora_rank"]),
            ("wkv", text["head_dim"], text["hidden_size"]),
        ):
            prefix = f"layers.{layer}.attn.{projection}"
            expect(prefix + ".weight", (rows, cols), "I8")
            for suffix in ("weight_scale", "weight_offset"):
                value = header(prefix + "." + suffix)
                if value.get("dtype") not in ("F32", "BF16") or value.get("shape") not in ([rows], [rows, 1]):
                    issues.append(dict(tensor=prefix + "." + suffix, actual=value,
                                       expected="per-output-channel FP32/BF16 metadata"))
    for layer in text["engram_layer_ids"]:
        prefix = f"layers.{layer}.engram.embed"
        value = header(prefix + ".weight")
        shape = value.get("shape") or []
        if len(shape) != 2 or shape[1] != 256 or value.get("dtype") != "I8":
            issues.append(dict(tensor=prefix + ".weight", actual=value, expected="INT8 [rows,256]"))
        elif shape[0] > 0:
            expect(prefix + ".scale", (shape[0], 8), "F32")
    quant = config.get("quantization_config") or {}
    if quant.get("model_quant_type") != "W8A8_DYNAMIC":
        issues.append(dict(field="config.json/quantization_config/model_quant_type", actual=quant.get("model_quant_type"),
                           expected="W8A8_DYNAMIC", explanation="Required by current serving loader routing"))
    identity = {}
    for name in ("config.json", index_path.name, "quant_model_description.json", "tokenizer.json"):
        path = model / name
        if path.is_file():
            identity[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(model=str(model.resolve()), weight_format="W8A8_DYNAMIC", identity=identity,
                header_shards_read=len(headers), projection_and_engram_layout_matches=not issues,
                layout_issues=issues, inspected_tensors=tensors,
                scope="header inspection; full paired-loader validation and model execution are separate")


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
    parser.add_argument(
        "--model", type=Path, default=Path(os.environ.get("TARGET_WEIGHTS") or DEFAULT_MODEL),
        help=f"Checkpoint directory. Defaults to TARGET_WEIGHTS or {DEFAULT_MODEL}",
    )
    parser.add_argument("--head", type=Path)
    parser.add_argument("--inspect", action="store_true", help="Inspect actual headers using the Python standard library")
    parser.add_argument("--output", type=Path, help="Save the layout report, including all mismatches")
    parser.add_argument("--lib", type=Path, help="Paired PyPTO Lib checkout. Defaults to PYPTO_LIB_ROOT.")
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.model)
        if args.output is not None:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.inspect:
            print(json.dumps(report, indent=2))
            return
        if report["layout_issues"]:
            first = report["layout_issues"][:4]
            raise ValueError(
                f"The checkpoint is W8A8_DYNAMIC, but {len(report['layout_issues'])} layout requirements "
                f"of the paired PyPTO loader differ. First mismatches: {json.dumps(first)}. "
                "Use --inspect --output checkpoint-layout.json to collect the complete tensor layout. "
                "The public checkpoint needs a loader/kernel adaptation."
            )
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
    except ModuleNotFoundError as exc:
        if exc.name in ("pypto_serving_a3", "torch", "pypto", "safetensors"):
            parser.exit(
                1, f"Missing environment package: {exc.name}. Run this check in the prepared "
                "PyPTO container with: bash online-speculation/scripts/dsv41_pypto_env.sh preflight. "
                "For a checkpoint-header report requiring only Python, use --inspect.\n"
            )
        parser.exit(1, f"PyPTO checkpoint preflight failed: {exc}\n")
    except (ImportError, KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"PyPTO checkpoint preflight failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
