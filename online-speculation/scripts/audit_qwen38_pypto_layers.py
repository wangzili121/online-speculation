#!/usr/bin/env python3
"""Replay real PyPTO draft layer inputs through the existing HF dual-view reference on CPU."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.cpu_threads < 1:
        parser.error("new output and a positive CPU thread count required")
    import torch
    import transformers
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.hf_reference import layer_reference
    from blockspec.pypto_qwen38.inspection import head_digest

    torch.set_num_threads(args.cpu_threads)
    report = dict(passed=False, scope="same_input_real_weight_hf_draft_layers", npu_execution=False,
                  transformers=transformers.__version__, torch_threads=args.cpu_threads,
                  layers=[], relative_l2_limit=.02)
    try:
        head = HeadSource(args.model, args.head_dir)
        parent = json.loads((args.capture_dir.parent / "result.json").read_text())
        if parent["head_manifest"] != json.loads((args.head_dir / "manifest.json").read_text()):
            raise ValueError("capture belongs to another head manifest")
        if parent["head_sha256"] != head_digest(head):
            raise ValueError("capture belongs to different draft parameters")
        config = Qwen3_5TextConfig.from_dict(json.loads((args.model / "config.json").read_text())["text_config"])
        config._attn_implementation = "sdpa"
        history = load_file(args.capture_dir / "history.safetensors")
        index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]

        def read(name):
            filename = Path(index[name])
            if filename.is_absolute() or len(filename.parts) != 1:
                raise ValueError("checkpoint shard names must be local basenames")
            with safe_open(args.model / filename, framework="pt", device="cpu") as file:
                return file.get_tensor(name)

        for layer in range(64):
            prefix = f"model.language_model.layers.{layer}."
            shared = {name: read(prefix + name) for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                      "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")}
            kind = config.layer_types[layer]
            state = ((history[f"key.{layer // 4}"], history[f"value.{layer // 4}"]) if kind == "full_attention"
                     else (history["conv"][layer - layer // 4], history["recurrent"][layer - layer // 4]))
            mixer = head.layer(layer)
            for backend in parent["signatures"]:
                capture = load_file(args.capture_dir / backend / f"layer{layer}.safetensors")
                expected = layer_reference(config, layer, capture["input"], shared, mixer, state,
                                           (history["cos"], history["sin"]))
                actual, expected = capture["output"].float(), expected.float()
                error = actual - expected
                relative = float(error.norm() / expected.norm().clamp_min(1e-12))
                row = dict(layer=layer, kind=kind, backend=backend,
                           finite=bool(actual.isfinite().all() and expected.isfinite().all()),
                           max_abs=float(error.abs().max()), relative_l2=relative,
                           exact_fraction=float((actual == expected).float().mean()))
                row["passed"] = row["finite"] and relative <= report["relative_l2_limit"]
                report["layers"].append(row)
                print(json.dumps(row), flush=True)
            args.output.with_suffix(".progress.json").write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = all(row["passed"] for row in report["layers"])
        report["head_manifest_sha256"] = hashlib.sha256((args.head_dir / "manifest.json").read_bytes()).hexdigest()
        if not report["passed"]:
            raise AssertionError("one or more real-weight layer checks exceeded the frozen relative-L2 gate")
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
