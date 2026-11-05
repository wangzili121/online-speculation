#!/usr/bin/env python3
"""Compare all rank-local head slices with independent full-tensor loading."""

import argparse
import hashlib
import json
from pathlib import Path


def reference_layer(head, layer):
    from safetensors import safe_open
    from blockspec.pypto_qwen38.head import merge_parameter, tensor_schema

    weights = {}
    for name in tensor_schema(layer):
        if head.directory:
            parts = []
            for rank in range(head.tp):
                with safe_open(head.directory / f"rank{rank}.safetensors", framework="pt", device="cpu") as file:
                    parts.append(file.get_tensor(f"layers.{layer}.attention.draft.{name}"))
        else:
            kind = "self_attn" if layer % 4 == 3 else "linear_attn"
            key = f"model.language_model.layers.{layer}.{kind}.{name}"
            filename = Path(head.index[key])
            if filename.is_absolute() or len(filename.parts) != 1:
                raise ValueError("checkpoint shard names must be local basenames")
            with safe_open(head.model_root / filename, framework="pt", device="cpu") as file:
                parts = [file.get_tensor(key)]
        weights[name] = merge_parameter(layer, name, parts)
    return weights


def digest(tensor):
    import torch
    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path)
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    import torch
    from blockspec.pypto_qwen38.baseline import native_module, ONESERVING_COMMIT
    from blockspec.pypto_qwen38.head import HeadSource, merge_parameter

    report = dict(passed=False, scope="real_head_rank_local_loading", npu_execution=False,
                  full_tp_inference=False, oneserving=ONESERVING_COMMIT, runtime_tp=args.tp, layers=[])
    try:
        torch.set_num_threads(4)
        native_module(args.source, "weight").verify_model_identity(args.model)
        source = HeadSource(args.model, args.head_dir)
        local = [HeadSource(args.model, args.head_dir, runtime_tp=args.tp, runtime_rank=rank)
                 for rank in range(args.tp)]
        report.update(source_tp=source.tp, initialization="trained" if args.head_dir else "AR_copy",
                      layouts=[dict(rank=h.layout.rank, heads=h.layout.head_ranges,
                                    history_shapes=h.layout.history_shapes(slots=1, pages=1)) for h in local])
        byte_counts = [0] * args.tp
        for layer in range(64):
            expected = reference_layer(source, layer)
            weights = [head.layer(layer) for head in local]
            for name, full in expected.items():
                actual = merge_parameter(layer, name, [part[name] for part in weights])
                if actual.dtype != full.dtype or not torch.equal(actual, full):
                    raise AssertionError(f"local head slices changed {layer}.{name}")
                for rank, part in enumerate(weights):
                    byte_counts[rank] += part[name].numel() * part[name].element_size()
            row = dict(layer=layer, parameter_count=len(expected),
                       full_sha256={name:digest(value) for name, value in expected.items()},
                       local_sha256=[{name:digest(value) for name,value in part.items()} for part in weights])
            report["layers"].append(row)
            print(json.dumps(dict(layer=layer, passed=True)), flush=True)
        report.update(rank_parameter_bytes=byte_counts, passed=True)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
