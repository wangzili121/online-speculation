#!/usr/bin/env python3
"""Inspect and replay captured native C4 calls without loading model weights."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def tensor_summary(value):
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    result = dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                  sha256=hashlib.sha256(raw).hexdigest())
    if value.numel() <= 256:
        result["values"] = value.tolist()
    return result


def describe(value):
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    return value


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, device) for item in value)
    return value


def difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(compatible=False, shapes=[list(left.shape), list(right.shape)],
                    dtypes=[str(left.dtype), str(right.dtype)])
    error = (left.double() - right.double()).abs()
    return dict(equal=torch.equal(left, right), different=torch.count_nonzero(left != right).item(),
                max_abs=error.max().item() if error.numel() else 0)


def reverse_physical_pages(kwargs):
    result = dict(kwargs)
    for kind in ("ori", "cmp"):
        cache, table = kwargs[kind + "_kv"], kwargs[kind + "_block_table"]
        result[kind + "_kv"] = cache.flip(0)
        result[kind + "_block_table"] = torch.where(table >= 0, cache.shape[0] - 1 - table, table)
    return result


def canonical_physical_pages(kwargs):
    result = dict(kwargs)
    for kind in ("ori", "cmp"):
        cache, table = kwargs[kind + "_kv"], kwargs[kind + "_block_table"]
        mapping, logical = {}, []
        for old in table.cpu().flatten().tolist():
            if old < 0:
                logical.append(old)
                continue
            if old not in mapping:
                mapping[old] = len(mapping)
            logical.append(mapping[old])
        packed = torch.zeros_like(cache)
        old_pages = torch.tensor(list(mapping), device=cache.device, dtype=torch.long)
        packed[:len(mapping)] = cache[old_pages]
        result[kind + "_kv"] = packed
        result[kind + "_block_table"] = torch.tensor(logical, device=table.device,
                                                     dtype=table.dtype).reshape(table.shape)
    return result


def compact_physical_pages(kwargs):
    result = dict(kwargs)
    for kind in ("ori", "cmp"):
        cache, table = kwargs[kind + "_kv"], kwargs[kind + "_block_table"]
        old_pages = table[table >= 0].unique(sorted=True).long()
        new_table = table.clone()
        for new, old in enumerate(old_pages.tolist()):
            new_table[table == old] = new
        result[kind + "_kv"] = cache.index_select(0, old_pages)
        result[kind + "_block_table"] = new_table
    return result


def reference_value(snapshot, row, head, column):
    query, kwargs = snapshot["args"][0], snapshot["kwargs"]
    starts = kwargs["cu_seqlens_q"].tolist()
    request = next(i for i, end in enumerate(starts[1:]) if row < end)
    position = int(kwargs["seqused_kv"][request]) - (starts[request + 1] - starts[request]) + row - starts[request]

    def gather(kind, indices):
        cache, table = kwargs[kind + "_kv"], kwargs[kind + "_block_table"]
        pages = table[request, indices // cache.shape[1]].long()
        return cache[pages, indices % cache.shape[1], 0].double()

    original = torch.arange(max(0, position - kwargs["ori_win_left"]), position + 1)
    compressed = kwargs["cmp_sparse_indices"][row, 0].long()
    compressed = compressed[(compressed >= 0) & (compressed < (position + 1) // kwargs["cmp_ratio"])]
    kv = torch.cat((gather("ori", original), gather("cmp", compressed)))
    # The kernel receives softmax_scale as a float attribute.
    scale = float(torch.tensor(kwargs["softmax_scale"], dtype=torch.float32))
    scores = kv @ query[row, head].double() * scale
    sink = kwargs["sinks"][head].double().reshape(1)
    probabilities = torch.softmax(torch.cat((scores, sink)), dim=0)[:-1]
    value = (probabilities * kv[:, column]).sum().item()
    return dict(position=position, original_keys=len(original), compressed_keys=len(compressed),
                fp64=value, rounded_bf16=torch.tensor(value, dtype=torch.bfloat16).item(),
                captured=snapshot["output"][0][row, head, column].item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--probe-layout", action="store_true")
    parser.add_argument("--reference-element", nargs=3, type=int, metavar=("ROW", "HEAD", "COLUMN"))
    parser.add_argument("--compact-output", type=Path,
                        help="save only referenced physical pages, preserving their relative address order")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.output.exists():
        parser.error("output must not exist")
    if not args.cpu_only:
        import torch_npu  # noqa: F401
        from vllm_ascend.utils import bootstrap_custom_op_env

        # Replay the captured operator explicitly, not a policy-selected substitute.
        bootstrap_custom_op_env()
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        torch.npu.set_device(0)
    report, outputs = {}, {}
    for path in args.snapshots:
        snapshot = torch.load(path, weights_only=True)
        if args.compact_output is not None:
            args.compact_output.mkdir(parents=True, exist_ok=True)
            compact_path = args.compact_output / path.name
            if compact_path.exists():
                raise FileExistsError(compact_path)
            compact = dict(snapshot, kwargs=compact_physical_pages(snapshot["kwargs"]))
            torch.save(compact, compact_path)
            del compact
        entry = dict(inputs=describe(dict(args=snapshot["args"], kwargs=snapshot["kwargs"])))
        if args.reference_element is not None:
            entry["reference_element"] = reference_value(snapshot, *args.reference_element)
        if not args.cpu_only:
            positional = to_device(snapshot["args"], "npu")
            kwargs = to_device(snapshot["kwargs"], "npu")
            expected = snapshot["output"][0]
            entry["replays"] = []
            first = None
            for _ in range(args.repeats):
                actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv(*positional, **kwargs)[0].cpu()
                if first is None:
                    first = actual.clone()
                entry["replays"].append(dict(vs_captured=difference(expected, actual),
                                              vs_first=difference(first, actual)))
            outputs[str(path)] = first
            if args.probe_layout:
                variants = {}
                for other_path in args.snapshots:
                    other = torch.load(other_path, weights_only=True)
                    for key in ("cu_seqlens_q", "seqused_kv"):
                        if not torch.equal(snapshot["kwargs"][key], other["kwargs"][key]):
                            raise ValueError("metadata swap requires identical sequence lengths")
                    swapped = dict(kwargs, metadata=other["kwargs"]["metadata"].to("npu"))
                    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv(*positional, **swapped)[0].cpu()
                    variants["metadata:" + str(other_path)] = difference(first, actual)
                    del other, swapped
                reversed_kwargs = reverse_physical_pages(kwargs)
                actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv(*positional, **reversed_kwargs)[0].cpu()
                variants["reversed_physical_pages"] = difference(first, actual)
                del reversed_kwargs
                canonical_kwargs = canonical_physical_pages(kwargs)
                actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv(*positional, **canonical_kwargs)[0].cpu()
                variants["canonical_physical_pages"] = difference(first, actual)
                outputs[str(path) + ":canonical"] = actual.clone()
                del canonical_kwargs
                entry["variants"] = variants
            del positional, kwargs
            torch.npu.empty_cache()
        report[str(path)] = entry
        print(json.dumps(dict(snapshot=str(path), replays=entry.get("replays"))), flush=True)
        del snapshot
    if outputs:
        first_path = next(iter(outputs))
        report["output_comparisons"] = {path: difference(outputs[first_path], value)
                                         for path, value in outputs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
