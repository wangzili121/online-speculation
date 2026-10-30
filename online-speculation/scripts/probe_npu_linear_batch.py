#!/usr/bin/env python3
"""Reproduce BF16 linear batch-shape differences without model weights."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-features", type=int, default=5120)
    parser.add_argument("--out-features", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.in_features, args.out_features) < 1:
        parser.error("linear dimensions must be positive")
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(args.seed)
    weight = (torch.randn(args.out_features, args.in_features, generator=generator) * .02).to(
        device=device, dtype=torch.bfloat16)
    vector = torch.randn(args.in_features, generator=generator).to(device=device, dtype=torch.bfloat16)
    rows = []
    with torch.inference_mode():
        baseline = torch.nn.functional.linear(vector[None], weight)[0].float()
        for width in (1, 2, 8, 16, 17, 18, 32):
            # 每行均为同一个向量，第一行的数学结果与 batch 大小无关。
            inputs = vector[None].expand(width, -1).contiguous()
            actual = torch.nn.functional.linear(inputs, weight)[0].float()
            rows.append(dict(width=width, equal=torch.equal(actual, baseline),
                             max_abs=float((actual - baseline).abs().max()),
                             changed_elements=int((actual != baseline).sum())))
    report = dict(operator="torch.nn.functional.linear", dtype="bfloat16", seed=args.seed,
                  weight_shape=list(weight.shape), rows=rows, diagnostic_only=True)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
