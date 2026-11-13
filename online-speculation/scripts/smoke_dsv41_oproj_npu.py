#!/usr/bin/env python3
"""Validate the grouped DeepSeek-V4.1 output projection on one Ascend NPU.

The draft attention returns ``[tokens, local_heads, head_dim]``.  V4.1 first
splits those heads into output groups, applies one low-rank projection per
group, concatenates the ranks, then runs the shared second projection.  This
smoke isolates the grouped first projection used by ``OrthrusV41Attention``
and compares the fused NPU result with an FP32 CPU contraction.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--o-rank", type=int, default=1024)
    args = parser.parse_args()
    if args.local_heads % args.groups:
        raise ValueError("local-heads must be divisible by groups")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    generator = torch.Generator().manual_seed(41041)
    width = args.local_heads * args.head_dim // args.groups

    # Keep the activation scale close to a normalized transformer residual and
    # use fan-in scaling for the projection, so the error is representative.
    grouped_cpu = torch.randn(
        args.tokens, args.groups, width, generator=generator, dtype=torch.float32
    ).mul_(0.25)
    weight_cpu = torch.randn(
        args.groups, width, args.o_rank, generator=generator, dtype=torch.float32
    ).mul_(width**-0.5)
    expected = torch.einsum("tgw,gwr->tgr", grouped_cpu, weight_cpu)

    grouped = grouped_cpu.to(device=device, dtype=torch.bfloat16)
    weight = weight_cpu.to(device=device, dtype=torch.bfloat16)
    actual = torch_npu.npu_transpose_batchmatmul(
        grouped,
        weight,
        bias=None,
        scale=None,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        batch_split_factor=1,
    ).float().cpu()

    error = (actual - expected).abs()
    relative = error / expected.abs().clamp_min(1e-3)
    result = {
        "device": torch.npu.get_device_name(device),
        "shape": {
            "tokens": args.tokens,
            "local_heads": args.local_heads,
            "head_dim": args.head_dim,
            "groups": args.groups,
            "o_rank": args.o_rank,
        },
        "output_shape": list(actual.shape),
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "p99_abs_error": torch.quantile(error.flatten(), 0.99).item(),
        "max_relative_error_above_1e-3": relative.max().item(),
        "all_finite": bool(torch.isfinite(actual).all()),
    }
    # BF16 has roughly 7 fraction bits.  This checks layout and catches an
    # accidental transpose/group permutation while allowing normal BF16 GEMM
    # rounding against an FP32 reference.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2.01 / 128)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
