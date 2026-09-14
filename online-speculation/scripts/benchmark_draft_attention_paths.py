"""Compare the current expanded-GQA draft attention with CANN native GQA."""

from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch_npu
from torch.nn import functional as F

from blockspec.reporting import dumps


def synchronize():
    torch.npu.synchronize()


def expanded_sdpa(query, key, value, repeats, scale):
    key = key.repeat_interleave(repeats, dim=2).transpose(1, 2)
    value = value.repeat_interleave(repeats, dim=2).transpose(1, 2)
    query = query.transpose(1, 2)
    return F.scaled_dot_product_attention(
        query, key, value, is_causal=False, scale=scale).transpose(1, 2)


def native_gqa(query, key, value, heads, scale):
    return torch_npu.npu_fusion_attention_v3(
        query, key, value, heads, "BSND", scale=scale,
        keep_prob=1.0, sparse_mode=0)[0]


def measure(function, *, warmup, iterations):
    for _ in range(warmup):
        function()
    synchronize()
    torch.npu.reset_peak_memory_stats()
    start_memory = torch.npu.memory_allocated()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize()
    elapsed = time.perf_counter() - start
    return {
        "mean_ms": elapsed * 1000 / iterations,
        "start_allocated_bytes": start_memory,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "incremental_peak_bytes": torch.npu.max_memory_allocated() - start_memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=32768)
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.query_heads % args.kv_heads:
        parser.error("query heads must be divisible by KV heads")

    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    total_kv = args.history + args.query_tokens
    query = torch.randn(
        1, args.query_tokens, args.query_heads, args.head_dim,
        dtype=dtype, device=device)
    key = torch.randn(
        1, total_kv, args.kv_heads, args.head_dim,
        dtype=dtype, device=device)
    value = torch.randn_like(key)
    repeats = args.query_heads // args.kv_heads
    scale = 1.0 / math.sqrt(args.head_dim)

    with torch.inference_mode():
        expected = expanded_sdpa(query, key, value, repeats, scale)
        actual = native_gqa(query, key, value, args.query_heads, scale)
        synchronize()
        difference = (actual.float() - expected.float()).abs()
        correctness = {
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "allclose_rtol_0.02_atol_0.02": torch.allclose(
                actual.float(), expected.float(), rtol=0.02, atol=0.02),
        }
        del expected, actual
        gc.collect()
        torch.npu.empty_cache()
        expanded = measure(
            lambda: expanded_sdpa(query, key, value, repeats, scale),
            warmup=args.warmup, iterations=args.iterations)
        gc.collect()
        torch.npu.empty_cache()
        native = measure(
            lambda: native_gqa(query, key, value, args.query_heads, scale),
            warmup=args.warmup, iterations=args.iterations)

    result = {
        "shape": vars(args),
        "correctness": correctness,
        "expanded_sdpa": expanded,
        "native_gqa": native,
        "speedup": expanded["mean_ms"] / native["mean_ms"],
        "incremental_peak_reduction": (
            expanded["incremental_peak_bytes"] -
            native["incremental_peak_bytes"]),
    }
    print(dumps(result, indent=2))


if __name__ == "__main__":
    main()
