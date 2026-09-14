"""Benchmark exact paged-old plus dense-local Orthrus draft attention."""

from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch_npu

from blockspec.reporting import dumps


def synchronize():
    torch.npu.synchronize()


def materialized_attention(query, key_cache, value_cache, block_table,
                           history, local_key, local_value, heads, scale):
    block_size = key_cache.shape[1]
    block_count = (history + block_size - 1) // block_size
    block_ids = block_table[0, :block_count].long()
    old_key = key_cache.index_select(0, block_ids).reshape(
        -1, key_cache.shape[-2], key_cache.shape[-1])[:history]
    old_value = value_cache.index_select(0, block_ids).reshape(
        -1, value_cache.shape[-2], value_cache.shape[-1])[:history]
    key = torch.cat((old_key, local_key), dim=0)
    value = torch.cat((old_value, local_value), dim=0)
    return torch_npu.npu_fusion_attention_v3(
        query[None], key[None], value[None], heads, "BSND",
        scale=scale, keep_prob=1.0, sparse_mode=0)[0][0]


def segmented_attention(query, key_cache, value_cache, block_table,
                        history, local_key, local_value, heads, kv_heads,
                        scale):
    query_tokens = query.shape[0]
    block_size = key_cache.shape[1]
    block_count = (history + block_size - 1) // block_size
    old_output, old_lse = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key_cache.view(key_cache.shape[0], block_size, -1),
        value=value_cache.view(value_cache.shape[0], block_size, -1),
        block_table=block_table[:, :block_count],
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=[query_tokens],
        actual_seq_lengths_kv=[history],
        num_key_value_heads=kv_heads,
        num_heads=heads,
        scale=scale,
        sparse_mode=0,
        softmax_lse_flag=True,
    )
    local_output, local_lse = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=local_key,
        value=local_value,
        input_layout="TND",
        actual_seq_lengths=[query_tokens],
        actual_seq_lengths_kv=[query_tokens],
        num_key_value_heads=kv_heads,
        num_heads=heads,
        scale=scale,
        sparse_mode=0,
        softmax_lse_flag=True,
    )
    merged_lse = torch.logaddexp(old_lse, local_lse)
    merged = (
        old_output.float() * torch.exp(old_lse - merged_lse)
        + local_output.float() * torch.exp(local_lse - merged_lse)
    )
    return merged.to(old_output.dtype)


def measure(function, *, warmup, iterations):
    for _ in range(warmup):
        function()
    synchronize()
    gc.collect()
    torch.npu.empty_cache()
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
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--extra-physical-blocks", type=int, default=7)
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
    logical_blocks = math.ceil(args.history / args.block_size)
    physical_blocks = logical_blocks + args.extra_physical_blocks
    query = torch.randn(
        args.query_tokens, args.query_heads, args.head_dim,
        dtype=dtype, device=device)
    key_cache = torch.randn(
        physical_blocks, args.block_size, args.kv_heads, args.head_dim,
        dtype=dtype, device=device)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.randperm(
        physical_blocks, dtype=torch.int32, device=device
    )[:logical_blocks].unsqueeze(0)
    local_key = torch.randn(
        args.query_tokens, args.kv_heads, args.head_dim,
        dtype=dtype, device=device)
    local_value = torch.randn_like(local_key)
    scale = 1.0 / math.sqrt(args.head_dim)
    common = (
        query, key_cache, value_cache, block_table, args.history,
        local_key, local_value, args.query_heads)

    with torch.inference_mode():
        expected = materialized_attention(*common, scale)
        actual = segmented_attention(
            *common, args.kv_heads, scale)
        synchronize()
        difference = (actual.float() - expected.float()).abs()
        correctness = {
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "allclose_rtol_0.02_atol_0.02": torch.allclose(
                actual.float(), expected.float(), rtol=0.02, atol=0.02),
        }
        del expected, actual
        materialized = measure(
            lambda: materialized_attention(*common, scale),
            warmup=args.warmup, iterations=args.iterations)
        segmented = measure(
            lambda: segmented_attention(*common, args.kv_heads, scale),
            warmup=args.warmup, iterations=args.iterations)

    result = {
        "shape": vars(args),
        "correctness": correctness,
        "materialized_native_gqa": materialized,
        "paged_segmented": segmented,
        "speedup": materialized["mean_ms"] / segmented["mean_ms"],
        "incremental_peak_reduction": (
            materialized["incremental_peak_bytes"]
            - segmented["incremental_peak_bytes"]),
    }
    print(dumps(result, indent=2))


if __name__ == "__main__":
    main()
