"""Compare Orthrus per-request KV materialization with batched paged attention."""

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


def materialized_batch(query, key_cache, value_cache, block_table, histories,
                       local_key, local_value, heads, scale):
    block_size = key_cache.shape[1]
    rows = []
    for index, history in enumerate(histories):
        block_count = math.ceil(history / block_size)
        block_ids = block_table[index, :block_count].long()
        old_key = key_cache.index_select(0, block_ids).reshape(
            -1, key_cache.shape[-2], key_cache.shape[-1])[:history]
        old_value = value_cache.index_select(0, block_ids).reshape(
            -1, value_cache.shape[-2], value_cache.shape[-1])[:history]
        key = torch.cat((old_key, local_key[index]), dim=0)
        value = torch.cat((old_value, local_value[index]), dim=0)
        rows.append(torch_npu.npu_fusion_attention_v3(
            query[index:index + 1], key[None], value[None], heads, "BSND",
            scale=scale, keep_prob=1.0, sparse_mode=0)[0][0])
    return torch.stack(rows)


def batched_segmented(query, key_cache, value_cache, block_table, histories,
                      local_key, local_value, heads, kv_heads, scale):
    batch, query_tokens, _, _ = query.shape
    block_size = key_cache.shape[1]
    cumulative = [query_tokens * (index + 1) for index in range(batch)]
    common = {
        "query": query.reshape(-1, heads, query.shape[-1]),
        "input_layout": "TND",
        "actual_seq_lengths": cumulative,
        "num_key_value_heads": kv_heads,
        "num_heads": heads,
        "scale": scale,
        "sparse_mode": 0,
        "softmax_lse_flag": True,
    }
    old_output, old_lse = torch_npu.npu_fused_infer_attention_score(
        key=key_cache.view(key_cache.shape[0], block_size, -1),
        value=value_cache.view(value_cache.shape[0], block_size, -1),
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths_kv=histories,
        **common,
    )
    local_output, local_lse = torch_npu.npu_fused_infer_attention_score(
        key=local_key.reshape(-1, kv_heads, local_key.shape[-1]),
        value=local_value.reshape(-1, kv_heads, local_value.shape[-1]),
        actual_seq_lengths_kv=cumulative,
        **common,
    )
    merged_lse = torch.logaddexp(old_lse, local_lse)
    output = (
        old_output.float() * torch.exp(old_lse - merged_lse)
        + local_output.float() * torch.exp(local_lse - merged_lse)
    )
    return output.to(old_output.dtype).view_as(query)


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


def parse_histories(value):
    histories = [int(item) for item in value.split(",") if item]
    if not histories or any(item <= 0 for item in histories):
        raise argparse.ArgumentTypeError("histories must be positive integers")
    return histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--histories", type=parse_histories,
        default=parse_histories("7668,7291,6821,6637,10110,39290,38109,7521"))
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
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
    logical_blocks = [math.ceil(item / args.block_size)
                      for item in args.histories]
    physical_blocks = sum(logical_blocks) + len(logical_blocks)
    max_blocks = max(logical_blocks)
    key_cache = torch.randn(
        physical_blocks, args.block_size, args.kv_heads, args.head_dim,
        dtype=dtype, device=device)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.full(
        (len(args.histories), max_blocks), -1,
        dtype=torch.int32, device=device)
    cursor = 0
    for index, count in enumerate(logical_blocks):
        block_table[index, :count] = torch.arange(
            cursor, cursor + count, dtype=torch.int32, device=device)
        cursor += count
    query = torch.randn(
        len(args.histories), args.query_tokens, args.query_heads, args.head_dim,
        dtype=dtype, device=device)
    local_key = torch.randn(
        len(args.histories), args.query_tokens, args.kv_heads, args.head_dim,
        dtype=dtype, device=device)
    local_value = torch.randn_like(local_key)
    scale = 1.0 / math.sqrt(args.head_dim)
    common = (
        query, key_cache, value_cache, block_table, args.histories,
        local_key, local_value, args.query_heads)

    with torch.inference_mode():
        expected = materialized_batch(*common, scale)
        actual = batched_segmented(*common, args.kv_heads, scale)
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
            lambda: materialized_batch(*common, scale),
            warmup=args.warmup, iterations=args.iterations)
        segmented = measure(
            lambda: batched_segmented(*common, args.kv_heads, scale),
            warmup=args.warmup, iterations=args.iterations)

    result = {
        "shape": {**vars(args), "batch": len(args.histories)},
        "correctness": correctness,
        "per_request_materialized": materialized,
        "batched_paged_segmented": segmented,
        "speedup": materialized["mean_ms"] / segmented["mean_ms"],
        "incremental_peak_reduction": (
            materialized["incremental_peak_bytes"]
            - segmented["incremental_peak_bytes"]),
    }
    print(dumps(result, indent=2))


if __name__ == "__main__":
    main()
