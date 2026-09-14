"""Measure page-sparse Orthrus draft attention and selection overhead on NPU."""

from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch_npu

from blockspec.reporting import dumps
from blockspec.vllm_ascend.sparse_draft import (
    recent_page_plan,
    vegas_page_plan,
    verifier_guided_page_plan,
)


def synchronize():
    torch.npu.synchronize()


def parse_histories(value):
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("histories must be positive integers")
    return result


def paged_attention(query, key, value, table, lengths, *, heads, kv_heads):
    batch, query_tokens, _, head_dim = query.shape
    cumulative = [query_tokens * (index + 1) for index in range(batch)]
    return torch_npu.npu_fused_infer_attention_score(
        query=query.reshape(-1, heads, head_dim),
        key=key.view(key.shape[0], key.shape[1], -1),
        value=value.view(value.shape[0], value.shape[1], -1),
        block_table=table,
        block_size=key.shape[1],
        actual_seq_lengths=cumulative,
        actual_seq_lengths_kv=lengths,
        num_key_value_heads=kv_heads,
        num_heads=heads,
        input_layout="TND",
        scale=1 / math.sqrt(head_dim),
        sparse_mode=0,
        softmax_lse_flag=True,
    )[0].view_as(query)


def measure(function, warmup, iterations):
    for _ in range(warmup):
        function()
    synchronize()
    gc.collect()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def similarity(actual, expected):
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    error = actual_float - expected_float
    return {
        "cosine": torch.nn.functional.cosine_similarity(
            actual_float, expected_float, dim=0).item(),
        "relative_l2": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected_float)).item(),
    }


def selected_attention(build, query, key, value, *, heads, kv_heads):
    plan = build()
    return paged_attention(
        query, key, value, plan.block_table, plan.sequence_lengths,
        heads=heads, kv_heads=kv_heads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--histories", type=parse_histories,
        default=parse_histories("7668,7291,6821,6637,10110,39290,38109,7521"))
    parser.add_argument("--ratios", default="0.025,0.05,0.1,0.25")
    parser.add_argument("--dense-below-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    ratios = [float(item) for item in args.ratios.split(",")]
    if args.query_heads % args.kv_heads:
        parser.error("query heads must be divisible by KV heads")

    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    histories = args.histories
    counts = [math.ceil(length / args.block_size) for length in histories]
    total_blocks = sum(counts)
    key = torch.randn(
        total_blocks, args.block_size, args.kv_heads, args.head_dim,
        dtype=torch.bfloat16, device="npu")
    value = torch.randn_like(key)
    query = torch.randn(
        len(histories), args.query_tokens, args.query_heads, args.head_dim,
        dtype=torch.bfloat16, device="npu")
    table = torch.full(
        (len(histories), max(counts)), -1,
        dtype=torch.int32, device="npu")
    cursor = 0
    for row, count in enumerate(counts):
        table[row, :count] = torch.arange(
            cursor, cursor + count, dtype=torch.int32, device="npu")
        cursor += count
    repeats = args.query_heads // args.kv_heads
    verifier_query = query[:, -1].view(
        len(histories), args.kv_heads, repeats, args.head_dim).mean(dim=2)
    verifier_queries = query[:, [0, -1]]

    with torch.inference_mode():
        dense = paged_attention(
            query, key, value, table, histories,
            heads=args.query_heads, kv_heads=args.kv_heads)
        dense_ms = measure(
            lambda: paged_attention(
                query, key, value, table, histories,
                heads=args.query_heads, kv_heads=args.kv_heads),
            args.warmup, args.iterations)
        rows = []
        for ratio in ratios:
            builders = {
                "recent": lambda: recent_page_plan(
                    table, histories, args.block_size,
                    budget_ratio=ratio, min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens,
                    dense_below_tokens=args.dense_below_tokens),
                "verifier": lambda: verifier_guided_page_plan(
                    verifier_query, key, table, histories,
                    budget_ratio=ratio, min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens,
                    dense_below_tokens=args.dense_below_tokens),
                "vegas": lambda: vegas_page_plan(
                    verifier_queries, key, table, histories,
                    budget_ratio=ratio, min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens,
                    dense_below_tokens=args.dense_below_tokens),
            }
            for policy, build in builders.items():
                plan = build()
                output = paged_attention(
                    query, key, value, plan.block_table,
                    plan.sequence_lengths, heads=args.query_heads,
                    kv_heads=args.kv_heads)
                selection_ms = measure(build, args.warmup, args.iterations)
                attention_ms = measure(
                    lambda plan=plan: paged_attention(
                        query, key, value, plan.block_table,
                        plan.sequence_lengths, heads=args.query_heads,
                        kv_heads=args.kv_heads),
                    args.warmup, args.iterations)
                combined_ms = measure(
                    lambda build=build: selected_attention(
                        build, query, key, value, heads=args.query_heads,
                        kv_heads=args.kv_heads),
                    args.warmup, args.iterations)
                rows.append({
                    "policy": policy,
                    "budget_ratio": ratio,
                    "selected_blocks": plan.selected_blocks,
                    "dense_blocks": plan.total_blocks,
                    "selection_ms": selection_ms,
                    "attention_ms": attention_ms,
                    "combined_ms": combined_ms,
                    "attention_speedup": dense_ms / attention_ms,
                    "combined_speedup": dense_ms / combined_ms,
                    "estimated_32_layer_attention_ms": (
                        (32 * selection_ms if policy == "vegas"
                         else selection_ms) + 32 * attention_ms),
                    "estimated_32_layer_attention_speedup": (
                        32 * dense_ms
                        / ((32 * selection_ms if policy == "vegas"
                            else selection_ms) + 32 * attention_ms)),
                    **similarity(output, dense),
                })

    print(dumps({
        "shape": {**vars(args), "batch": len(histories)},
        "dense_attention_ms": dense_ms,
        "rows": rows,
        "note": (
            "Sparse output similarity is a draft-quality proxy only; full "
            "verification and exact rejection preserve the target distribution."),
    }, indent=2))


if __name__ == "__main__":
    main()
