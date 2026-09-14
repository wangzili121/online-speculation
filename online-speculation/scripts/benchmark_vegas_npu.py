"""Probe Ascend's native top-k indexer and token-sparse attention for VEGAS."""

from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch_npu

from blockspec.reporting import dumps


def measure(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    gc.collect()
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = function()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--history", type=int, default=32768)
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sparse-counts", default="512,2048,6144")
    parser.add_argument(
        "--index-mode", choices=("first_head", "flattened_gqa"),
        default="first_head")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if args.query_heads % args.kv_heads:
        parser.error("query heads must be divisible by KV heads")
    sparse_counts = [int(value) for value in args.sparse_counts.split(",")]
    if any(value > args.history for value in sparse_counts):
        parser.error("sparse counts cannot exceed history length")

    torch.npu.set_device(0)
    torch.manual_seed(20260914)
    pages = math.ceil(args.history / args.block_size)
    total_pages = args.batch * pages
    key = torch.randn(
        total_pages, args.block_size, args.kv_heads, args.head_dim,
        dtype=torch.bfloat16, device="npu")
    value = torch.randn_like(key)
    verifier_query = torch.randn(
        args.batch, 2, args.query_heads, args.head_dim,
        dtype=torch.bfloat16, device="npu")
    draft_query = torch.randn(
        args.batch, args.query_tokens, args.query_heads, args.head_dim,
        dtype=torch.bfloat16, device="npu")
    query_rope = torch.zeros(
        args.batch, args.query_tokens, args.query_heads, 64,
        dtype=torch.bfloat16, device="npu")
    key_rope = torch.zeros(
        total_pages, args.block_size, 1, 64,
        dtype=torch.bfloat16, device="npu")
    repeats = args.query_heads // args.kv_heads
    grouped_query = verifier_query.view(
        args.batch, 2, args.kv_heads, repeats, args.head_dim)
    if args.index_mode == "first_head":
        index_query = grouped_query[:, :, 0].contiguous()
        index_key = key[:, :, 0:1].contiguous()
        index_heads = repeats
    else:
        index_query = grouped_query.mean(dim=3).reshape(
            args.batch, 2, 1, args.kv_heads * args.head_dim)
        index_key = key.view(
            total_pages, args.block_size, 1,
            args.kv_heads * args.head_dim)
        index_heads = 1
    weights = torch.full(
        (args.batch, 2, index_heads), 1 / index_heads,
        dtype=torch.bfloat16, device="npu")
    table = torch.arange(
        total_pages, dtype=torch.int32, device="npu").view(args.batch, pages)
    query_lengths = torch.full(
        (args.batch,), 2, dtype=torch.int32, device="npu")
    key_lengths = torch.full(
        (args.batch,), args.history, dtype=torch.int32, device="npu")

    def index(sparse_count):
        return torch_npu.npu_lightning_indexer(
            query=index_query,
            key=index_key,
            weights=weights,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_key=key_lengths,
            block_table=table,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            sparse_mode=0,
        )

    rows = []
    with torch.inference_mode():
        for sparse_count in sparse_counts:
            index_ms, result = measure(
                lambda count=sparse_count: index(count),
                args.warmup, args.iterations)
            indices = result[0]
            row = {
                "sparse_count": sparse_count,
                "index_ms": index_ms,
                "index_shape": list(indices.shape),
                "index_dtype": str(indices.dtype),
                "index_min": int(indices.min().item()),
                "index_max": int(indices.max().item()),
            }
            try:
                selected = indices
                if selected.ndim == 4:
                    selected = selected[:, -1, 0]
                elif selected.ndim == 3:
                    selected = selected[:, -1]
                selected = selected[:, None, None].expand(
                    args.batch, args.query_tokens, args.kv_heads,
                    sparse_count).contiguous()

                def sparse_attention():
                    return torch_npu.npu_sparse_flash_attention(
                        draft_query, key, value, selected,
                        1 / math.sqrt(args.head_dim),
                        block_table=table,
                        actual_seq_lengths_query=torch.full(
                            (args.batch,), args.query_tokens,
                            dtype=torch.int32, device="npu"),
                        actual_seq_lengths_kv=key_lengths,
                        query_rope=query_rope,
                        key_rope=key_rope,
                        sparse_block_size=1,
                        layout_query="BSND",
                        layout_kv="PA_BSND",
                        sparse_mode=0,
                        attention_mode=2,
                        return_softmax_lse=False,
                    )

                sparse_ms, sparse_result = measure(
                    sparse_attention, args.warmup, args.iterations)
                row.update({
                    "sparse_attention_supported": True,
                    "sparse_attention_ms": sparse_ms,
                    "sparse_output_shapes": [
                        list(value.shape) for value in sparse_result],
                })
            except Exception as error:  # Probe unsupported shape contracts.
                print(f"SPARSE_ATTENTION_ERROR: {error}")
                row["sparse_attention_supported"] = False
                row["sparse_attention_error_type"] = type(error).__name__
                row["sparse_attention_error_summary"] = str(error).splitlines()[0]
            rows.append(row)

    print(dumps({"shape": vars(args), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
