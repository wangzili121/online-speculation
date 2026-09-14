"""Estimate PIVOT's DSV4 indexer opportunity with native Ascend operators.

This is an operator-level feasibility probe, not an end-to-end model result.
It uses DSV4's 64 x 128 lightning-indexer shape and compares independent
full-prefix scans with one proxy scan shared by a DSpark-sized query group.
"""

from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch_npu

from blockspec.reporting import dumps


def measure(function, warmup: int, iterations: int):
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


def normalize_indices(indices: torch.Tensor) -> torch.Tensor:
    """Convert native indexer output to [batch, query_tokens, topk]."""

    if indices.ndim == 4:
        if indices.shape[2] != 1:
            raise ValueError("expected singleton index head")
        return indices[:, :, 0]
    if indices.ndim == 3:
        return indices
    raise ValueError(f"unsupported index shape: {tuple(indices.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--history", type=int, default=32768)
    parser.add_argument("--groups", default="6,8")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--candidate-ratios", default="1.25,1.5,2.0")
    parser.add_argument("--query-noise", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    groups = [int(value) for value in args.groups.split(",")]
    candidate_ratios = [
        float(value) for value in args.candidate_ratios.split(",")
    ]
    if any(group <= 1 for group in groups):
        parser.error("groups must be greater than one")
    if args.topk > args.history:
        parser.error("topk cannot exceed history")

    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    pages = math.ceil(args.history / args.block_size)
    padded_history = pages * args.block_size
    total_pages = args.batch * pages
    key = torch.randn(
        total_pages,
        args.block_size,
        1,
        args.head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    table = torch.arange(
        total_pages, dtype=torch.int32, device="npu"
    ).view(args.batch, pages)
    key_lengths = torch.full(
        (args.batch,), args.history, dtype=torch.int32, device="npu"
    )
    batch_offsets = (
        torch.arange(args.batch, dtype=torch.long, device="npu")
        * padded_history
    )
    flat_key = key.view(-1, args.head_dim)

    def recall(reference: torch.Tensor, candidate: torch.Tensor) -> float:
        """Return mean top-k set recall without materializing an NxN matrix."""

        if candidate.shape[1] == 1 and reference.shape[1] != 1:
            candidate = candidate.expand(-1, reference.shape[1], -1)
        sorted_candidate = candidate.sort(dim=-1).values
        positions = torch.searchsorted(sorted_candidate, reference)
        positions.clamp_max_(sorted_candidate.shape[-1] - 1)
        matches = sorted_candidate.gather(-1, positions).eq(reference)
        return float(matches.float().mean().cpu())

    def run_indexer(
        query: torch.Tensor,
        weights: torch.Tensor,
        sparse_count: int,
        *,
        index_key: torch.Tensor = key,
        index_table: torch.Tensor = table,
        index_key_lengths: torch.Tensor = key_lengths,
    ) -> torch.Tensor:
        query_tokens = query.shape[1]
        query_lengths = torch.full(
            (args.batch,), query_tokens, dtype=torch.int32, device="npu"
        )
        result = torch_npu.npu_lightning_indexer(
            query=query,
            key=index_key,
            weights=weights,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_key=index_key_lengths,
            block_table=index_table,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            sparse_mode=0,
        )
        return normalize_indices(result[0])

    rows = []
    with torch.inference_mode():
        for group in groups:
            anchor_query = torch.randn(
                args.batch,
                1,
                args.heads,
                args.head_dim,
                dtype=torch.bfloat16,
                device="npu",
            )
            query = anchor_query + args.query_noise * torch.randn(
                args.batch,
                group,
                args.heads,
                args.head_dim,
                dtype=torch.bfloat16,
                device="npu",
            )
            anchor_weight_logits = torch.randn(
                args.batch,
                1,
                args.heads,
                dtype=torch.float32,
                device="npu",
            )
            weights = torch.softmax(
                anchor_weight_logits + args.query_noise * torch.randn(
                    args.batch,
                    group,
                    args.heads,
                    dtype=torch.float32,
                    device="npu",
                ),
                dim=-1,
            ).to(torch.bfloat16)

            # PIVOT's exact aggregation is gate-aware. The mean proxy is enough
            # for this cost probe because it preserves the native op shape.
            proxy_query = query.mean(dim=1, keepdim=True)
            proxy_weights = weights.mean(dim=1, keepdim=True)

            baseline_ms, baseline_indices = measure(
                lambda: run_indexer(query, weights, args.topk),
                args.warmup,
                args.iterations,
            )
            reuse_ms, reuse_indices = measure(
                lambda: run_indexer(proxy_query, proxy_weights, args.topk),
                args.warmup,
                args.iterations,
            )
            row = {
                "group": group,
                "baseline_ms": baseline_ms,
                "reuse_ms": reuse_ms,
                "reuse_indexer_speedup": baseline_ms / reuse_ms,
                "reuse_topk_recall": recall(
                    baseline_indices, reuse_indices
                ),
                "baseline_shape": list(baseline_indices.shape),
                "reuse_shape": list(reuse_indices.shape),
                "refine": [],
            }

            for ratio in candidate_ratios:
                candidate_count = min(
                    args.history, math.ceil(args.topk * ratio)
                )
                # The current Ascend LightningIndexer contract accepts
                # arbitrary supported counts through 2048, then 1024-token
                # buckets up to 8192.
                sparse_alignment = (
                    1024 if candidate_count > 2048 else args.block_size
                )
                candidate_count = min(
                    args.history,
                    math.ceil(candidate_count / sparse_alignment)
                    * sparse_alignment,
                )

                def proxy_refine():
                    proxy = run_indexer(
                        proxy_query, proxy_weights, candidate_count
                    )[:, 0]
                    physical = proxy.long() + batch_offsets[:, None]
                    candidates = flat_key.index_select(
                        0, physical.reshape(-1)
                    ).view(args.batch, candidate_count, args.head_dim)
                    scores = torch.einsum(
                        "bghd,bcd->bghc", query.float(), candidates.float()
                    ).relu_()
                    scores.mul_(weights.float().unsqueeze(-1))
                    reduced = scores.sum(dim=2)
                    local = torch.topk(
                        reduced, k=args.topk, dim=-1, sorted=False
                    ).indices
                    return proxy.gather(1, local.reshape(args.batch, -1)).view(
                        args.batch, group, args.topk
                    )

                refine_ms, refine_indices = measure(
                    proxy_refine, args.warmup, args.iterations
                )

                candidate_pages = math.ceil(
                    candidate_count / args.block_size
                )
                candidate_table = torch.arange(
                    args.batch * candidate_pages,
                    dtype=torch.int32,
                    device="npu",
                ).view(args.batch, candidate_pages)
                candidate_key_lengths = torch.full(
                    (args.batch,),
                    candidate_count,
                    dtype=torch.int32,
                    device="npu",
                )

                proxy_candidates = run_indexer(
                    proxy_query, proxy_weights, candidate_count
                )[:, 0]

                def pack_candidates():
                    physical = (
                        proxy_candidates.long() + batch_offsets[:, None]
                    )
                    return flat_key.index_select(
                        0, physical.reshape(-1)
                    ).view(
                        args.batch * candidate_pages,
                        args.block_size,
                        1,
                        args.head_dim,
                    )

                packed_candidates = pack_candidates()
                proxy_candidate_ms, _ = measure(
                    lambda: run_indexer(
                        proxy_query, proxy_weights, candidate_count
                    ),
                    args.warmup,
                    args.iterations,
                )
                candidate_pack_ms, _ = measure(
                    pack_candidates, args.warmup, args.iterations
                )
                candidate_rescore_ms, _ = measure(
                    lambda: run_indexer(
                        query,
                        weights,
                        args.topk,
                        index_key=packed_candidates,
                        index_table=candidate_table,
                        index_key_lengths=candidate_key_lengths,
                    ),
                    args.warmup,
                    args.iterations,
                )

                def native_proxy_refine():
                    proxy = run_indexer(
                        proxy_query, proxy_weights, candidate_count
                    )[:, 0]
                    physical = proxy.long() + batch_offsets[:, None]
                    packed_key = flat_key.index_select(
                        0, physical.reshape(-1)
                    ).view(
                        args.batch * candidate_pages,
                        args.block_size,
                        1,
                        args.head_dim,
                    )
                    local = run_indexer(
                        query,
                        weights,
                        args.topk,
                        index_key=packed_key,
                        index_table=candidate_table,
                        index_key_lengths=candidate_key_lengths,
                    )
                    return proxy.gather(
                        1, local.reshape(args.batch, -1)
                    ).view(args.batch, group, args.topk)

                native_ms, native_indices = measure(
                    native_proxy_refine, args.warmup, args.iterations
                )
                row["refine"].append({
                    "candidate_ratio": ratio,
                    "candidate_count": candidate_count,
                    "torch_ms": refine_ms,
                    "torch_speedup": baseline_ms / refine_ms,
                    "torch_topk_recall": recall(
                        baseline_indices, refine_indices
                    ),
                    "native_two_pass_ms": native_ms,
                    "native_two_pass_speedup": baseline_ms / native_ms,
                    "native_topk_recall": recall(
                        baseline_indices, native_indices
                    ),
                    "native_breakdown_ms": {
                        "proxy_candidate_scan": proxy_candidate_ms,
                        "candidate_pack": candidate_pack_ms,
                        "candidate_rescore": candidate_rescore_ms,
                        "component_sum": (
                            proxy_candidate_ms + candidate_pack_ms
                            + candidate_rescore_ms
                        ),
                    },
                    "shape": list(refine_indices.shape),
                    "native_shape": list(native_indices.shape),
                })
            rows.append(row)

    print(
        dumps(
            {
                "shape": {
                    **vars(args),
                    "compressed_history_tokens": args.history,
                    "raw_context_equivalent_c4": args.history * 4,
                },
                "rows": rows,
                "scope": (
                    "Indexer-only synthetic DSV4 shape. It does not measure "
                    "top-k fidelity, DSpark acceptance, or end-to-end serving."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
