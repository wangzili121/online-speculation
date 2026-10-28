#!/usr/bin/env python3
"""Exercise Orthrus K32/FIA/paged-cache invariants on one Ascend NPU."""

from __future__ import annotations

import argparse
import time

from blockspec.reporting import dumps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    import torch

    from blockspec.vllm_ascend.dsv4 import _fia_segment
    from blockspec.vllm_ascend.dsv4_hybrid import (
        assert_cache_unchanged,
        cache_checksum,
        compressed_logical_indices,
        gather_paged_cache,
        merge_attention_segments,
        merge_attention_outputs,
    )

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(20260915)
    batch, queries, heads, dim = 2, 32, args.heads, args.head_dim
    scale = dim**-0.5
    query = torch.randn(
        batch, queries, heads, dim, device=device, dtype=torch.bfloat16
    )

    def reference(
        key_value: torch.Tensor, key_lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = key_value.expand(-1, -1, heads, -1)
        scores = torch.einsum(
            "bqhd,bkhd->bqhk", query.float(), expanded.float()
        ) * scale
        offsets = torch.arange(key_value.shape[1], device=device)
        valid = offsets[None, :] < torch.tensor(key_lengths, device=device)[:, None]
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        output = torch.einsum(
            "bqhk,bkhd->bqhd", scores.softmax(-1), expanded.float()
        )
        empty = torch.tensor(
            [length == 0 for length in key_lengths], device=device
        )
        output = torch.where(
            empty.view(-1, 1, 1, 1), torch.zeros_like(output), output
        )
        return output, lse

    segments = []
    lses = []
    references = []
    reference_lses = []
    all_keys = []
    all_lengths = []
    for width, lengths in ((128, [128, 13]), (1024, [0, 1024]), (32, [32, 32])):
        key_value = torch.randn(
            batch, width, 1, dim, device=device, dtype=torch.bfloat16
        )
        output, lse = _fia_segment(
            query,
            key_value,
            [queries] * batch,
            lengths,
            scale,
        )
        expected_output, expected_lse = reference(key_value, lengths)
        torch.testing.assert_close(
            output.float(), expected_output, atol=args.atol, rtol=args.rtol
        )
        torch.testing.assert_close(
            lse.float(), expected_lse, atol=args.atol, rtol=args.rtol
        )
        segments.append(output)
        lses.append(lse)
        references.append(expected_output)
        reference_lses.append(expected_lse)
        all_keys.append(key_value)
        all_lengths.append(lengths)

    # Count the sink exactly once. Its value is zero and its logit is per head.
    sink = torch.linspace(-1, 1, heads, device=device).view(1, 1, heads, 1)
    segments.append(torch.zeros_like(segments[0]))
    lses.append(sink.expand_as(lses[0]).contiguous())
    references.append(torch.zeros_like(references[0]))
    reference_lses.append(lses[-1])

    merged, merged_lse = merge_attention_segments(segments, lses)
    expected_merged, expected_merged_lse = merge_attention_segments(
        references, reference_lses
    )
    torch.testing.assert_close(
        merged.float(), expected_merged.float(), atol=args.atol, rtol=args.rtol
    )
    torch.testing.assert_close(
        merged_lse.float(),
        expected_merged_lse.float(),
        atol=args.atol,
        rtol=args.rtol,
    )

    native = merge_attention_outputs(segments, lses)
    # Independent dense reference: one softmax over all history, draft and sink.
    keys = torch.cat(all_keys, dim=1).expand(-1, -1, heads, -1).float()
    valid = torch.cat([
        torch.arange(k.shape[1], device=device)[None, :] <
        torch.tensor(lengths, device=device)[:, None]
        for k, lengths in zip(all_keys, all_lengths)
    ], dim=1)
    scores = torch.einsum("bqhd,bkhd->bqhk", query.float(), keys) * scale
    scores.masked_fill_(~valid[:, None, None, :], float("-inf"))
    scores = torch.cat([scores, sink.expand(batch, queries, heads, 1)], dim=-1)
    dense = torch.einsum("bqhk,bkhd->bqhd", scores.softmax(-1)[..., :-1], keys)
    torch.testing.assert_close(native.float(), dense, atol=args.atol, rtol=args.rtol)

    timings = {}
    for name, fn in (("torch_merge", merge_attention_segments),
                     ("native_merge", merge_attention_outputs)):
        for _ in range(10):
            fn(segments, lses)
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(args.iterations):
            fn(segments, lses)
        torch.npu.synchronize()
        timings[name + "_us"] = (time.perf_counter() - started) * 1e6 / args.iterations

    cache = torch.randn(7, 16, 1, dim, device=device, dtype=torch.bfloat16)
    before = cache_checksum(cache)
    block_table = torch.tensor(
        [[3, 6, 1, 4], [5, 0, 2, 6]], device=device, dtype=torch.int32
    )
    logical = torch.tensor(
        [[0, 15, 16, 47, -1], [1, 17, 31, 48, -1]],
        device=device,
        dtype=torch.int64,
    )
    gathered = gather_paged_cache(cache, block_table, logical, validate=True)
    if gathered.shape != (batch, 5, 1, dim):
        raise RuntimeError(f"unexpected paged gather shape: {tuple(gathered.shape)}")
    if torch.count_nonzero(gathered[:, -1]).item() != 0:
        raise RuntimeError("invalid paged indices were not zero-filled")
    assert_cache_unchanged(before, cache, "synthetic-paged-cache")

    _, c4_lengths = compressed_logical_indices(
        torch.tensor([3, 9, 260], device=device), 4
    )
    _, c128_lengths = compressed_logical_indices(
        torch.tensor([127, 128, 385], device=device), 128
    )
    if c4_lengths.tolist() != [0, 2, 65]:
        raise RuntimeError(f"unexpected C4 lengths: {c4_lengths.tolist()}")
    if c128_lengths.tolist() != [0, 1, 3]:
        raise RuntimeError(f"unexpected C128 lengths: {c128_lengths.tolist()}")

    torch.npu.synchronize()
    print(
        dumps(
            {
                "status": "ok",
                "device": str(device),
                "block_size": queries,
                "query_shape": list(query.shape),
                "native_merge_dense_max_abs_error": float((native.float() - dense).abs().max()),
                "merge_timings": timings,
                "fia_lse_layout": "B-Q-H-1",
                "segments_merged": len(segments),
                "paged_boundary_checked": True,
                "cache_checksum_unchanged": True,
                "c4_completed_groups": c4_lengths.tolist(),
                "c128_completed_groups": c128_lengths.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
