#!/usr/bin/env python3
"""Exercise Orthrus K32/FIA/paged-cache invariants on one Ascend NPU."""

from __future__ import annotations

import argparse

from blockspec.reporting import dumps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()

    import torch

    from blockspec.vllm_ascend.dsv4 import _fia_segment
    from blockspec.vllm_ascend.dsv4_hybrid import (
        assert_cache_unchanged,
        cache_checksum,
        compressed_logical_indices,
        gather_paged_cache,
        merge_attention_segments,
    )

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(20260915)
    batch, queries, heads, dim = 2, 32, 4, 128
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
    for width, lengths in ((19, [19, 13]), (5, [0, 5]), (32, [32, 32])):
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
