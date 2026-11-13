#!/usr/bin/env python3
"""Check V4.1 native read-only QLI on C2A and C1A index pages."""

import json

import torch
import torch_npu
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

from blockspec.vllm_ascend.dsv41_native_index import IndexPages, select_projected_history


def main():
    torch_npu.npu.set_device(0)
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators are unavailable")
    init_device_properties_triton()
    device = "npu"
    block, heads, dim = 32, 32, 128
    batch, page_size, page_count = 3, 16, 35
    width = page_size * page_count
    # Positive, distinct index scores for odd/even rows. Future key scores
    # are deliberately higher on one query, so leakage is easy to detect.
    key = torch.zeros(batch, width, 1, dim, dtype=torch.int8)
    key[..., 0, 0] = 1
    key = key.reshape(-1, page_size, 1, dim)
    key_scale = torch.arange(1, width + 1, dtype=torch.float16).repeat(batch).reshape(-1, page_size, 1)
    # Reverse score order on alternate queries using signed index weights.
    query = torch.zeros(batch, block, heads, dim, dtype=torch.bfloat16)
    query[:, :, 0, 0] = 1
    weights = torch.zeros(batch, block, heads, dtype=torch.float16)
    weights[:, ::2, 0] = 1
    weights[:, 1::2, 0] = -1
    query, weights = query.to(device), weights.to(device)
    page_order = torch.randperm(batch * page_count, generator=torch.Generator().manual_seed(41))
    table = page_order.argsort().int().reshape(batch, page_count)
    pages = IndexPages(key[page_order].to(device), key_scale[page_order].to(device), table.to(device))
    c2_anchors = torch.tensor([0, 33, 36], device=device)
    selected, _ = select_projected_history(query, weights, c2_anchors, pages, ratio=2, topk=16)
    expected = torch.full((batch, block, 16), -1, dtype=torch.int32)
    for b, length in enumerate([0, 16, 18]):
        used = min(16, length)
        expected[b, ::2, :used] = torch.arange(length - used, length)
        expected[b, 1::2, :used] = torch.arange(used)
    torch.testing.assert_close(selected.cpu(), expected, rtol=0, atol=0)

    anchors = torch.tensor([0, 535, 541], device=device)
    kwargs = dict(ratio=1, topk=512, candidate_topk_blocks=64, candidate_block_size=8)
    full, candidates = select_projected_history(query, weights, anchors, pages, candidate_mode="produce", **kwargs)
    reverse = -weights
    reindex, _ = select_projected_history(
        query, reverse, anchors, pages, candidate_mode="consume", candidates=candidates, **kwargs,
    )
    full, candidates, reindex = full.cpu(), candidates.cpu(), reindex.cpu()
    assert full[0].eq(-1).all() and reindex[0].eq(-1).all()
    checked_candidates = 0
    for b, length in enumerate([535, 541], start=1):
        for i in range(block):
            expected_full = list(range(length - 512, length)) if i % 2 == 0 else list(range(512))
            assert full[b, i].tolist() == expected_full
            blocks = candidates[b * block + i, 0].tolist()
            latest = (length - 1) // 8
            expected_blocks = set(range(latest - 63, latest + 1)) if i % 2 == 0 else set(range(63)) | {latest}
            assert set(blocks) == expected_blocks, (i, blocks, sorted(expected_blocks))
            rows = sorted(row for row in range(length) if row // 8 in expected_blocks)
            actual = reindex[b, i].tolist()
            expected_rows = rows + [-1] * (512 - len(rows))
            assert actual == expected_rows, (
                b, i, len(rows), sum(row >= 0 for row in actual),
                sorted(set(actual) - set(expected_rows))[:32],
                sorted(set(expected_rows) - set(actual))[:32],
            )
            checked_candidates += 1
    assert torch.equal(pages.key.cpu(), key[page_order])
    assert torch.equal(pages.scale.cpu(), key_scale[page_order])
    assert torch.equal(pages.block_table.cpu(), table)
    print(json.dumps({
        "device": torch_npu.npu.get_device_name(0),
        "c2a_rows_checked": batch * block,
        "c1a_rows_checked": batch * block,
        "candidate_pools_checked": checked_candidates,
        "target_index_pages_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
