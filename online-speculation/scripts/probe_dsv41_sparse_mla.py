#!/usr/bin/env python3
"""Probe the native V4.1 attention kernel for an Orthrus draft block."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu

from blockspec.vllm_ascend.dsv41_attention import PagedRows
from blockspec.vllm_ascend.dsv41_native_attention import (
    _merge_fp32,
    _native_sparse_plane,
    temporary_block,
    native_draft_attention,
)
from blockspec.vllm_ascend.dsv41_precise_attention import precise_draft_attention


def _physical_pages(rows: torch.Tensor, page_size: int) -> PagedRows:
    batch, width, dim = rows.shape
    if width % page_size:
        raise ValueError("cache width must be a multiple of the page size")
    pages_per_request = width // page_size
    pages = rows.reshape(batch * pages_per_request, page_size, 1, dim).contiguous()
    table = torch.arange(batch * pages_per_request, dtype=torch.int32).view(
        batch, pages_per_request
    )
    return PagedRows(pages, table)


def _logical_rows(lengths: torch.Tensor, width: int) -> torch.Tensor:
    offsets = torch.arange(width)
    starts = (lengths - width).clamp_min(0)
    rows = starts[:, None] + offsets
    return torch.where(offsets < lengths.clamp_max(width)[:, None], rows, -1)


def _reference(
    query: torch.Tensor,
    ori: torch.Tensor,
    ori_indices: torch.Tensor,
    cmp: torch.Tensor,
    cmp_indices: torch.Tensor,
    temporary: torch.Tensor,
    sink: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, block, heads, dim = query.shape
    outputs = torch.empty(batch, block, heads, dim, dtype=torch.float32)
    lses = torch.empty(batch, block, heads, 1, dtype=torch.float32)
    for request in range(batch):
        local = ori[request, ori_indices[request][ori_indices[request] >= 0]]
        for row in range(block):
            sparse = cmp[request, cmp_indices[request, row][cmp_indices[request, row] >= 0]]
            kv = torch.cat((local, sparse, temporary[request]), dim=0).float()
            logits = torch.cat(
                (
                    torch.einsum("hd,rd->hr", query[request, row].float(), kv)
                    * dim**-0.5,
                    sink.float()[:, None],
                ),
                dim=-1,
            )
            lse = logits.logsumexp(-1, keepdim=True)
            outputs[request, row] = torch.einsum("hr,rd->hd", (logits[:, :-1] - lse).exp(), kv)
            lses[request, row] = lse
    return outputs, lses


def _elapsed(call, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    generator = torch.Generator().manual_seed(41017)
    batch, block, heads, dim = 3, args.block, args.heads, args.head_dim
    page_size, cache_width = 64, 2048
    anchors = torch.tensor([0, 1023, 2047], dtype=torch.int32)
    cmp_lengths = anchors // 2

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).bfloat16()

    query = sample(batch, block, heads, dim)
    ori = sample(batch, cache_width, dim)
    cmp = sample(batch, cache_width // 2, dim)
    temporary = sample(batch, block, dim)
    sink = torch.linspace(-4, 4, heads, dtype=torch.float32)
    ori_indices = _logical_rows(anchors, 128)
    offsets = torch.arange(args.topk)
    cmp_indices = torch.stack(
        [
            torch.where(offsets < length.clamp_max(32), (offsets * 3) % length.clamp_min(1), -1)
            for length in cmp_lengths
        ]
    )[:, None].expand(-1, block, -1).contiguous()
    # Make each query's sparse history different while keeping it within the anchor prefix.
    cmp_indices = torch.where(
        cmp_indices >= 0,
        (cmp_indices + torch.arange(block)[None, :, None]) % cmp_lengths[:, None, None].clamp_min(1),
        cmp_indices,
    )
    expected = _reference(query, ori, ori_indices, cmp, cmp_indices, temporary, sink)
    ori_pages, cmp_pages = _physical_pages(ori, page_size), _physical_pages(cmp, page_size)
    query_d, ori_indices_d, anchors_d, sink_d, cmp_indices_d, cmp_lengths_d = [
        value.to(device)
        for value in (query, ori_indices, anchors, sink, cmp_indices, cmp_lengths)
    ]
    ori_pages_d = PagedRows(ori_pages.values.to(device), ori_pages.table.to(device))
    cmp_pages_d = PagedRows(cmp_pages.values.to(device), cmp_pages.table.to(device))
    precise_ori = PagedRows(ori_pages_d.values.squeeze(2), ori_pages_d.table)
    precise_cmp = PagedRows(cmp_pages_d.values.squeeze(2), cmp_pages_d.table)

    disabled_sink = torch.full_like(sink_d, -torch.inf)

    def history_call():
        swa = _native_sparse_plane(
            query_d,
            ori_pages_d,
            ori_indices_d,
            anchors_d,
            sink_d,
            maximum_length=int(anchors.max()),
        )
        compressed = _native_sparse_plane(
            query_d,
            cmp_pages_d,
            cmp_indices_d,
            cmp_lengths_d,
            disabled_sink,
            maximum_length=int(cmp_lengths.max()),
        )
        return _merge_fp32(
            (swa[0].float(), compressed[0].float()),
            (swa[1].float(), compressed[1].float()),
        )

    def block_call():
        return temporary_block(query_d, temporary.to(device))

    def full_call():
        return native_draft_attention(
            query_d,
            temporary.to(device),
            ori_pages_d,
            anchors_d,
            sink_d,
            window=128,
            global_kv=cmp_pages_d,
            global_indices=cmp_indices_d,
            global_lengths=cmp_lengths_d,
            max_swa_history=int(anchors.max()),
            max_global_history=int(cmp_lengths.max()),
        )

    def precise_call():
        return precise_draft_attention(
            query_d,
            temporary.to(device),
            precise_ori,
            anchors_d,
            sink_d,
            window=128,
            global_kv=precise_cmp,
            global_indices=cmp_indices_d,
        )

    history = history_call()
    merged = full_call()
    actual = tuple(part.cpu() for part in merged)
    precise = tuple(part.cpu() for part in precise_call())
    torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=2.01 / 128)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-2, rtol=2.01 / 128)
    torch.testing.assert_close(precise[0], expected[0], atol=1e-5, rtol=1.3e-6)
    torch.testing.assert_close(precise[1], expected[1], atol=1e-5, rtol=1.3e-6)
    for current, original in ((ori_pages_d, ori_pages), (cmp_pages_d, cmp_pages)):
        torch.testing.assert_close(current.values.cpu(), original.values, atol=0, rtol=0)
        torch.testing.assert_close(current.table.cpu(), original.table, atol=0, rtol=0)

    report = {
        "device": torch.npu.get_device_name(device),
        "torch_npu": torch_npu.__version__,
        "shape": {"batch": batch, "block": block, "heads": heads, "head_dim": dim},
        "history_lse_shape": list(history[1].shape),
        "max_output_abs_error": float((actual[0] - expected[0]).abs().max()),
        "max_lse_abs_error": float((actual[1] - expected[1]).abs().max()),
        "precise_max_output_abs_error": float((precise[0] - expected[0]).abs().max()),
        "precise_max_lse_abs_error": float((precise[1] - expected[1]).abs().max()),
        "history_ms": _elapsed(history_call, args.warmup, args.iterations),
        "temporary_block_ms": _elapsed(block_call, args.warmup, args.iterations),
        "full_attention_ms": _elapsed(full_call, args.warmup, args.iterations),
        "precise_attention_ms": _elapsed(precise_call, args.warmup, args.iterations),
        "target_cache_unchanged": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
