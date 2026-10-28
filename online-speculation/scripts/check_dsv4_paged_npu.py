#!/usr/bin/env python3
"""Compare direct paged C128 FIA with gather + FIA using real DSV4 shapes."""

import json
import time

import torch
import torch_npu  # noqa: F401

from blockspec.vllm_ascend.dsv4 import _c128_segment, _fia_segment
from blockspec.vllm_ascend.dsv4_hybrid import (
    assert_cache_unchanged, cache_checksum, gather_paged_cache,
)


def run(heads, block_size, lengths):
    width = max(max(lengths), 1)
    pages = (width + block_size - 1) // block_size
    query = torch.randn(2, 32, heads, 512, device="npu", dtype=torch.bfloat16)
    cache = torch.randn(2 * pages, block_size, 1, 512, device="npu", dtype=torch.bfloat16)
    before = cache_checksum(cache, full=True)
    table = torch.randperm(2 * pages, device="npu", dtype=torch.int32).view(2, pages)
    lens = torch.tensor(lengths, device="npu", dtype=torch.int32)
    offsets = torch.arange(width, device="npu").expand(2, -1)
    indices = torch.where(offsets < lens[:, None], offsets, -1)

    def gathered():
        kv = gather_paged_cache(cache, table, indices)
        return _fia_segment(query, kv, [32, 32], lens, 512**-0.5, allow_empty=True)

    def paged():
        return _c128_segment(query, cache, table, lens, width, 512**-0.5)

    expected, expected_lse = gathered()
    output, lse = paged()
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.02)
    torch.testing.assert_close(lse, expected_lse, atol=0.001, rtol=0.001)
    assert_cache_unchanged(before, cache, "paged-c128")
    timings = {}
    for name, fn in (("gather_fia", gathered), ("selected_backend", paged)):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            fn()
        torch.npu.synchronize()
        timings[name + "_us"] = (time.perf_counter() - start) * 1e6 / 30
    print(json.dumps(dict(heads=heads, block_size=block_size, lengths=lengths,
                          backend="paged_fia" if block_size % 128 == 0 else "gather_fia",
                          max_abs_error=float((output - expected).abs().max()),
                          timings=timings, passed=True)), flush=True)


if __name__ == "__main__":
    torch.npu.set_device(0)
    torch.manual_seed(20261028)
    for heads in (8, 16):
        for block_size in (32, 128):
            for lengths in ([0, 2], [1, 128], [1024, 567], [8192, 8191]):
                run(heads, block_size, lengths)
