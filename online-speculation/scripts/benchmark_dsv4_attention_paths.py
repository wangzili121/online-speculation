#!/usr/bin/env python3
"""Compare existing NPU operators for the same Orthrus attention semantics.

Timings include temporary buffers and host dispatch; they are not model speedups.
Unsupported sparse APIs are reported explicitly, never silently substituted.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch_npu

from blockspec.vllm_ascend.dsv4 import (
    _c4_segment, _c128_segment, _fia_segment, _normalize_fia_lse, _swa_block_segment,
)
from blockspec.vllm_ascend.dsv4_hybrid import (
    assert_cache_unchanged, cache_checksum, gather_paged_cache, history_and_block_mask,
    merge_attention_outputs,
)


def measure(fn, iterations):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return {"median_us": statistics.median(samples), "rounds_us": samples}


def check_joined(args):
    for batch in (1, 2, 4):
        for heads in (8, 16, 32):
            q = torch.randn(batch, 32, heads, 512, device="npu", dtype=torch.bfloat16)
            history = torch.randn(batch, 128, 1, 512, device="npu", dtype=q.dtype)
            local = torch.randn(batch, 32, 1, 512, device="npu", dtype=q.dtype)
            sink_out = torch.zeros_like(q)
            sink_lse = torch.linspace(-1, 1, heads, device="npu").view(1,1,heads,1)
            sink_lse = sink_lse.expand(batch, 32, -1, -1).contiguous()
            for length in (0, 1, 127, 128):
                lengths = torch.tensor([max(length-i, 0) for i in range(batch)],
                                       device="npu", dtype=torch.int32)

                def split():
                    o1, l1 = _fia_segment(q, history, [32]*batch, lengths, 512**-0.5)
                    o2, l2 = _fia_segment(q, local, [32]*batch, [32]*batch, 512**-0.5)
                    return merge_attention_outputs([o1, o2, sink_out], [l1, l2, sink_lse])

                def joined():
                    out, lse = _swa_block_segment(q, history, local, lengths, 512**-0.5)
                    return merge_attention_outputs([out, sink_out], [lse, sink_lse])

                expected = split()
                output = joined()
                _, lse = _swa_block_segment(q, history, local, lengths, 512**-0.5)
                # Independent FP32 reference includes no intermediate BF16 normalization.
                kv = torch.cat((history, local), dim=1).float()
                scores = torch.einsum("bqhd,bkhd->bqhk", q.float(), kv) * 512**-0.5
                offsets = torch.arange(160, device="npu")
                valid = (offsets[None, :] < lengths[:, None]) | (offsets[None, :] >= 128)
                scores.masked_fill_(~valid[:, None, None, :], -torch.inf)
                combined_lse = torch.logaddexp(scores.logsumexp(-1, keepdim=True), sink_lse)
                reference = torch.einsum("bqhk,bkhd->bqhd", (scores-combined_lse).exp(), kv)
                torch.testing.assert_close(output.float(), reference, atol=0.015, rtol=0.02)
                torch.testing.assert_close(lse, scores.logsumexp(-1, keepdim=True),
                                           atol=0.002, rtol=0.001)
                torch.testing.assert_close(output, expected, atol=0.015, rtol=0.02)
                print(json.dumps({"case": "joined_swa_block", "batch": batch,
                                  "heads": heads, "lengths": lengths.cpu().tolist(),
                                  "max_abs_error": (output.float()-reference).abs().max().item(),
                                  "split": measure(split, args.iterations),
                                  "joined": measure(joined, args.iterations),
                                  "passed": True}), flush=True)


def advanced_rows(cache, table, indices):
    safe = indices.clamp_min(0).long()
    rows = torch.arange(indices.shape[0], device=indices.device).view(
        (indices.shape[0],) + (1,) * (indices.ndim-1))
    blocks = table[rows, safe // cache.shape[1]].clamp_min(0).long()
    result = cache[blocks, safe % cache.shape[1]]
    return torch.where((indices >= 0).view(*indices.shape, 1, 1),
                       result, torch.zeros_like(result))


def check_gather(args):
    for batch in (1, 2, 4):
        for width in (128, 512):
            shape = (batch, width) if width == 128 else (batch, 32, width)
            cache = torch.randn(batch*128, 32, 1, 512, device="npu", dtype=torch.bfloat16)
            table = torch.randperm(batch*128, device="npu", dtype=torch.int32).view(batch, 128)
            indices = torch.randint(0, 4096, shape, device="npu")
            indices[..., -5:] = -1
            current = lambda: advanced_rows(cache, table, indices)
            selected = lambda: gather_paged_cache(cache, table, indices)
            torch.testing.assert_close(current(), selected(), atol=0, rtol=0)
            print(json.dumps({"case": "gather", "batch": batch, "shape": shape,
                              "advanced_index": measure(current, args.iterations),
                              "selected_backend": measure(selected, args.iterations),
                              "passed": True}), flush=True)


def check_sparse(args):
    width, dim, topk = 32, 512, 512
    for batch, heads, history in ((1,8,0), (2,16,3), (2,8,128), (2,16,1024),
                                  (4,32,8192), (1,8,262144)):
        pages = max((history+31)//32, 1)
        q = torch.randn(batch, width, heads, dim, device="npu", dtype=torch.bfloat16)
        cache = torch.randn(batch*pages, 32, 1, dim, device="npu", dtype=q.dtype)
        table = torch.randperm(batch*pages, device="npu", dtype=torch.int32).view(batch, pages)
        lengths = torch.tensor([max(history-i, 0) for i in range(batch)], device="npu", dtype=torch.int32)
        order = torch.arange(topk, device="npu", dtype=torch.int32).view(1,1,-1)
        rows = torch.arange(width, device="npu", dtype=torch.int32).view(1,-1,1)
        valid = order < lengths[:, None, None]
        indices = (order + rows*19) % lengths[:, None, None].clamp_min(1)
        indices = torch.where(valid, indices, -1).contiguous()
        counts = (indices >= 0).sum(-1).flatten()
        before = cache_checksum(cache, full=True)

        def sparse():
            os.environ["ORTHRUS_C4_BACKEND"] = "sparse"
            return _c4_segment(q, cache, table, indices, lengths, dim**-0.5)

        def gathered():
            os.environ["ORTHRUS_C4_BACKEND"] = "gather"
            return _c4_segment(q, cache, table, indices, lengths, dim**-0.5)

        try:
            expected, expected_lse = gathered()
            output, lse = sparse()
            torch.testing.assert_close(output, expected, atol=0.015, rtol=0.02)
            torch.testing.assert_close(lse, expected_lse, atol=0.002, rtol=0.001)
            # Compare to one FP32 softmax, not only to another NPU attention op.
            kv = gather_paged_cache(cache, table, indices).float()
            scores = torch.einsum("bqhd,bqknd->bqhk", q.float(), kv) * dim**-0.5
            scores.masked_fill_((indices < 0)[:, :, None, :], -torch.inf)
            reference = torch.einsum("bqhk,bqknd->bqhd", scores.softmax(-1), kv)
            reference.masked_fill_((counts.view(batch,width)==0)[..., None, None], 0)
            torch.testing.assert_close(output.float(), reference, atol=0.015, rtol=0.02)
            assert_cache_unchanged(before, cache, "native-sparse-c4")
            extra_memory = {}
            for name, fn in (("gather", gathered), ("sparse", sparse)):
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                allocated = torch.npu.memory_allocated()
                fn()
                torch.npu.synchronize()
                extra_memory[name] = torch.npu.max_memory_allocated() - allocated
            print(json.dumps({"case": "paged_sparse_lse", "batch": batch, "heads": heads,
                              "history": history, "max_abs_error": (output-expected).abs().max().item(),
                              "dense_max_abs_error": (output.float()-reference).abs().max().item(),
                              "temporary_peak_bytes": extra_memory, "cache_unchanged": True,
                              "gather_fia": measure(gathered, args.iterations),
                              "native_sparse": measure(sparse, args.iterations),
                              "passed": True}), flush=True)
        except (RuntimeError, AssertionError) as error:
            print(json.dumps({"case": "paged_sparse_lse", "batch": batch, "heads": heads,
                              "history": history, "passed": False, "error": str(error)}), flush=True)
            raise


def check_c128_sparse(args):
    """Can native sparse attention avoid the small-page C128 gather fallback?"""
    os.environ["ORTHRUS_C4_BACKEND"] = "sparse"
    for batch, heads, history in ((1,8,0), (2,32,1), (1,8,128), (2,32,512),
                                  (1,8,1024), (1,8,4096), (1,8,8192), (4,32,8192)):
        pages = max((history + 31) // 32, 1)
        q = torch.randn(batch, 32, heads, 512, device="npu", dtype=torch.bfloat16)
        cache = torch.randn(batch * pages, 32, 1, 512, device="npu", dtype=q.dtype)
        table = torch.randperm(batch * pages, device="npu", dtype=torch.int32).view(batch, pages)
        lengths = torch.tensor([max(history-i, 0) for i in range(batch)], device="npu", dtype=torch.int32)
        before = cache_checksum(cache, full=True)

        def gathered():
            return _c128_segment(q, cache, table, lengths, max(history, 1), 512**-0.5, backend="gather")

        def sparse():
            # Padding and index construction belong in the timed path.
            offsets = torch.arange(max(128, (history + 127)//128*128), device="npu", dtype=torch.int32)
            indices = torch.where(offsets[None, None, :] < lengths[:, None, None], offsets, -1)
            indices = indices.expand(-1, 32, -1).contiguous()
            return _c4_segment(q, cache, table, indices, lengths, 512**-0.5)

        expected, expected_lse = gathered()
        output, lse = sparse()
        torch.testing.assert_close(output, expected, atol=0.015, rtol=0.02)
        torch.testing.assert_close(lse, expected_lse, atol=0.002, rtol=0.001)
        offsets = torch.arange(max(history, 1), device="npu")
        rows = torch.where(offsets[None, :] < lengths[:, None], offsets, -1)
        kv = gather_paged_cache(cache, table, rows).float()
        scores = torch.einsum("bqhd,bknd->bqhk", q.float(), kv) * 512**-0.5
        scores.masked_fill_((rows < 0)[:, None, None, :], -torch.inf)
        reference = torch.einsum("bqhk,bknd->bqhd", scores.softmax(-1), kv)
        reference.masked_fill_((lengths == 0)[:, None, None, None], 0)
        torch.testing.assert_close(output.float(), reference, atol=0.015, rtol=0.02)
        torch.testing.assert_close(lse, scores.logsumexp(-1, keepdim=True), atol=0.002, rtol=0.001)
        production, production_lse = _c128_segment(q, cache, table, lengths, max(history, 1), 512**-0.5)
        torch.testing.assert_close(output, production, atol=0, rtol=0)
        torch.testing.assert_close(lse, production_lse, atol=0, rtol=0)
        assert_cache_unchanged(before, cache, "c128-native-sparse")
        print(json.dumps({"case": "c128_small_page", "batch": batch, "heads": heads,
                          "history": history, "max_abs": (output-expected).abs().max().item(),
                          "fp32_max_abs": (output.float()-reference).abs().max().item(),
                          "gather_fia": measure(gathered, args.iterations),
                          "native_sparse": measure(sparse, args.iterations), "passed": True}), flush=True)


def check_sink_v2(args):
    """Check fused sink support, including the LSE required for later merging."""
    for batch, heads, history_len in ((1,8,0), (1,8,128), (2,32,1), (2,32,128)):
        q = torch.randn(batch, 32, heads, 512, device="npu", dtype=torch.bfloat16)
        history = torch.randn(batch, 128, 1, 512, device="npu", dtype=q.dtype)
        local = torch.randn(batch, 32, 1, 512, device="npu", dtype=q.dtype)
        lengths = torch.full((batch,), history_len, device="npu", dtype=torch.int32)
        sink = torch.linspace(-1, 5, heads, device="npu")
        sink_lse = sink.view(1,1,heads,1).expand(batch,32,-1,-1)
        zero = torch.zeros_like(q)

        def current():
            o, l = _swa_block_segment(q, history, local, lengths, 512**-0.5)
            return merge_attention_outputs([o, zero], [l, sink_lse])

        def fused():
            kv = torch.cat((history, local), dim=1)
            mask = history_and_block_mask(lengths, 128, 32)
            o, l = torch_npu.npu_fused_infer_attention_score_v2(
                q, kv, kv, input_layout="BSND", num_query_heads=heads, num_key_value_heads=1,
                softmax_scale=512**-0.5, sparse_mode=0, atten_mask=mask,
                actual_seq_qlen=[32]*batch, actual_seq_kvlen=[160]*batch,
                learnable_sink=sink.to(q.dtype), return_softmax_lse=True)
            return o, _normalize_fia_lse(l, batch, 32)

        expected = current()
        _, old_lse = _swa_block_segment(q, history, local, lengths, 512**-0.5)
        expected_lse = torch.logaddexp(old_lse, sink_lse)
        output, lse = fused()
        torch.testing.assert_close(output, expected, atol=0.015, rtol=0.02)
        torch.testing.assert_close(lse, expected_lse, atol=0.002, rtol=0.001)
        print(json.dumps({"case": "sink_v2", "batch": batch, "heads": heads,
                          "history": history_len, "max_abs": (output-expected).abs().max().item(),
                          "current": measure(current, args.iterations),
                          "fused_sink": measure(fused, args.iterations), "passed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("joined", "gather", "sparse", "c128-sparse", "sink-v2"), required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261028)
    {"joined": check_joined, "gather": check_gather, "sparse": check_sparse,
     "c128-sparse": check_c128_sparse, "sink-v2": check_sink_v2}[args.case](args)
