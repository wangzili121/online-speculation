#!/usr/bin/env python3
"""Compare existing FAS GQA against materializing repeated history heads."""

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu


@torch.no_grad()
def main():
    torch.manual_seed(73)
    rows = []
    for past in (0, 491, 4096, 8160):
        q = torch.randn(1, 12, 32, 256, device="npu", dtype=torch.bfloat16)
        k, v = [torch.randn(1, 2, past + 32, 256, device="npu", dtype=torch.bfloat16) for _ in range(2)]
        mask = torch.zeros(1, 1, 32, past + 32, device="npu", dtype=torch.bool)

        def attention(expanded, q=q, k=k, v=v, mask=mask):
            key, value = (k.repeat_interleave(6, dim=1), v.repeat_interleave(6, dim=1)) if expanded else (k, v)
            return torch_npu.npu_fusion_attention(q, key, value, 12, input_layout="BNSD", atten_mask=mask,
                                                 scale=256 ** -.5, keep_prob=1., sparse_mode=0)[0]

        old, new = attention(True), attention(False)
        relative = float((old.float() - new.float()).norm() / old.float().norm().clamp_min(1e-12))
        if not torch.isfinite(new).all() or relative > .001:
            raise AssertionError(f"GQA forward mismatch: {relative}")
        del old, new
        for expanded in (False, True):
            for _ in range(10):
                attention(expanded)
        torch.npu.synchronize()
        timings, peaks, iterations = {False: [], True: []}, {False: [], True: []}, 100
        for repeat in range(5):
            for expanded in ((False, True) if repeat % 2 == 0 else (True, False)):
                torch.npu.reset_peak_memory_stats()
                allocated = torch.npu.memory_allocated()
                start = time.perf_counter()
                for _ in range(iterations):
                    attention(expanded)
                torch.npu.synchronize()
                timings[expanded].append((time.perf_counter() - start) * 1000 / iterations)
                peaks[expanded].append(torch.npu.max_memory_allocated() - allocated)
        rows.append(dict(past=past, query_heads=12, kv_heads=2, query_length=32,
                         forward_relative_l2=relative, forward_equal=relative == 0,
                         expanded_ms=timings[True], grouped_ms=timings[False],
                         expanded_median_ms=statistics.median(timings[True]),
                         grouped_median_ms=statistics.median(timings[False]),
                         expanded_peak_extra_bytes=max(peaks[True]), grouped_peak_extra_bytes=max(peaks[False])))
    report = dict(scope="attention microbenchmark, not end-to-end throughput",
                  torch=torch.__version__, device=torch.npu.get_device_name(0), rows=rows)
    Path(os.environ["ORTHRUS_GQA_BENCH_OUTPUT"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
