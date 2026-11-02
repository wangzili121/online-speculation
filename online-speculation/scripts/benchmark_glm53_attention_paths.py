#!/usr/bin/env python3
"""对照已有 FIA/卷积调用，计时包含临时状态及布局转换，不代表服务提速。"""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa: F401

from blockspec.vllm_ascend.glm53_ops import _block_segment, _segment, kda_convolve


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
    torch.npu.reset_peak_memory_stats()
    allocated = torch.npu.memory_allocated()
    result = fn()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - allocated
    del result
    return dict(median_us=statistics.median(samples), samples_us=samples, peak_bytes=peak)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    assert enable_custom_op()
    torch.manual_seed(5303)
    results = []
    for length, heads in ((2, 8), (8, 8), (8, 32), (8, 64), (32, 8)):
        query = torch.randn(length, heads, 512, device="npu", dtype=torch.bfloat16)
        block = torch.randn(length, 512, device="npu", dtype=torch.bfloat16)
        old = partial(_segment, query, block.expand(length, -1, -1), 256**-.5, "npu")
        new = partial(_block_segment, query, block, 256**-.5, "npu")
        actual, lse = new()
        expected, expected_lse = _block_segment(query, block, 256**-.5, "torch")
        torch.testing.assert_close(actual, expected, atol=.015, rtol=.02)
        torch.testing.assert_close(lse, expected_lse, atol=.002, rtol=.001)
        results.append(dict(case="shared_block", length=length, heads=heads,
                            expanded=measure(old, args.iterations), shared=measure(new, args.iterations),
                            max_abs=float((actual - expected).abs().max()), passed=True))
        channels = 3 * heads * 128
        rows = torch.randn(length, channels + heads + 256, device="npu", dtype=torch.bfloat16)[:, :channels]
        tail = torch.randn(3, channels, device="npu", dtype=rows.dtype)
        for weight_dtype in (torch.bfloat16, torch.float32):
            weight = torch.randn(channels, 4, device="npu", dtype=weight_dtype) * .1
            old_conv = partial(kda_convolve, rows, tail, weight)
            new_conv = partial(kda_convolve, rows, tail, weight, backend="npu")
            actual, expected = new_conv(), old_conv()
            torch.testing.assert_close(actual, expected, atol=.015, rtol=.02)
            results.append(dict(case="fused_conv", length=length, heads=heads, weight_dtype=str(weight_dtype),
                                input_stride=list(rows.stride()),
                                composed=measure(old_conv, args.iterations), fused=measure(new_conv, args.iterations),
                                max_abs=float((actual - expected).abs().max()), passed=True))
    args.output.write_text(json.dumps(dict(scope="local operator calls, synthetic inputs", results=results), indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
