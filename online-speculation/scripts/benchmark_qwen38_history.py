#!/usr/bin/env python3
"""Compare existing operators for owned, head-major native KV snapshots."""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch

from blockspec.vllm_ascend.qwen38_history import paged_history


def selected_pages(cache, table, length):
    pages = (length + cache.shape[1] - 1) // cache.shape[1]
    indices = table[0, :pages].to(device=cache.device, dtype=torch.long)
    return cache.index_select(0, indices).flatten(0, 1)[:length].transpose(0, 1)[None].contiguous()


def selected_tokens(cache, table, length, *, gather=False):
    pages = (length + cache.shape[1] - 1) // cache.shape[1]
    indices = table[0, :pages].to(device=cache.device, dtype=torch.long)
    slots = (indices[:, None] * cache.shape[1]
             + torch.arange(cache.shape[1], device=cache.device)).flatten()[:length]
    source = cache.flatten(0, 1).transpose(0, 1)
    if gather:
        return source.gather(1, slots[None, :, None].expand(source.shape[0], -1, source.shape[2]))[None]
    return source.index_select(1, slots)[None]


def measure(function, synchronize, *, repeats, iterations, device):
    for _ in range(5):
        function()
    synchronize()
    elapsed, peaks = [], []
    for _ in range(repeats):
        if device.type == "npu":
            torch.npu.reset_peak_memory_stats()
            before = torch.npu.memory_allocated()
        start = time.perf_counter()
        for _ in range(iterations):
            result = function()
            del result
        synchronize()
        elapsed.append((time.perf_counter() - start) * 1000 / iterations)
        if device.type == "npu":
            peaks.append(torch.npu.max_memory_allocated() - before)
    return dict(median_ms=statistics.median(elapsed), samples_ms=elapsed,
                peak_extra_bytes=max(peaks, default=None))


def convolution_tails(device, args, synchronize):
    rows = []
    conv = torch.randn(40, 34, 5120, device=device, dtype=torch.bfloat16)
    index = torch.tensor([27], device=device, dtype=torch.long)
    for accepted in (1, 17, 32):
        offset = accepted - 1
        functions = dict(
            row_then_slice=lambda offset=offset: conv.index_select(0, index)[:, offset:offset + 3],
            slice_then_row=lambda offset=offset: conv[:, offset:offset + 3].index_select(0, index),
        )
        torch.testing.assert_close(functions["row_then_slice"](), functions["slice_then_row"](), rtol=0, atol=0)
        rows.append(dict(accepted=accepted, results={name: measure(fn, synchronize, repeats=args.repeats,
                         iterations=args.iterations, device=device) for name, fn in functions.items()}))
    return rows


def attention_masks(device, args, synchronize):
    if device.type != "npu":
        return []
    import torch_npu
    rows = []
    for width, length in ((1, 1), (2, 7), (8, 491), (17, 4096), (32, 32), (32, 4096), (32, 8192)):
        q = torch.randn(1, 12, width, 256, device=device, dtype=torch.bfloat16)
        k, v = [torch.randn(1, 2, length, 256, device=device, dtype=torch.bfloat16) for _ in range(2)]
        mask = torch.zeros(1, 1, width, length, device=device, dtype=torch.bfloat16)

        def operation(masked, q=q, k=k, v=v, mask=mask):
            blocked = torch.isneginf(mask).contiguous() if masked else None
            return torch_npu.npu_fusion_attention(q, k, v, 12, input_layout="BNSD", atten_mask=blocked,
                                                 scale=256 ** -.5, keep_prob=1., sparse_mode=0)[0]

        torch.testing.assert_close(operation(True), operation(False), rtol=0, atol=0)
        gradients = []
        with torch.enable_grad():
            for masked in (True, False):
                qi, ki, vi = [x.detach().clone().requires_grad_() for x in (q, k, v)]
                blocked = torch.isneginf(mask).contiguous() if masked else None
                result = torch_npu.npu_fusion_attention(qi, ki, vi, 12, input_layout="BNSD",
                    atten_mask=blocked, scale=256 ** -.5, keep_prob=1., sparse_mode=0)[0]
                result.float().square().mean().backward()
                gradients.append([x.grad for x in (qi, ki, vi)])
        gradient_errors = []
        for old, new in zip(*gradients, strict=True):
            torch.testing.assert_close(old, new, rtol=1e-2, atol=1e-7)
            relative = float((old.float() - new.float()).norm() / old.float().norm().clamp_min(1e-12))
            if relative > 1e-3:
                raise AssertionError(f"attention gradient relative L2 mismatch: {relative}")
            gradient_errors.append(dict(relative_l2=relative, max_abs=float((old.float() - new.float()).abs().max()),
                                        bitwise_equal=torch.equal(old, new)))
        measurements = {name: measure(partial(operation, masked), synchronize, repeats=args.repeats,
                        iterations=args.iterations, device=device)
                        for name, masked in (("explicit_mask", True), ("unmasked", False))}
        rows.append(dict(query_length=width, key_length=length, forward_equal=True,
                         gradient_errors=gradient_errors, results=measurements))
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    synchronize = torch.npu.synchronize if device.type == "npu" else lambda: None
    torch.manual_seed(73)
    rows = []
    for length in (1, 491, 4096, 8160, 32768):
        pages = (length + 127) // 128
        cache = torch.randn(pages * 2 + 3, 128, 2, 256, device=device, dtype=torch.bfloat16)
        table = torch.randperm(cache.shape[0], device=device)[:pages][None].to(torch.int32)
        expected = selected_pages(cache, table, length)
        before = cache.clone()
        functions = dict(pages=partial(selected_pages, cache, table, length),
                         tokens=partial(selected_tokens, cache, table, length),
                         gather=partial(selected_tokens, cache, table, length, gather=True),
                         production=partial(paged_history, cache, table, length))
        measurements = {}
        for name, operation in functions.items():
            actual = operation()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if not actual.is_contiguous():
                raise AssertionError(f"{name}: noncontiguous attention input")
            actual.zero_()
            torch.testing.assert_close(cache, before, rtol=0, atol=0)
            del actual
            measurements[name] = measure(operation, synchronize, repeats=args.repeats,
                                         iterations=args.iterations, device=device)
        rows.append(dict(length=length, physical_pages=cache.shape[0], results=measurements,
                         bitwise_equal=True, independent_snapshot=True))
        print(json.dumps(rows[-1]), flush=True)
        del cache, expected, before
    report = dict(scope="history and attention microbenchmarks, not end-to-end model throughput",
                  torch=torch.__version__, device=str(device),
                  device_name=torch.npu.get_device_name(0) if device.type == "npu" else "CPU",
                  repeats=args.repeats, iterations=args.iterations, rows=rows,
                  convolution=convolution_tails(device, args, synchronize),
                  attention_masks=attention_masks(device, args, synchronize))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
