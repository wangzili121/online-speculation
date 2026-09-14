"""Benchmark exact request-local categorical samplers for Orthrus drafts."""

from __future__ import annotations

import argparse
import gc
import time

import torch

from blockspec.reporting import dumps


def generators(batch, seed):
    result = []
    for index in range(batch):
        generator = torch.Generator(device="npu")
        generator.manual_seed(seed + index)
        result.append(generator)
    return result


def exponential_race(probabilities, request_generators):
    rows = []
    for probability, generator in zip(probabilities, request_generators):
        exponential = torch.empty_like(probability).exponential_(
            generator=generator)
        rows.append((probability / exponential.clamp_min(
            torch.finfo(torch.float32).tiny)).argmax(-1))
    return torch.stack(rows)


def native_multinomial(probabilities, request_generators):
    return torch.stack([
        torch.multinomial(
            probability, 1, replacement=True, generator=generator
        ).squeeze(-1)
        for probability, generator in zip(probabilities, request_generators)
    ])


def inverse_cdf(probabilities, request_generators):
    rows = []
    for probability, generator in zip(probabilities, request_generators):
        uniform = torch.rand(
            probability.shape[0], 1, device=probability.device,
            generator=generator)
        rows.append(torch.searchsorted(
            probability.cumsum(-1), uniform).squeeze(-1))
    return torch.stack(rows)


def batched_inverse_cdf(probabilities, request_generators):
    uniform = torch.stack([
        torch.rand(
            probabilities.shape[1], 1, device=probabilities.device,
            generator=generator)
        for generator in request_generators
    ])
    return torch.searchsorted(
        probabilities.cumsum(-1), uniform
    ).squeeze(-1).clamp_max(probabilities.shape[-1] - 1)


def measure(function, probabilities, *, batch, seed, warmup, iterations):
    request_generators = generators(batch, seed)
    for _ in range(warmup):
        function(probabilities, request_generators)
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    start_memory = torch.npu.memory_allocated()
    start = time.perf_counter()
    for _ in range(iterations):
        output = function(probabilities, request_generators)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "mean_ms": elapsed * 1000 / iterations,
        "start_allocated_bytes": start_memory,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "incremental_peak_bytes": torch.npu.max_memory_allocated() - start_memory,
        "output_shape": list(output.shape),
        "output_in_range": bool((output >= 0).all() and
                                (output < probabilities.shape[-1]).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=23)
    parser.add_argument("--draft-tokens", type=int, default=31)
    parser.add_argument("--vocabulary", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    logits = torch.randn(
        args.batch, args.draft_tokens, args.vocabulary,
        dtype=torch.float32, device="npu")
    probabilities = logits.softmax(-1)
    del logits
    methods = {
        "exponential_race": exponential_race,
        "native_multinomial": native_multinomial,
        "inverse_cdf": inverse_cdf,
        "batched_inverse_cdf": batched_inverse_cdf,
    }
    results = {}
    with torch.inference_mode():
        for name, function in methods.items():
            results[name] = measure(
                function, probabilities, batch=args.batch, seed=args.seed,
                warmup=args.warmup, iterations=args.iterations)
    baseline = results["exponential_race"]["mean_ms"]
    print(dumps({
        "shape": vars(args),
        "methods": results,
        "speedup_over_exponential": {
            name: baseline / result["mean_ms"]
            for name, result in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
