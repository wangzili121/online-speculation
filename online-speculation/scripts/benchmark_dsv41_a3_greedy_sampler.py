#!/usr/bin/env python3
"""Compare the resident A3 greedy sampler with a pinned Lib checkout."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
from statistics import median
import sys
from time import perf_counter_ns

import pypto.language as pl
import torch
from models.deepseek_v4_1_flash_a3.config import FLASH, MAX_LOGIT_ROWS, SAMPLED_FINITE_COLUMN, SAMPLED_IDS_PAD
from models.deepseek_v4_1_flash_a3.lm_head import greedy_sample
from pypto.backend import BackendType
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig


ROWS, VOCAB, PAD = MAX_LOGIT_ROWS, FLASH.vocab_size, SAMPLED_IDS_PAD


def baseline_entry(path):
    module_name = "models.deepseek_v4_1_flash_a3._greedy_benchmark_baseline"
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    prior_sample = mod.greedy_sample

    @pl.jit
    def legacy(logits: pl.Tensor[[ROWS, VOCAB], pl.FP32], ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]]):
        return prior_sample(logits, ids)

    return legacy


@pl.jit
def vector(
    logits: pl.Tensor[[ROWS, VOCAB], pl.FP32],
    row_indices: pl.Tensor[[ROWS], pl.INT32],
    ids: pl.Out[pl.Tensor[[ROWS, PAD], pl.INT32]],
):
    return greedy_sample(logits, row_indices, ids)


def fixture():
    logits = torch.randn((ROWS, VOCAB), generator=torch.Generator().manual_seed(205))
    logits[0, 1279] = logits[0, 1280] = logits[0, -1] = 100
    logits[1, 1279] = logits[1, 2559] = 100
    logits[2, 55] = torch.finfo(torch.float32).max
    logits[3].fill_(-torch.finfo(torch.float32).max)
    logits[4].fill_(-1)
    logits[4, 0], logits[4, 1] = -0., 0.
    return logits


def device_samples(path, cases, warmup, rounds):
    pattern = re.compile(r"inv=(\d+).*name=chip.run.runner_run.device_wall ts=\d+ dur=(\d+) clk=dev")
    spans = {}
    with path.open(errors="replace") as log:
        for line in log:
            match = pattern.search(line)
            if match:
                spans[int(match[1])] = int(match[2]) / 1000
    steps = warmup + rounds
    if set(spans) != set(range(1, 2 * len(cases) * steps + 2)):
        raise ValueError("Device trace must contain exactly one window per benchmark dispatch")
    for index, case in enumerate(cases.values()):
        samples = dict(prior=[], vector=[])
        for step in range(warmup, steps):
            first = 2 * (index * steps + step) + 1
            tags = ("prior", "vector") if step % 2 == 0 else ("vector", "prior")
            for offset, tag in enumerate(tags):
                samples[tag].append(spans[first + offset])
        case["device_samples_us"] = samples
        case["device_median_us"] = {name: median(times) for name, times in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Path to the pinned baseline lm_head.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-log", type=Path, help="File receiving stdout and stderr with device tracing enabled")
    args = parser.parse_args()
    torch.set_num_threads(1)
    original = fixture()
    logits = original.clone()
    indices = torch.zeros(ROWS, dtype=torch.int32)
    ids = torch.empty(ROWS, PAD, dtype=torch.int32)
    cfg = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=args.device)
    prior = baseline_entry(args.baseline).compile(logits, ids, config=cfg)
    current = vector.compile(logits, indices, ids, config=cfg)
    cases = {}
    with ChipWorker(cfg) as worker:
        source = worker.alloc_tensor(logits.shape, logits.dtype)
        mask = worker.alloc_tensor(indices.shape, indices.dtype)
        output = worker.alloc_tensor(ids.shape, ids.dtype)

        def upload():
            for tensor, device in ((logits, source), (indices, mask)):
                worker.copy_to(device.data_ptr, tensor.data_ptr(), tensor.nbytes)

        def read():
            worker.copy_from(ids.data_ptr(), output.data_ptr, ids.nbytes)

        programs = ((prior, (source, output), "prior"), (current, (source, mask, output), "vector"))
        for active in (0, 1, 8, ROWS):
            logits.copy_(original)
            logits[active:].zero_()
            indices[:active].zero_()
            indices[active:].fill_(-1)
            upload()
            expected = logits.argmax(dim=1)
            samples = dict(prior=[], vector=[])
            for step in range(args.rounds + args.warmup):
                for program, operands, name in (programs if step % 2 == 0 else tuple(reversed(programs))):
                    start = perf_counter_ns()
                    worker.run(program, *operands, config=cfg)
                    duration = (perf_counter_ns() - start) / 1000
                    if step >= args.warmup:
                        samples[name].append(duration)
                    read()
                    assert torch.equal(ids[:, 0], expected), (name, active, ids[:, 0])
                    assert torch.all(ids[:, SAMPLED_FINITE_COLUMN] == 1)
            cases[active] = dict(samples_us=samples, median_us={name: median(times) for name, times in samples.items()})

        logits.copy_(original)
        indices.fill_(-1)
        indices[[0, 2, 4, 6, 8, 10, 12, 14]] = 0
        for index, bad in ((2, float("nan")), (4, float("inf")), (6, float("-inf"))):
            logits[index, 1280] = logits[index + 1, 1280] = bad
        upload()
        worker.run(current, source, mask, output, config=cfg)
        read()
        flags = torch.ones(ROWS, dtype=torch.int32)
        flags[[2, 4, 6]] = 0
        assert torch.equal(ids[:, SAMPLED_FINITE_COLUMN], flags)
        assert torch.all(ids[1::2, 0] == 0)
        good = (indices >= 0) & flags.bool()
        assert torch.equal(ids[good, 0], logits[good].argmax(dim=1))

    report = dict(passed=True, vocab=VOCAB, physical_rows=ROWS, rounds=args.rounds, warmup=args.warmup,
                  scope="blocking execution of one resident sampler on a single 910B3", cases=cases)
    if args.trace_log is not None:
        device_samples(args.trace_log, cases, args.warmup, args.rounds)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({active: result["median_us"] for active, result in cases.items()}), flush=True)


if __name__ == "__main__":
    main()
