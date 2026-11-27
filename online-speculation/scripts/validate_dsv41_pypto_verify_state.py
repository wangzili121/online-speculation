#!/usr/bin/env python3
"""Check C2A Decode state after discarding a speculative suffix, on one NPU."""

import argparse
import ctypes
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch


def resident_commit_entries(capacity, blocks, state_blocks):
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.compressor import (
        c2a_compressor_pool, c2a_compressor_project, c2a_compressor_publish, c2a_compressor_state_write,
    )
    from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE, FLASH, STATE_CAPACITY

    hidden = FLASH.hidden_size
    head = FLASH.head_dim
    rope = FLASH.qk_rope_head_dim // 2

    @pl.jit(auto_scope=False)
    def verify_pending(
        source: pl.Tensor[[capacity, hidden], pl.BF16],
        wkv: pl.Tensor[[hidden, head], pl.BF16],
        wgate: pl.Tensor[[hidden, head], pl.BF16],
        norm: pl.Tensor[[head], pl.BF16],
        positions: pl.Tensor[[capacity], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        starts: pl.Tensor[[2], pl.INT32],
        state_table: pl.Tensor[[1, 1], pl.INT32],
        slots: pl.Tensor[[capacity], pl.INT64],
        cos: pl.Tensor[[capacity, rope], pl.FP32],
        sin: pl.Tensor[[capacity, rope], pl.FP32],
        width: pl.Scalar[pl.INT32],
        state: pl.Tensor[[state_blocks, STATE_CAPACITY, 2 * head], pl.FP32],
        latent: pl.InOut[pl.Tensor[[capacity, head], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[blocks, BLOCK_SIZE, 1, head // 2], pl.UINT8]],
        scales: pl.InOut[pl.Tensor[[blocks, BLOCK_SIZE, 1, head // 16], pl.BF16]],
        kv: pl.Out[pl.Tensor[[capacity, head], pl.FP32]],
        score: pl.Out[pl.Tensor[[capacity, head], pl.FP32]],
    ):
        entry = pl.system.task_dummy(deps=[])
        c2a_compressor_project(source, wkv, wgate, kv, score, width)
        pooled = c2a_compressor_pool(kv, score, state, positions, norm, latent, width,
                                     starts, requests, state_table, state_blocks, entry)
        c2a_compressor_publish(latent, cos, sin, slots, cache, scales, width, entry, pooled, max_workers=4)

    @pl.jit(auto_scope=False)
    def commit_prefix(
        kv: pl.Tensor[[capacity, head], pl.FP32],
        score: pl.Tensor[[capacity, head], pl.FP32],
        starts: pl.Tensor[[2], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_table: pl.Tensor[[1, 1], pl.INT32],
        state: pl.InOut[pl.Tensor[[state_blocks, STATE_CAPACITY, 2 * head], pl.FP32]],
        keep: pl.Scalar[pl.INT32],
    ):
        entry = pl.system.task_dummy(deps=[])
        c2a_compressor_state_write(kv, score, starts, positions, requests, state_table, state, keep, entry)

    return verify_pending, commit_prefix


def validate(checkpoint_path, device, resident_commit=False, prefix_sweep=False):
    from models.deepseek_v4_1_flash_a3.compressor import l2_compressor
    from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE, FLASH, STATE_CAPACITY
    from models.deepseek_v4_1_flash_a3.utils import paged_slots
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint
    from pypto.backend import BackendType
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

    torch.set_num_threads(2)
    checkpoint = Checkpoint(checkpoint_path)
    layer = 2
    prefix = f"layers.{layer}.attn.compressor."
    weights = [checkpoint.tensor(prefix + name).t().contiguous() for name in ("wkv.weight", "wgate.weight")]
    weights.append(checkpoint.tensor(prefix + "norm.weight"))
    head = FLASH.head_dim
    capacity = 16
    generator = torch.Generator().manual_seed(41127)
    pages = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    table = torch.tensor([[1]], dtype=torch.int32)
    blank = [
        torch.randn((2, STATE_CAPACITY, 2 * head), generator=generator),
        torch.zeros((capacity, head), dtype=torch.bfloat16),
        torch.full((3, BLOCK_SIZE, 1, head // 2), 199, dtype=torch.uint8),
        torch.full((3, BLOCK_SIZE, 1, head // 16), 2, dtype=torch.bfloat16),
    ]
    samples = torch.randn((capacity, FLASH.hidden_size), generator=generator).bfloat16()
    replacement = torch.randn((capacity, FLASH.hidden_size), generator=generator).bfloat16()
    angle_steps = FLASH.compress_rope_theta ** (
        -torch.arange(0, FLASH.qk_rope_head_dim, 2).float() / FLASH.qk_rope_head_dim
    )

    def inputs(start, data):
        live = len(data)
        position = torch.full((capacity,), -1, dtype=torch.int32)
        position[:live] = torch.arange(start, start + live, dtype=torch.int32)
        requests = torch.full_like(position, -1)
        requests[:live] = 0
        slots = torch.full((capacity,), -1, dtype=torch.int64)
        slots[:live] = paged_slots(position[:live], requests[:live], pages, BLOCK_SIZE, 2, True)
        angles = (position.clamp_min(0) // 2 * 2).float()[:, None] * angle_steps
        source = torch.zeros_like(samples)
        source[:live].copy_(data)
        return [source, *weights, position, requests, torch.tensor([0, live], dtype=torch.int32),
                table, slots, angles.cos(), angles.sin()]

    def physical_rows(start, end):
        # The compressed slot for a pair becomes visible when its odd token
        # is committed. Use a permuted physical page table to cover rollover.
        return [int(pages[0, row // BLOCK_SIZE]) * BLOCK_SIZE + row % BLOCK_SIZE
                for row in range(start // 2, end // 2)]

    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=device)
    example = inputs(31, samples)
    compiled = l2_compressor.compile(*example, capacity, 2, *blank, config=config)
    if resident_commit:
        projections = [torch.empty((capacity, head), dtype=torch.float32) for _ in range(2)]
        verify_pending, commit_prefix = resident_commit_entries(capacity, 3, 2)
        pending = verify_pending.compile(*example, capacity, *blank, *projections, config=config)
        commit = commit_prefix.compile(*projections, example[6], example[4], example[5], example[7],
                                       blank[0], 1, config=config)
    report = []
    overhead = {}
    with ChipWorker(config) as worker:
        device_inputs = [worker.alloc_tensor(tuple(value.shape), value.dtype, init=value) for value in example]
        state = [worker.alloc_tensor(tuple(value.shape), value.dtype) for value in blank]
        changing = (0, 4, 5, 6, 8, 9, 10)

        def stage(start, data):
            values = inputs(start, data)
            for index in changing:
                value = values[index]
                worker.copy_to(device_inputs[index].data_ptr, value.data_ptr(), value.nbytes)

        def launch(start, data):
            stage(start, data)
            worker.run(compiled, *device_inputs, len(data), 2, *state, config=config)

        def read(index):
            result = torch.empty_like(blank[index])
            worker.copy_from(result.data_ptr(), state[index].data_ptr, result.nbytes)
            return result

        def reset():
            for host, target in zip(blank, state, strict=True):
                worker.copy_to(target.data_ptr, host.data_ptr(), host.nbytes)

        if resident_commit:
            saved = [worker.alloc_tensor(tuple(value.shape), value.dtype) for value in projections]

            def pending_run(width):
                worker.run(pending, *device_inputs, width, *state, *saved, config=config)

            def commit_run(keep):
                positions, requests, starts, state_table = device_inputs[4:8]
                worker.run(commit, *saved, starts, positions, requests, state_table, state[0], keep, config=config)

        cases = ((31, 8, 1), (32, 16, 1), (127, 8, 2), (253, 16, 2), (255, 16, 8))
        if prefix_sweep:
            cases += tuple((start, width, keep) for start, width in ((127, 8), (255, 16))
                           for keep in range(1, width + 1))
        for start, width, keep in cases:
            continuation = start + keep
            kept_rows = physical_rows(start, continuation)
            suffix_rows = physical_rows(continuation, start + width)
            next_rows = physical_rows(continuation, continuation + capacity)
            observed = []
            paths = ("serial", "verify", "pending_commit") if resident_commit else ("serial", "verify")
            for path in paths:
                reset()
                if path == "pending_commit":
                    stage(start, samples[:width])
                    pending_run(width)
                    torch.testing.assert_close(read(0), blank[0], atol=0, rtol=0, msg="Verify modified the committed ring")
                    commit_run(keep)
                elif path == "verify":
                    launch(start, samples[:width])
                else:
                    for offset in range(keep):
                        launch(start + offset, samples[offset:offset + 1])
                predecessor = read(0)[1, (continuation - 1) % STATE_CAPACITY]
                cache = read(2).flatten(0, 1)
                if suffix_rows:
                    assert (not torch.equal(cache[suffix_rows], blank[2].flatten(0, 1)[suffix_rows])) == (path != "serial")
                committed = (cache[kept_rows], read(3).flatten(0, 1)[kept_rows])
                launch(continuation, replacement)
                observed.append((predecessor, *committed, read(1), read(2).flatten(0, 1)[next_rows],
                                 read(3).flatten(0, 1)[next_rows]))
            labels = (
                "committed ring predecessor", "committed FP4 payload", "committed scales",
                "next latent", "next FP4 payload", "next scales",
            )
            serial, unprotected = observed[:2]
            if resident_commit:
                for label, reference, actual in zip(labels, serial, observed[2], strict=True):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0, msg=f"{label} at {start}")
            else:
                for label, reference, actual in zip(labels, serial, unprotected, strict=True):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0, msg=f"{label} at {start}")
            report.append(dict(anchor=start, width=width, committed=keep, continuation=continuation,
                               kept_fp4_rows=len(kept_rows), rejected_fp4_rows=len(suffix_rows),
                               continuation_fp4_rows=len(next_rows), exact=True,
                               unprotected_differences=[label for label, ref, actual in
                                                        zip(labels, serial, unprotected, strict=True)
                                                        if not torch.equal(ref, actual)]))

        if resident_commit or prefix_sweep:
            # Compare the ring commit with the current journal/replay approach
            # on the same single-layer workload. Journal the whole ring and
            # only the contiguous compressed rows this case overwrites.
            start, width, keep = cases[0]
            touched = physical_rows(start, start + width)
            page, row = divmod(touched[0], BLOCK_SIZE)
            assert touched == list(range(touched[0], touched[0] + len(touched)))
            journal = [(state[0], torch.empty_like(blank[0]), 0)]
            for i in (2, 3):
                view = blank[i][page, row:row + len(touched)]
                journal.append((state[i], torch.empty_like(view), view.data_ptr() - blank[i].data_ptr()))
            # ChipWorker accepts allocation base addresses. ACL handles the
            # same exact ranges as the serving journal's offset copies.
            copy = ctypes.CDLL("libascendcl.so").aclrtMemcpy
            copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

            def copy_region(destination, source, size, direction):
                status = copy(destination, size, source, size, direction)
                if status:
                    raise RuntimeError(f"aclrtMemcpy failed with code {status}")

            modes = ("journal_replay", "resident_commit" if resident_commit else "direct_prefix")
            samples_us = {mode: [] for mode in modes}
            for index in range(33):
                for mode in modes[index % 2:] + modes[:index % 2]:
                    reset()
                    stage(start, samples[:width])
                    begin = perf_counter_ns()
                    if mode == "journal_replay":
                        for target, host, offset in journal:
                            copy_region(host.data_ptr(), target.data_ptr + offset, host.nbytes, 2)
                        worker.run(compiled, *device_inputs, width, 2, *state, config=config)
                        for target, host, offset in journal:
                            copy_region(target.data_ptr + offset, host.data_ptr(), host.nbytes, 1)
                        launch(start, samples[:keep])
                    elif mode == "resident_commit":
                        pending_run(width)
                        commit_run(keep)
                    else:
                        worker.run(compiled, *device_inputs, width, 2, *state, config=config)
                    elapsed = (perf_counter_ns() - begin) / 1000
                    if index >= 3:
                        samples_us[mode].append(elapsed)
            overhead = dict(case=dict(anchor=start, width=width, committed=keep), warmup=3, rounds=30,
                            wall_median_us={mode: round(median(values), 2) for mode, values in samples_us.items()},
                            samples_us=samples_us)
    return dict(passed=True, executed=True, layer=layer, resident_commit=resident_commit, cases=report,
                benchmark=overhead)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resident-commit", action="store_true",
                        help="Validate an isolated Verify and resident prefix commit against serial Decode")
    parser.add_argument("--prefix-sweep", action="store_true",
                        help="Check all accepted prefix lengths across K8 and K16 page boundaries")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.checkpoint, args.device, args.resident_commit, args.prefix_sweep)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
