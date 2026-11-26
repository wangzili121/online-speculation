#!/usr/bin/env python3
"""Check C2A Decode state after discarding a speculative suffix, on one NPU."""

import argparse
import json
from pathlib import Path

import torch


def validate(checkpoint_path, device):
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
    report = []
    with ChipWorker(config) as worker:
        device_inputs = [worker.alloc_tensor(tuple(value.shape), value.dtype, init=value) for value in example]
        state = [worker.alloc_tensor(tuple(value.shape), value.dtype) for value in blank]
        changing = (0, 4, 5, 6, 8, 9, 10)

        def launch(start, data):
            values = inputs(start, data)
            for index in changing:
                value = values[index]
                worker.copy_to(device_inputs[index].data_ptr, value.data_ptr(), value.nbytes)
            worker.run(compiled, *device_inputs, len(data), 2, *state, config=config)

        def read(index):
            result = torch.empty_like(blank[index])
            worker.copy_from(result.data_ptr(), state[index].data_ptr, result.nbytes)
            return result

        for start, width, keep in ((31, 8, 1), (32, 16, 1), (127, 8, 2), (253, 16, 2), (255, 16, 8)):
            continuation = start + keep
            kept_rows = physical_rows(start, continuation)
            suffix_rows = physical_rows(continuation, start + width)
            next_rows = physical_rows(continuation, continuation + capacity)
            observed = []
            for speculative in (False, True):
                for host, target in zip(blank, state, strict=True):
                    worker.copy_to(target.data_ptr, host.data_ptr(), host.nbytes)
                if speculative:
                    launch(start, samples[:width])
                else:
                    for offset in range(keep):
                        launch(start + offset, samples[offset:offset + 1])
                predecessor = read(0)[1, (continuation - 1) % STATE_CAPACITY]
                cache = read(2).flatten(0, 1)
                assert (not torch.equal(cache[suffix_rows], blank[2].flatten(0, 1)[suffix_rows])) == speculative
                committed = (cache[kept_rows], read(3).flatten(0, 1)[kept_rows])
                launch(continuation, replacement)
                observed.append((predecessor, *committed, read(1), read(2).flatten(0, 1)[next_rows],
                                 read(3).flatten(0, 1)[next_rows]))
            for stage, serial, packed in zip(
                ("committed ring predecessor", "committed FP4 payload", "committed scales",
                 "next latent", "next FP4 payload", "next scales"), *observed,
                strict=True,
            ):
                torch.testing.assert_close(packed, serial, atol=0, rtol=0, msg=f"{stage} at {start}")
            report.append(dict(anchor=start, width=width, committed=keep, continuation=continuation,
                               kept_fp4_rows=len(kept_rows), rejected_fp4_rows=len(suffix_rows),
                               continuation_fp4_rows=len(next_rows), exact=True))
    return dict(passed=True, executed=True, layer=layer, cases=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.checkpoint, args.device)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
