#!/usr/bin/env python3
"""Check that rejected cache rows are invisible to the A3 Orthrus draft."""

import argparse
import json
from pathlib import Path

import torch
from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE, FLASH, TP_SIZE
from models.deepseek_v4_1_flash_a3.orthrus_draft.attention_history import make_packed_swa_chain, packed_swa_controls
from models.deepseek_v4_1_flash_a3.orthrus_draft.history import packed_history_lookup
from models.deepseek_v4_1_flash_a3.orthrus_draft.packed_indexer import make_packed_draft_indexer, packed_index_controls
from models.deepseek_v4_1_flash_a3.orthrus_draft.sparse_attention import TOPK
from models.deepseek_v4_1_flash_a3.quant import quantize_mxfp4_cache, quantize_mxfp8_cache
from pypto.backend import BackendType
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig


def physical_prefix(pages, size):
    logical = torch.arange(size)
    return pages[logical // BLOCK_SIZE].long() * BLOCK_SIZE + logical % BLOCK_SIZE


def change_tail(values, table, visible):
    shape = values.shape
    mask = torch.ones(shape[0] * shape[1], dtype=torch.bool)
    mask[physical_prefix(table, visible)] = False
    changed = values.clone().flatten(0, 1)
    changed[mask] = changed[mask].flip(-1) * -3.0
    return changed.reshape(shape), int(mask.sum())


def validate(args):
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(31127)
    def random(shape, dtype=torch.float32):
        return torch.randn(shape, generator=generator).to(dtype)

    table = torch.tensor([3, 0, 5, 2, 7, 1, 6, 4], dtype=torch.int32)
    pages, rows, head = len(table), args.block, FLASH.head_dim
    capacity = max(16, rows)
    heads = FLASH.num_attention_heads // TP_SIZE
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=args.device)
    reports = []

    keys = random((pages, BLOCK_SIZE, 1, head))
    altered, dirty = change_tail(keys, table, 257)
    payloads = [tuple(quantize_mxfp8_cache(x)) for x in (keys, altered)]
    temporary = random((capacity, head), torch.bfloat16)
    temporary[rows:].zero_()
    common = (packed_history_lookup("fp8"), *packed_swa_controls(table, 257, pages),
              random((rows, heads * head), torch.bfloat16), temporary,
              torch.linspace(-4, 4, heads), torch.zeros((BLOCK_SIZE, head), dtype=torch.bfloat16))
    output = (torch.empty((rows * heads, head)), torch.empty((rows * heads, 1)))
    cases = [("swa", 257, 257, dirty, make_packed_swa_chain(pages, heads, rows),
              [[p.view(torch.uint8), scale, *common] for p, scale in payloads], output)]

    for ratio, anchor in ((1, 513), (2, 1027)):
        index_dim = FLASH.index_head_dim
        index_heads = FLASH.index_n_heads
        half_rope = FLASH.qk_rope_head_dim // 2
        keys = random((pages, BLOCK_SIZE, 1, index_dim))
        visible = anchor // ratio
        altered, dirty = change_tail(keys, table, visible)
        payloads = [quantize_mxfp4_cache(x, 32, "e8m0", round_to_even=True) for x in (keys, altered)]
        bucket, *control = packed_index_controls(table, anchor, ratio, rows)
        common = (
            random((capacity, FLASH.hidden_size), torch.bfloat16),
            random((capacity, FLASH.q_lora_rank), torch.bfloat16),
            random((FLASH.q_lora_rank, index_heads * index_dim), torch.bfloat16) * 0.01,
            random((FLASH.hidden_size, index_heads), torch.bfloat16) * 0.01,
            torch.ones((capacity, half_rope)), torch.zeros((capacity, half_rope)),
        )
        output = (torch.empty((rows, TOPK), dtype=torch.int32),
                  torch.empty((rows,), dtype=torch.int32), torch.empty((rows, TOPK)))
        cases.append((f"index_c{ratio}a", anchor, visible, dirty, make_packed_draft_indexer(pages, bucket, ratio, rows),
                      [[*common, p.view(torch.uint8), scale, *control] for p, scale in payloads], output))

    for name, anchor, visible, dirty, entry, variants, outputs in cases:
        program = entry.compile(*variants[0], *outputs, config=config)
        assert any(not torch.equal(a, b) for a, b in zip(*variants, strict=True))
        with ChipWorker(config) as worker:
            device_inputs = [worker.alloc_tensor(x.shape, x.dtype, init=x) for x in variants[0]]
            device_outputs = [worker.alloc_tensor(x.shape, x.dtype) for x in outputs]
            expected = None
            for inputs in variants:
                for current, initial, handle in zip(inputs, variants[0], device_inputs, strict=True):
                    if current is not initial:
                        worker.copy_to(handle.data_ptr, current.data_ptr(), current.nbytes)
                worker.run(program, *device_inputs, *device_outputs, config=config)
                for host, handle in zip(outputs, device_outputs, strict=True):
                    worker.copy_from(host.data_ptr(), handle.data_ptr, host.nbytes)
                if expected is None:
                    expected = tuple(x.clone() for x in outputs)
                else:
                    for got, want in zip(outputs, expected, strict=True):
                        torch.testing.assert_close(got, want, atol=0, rtol=0)
                    if name.startswith("index"):
                        visible_rows = physical_prefix(table, visible)
                        selected, lengths, _ = outputs
                        assert torch.all(lengths == TOPK)
                        assert bool(torch.isin(selected, visible_rows).all())
        reports.append(dict(operator=name, anchor=anchor, committed=visible, changed_tail=dirty, exact=True))
    return dict(passed=True, block=rows, cases=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
