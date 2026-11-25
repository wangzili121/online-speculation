#!/usr/bin/env python3
"""Compile and check Orthrus read-only gathers against the A3 cache codec."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from blockspec.pypto_dsv41_packed_history import (
    PAGE_ROWS, ROW_TILE, compressed_plan, make_packed_history_gather, window_plan,
)
from blockspec.pypto_dsv41_swa import HEAD_DIM, WINDOW


def validate_swa(args, config, code, scale, golden, page_table):
    from pypto.runtime import ChipWorker

    from blockspec.pypto_dsv41_packed_swa import (
        packed_swa_controls, packed_swa_programs, packed_swa_views, run_packed_swa,
    )
    from blockspec.vllm_ascend.dsv41_attention import PagedRows, draft_attention

    pages = code.shape[0]
    generator = torch.Generator().manual_seed(41123)
    heads = 16
    reports = []
    for block in (8, 16, 32):
        anchor = 129
        query = torch.randn((block, heads, HEAD_DIM), generator=generator).bfloat16()
        temporary = torch.randn((max(16, block), HEAD_DIM), generator=generator).bfloat16()
        sink = torch.linspace(-4, 4, heads)
        zeros = torch.zeros((WINDOW, HEAD_DIM), dtype=torch.bfloat16)
        out = torch.empty((heads * block, HEAD_DIM), dtype=torch.float32)
        lse = torch.empty((heads * block, 1), dtype=torch.float32)
        scratch = torch.empty_like(zeros)
        rows, starts, spans, bias = packed_swa_controls(page_table, anchor, pages)
        controls = [rows, starts, spans, bias]
        q = query.flatten(1)
        gather, attention = packed_swa_programs(pages, heads, block)
        programs = (
            gather.compile(code, scale, rows, starts, scratch, config=config),
            attention.compile(q, scratch, spans, temporary, bias, sink, zeros, out, lse, config=config),
        )
        if args.compile_only:
            reports.append(dict(kind="swa", block=block, compiled=True))
            continue
        fields = [code[None], scale[None], *controls, scratch, q, temporary, sink, zeros, out, lse]
        resident = [field.clone() for field in fields]
        with ChipWorker(config) as worker:
            handles = [worker.alloc_tensor(tuple(field.shape), field.dtype) for field in resident]
            for host, device in zip(resident[:-2], handles[:-2], strict=True):
                worker.copy_to(device.data_ptr, host.data_ptr(), host.nbytes)
            caches = dict(zip(("window_cache", "window_scales"), handles[:2], strict=True))
            packed, scales = packed_swa_views(worker, caches, 0, worker_id=0)
            run_packed_swa(worker, programs, packed, scales, handles[2:6], *handles[6:], config=config)
            for host, device in zip(resident[-2:], handles[-2:], strict=True):
                worker.copy_from(host.data_ptr(), device.data_ptr, host.nbytes)

        pv, plse = draft_attention(
            query[None], temporary[None, :block],
            PagedRows(golden.squeeze(-2).bfloat16(), torch.tensor([page_table])),
            torch.tensor([anchor]), sink, window=WINDOW,
        )
        expected_out = pv.transpose(1, 2).reshape_as(out)
        expected_lse = plse.transpose(1, 2).reshape_as(lse)
        torch.testing.assert_close(resident[-2], expected_out, atol=1e-5, rtol=1.3e-6)
        torch.testing.assert_close(resident[-1], expected_lse, atol=2e-5, rtol=2e-6)
        reports.append(dict(kind="swa", block=block, anchor=anchor,
                            max_abs=(resident[-2] - expected_out).abs().max().item(),
                            lse_max_abs=(resident[-1] - expected_lse).abs().max().item()))
    return reports


def validate(args):
    from models.deepseek_v4_1_flash_a3.quant import (
        dequantize_mxfp4_cache, dequantize_mxfp8_cache,
        quantize_mxfp4_cache, quantize_mxfp8_cache,
    )
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(41122)
    pages = args.pages
    page_table = (pages - 2, 0, pages - 1)
    values = torch.randn(pages, PAGE_ROWS, 1, HEAD_DIM, generator=generator)
    mode = ExecutionMode.SIM if args.simulator else ExecutionMode.ONBOARD
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=mode, device_id=args.device,
                       distributed_config=DistributedConfig(device_ids=[args.device]))
    reports = []
    for kind, items in (
        ("fp8", [(anchor, window_plan(page_table, anchor, pages)) for anchor in (0, 19, 128, 129, 271)]),
        ("fp4", [(anchor, compressed_plan(page_table, anchor, ratio, pages, selected))
                 for anchor, ratio, selected in (
            (0, 2, [-1] * args.topk),
            (513, 2, (list(range(128, 160)) + [255, 1, -1, 128] * args.topk)[:args.topk]),
            (700, 2, (list(range(256, 288)) + [255, 1, -1, 128] * args.topk)[:args.topk]),
            (384, 1, (list(range(384)) + [-1] * args.topk)[:args.topk]),
        )]),
    ):
        if kind == "fp8":
            code, scale = quantize_mxfp8_cache(values)
            golden = dequantize_mxfp8_cache(code, scale)
        else:
            code, scale = quantize_mxfp4_cache(values, 16, "bf16")
            # The FP32 scale product must also work outside the FP16 range.
            scale[0, 3, 0, :].fill_(2**20)
            golden = dequantize_mxfp4_cache(code, scale, 16, "bf16")
        code = code.view(torch.uint8)
        flat = golden.reshape(-1, HEAD_DIM).bfloat16()
        count = len(items[0][1].rows)
        output = torch.empty(count, HEAD_DIM, dtype=torch.bfloat16)
        rows = torch.empty((count // ROW_TILE, ROW_TILE), dtype=torch.int32)
        starts = torch.empty((count // ROW_TILE,), dtype=torch.int32)
        gather = make_packed_history_gather(kind, pages, count).compile(
            code, scale, rows, starts, output, config=config,
        )
        if kind == "fp8" and args.swa:
            reports.extend(validate_swa(args, config, code, scale, golden, page_table))
        if args.compile_only:
            reports.append(dict(kind=kind, compiled=True))
            continue
        with ChipWorker(config) as worker:
            fields = (code, scale, rows, starts, output)
            buffers = [worker.alloc_tensor(tuple(field.shape), field.dtype) for field in fields]
            for host, device in zip(fields[:2], buffers[:2], strict=True):
                worker.copy_to(device.data_ptr, host.data_ptr(), host.nbytes)
            for anchor, plan in items:
                selected = torch.tensor(plan.rows)
                rows.copy_(selected.reshape_as(rows))
                starts.copy_(torch.tensor(plan.contiguous_starts))
                expected = flat[selected.clamp_min(0)].masked_fill((selected < 0)[:, None], 0)
                for host, device in zip(fields[2:4], buffers[2:4], strict=True):
                    worker.copy_to(device.data_ptr, host.data_ptr(), host.nbytes)
                worker.run(gather, *buffers, config=config)
                timings = []
                for _ in range(args.iterations):
                    start = perf_counter()
                    worker.run(gather, *buffers, config=config)
                    timings.append(perf_counter() - start)
                    worker.copy_from(output.data_ptr(), buffers[-1].data_ptr, output.nbytes)
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)
                reports.append(dict(kind=kind, anchor=anchor, max_abs=0,
                                    contiguous_tiles=sum(start >= 0 for start in plan.contiguous_starts),
                                    resident_call_median_us=median(timings) * 1e6))
            for host, device in zip(fields[:2], buffers[:2], strict=True):
                retained = torch.empty_like(host)
                worker.copy_from(retained.data_ptr(), device.data_ptr, retained.nbytes)
                torch.testing.assert_close(retained, host, atol=0, rtol=0)
    return dict(passed=True, npu_execution=not args.compile_only and not args.simulator,
                simulator_execution=not args.compile_only and args.simulator, cases=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--topk", type=int, choices=(128, 512), default=512)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--swa", action="store_true", help="Also check the device-resident SWA chain")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.pages < 4:
        parser.error("require positive iterations and at least four physical pages")
    payload = json.dumps(validate(args), indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
