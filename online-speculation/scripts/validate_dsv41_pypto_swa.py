#!/usr/bin/env python3
"""Compare the PyPTO V4.1 K32 SWA kernel with the Torch FP32 reference."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import time

import torch

from blockspec.pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW, make_swa_program, swa_spans
from blockspec.vllm_ascend.dsv41_attention import PagedRows, draft_attention


def fixture(anchor, heads, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn((1, BLOCK, heads, HEAD_DIM), generator=generator).bfloat16()
    temporary = torch.randn((1, BLOCK, HEAD_DIM), generator=generator).bfloat16()
    sink = torch.linspace(-8, 8, heads)
    pages = torch.full((5, WINDOW, HEAD_DIM), float("nan"), dtype=torch.bfloat16)
    table = torch.full((1, anchor // WINDOW + 1), -1, dtype=torch.int32)
    # Use nonconsecutive physical page IDs, and poison every row outside the window.
    cursor = max(0, anchor - WINDOW)
    for physical in (3, 1):
        if cursor == anchor:
            break
        logical, offset = divmod(cursor, WINDOW)
        count = min(WINDOW - offset, anchor - cursor)
        table[0, logical] = physical
        pages[physical, offset:offset + count] = torch.randn((count, HEAD_DIM), generator=generator)
        cursor += count
    assert cursor == anchor
    spans = torch.tensor(swa_spans(table[0].tolist(), anchor, pages.shape[0]), dtype=torch.int32)
    history_bias = torch.full((1, WINDOW), -torch.inf)
    history_bias[:, :min(anchor, WINDOW)] = 0
    expected = draft_attention(query, temporary, PagedRows(pages, table), torch.tensor([anchor]), sink, window=WINDOW)
    packed_q = query.permute(0, 2, 1, 3).reshape(heads * BLOCK, HEAD_DIM).contiguous()
    tensors = [
        packed_q, pages.flatten(0, 1), spans, temporary[0], history_bias, sink,
        torch.zeros((WINDOW, HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((heads * BLOCK, HEAD_DIM), dtype=torch.float32),
        torch.empty((heads * BLOCK, 1), dtype=torch.float32),
    ]
    expected = tuple(x.transpose(1, 2).reshape(heads * BLOCK, -1) for x in expected)
    return tensors, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, choices=(8, 64), default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.device < 0 or args.output.exists():
        parser.error("need positive iterations, a nonnegative device and a new output path")

    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig

    torch.set_num_threads(4)
    mode = ExecutionMode.SIM if args.simulate else ExecutionMode.ONBOARD
    config = RunConfig(
        backend_type=BackendType.Ascend910B, execution_mode=mode,
        distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
    )
    if config.execution_mode != mode:
        raise RuntimeError("PyPTO overrode the requested execution mode")
    report = {
        "passed": False, "compiled": False, "npu_execution": False, "simulator_execution": False,
        "platform": config.platform, "torch": torch.__version__, "pypto": version("pypto"),
        "shape": {"block": BLOCK, "window": WINDOW, "heads": args.heads, "dim": HEAD_DIM},
        "output_tolerance": {"atol": 1e-5, "rtol": 1.3e-6},
        "lse_tolerance": {"atol": 2e-5, "rtol": 2e-6},
        "cases": [],
    }
    worker, handles = None, []
    anchors = (0, 1, 31, 127, 128, 129, 255, 257, 8194)
    try:
        sample, _ = fixture(anchors[0], args.heads, 173)
        resident = [tensor.share_memory_() for tensor in sample]
        program = make_swa_program(args.heads).compile(*resident, config=config)
        report["compiled"] = True
        if args.compile_only:
            report["passed"] = True
            return
        worker = DistributedWorker([program], config=config, inherited_host_tensors=resident)
        for tensor in resident:
            handles.append(worker.alloc_tensor(tuple(tensor.shape), tensor.dtype))
        before = [torch.empty_like(tensor).share_memory_() for tensor in resident[:-2]]
        for index, anchor in enumerate(anchors):
            tensors, expected = fixture(anchor, args.heads, 173 + index)
            for src, host, handle in zip(tensors[:-2], resident[:-2], handles[:-2], strict=True):
                host.copy_(src)
                worker.copy_to(handle.data_ptr, host.data_ptr(), host.nbytes)
            worker.run(program, *handles, config=config)
            elapsed = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                worker.run(program, *handles, config=config)
                elapsed.append(time.perf_counter() - start)
            for host, handle in zip(resident[-2:], handles[-2:], strict=True):
                worker.copy_from(host.data_ptr(), handle.data_ptr, host.nbytes)
            for saved, tensor, handle in zip(before, resident[:-2], handles[:-2], strict=True):
                worker.copy_from(saved.data_ptr(), handle.data_ptr, saved.nbytes)
                if not torch.equal(saved.view(torch.uint8), tensor.view(torch.uint8)):
                    raise AssertionError("PyPTO modified a read-only input")
            output, lse = resident[-2:]
            case = {
                "anchor": anchor, "history": min(anchor, WINDOW), "inputs_byte_exact": True,
                "output_max_abs": float((output - expected[0]).abs().max()),
                "lse_max_abs": float((lse - expected[1]).abs().max()),
                "resident_call_median_us": None if args.simulate else statistics.median(elapsed) * 1e6,
            }
            report["cases"].append(case)
            torch.testing.assert_close(output, expected[0], **report["output_tolerance"])
            torch.testing.assert_close(lse, expected[1], **report["lse_tolerance"])
            print(json.dumps(case), flush=True)
        report.update(passed=True, npu_execution=not args.simulate, simulator_execution=args.simulate)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                try:
                    for handle in reversed(handles):
                        worker.free_tensor(handle)
                finally:
                    worker.close()
        except BaseException as error:
            report.update(passed=False, cleanup_error=repr(error))
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
