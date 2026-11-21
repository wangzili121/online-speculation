#!/usr/bin/env python3
"""Compare PyPTO V4.1 K32 SWA and compressed history with the Torch FP32 reference."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from blockspec.pypto_dsv41_sparse import HEAD_TILE, TILE, TOPK, compressed_physical_rows, make_sparse_program
from blockspec.pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW, make_swa_program, swa_spans
from blockspec.vllm_ascend.dsv41_attention import PagedRows, draft_attention


def fixture(anchor, heads, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn((1, BLOCK, heads, HEAD_DIM), generator=generator).bfloat16()
    temporary = torch.randn((1, BLOCK, HEAD_DIM), generator=generator).bfloat16()
    sink = torch.linspace(-8, 8, heads)
    pages = torch.full((5, WINDOW, HEAD_DIM), float("nan"), dtype=torch.bfloat16)
    # Use nonconsecutive physical page IDs, and poison every row outside the window.
    table = swa_table(anchor)
    metadata = swa_spans(table[0].tolist(), anchor, pages.shape[0])
    flat = pages.flatten(0, 1)
    for _, source, count in metadata:
        flat[source:source + count] = torch.randn((count, HEAD_DIM), generator=generator)
    spans = torch.tensor(metadata, dtype=torch.int32)
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


def compressed_fixture(anchor, ratio, seed, local):
    heads = local[5].numel()
    query = local[0].reshape(heads, BLOCK, HEAD_DIM).transpose(0, 1).unsqueeze(0)
    swa = PagedRows(local[1].unflatten(0, (-1, WINDOW)), swa_table(anchor))
    count = anchor // ratio
    generator = torch.Generator().manual_seed(seed + 120)
    page_rows = WINDOW // ratio
    pool_stride = WINDOW
    pool = torch.full((20, pool_stride, HEAD_DIM), float("nan"), dtype=torch.bfloat16)
    pages = pool[:, :page_rows]
    table = torch.randperm(pages.shape[0], generator=generator, dtype=torch.int32)
    logical = torch.arange(count)
    physical = table[logical // page_rows] * pool_stride + logical % page_rows
    pool.flatten(0, 1)[physical.long()] = torch.randn((count, HEAD_DIM), generator=generator).bfloat16()
    lengths = torch.tensor((0, 1, 31, 127, 128, 129, 333, TOPK) * 4, dtype=torch.int32).clamp_max(count)
    bias = torch.full((BLOCK, TOPK), -torch.inf)
    selected = torch.full((1, BLOCK, TOPK), -1, dtype=torch.long)
    for token, take in enumerate(lengths.tolist()):
        rows = torch.randperm(count, generator=generator)[:take].sort().values
        selected[0, token, :take] = rows
        bias[token, :take] = 0
    physical_rows, lengths = compressed_physical_rows(
        selected[0], table, anchor=anchor, ratio=ratio, physical_pages=pool.shape[0], stride_rows=pool_stride,
    )
    gold = draft_attention(
        query, local[3][None], swa, torch.tensor([anchor]), local[5], window=WINDOW,
        global_kv=PagedRows(pages, table[None]), global_indices=selected,
    )
    tensors = [
        F.pad(query[0], (0, 0, 0, HEAD_TILE - heads)).reshape(-1, HEAD_DIM).contiguous(),
        pool.flatten(0, 1), physical_rows, lengths, bias,
        torch.zeros((TILE, HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((BLOCK * HEAD_TILE, HEAD_DIM)), torch.empty((BLOCK * HEAD_TILE, 1)),
        torch.empty((BLOCK * HEAD_TILE, HEAD_DIM)), torch.empty((BLOCK * HEAD_TILE, 1)),
    ]
    return tensors, tuple(tensor[0] for tensor in gold)


def swa_table(anchor):
    table = torch.full((1, anchor // WINDOW + 1), -1, dtype=torch.int32)
    start, end = max(0, anchor - WINDOW) // WINDOW, (anchor + WINDOW - 1) // WINDOW
    table[0, start:end] = torch.tensor((3, 1)[:end - start])
    return table


def upload(worker, tensors, resident, handles):
    for src, host, handle in zip(tensors, resident, handles, strict=True):
        host.copy_(src)
        worker.copy_to(handle.data_ptr, host.data_ptr(), host.nbytes)


def download(worker, resident, handles):
    for host, handle in zip(resident, handles, strict=True):
        worker.copy_from(host.data_ptr(), handle.data_ptr, host.nbytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, choices=(8, 16, 64), default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--check-every-run", action="store_true", help="Check each timed run for intermittent errors")
    parser.add_argument("--fused-swa", action="store_true", help="Compare against the former mixed-core SWA")
    parser.add_argument("--serving-layout", action=argparse.BooleanOptionalAction, default=False,
                        help="Use the token-major query layout from serving")
    parser.add_argument("--anchor", type=int, help="Run only the built-in cases at this anchor, preserving their seeds")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compressed", action="store_true", help="verify selected C1A/C2A history and LSE merge")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.device < 0 or args.output.exists():
        parser.error("need positive iterations, a nonnegative device and a new output path")
    if args.compressed and args.heads > HEAD_TILE:
        parser.error("compressed history probe supports up to sixteen local heads")

    from pypto import __version__ as pypto_version
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
        "platform": config.platform, "torch": torch.__version__, "pypto": pypto_version,
        "swa": "fused" if args.fused_swa else "staged",
        "query_layout": "token_major" if args.serving_layout else "head_major",
        "shape": {"block": BLOCK, "window": WINDOW, "heads": args.heads, "dim": HEAD_DIM},
        "output_tolerance": {"atol": 1e-5, "rtol": 1.3e-6},
        "lse_tolerance": {"atol": 2e-5, "rtol": 2e-6},
        "cases": [],
    }
    worker, handles, swa_handles = None, [], []
    local_tensors, _ = fixture(0, args.heads, 173)

    def serving_query(tensor):
        return tensor.reshape(args.heads, BLOCK, HEAD_DIM).transpose(0, 1).reshape(BLOCK, -1).contiguous()

    def refresh_swa(tensors):
        worker.run(swa_program, *swa_handles, config=config)
        download(worker, swa_resident[-2:], swa_handles[-2:])
        for host, segment in zip(tensors[-4:-2], swa_resident[-2:], strict=True):
            active = segment.reshape(args.heads, BLOCK, -1).transpose(0, 1)
            host.copy_(F.pad(active, (0, 0, 0, HEAD_TILE - args.heads)).reshape_as(host))

    swa_resident = [tensor.share_memory_() for tensor in local_tensors]
    if args.serving_layout:
        swa_resident[0] = serving_query(local_tensors[0]).share_memory_()
    if args.compressed:
        cases = ((0, 1), (31, 2), (129, 1), (257, 2), (1025, 1), (1025, 2))
        program_entry = make_sparse_program()
        report["shape"]["topk"] = TOPK
        report["swa_input"] = "pypto_swa_output_with_host_layout_conversion"
    else:
        cases = tuple((anchor, None) for anchor in (0, 1, 31, 127, 128, 129, 255, 257, 8194))
    if args.anchor is not None and all(anchor != args.anchor for anchor, _ in cases):
        parser.error("requested anchor is not in the selected fixture set")
    try:
        swa_program = make_swa_program(
            args.heads, token_major_query=args.serving_layout, fused=args.fused_swa
        ).compile(*swa_resident, config=config)
        programs = [swa_program]
        program, resident = swa_program, swa_resident
        if args.compressed:
            sample, _ = compressed_fixture(*cases[0], 173, local_tensors)
            resident = [tensor.share_memory_() for tensor in sample]
            program = program_entry.compile(*resident, config=config)
            programs.append(program)
        report["compiled"] = True
        if args.compile_only:
            report["passed"] = True
            return
        host_tensors = swa_resident + resident if args.compressed else resident
        worker = DistributedWorker(programs, config=config, inherited_host_tensors=host_tensors)
        for tensor in resident:
            handles.append(worker.alloc_tensor(tuple(tensor.shape), tensor.dtype))
        if args.compressed:
            for tensor in swa_resident:
                swa_handles.append(worker.alloc_tensor(tuple(tensor.shape), tensor.dtype))
        before = [torch.empty_like(tensor).share_memory_() for tensor in resident[:-2]]
        for index, (anchor, ratio) in enumerate(cases):
            if args.anchor is not None and anchor != args.anchor:
                continue
            local, swa_expected = fixture(anchor, args.heads, 173 + index)
            expected = swa_expected
            swa_errors = {}
            tensors = local
            if args.compressed:
                tensors, expected = compressed_fixture(anchor, ratio, 173 + index, local)
            if args.serving_layout:
                local[0] = serving_query(local[0])
            if args.compressed:
                upload(worker, local[:-2], swa_resident[:-2], swa_handles[:-2])
                refresh_swa(tensors)
            upload(worker, tensors[:-2], resident[:-2], handles[:-2])
            worker.run(program, *handles, config=config)
            output, lse = resident[-2:]
            if args.compressed:
                output = output.unflatten(0, (BLOCK, HEAD_TILE))[:, :args.heads]
                lse = lse.unflatten(0, (BLOCK, HEAD_TILE))[:, :args.heads]
            elapsed = []
            for _ in range(args.iterations):
                if args.compressed and args.check_every_run:
                    refresh_swa(tensors)
                    upload(worker, tensors[-4:-2], resident[-4:-2], handles[-4:-2])
                start = time.perf_counter()
                worker.run(program, *handles, config=config)
                elapsed.append(time.perf_counter() - start)
                if args.check_every_run:
                    download(worker, resident[-2:], handles[-2:])
                    if args.compressed and (
                        not torch.allclose(swa_resident[-2], swa_expected[0], **report["output_tolerance"])
                        or not torch.allclose(swa_resident[-1], swa_expected[1], **report["lse_tolerance"])
                    ):
                        break
                    if (not torch.allclose(output, expected[0], **report["output_tolerance"])
                            or not torch.allclose(lse, expected[1], **report["lse_tolerance"])):
                        break
            if not args.check_every_run:
                download(worker, resident[-2:], handles[-2:])
            report.update(npu_execution=not args.simulate, simulator_execution=args.simulate)
            if args.compressed:
                swa_errors = dict(
                    swa_output_max_abs=float((swa_resident[-2] - swa_expected[0]).abs().max()),
                    swa_lse_max_abs=float((swa_resident[-1] - swa_expected[1]).abs().max()),
                )
                download(worker, swa_resident[:-2], swa_handles[:-2])
                for saved, original in zip(swa_resident[:-2], local[:-2], strict=True):
                    if not torch.equal(saved.view(torch.uint8), original.view(torch.uint8)):
                        raise AssertionError("SWA modified a read-only input")
            for saved, tensor, handle in zip(before, resident[:-2], handles[:-2], strict=True):
                worker.copy_from(saved.data_ptr(), handle.data_ptr, saved.nbytes)
                if not torch.equal(saved.view(torch.uint8), tensor.view(torch.uint8)):
                    raise AssertionError("PyPTO modified a read-only input")
            case = {
                "anchor": anchor, "history": min(anchor, WINDOW), "inputs_byte_exact": True,
                "output_max_abs": float((output - expected[0]).abs().max()),
                "lse_max_abs": float((lse - expected[1]).abs().max()),
                "runs_checked": len(elapsed) if args.check_every_run else 1,
                "resident_call_median_us": None if args.simulate else statistics.median(elapsed) * 1e6,
            }
            if args.compressed:
                case.update(ratio=ratio, max_selected=min(anchor // ratio, TOPK), **swa_errors)
            report["cases"].append(case)
            try:
                torch.testing.assert_close(output, expected[0], **report["output_tolerance"])
                torch.testing.assert_close(lse, expected[1], **report["lse_tolerance"])
            except AssertionError:
                bad_rows = ~torch.isclose(output, expected[0], **report["output_tolerance"]).all(dim=-1)
                if not args.compressed:
                    bad_rows = bad_rows.reshape(args.heads, BLOCK).transpose(0, 1)
                case["output_failed_token_heads"] = bad_rows.nonzero().tolist()
                if args.compressed:
                    case["selected_rows_by_failed_token"] = {
                        int(token): int(tensors[3][token]) for token in bad_rows.any(dim=-1).nonzero().flatten()
                    }
                raise
            if args.compressed:
                torch.testing.assert_close(swa_resident[-2], swa_expected[0], **report["output_tolerance"])
                torch.testing.assert_close(swa_resident[-1], swa_expected[1], **report["lse_tolerance"])
            print(json.dumps(case), flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                try:
                    for handle in reversed(handles + swa_handles):
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
