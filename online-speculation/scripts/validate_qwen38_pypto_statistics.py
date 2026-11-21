#!/usr/bin/env python3
"""真实 last-1/4 flat 形状：合并统计与逐次调用逐位对照，并测试溢出保护。"""

import argparse
import json
from pathlib import Path
from statistics import median
import time


def main(guard=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh result path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.training_flat import FlatLayout, make_summary
    from blockspec.pypto_qwen38.training_optimizer_programs import make_stats, make_reduce
    from blockspec.pypto_qwen38.training_statistics import make_statistics
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph

    torch.set_num_threads(4)
    torch.manual_seed(214)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    meta = lambda shape: torch.empty(shape, dtype=torch.float32, device="meta")
    compiled, programs = [], []
    report = dict(completed=False, clean_shutdown=False, passed=False, cases=[],
        scope="Independent single-card exact optimizer geometry, synchronous call latency; not online throughput")
    worker, handles = None, []
    try:
        for layers in (1, 4):
            shape = FlatLayout(SuffixGraph(layers).parameters).shape
            stat, rows = make_stats(shape)
            counts = [rows]
            stages = [stat.compile(meta(shape), meta((rows, 8)), meta((rows, 8)), config=config)]
            while rows > 1:
                reducer, count = make_reduce(rows)
                stages.append(reducer.compile(meta((rows, 8)), meta((rows, 8)),
                    meta((count, 8)), meta((count, 8)), config=config))
                counts.append(count)
                rows = count
            stages.append(make_summary(1.).compile(meta((1, 8)), meta((1, 8)), meta((3, 8)), config=config))
            fused, fused_counts = make_statistics(shape, 1.)
            assert fused_counts == counts
            merged = fused.compile(meta(shape), *[meta((n, 8)) for n in counts for _ in range(2)],
                meta((3, 8)), config=config)
            compiled.append((layers, shape, counts, stages, merged))
            programs.extend([*stages, merged])
        report["compiled"] = True
        if args.compile_only:
            report.update(passed=True, completed=True, physical_device_count=0,
                          compiled_paths=[str(p.output_dir) for p in programs])
            return
        worker = DistributedWorker(programs, config=config)
        if guard is not None:
            guard.release()
            guard.close()
        def alloc(shape, init=None):
            value = worker.alloc_tensor(shape, torch.float32, **({"init": init} if init is not None else {}))
            handles.append(value)
            return value
        for layers, shape, counts, stages, merged in compiled:
            original = torch.randn(shape).mul_(.001).contiguous()
            value = alloc(shape, original)
            buffers = [[[alloc((n, 8)) for _ in range(2)] for n in counts] for _ in range(2)]
            outputs = [alloc((3, 8)) for _ in range(2)]
            def run(index, *, buffers=buffers, merged=merged, value=value,
                    outputs=outputs, stages=stages, counts=counts):
                pairs = buffers[index]
                if index:
                    worker.run(merged, value, *(t for pair in pairs for t in pair), outputs[index], config=config)
                else:
                    worker.run(stages[0], value, *pairs[0], config=config)
                    for i in range(1, len(counts)):
                        worker.run(stages[i], *pairs[i-1], *pairs[i], config=config)
                    worker.run(stages[-1], *pairs[-1], outputs[index], config=config)
            def read(handle, shape):
                host = torch.empty(shape, dtype=torch.float32)
                worker.copy_from(host.data_ptr(), handle.data_ptr, host.nbytes)
                return host
            for kind in ("clipped", "zero", "unclipped", "nan", "positive_inf", "negative_inf", "square_overflow"):
                current = original.clone() if kind == "clipped" else torch.zeros(shape)
                if kind == "unclipped":
                    current[0, :8] = .001
                elif kind in {"nan", "positive_inf", "negative_inf", "square_overflow"}:
                    current[0, -1] = dict(nan=float("nan"), positive_inf=float("inf"),
                        negative_inf=-float("inf"), square_overflow=1e20)[kind]
                worker.copy_to(value.data_ptr, current.data_ptr(), current.nbytes)
                run(0)
                run(1)
                reference, actual = [read(out, (3, 8)) for out in outputs]
                assert torch.equal(reference.view(torch.int32), actual.view(torch.int32)), (layers, kind)
                for left, right in zip(buffers[0], buffers[1], strict=True):
                    for a, b in zip(left, right, strict=True):
                        assert torch.equal(read(a, tuple(a.shape)).view(torch.int32),
                                           read(b, tuple(b.shape)).view(torch.int32)), (layers, kind, a.shape)
                good = kind in {"clipped", "zero", "unclipped"}
                assert bool(actual[2, 0] == 1.) == good, (layers, kind, actual)
                if good:
                    norm = current.double().norm().item()
                    assert abs(actual[0, 0].item() - norm) / max(norm, 1e-30) < 2e-6
                    assert abs(actual[1, 0].item() - min(1., 1. / (norm + 1e-6))) < 2e-6
                row = dict(last_layers=layers, shape=shape, case=kind, counts=counts,
                    summary_bitwise=True, every_intermediate_bitwise=True,
                    finite_guard_correct=True, calls_before=len(stages), calls_after=1)
                if kind == "clipped":
                    assert torch.equal(read(value, shape).view(torch.int32), current.view(torch.int32))
                    row["input_byte_unchanged"] = True
                    for _ in range(4):
                        run(0)
                        run(1)
                    samples = [[], []]
                    for _ in range(20):
                        for index in (0, 1):
                            start = time.monotonic_ns()
                            run(index)
                            samples[index].append(time.monotonic_ns() - start)
                    row.update(chained_call_ns=samples[0], merged_call_ns=samples[1],
                        chained_median_ns=median(samples[0]), merged_median_ns=median(samples[1]))
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            while handles:
                worker.free_tensor(handles.pop())
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                while handles:
                    worker.free_tensor(handles.pop())
                worker.close()
                report["clean_shutdown"] = True
        finally:
            report["completed"] = report["passed"] and (report["clean_shutdown"] or args.compile_only)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
