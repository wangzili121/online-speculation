#!/usr/bin/env python3
"""比较融合合并与现有组合；使用真实投影形状，不代表在线吞吐。"""

import argparse
import json
from pathlib import Path
from statistics import median
import time

from validate_qwen38_pypto_lora_projection import section


def main():
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
    from blockspec.pypto_qwen38.training_executor import TrainingPrograms, TrainingWorkspace
    from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
    from blockspec.pypto_qwen38.training_graph import SuffixShape
    from blockspec.pypto_qwen38.training_lora_graph import LoRAMergeGraph, LoRASuffixGraph

    torch.set_num_threads(4)
    torch.manual_seed(177)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0))
    graph = LoRASuffixGraph(4)
    small = lambda rank: LoRASuffixGraph(1, SuffixShape(hidden=128, intermediate=256, vocabulary=128,
        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32),
        GDNShape(heads=8, query_heads=4, dimension=64), rank=rank)
    cases = [(graph, n) for n in ["l63_q_weight", "l63_k_weight", "l63_o_weight", "l60_qkv_weight",
                                  "l60_z_weight", "l60_a_weight"]]
    cases += [(small(r), "l63_k_weight") for r in (64, 128)]
    compiled, programs = [], []
    for current, name in cases:
        pair = []
        for fused in (False, True):
            merge = LoRAMergeGraph(current, fused=fused)
            part = section(merge, {name: merge.targets[name]}, set(merge.inputs))
            print("compiling", name, current.adapter_rank, "fused", fused, flush=True)
            entry = TrainingPrograms(part, config, combined=True)
            pair.append((merge, part, entry)); programs += list(entry.programs.values())
        compiled.append((current.adapters[name], pair))
    report = dict(completed=False, clean_shutdown=False, passed=False, cases=[],
        scope="Independent single-card real projection geometry; synchronous call timings, not whole model speed")
    if args.compile_only:
        report.update(completed=True, compiled=True, physical_device_count=0, program_count=len(programs),
                      compiled_paths=[str(p.output_dir) for p in programs])
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    worker, handles, workspaces = None, [], []
    def alloc(value):
        out = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value.contiguous())
        handles.append(out)
        return out
    def read(value, shape):
        host = torch.empty(shape, dtype=torch.bfloat16)
        worker.copy_from(host.data_ptr(), value.data_ptr, host.nbytes)
        return host
    def clear():
        while workspaces: workspaces.pop().close()
        while handles: worker.free_tensor(handles.pop())
    try:
        worker = DistributedWorker(programs, config=config)
        for adapter, pair in compiled:
            name, shape = adapter.projection.name, adapter.projection.shape
            rank = adapter.a.shape[0]
            base = torch.randn(shape).mul_(.02).bfloat16()
            a = torch.empty(adapter.a.shape).uniform_(-shape[1] ** -.5, shape[1] ** -.5).bfloat16()
            b = torch.randn(adapter.b.shape).mul_(.003).bfloat16()
            inputs = {"base_" + name: alloc(base), adapter.a.name: alloc(a), adapter.b.name: alloc(b)}
            spaces = []
            for merge, part, entry in pair:
                space = TrainingWorkspace(entry, worker); workspaces.append(space)
                spaces.append((space, part.gradients[name]))
            def run(index):
                space, _ = spaces[index]
                space.run(inputs, past=0, width=32, valid=31)
            def result(index):
                space, target = spaces[index]
                return read(space.values[target.name], target.shape)
            for zero_b in (True, False):
                bv = torch.zeros_like(b) if zero_b else b
                worker.copy_to(inputs[adapter.b.name].data_ptr, bv.data_ptr(), bv.nbytes)
                run(0); run(1)
                before, after = result(0), result(1)
                assert torch.equal(before.view(torch.int16), after.view(torch.int16)), (name, rank)
                expected = (base.float() + (bv @ a).float()).bfloat16()
                relative = float((after.float() - expected.float()).norm() / expected.float().norm())
                assert relative < .003 and float((after.float() - expected.float()).abs().max()) < .001
                if zero_b: assert torch.equal(after.view(torch.int16), base.view(torch.int16))
                assert torch.equal(read(inputs["base_" + name], shape).view(torch.int16), base.view(torch.int16))
                row = dict(name=name, shape=shape, rank=rank, zero_b=zero_b,
                           chained_fused_bitwise=True, base_unchanged=True, CPU_relative_L2=relative)
                if not zero_b:
                    for _ in range(4): run(0); run(1)
                    samples, chronology = [[], []], []
                    for iteration in range(16):
                        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                            start = time.monotonic_ns(); run(index); elapsed = time.monotonic_ns() - start
                            samples[index].append(elapsed); chronology.append(dict(fused=bool(index), ns=elapsed))
                    row.update(chained_call_ns=samples[0], fused_call_ns=samples[1],
                               chained_median_ns=median(samples[0]), fused_median_ns=median(samples[1]),
                               chronological_calls=chronology)
                    # 私有增量里的非有限值不能被融合路径悄悄吞掉。
                    if shape == (128, 128) and rank == 64:
                        row["nonfinite"] = []
                        for value in (float("nan"), float("inf"), -float("inf")):
                            bad = b.clone(); bad[0, 0] = value
                            worker.copy_to(inputs[adapter.b.name].data_ptr, bad.data_ptr(), bad.nbytes)
                            run(0); run(1)
                            x, y = result(0).float(), result(1).float()
                            assert not bool(torch.isfinite(y).all())
                            assert torch.equal(torch.isfinite(x), torch.isfinite(y))
                            row["nonfinite"].append(str(value))
                report["cases"].append(row)
            clear()
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                clear(); worker.close()
                report["clean_shutdown"] = True
        finally:
            report["completed"] = report["passed"] and report["clean_shutdown"]
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
