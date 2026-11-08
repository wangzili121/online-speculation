#!/usr/bin/env python3
"""GDN gate、norm/O projection 与只读卷积对照，包含独立参考与计时。"""

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import shutil
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--components", nargs="+", choices=("gates", "tail", "conv"), default=["gates", "tail", "conv"])
    a = p.parse_args()
    if a.output.exists() or a.device < 0 or a.repeats < 3 or len(set(a.components)) != len(a.components):
        p.error("assigned device, 3+ repeats and new output required")
    logging.getLogger("simpler").setLevel(logging.WARNING)
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, ExecutionMode
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.gdn_aux_programs import make_auxiliary_entries
    from blockspec.pypto_qwen38.gdn import convolution_golden
    from summarize_qwen38_pypto_profile import summarize

    report = dict(passed=False, scope="gdn_auxiliary_only", cases=[], runtime=runtime_identity(pypto))
    worker, handles = None, []
    try:
        torch.set_num_threads(4)
        config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                           distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
        if config.execution_mode != ExecutionMode.ONBOARD:
            raise RuntimeError("real NPU required")
        gen = torch.Generator().manual_seed(211)
        rand = lambda shape, scale=1.: (torch.randn(shape, generator=gen) * scale).bfloat16().share_memory_()
        av, bv = rand((16, 128)), rand((16, 128))
        al, dt = rand((1, 64), .2).float().share_memory_(), rand((1, 64), .2)
        decay, beta = [torch.empty((16, 48)).share_memory_() for _ in range(2)]
        core, z, nw, ow = rand((16, 48, 128)), rand((16, 6144)), rand((1, 128), .2), rand((5120, 6144), .01)
        gated, attention = rand((768, 128)), rand((16, 5120))
        projected, cw, history, convolved = rand((32, 10240)), rand((4, 10240), .1), rand((4, 10240)), rand((32, 10240))
        groups = dict(gates=[av, bv, al, dt, decay, beta], tail=[core, z, nw, ow, gated, attention],
                      conv=[projected, cw, history, convolved])
        groups = {name: groups[name] for name in a.components}
        outputs_per_group = dict(gates=2, tail=2, conv=1)
        programs = {}
        for backend in ("native", "fused"):
            for name, entry in make_auxiliary_entries(a.source, backend=backend).items():
                if name not in groups:
                    continue
                print(f"compiling {name} {backend}", flush=True)
                programs[name, backend] = entry.compile(*groups[name], config=config)
        tensors = [t for group in groups.values() for t in group]
        worker = DistributedWorker(list(programs.values()), config=config, inherited_host_tensors=tensors)
        report["npu_execution"] = True
        device = {}
        for name, group in groups.items():
            device[name] = [worker.alloc_tensor(tuple(t.shape), t.dtype, init=t) for t in group]
            handles += device[name]
        for case in range(3):
            if case:
                for t in (av, bv, al, dt, core, z, projected):
                    t.mul_(.5 if case == 1 else -1.)
            if case == 1:
                history.zero_()
            elif case == 2:
                history.copy_(rand(history.shape))
            # 保留中间 BF16 舍入，参考不调用待测 PyPTO 实现。
            gref = -al[:, :48].exp() * torch.log1p((av[:, :48].float() + dt[:, :48].float()).exp())
            bref = bv[:, :48].float().sigmoid().bfloat16().float()
            x = core.float().reshape(768, 128)
            normalized = (x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).bfloat16()
            weighted = (normalized.float() * nw.float()).bfloat16()
            gated_ref = (weighted.float() * torch.nn.functional.silu(z.float().reshape(768, 128))).bfloat16()
            attn_ref = gated_ref.reshape(16, 6144) @ ow.T
            refs = dict(gates=[gref, bref], tail=[gated_ref, attn_ref],
                        conv=[convolution_golden(projected, cw, history)])
            for name, group in groups.items():
                count = outputs_per_group[name]
                for t, handle in zip(group[:-count], device[name][:-count], strict=True):
                    worker.copy_to(handle.data_ptr, t.data_ptr(), t.numel() * t.element_size())
                timings, outputs = {}, {}
                for backend in ("native", "fused"):
                    timings[backend] = []
                    for _ in range(5):
                        worker.run(programs[name, backend], *device[name], config=config)
                for repeat in range(a.repeats):
                    for backend in (("native", "fused") if repeat % 2 == 0 else ("fused", "native")):
                        start = time.perf_counter()
                        worker.run(programs[name, backend], *device[name], config=config)
                        timings[backend].append(time.perf_counter() - start)
                for backend in ("native", "fused"):
                    program = programs[name, backend]
                    worker.run(program, *device[name], config=config)
                    for t, handle in zip(group[-count:], device[name][-count:], strict=True):
                        worker.copy_from(t.data_ptr(), handle.data_ptr, t.numel() * t.element_size())
                    for actual, expected in zip(group[-count:], refs[name], strict=True):
                        torch.testing.assert_close(actual.float(), expected.float(), rtol=.02, atol=.002)
                    outputs[backend] = [t.clone() for t in group[-count:]]
                    for t, handle in zip(group[:-count], device[name][:-count], strict=True):
                        returned = torch.empty_like(t)
                        worker.copy_from(returned.data_ptr(), handle.data_ptr, t.numel() * t.element_size())
                        if not torch.equal(returned.view(torch.uint8), t.view(torch.uint8)):
                            raise AssertionError("auxiliary kernel modified an input")
                    profiles = []
                    if case == 0:
                        for sample in range(5):
                            worker.run(program, *device[name], config=replace(config, enable_chip_swimlane=1))
                            trace = max(Path(program.output_dir).rglob("merged_swimlane*.json"), key=lambda p: p.stat().st_mtime_ns)
                            target = a.output.parent / (a.output.stem + "-profiles") / f"{name}-{backend}-{sample}.json"
                            target.parent.mkdir(exist_ok=True)
                            shutil.copyfile(trace, target)
                            data = target.read_bytes()
                            profiles.append(dict(path=str(target), sha256=hashlib.sha256(data).hexdigest(), **summarize(json.loads(data))))
                    row = dict(component=name, backend=backend, case=case, passed=True,
                               wall_median_seconds=statistics.median(timings[backend]), wall_samples_seconds=timings[backend],
                               max_abs=[float((t.float() - r.float()).abs().max()) for t, r in zip(group[-count:], refs[name], strict=True)],
                               device_profiles=profiles, inputs_byte_exact=True)
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
                if any(not torch.equal(x, y) for x, y in zip(outputs["native"], outputs["fused"], strict=True)):
                    raise AssertionError("auxiliary fusion changed native rounded results")
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        failures = []
        if worker is not None:
            for handle in reversed(handles):
                try:
                    worker.free_tensor(handle)
                except BaseException as error:
                    failures.append(repr(error))
            try:
                worker.close()
            except BaseException as error:
                failures.append(repr(error))
        if failures:
            report.update(passed=False, cleanup_errors=failures)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        if failures:
            raise RuntimeError("auxiliary cleanup failed")


if __name__ == "__main__":
    main()
