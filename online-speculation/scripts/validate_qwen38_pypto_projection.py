#!/usr/bin/env python3
"""仅测 K32 draft 准备阶段，不加载主模型或训练 head。"""

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
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--components", nargs="+", choices=("gdn_qkv", "gdn_z", "gdn_ab", "attention_o", "full_q", "full_kv", "mlp_up", "mlp_down", "norm"),
                   default=["gdn_qkv", "gdn_z", "gdn_ab", "attention_o", "full_q", "full_kv", "mlp_up", "mlp_down", "norm"])
    a = p.parse_args()
    if a.output.exists() or a.device < 0 or a.repeats < 3 or len(set(a.components)) != len(a.components):
        p.error("new output, assigned device and at least three repetitions required")
    logging.getLogger("simpler").setLevel(logging.WARNING)
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, ExecutionMode
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.projection_programs import make_projection, make_norm
    from summarize_qwen38_pypto_profile import summarize

    report = dict(passed=False, scope="standalone_head_projection_and_input_norm", runtime=runtime_identity(pypto),
                  cases=[], artifact_directories={})
    worker, handles = None, []
    try:
        torch.set_num_threads(4)
        config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                           distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
        assert config.execution_mode == ExecutionMode.ONBOARD
        gen = torch.Generator().manual_seed(233)
        rand = lambda shape, scale=.5: (torch.randn(shape, generator=gen) * scale).bfloat16().share_memory_()
        shapes = dict(gdn_qkv=(5120, 10240), gdn_z=(5120, 6144), gdn_ab=(5120, 128),
                      attention_o=(6144, 5120), full_q=(5120, 12288), full_kv=(5120, 1024),
                      mlp_up=(5120, 17408), mlp_down=(17408, 5120))
        groups, programs = {}, {}
        for name, (inputs, outputs) in shapes.items():
            if name not in a.components:
                continue
            x, w, y = rand((32, inputs)), rand((outputs, inputs), .02), rand((32, outputs))
            groups[name] = [x, w, y]
            candidates = {"native": dict(native=True)}
            candidates.update({f"n{n}": dict(tile=n) for n in (64, 128) if n <= outputs})
            programs[name] = {}
            for backend, options in candidates.items():
                print("compiling", name, backend, flush=True)
                program = make_projection(a.source, inputs, outputs, **options)[1].compile(*groups[name], config=config)
                programs[name][backend] = program
                report["artifact_directories"][name + "/" + backend] = str(program.output_dir)
        if "norm" in a.components:
            groups["norm"] = [rand((32, 5120)), (1 + rand((1, 5120), .2).float()).share_memory_(), rand((32, 5120))]
            programs["norm"] = {}
            for backend, options in dict(native=dict(native=True), rows1=dict(row_tile=1), rows2=dict(row_tile=2)).items():
                print("compiling norm", backend, flush=True)
                program = make_norm(a.source, **options)[1].compile(*groups["norm"], config=config)
                programs["norm"][backend] = program
                report["artifact_directories"]["norm/" + backend] = str(program.output_dir)
        all_tensors = [t for group in groups.values() for t in group]
        worker = DistributedWorker([p for group in programs.values() for p in group.values()],
                                   config=config, inherited_host_tensors=all_tensors)
        report["npu_execution"] = True
        device = {}
        for name, tensors in groups.items():
            device[name] = [worker.alloc_tensor(tuple(t.shape), t.dtype, init=t) for t in tensors]
            handles.extend(device[name])
        for case in range(2):
            for name, tensors in groups.items():
                x, w, output = tensors
                if case:
                    x.mul_(-2.)
                if name == "norm":
                    expected = (x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt() * w).bfloat16()
                else:
                    expected = x @ w.T
                worker.copy_to(device[name][0].data_ptr, x.data_ptr(), x.numel() * x.element_size())
                samples, results = {b: [] for b in programs[name]}, {}
                for program in programs[name].values():
                    for _ in range(5):
                        worker.run(program, *device[name], config=config)
                for repeat in range(a.repeats):
                    order = list(programs[name]) if repeat % 2 == 0 else list(reversed(programs[name]))
                    for backend in order:
                        start = time.perf_counter()
                        worker.run(programs[name][backend], *device[name], config=config)
                        samples[backend].append(time.perf_counter() - start)
                for backend, program in programs[name].items():
                    worker.run(program, *device[name], config=config)
                    worker.copy_from(output.data_ptr(), device[name][-1].data_ptr, output.numel() * output.element_size())
                    torch.testing.assert_close(output.float(), expected.float(), rtol=.02, atol=.002)
                    for original, handle in zip(tensors[:-1], device[name][:-1], strict=True):
                        returned = torch.empty_like(original)
                        worker.copy_from(returned.data_ptr(), handle.data_ptr, original.numel() * original.element_size())
                        if not torch.equal(original.view(torch.uint8), returned.view(torch.uint8)):
                            raise AssertionError("modified an input or projection parameter")
                    results[backend] = output.clone()
                    if name != "norm" and not torch.equal(output, results["native"]):
                        raise AssertionError("K32 projection changed the native accumulation result")
                    profiles = []
                    if case == 0:
                        for sample in range(5):
                            worker.run(program, *device[name], config=replace(config, enable_chip_swimlane=1))
                            source = max(Path(program.output_dir).rglob("merged_swimlane*.json"), key=lambda p: p.stat().st_mtime_ns)
                            target = a.output.parent / (a.output.stem + "-profiles") / f"{name}-{backend}-{sample}.json"
                            target.parent.mkdir(exist_ok=True)
                            shutil.copyfile(source, target)
                            data = target.read_bytes()
                            profiles.append(dict(path=str(target), sha256=hashlib.sha256(data).hexdigest(), **summarize(json.loads(data))))
                    row = dict(component=name, backend=backend, case=case, passed=True, inputs_byte_exact=True,
                               max_abs=float((output.float() - expected.float()).abs().max()),
                               relative_l2=float((output.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)),
                               native_max_abs=float((output.float() - results["native"].float()).abs().max()),
                               native_byte_exact=torch.equal(output, results["native"]),
                               wall_median_seconds=statistics.median(samples[backend]), wall_samples_seconds=samples[backend],
                               device_profiles=profiles)
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
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
            raise RuntimeError("projection cleanup failed")


if __name__ == "__main__":
    main()
