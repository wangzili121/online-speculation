#!/usr/bin/env python3
"""独立验证与计时 GDN 递推；投影、卷积、MLP 和参数上传不计入此入口。"""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=("native", "vector16", "vector32", "packed32"), default=["native", "vector16", "vector32"])
    parser.add_argument("--value-tile", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()
    if (args.output.exists() or args.device < 0 or args.repeats < 3 or args.profile_repeats < 0
            or len(set(args.backends)) != len(args.backends)):
        parser.error("require assigned device, 3+ repeats, unique backends and new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("simpler").setLevel(logging.WARNING)
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, ExecutionMode
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.gdn import recurrence_golden
    from blockspec.pypto_qwen38.gdn_programs import make_recurrence
    from summarize_qwen38_pypto_profile import summarize

    report = dict(passed=False, scope="gdn_recurrence_only", npu_execution=False, cases=[],
                  runtime=runtime_identity(pypto), value_tile=args.value_tile,
                  inputs="packed_qkv_shared_heads" if "packed32" in args.backends else "independent_head_inputs",
                  tolerances=dict(output_rtol=.02, output_atol=.002, state_rtol=.002, state_atol=.00002))
    worker, handles, packed_handles = None, [], []
    try:
        torch.set_num_threads(4)
        config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                           dump_passes=args.dump_passes, distributed_config=DistributedConfig(
            device_ids=[args.device], num_sub_workers=0))
        if config.execution_mode != ExecutionMode.ONBOARD:
            raise RuntimeError("the GDN benchmark requires real NPU execution")
        generator = torch.Generator().manual_seed(191)
        packed = (torch.randn((32, 10240), generator=generator) * .5).bfloat16().share_memory_()
        packed_original = packed.clone()
        if "packed32" in args.backends:
            q, k = [packed[:, start:start + 2048].reshape(32, 16, 128).repeat_interleave(3, dim=1).contiguous().share_memory_() for start in (0, 2048)]
            v = packed[:, 4096:].reshape(32, 48, 128).contiguous().share_memory_()
        else:
            q, k, v = [(torch.randn((32, 48, 128), generator=generator) * .5).bfloat16().share_memory_() for _ in range(3)]
        g = (-torch.rand((32, 48), generator=generator) * .2).share_memory_()
        b = torch.rand((32, 48), generator=generator).bfloat16().float().share_memory_()
        history = torch.empty((48, 128, 128), dtype=torch.float32).share_memory_()
        output, final = torch.empty_like(q).share_memory_(), torch.empty_like(history).share_memory_()
        tensors = [q, k, v, g, b, history, output, final]
        programs = {}
        for name in args.backends:
            print(f"compiling GDN {name}", flush=True)
            arguments = [packed, packed, packed, *tensors[3:]] if name == "packed32" else tensors
            programs[name] = make_recurrence(args.source, backend="vector32" if name == "packed32" else name,
                packed=name == "packed32", value_tile=args.value_tile)[1].compile(*arguments, 32, config=config)
        report["artifact_directories"] = {name: str(program.output_dir) for name, program in programs.items()}
        worker = DistributedWorker(list(programs.values()), config=config, inherited_host_tensors=[*tensors, packed])
        report["npu_execution"] = True
        for tensor in tensors:
            handles.append(worker.alloc_tensor(tuple(tensor.shape), tensor.dtype))
        packed_handle = worker.alloc_tensor(tuple(packed.shape), packed.dtype)
        packed_handles.append(packed_handle)
        calls = {name: [packed_handle, packed_handle, packed_handle, *handles[3:]] if name == "packed32" else handles for name in programs}
        originals = [x.clone() for x in tensors[:5]]
        returned = [torch.empty_like(x).share_memory_() for x in tensors[:6]]
        for nonzero in (False, True):
            seed = torch.randn(history.shape, generator=generator) * .01 if nonzero else torch.zeros_like(history)
            history.copy_(seed)
            for width in (1, 7, 16, 17, 31, 32):
                for tensor, original in zip(tensors[:5], originals, strict=True):
                    tensor.copy_(original)
                    tensor[width:] = float("nan")
                packed.copy_(packed_original)
                packed[width:] = float("nan")
                worker.copy_to(packed_handle.data_ptr, packed.data_ptr(), packed.numel() * packed.element_size())
                expected, expected_state = recurrence_golden(q, k, v, g, b, history, width)
                for tensor, handle in zip(tensors[:6], handles[:6], strict=True):
                    worker.copy_to(handle.data_ptr, tensor.data_ptr(), tensor.numel() * tensor.element_size())
                samples = {name: [] for name in programs}
                for name, program in programs.items():
                    for _ in range(5):
                        worker.run(program, *calls[name], width, config=config)
                for repeat in range(args.repeats):
                    for name in (list(programs) if repeat % 2 == 0 else list(reversed(programs))):
                        start = time.perf_counter()
                        worker.run(programs[name], *calls[name], width, config=config)
                        samples[name].append(time.perf_counter() - start)
                for name, program in programs.items():
                    worker.run(program, *calls[name], width, config=config)
                    for host, handle in zip(tensors[6:], handles[6:], strict=True):
                        worker.copy_from(host.data_ptr(), handle.data_ptr, host.numel() * host.element_size())
                    torch.testing.assert_close(output.float(), expected.float(), rtol=.02, atol=.002)
                    torch.testing.assert_close(final, expected_state, rtol=.002, atol=.00002)
                    for before, after, handle in zip(tensors[:6], returned, handles[:6], strict=True):
                        worker.copy_from(after.data_ptr(), handle.data_ptr, after.numel() * after.element_size())
                        if not torch.equal(before.view(torch.uint8), after.view(torch.uint8)):
                            raise AssertionError("GDN modified an input or target history")
                    returned_packed = torch.empty_like(packed)
                    worker.copy_from(returned_packed.data_ptr(), packed_handle.data_ptr, packed.numel() * packed.element_size())
                    if not torch.equal(packed.view(torch.uint8), returned_packed.view(torch.uint8)):
                        raise AssertionError("modified packed QKV")
                    row = dict(backend=name, width=width, nonzero_history=nonzero, passed=True,
                               inputs_byte_exact=True, poisoned_inactive_inputs=True,
                               output_max_abs=float((output.float() - expected.float()).abs().max()),
                               state_max_abs=float((final - expected_state).abs().max()),
                               resident_dispatch_median_seconds=statistics.median(samples[name]),
                               resident_dispatch_samples_seconds=samples[name], device_profiles=[])
                    report["cases"].append(row)
                    if nonzero and width in (17, 32):
                        for sample in range(args.profile_repeats):
                            worker.run(program, *calls[name], width, config=replace(config, enable_chip_swimlane=1))
                            paths = list(Path(program.output_dir).rglob("merged_swimlane*.json"))
                            if not paths:
                                raise FileNotFoundError("missing GDN device profile")
                            path = max(paths, key=lambda p: p.stat().st_mtime_ns)
                            data = path.read_bytes()
                            saved = args.output.parent / (args.output.stem + "-profiles") / f"w{width}-{name}-{sample}.json"
                            saved.parent.mkdir(exist_ok=True)
                            if saved.exists():
                                raise FileExistsError(saved)
                            shutil.copyfile(path, saved)
                            row["device_profiles"].append(dict(path=str(saved), sha256=hashlib.sha256(data).hexdigest(),
                                                               **summarize(json.loads(data))))
                    print(json.dumps(row), flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        failures = []
        if worker is not None:
            for handle in reversed([*handles, *packed_handles]):
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
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if failures:
            raise RuntimeError("GDN cleanup failed")


if __name__ == "__main__":
    main()
