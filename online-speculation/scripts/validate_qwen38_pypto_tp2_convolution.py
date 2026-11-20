#!/usr/bin/env python3
"""TP2 draft 卷积的独立正确性与设备计时，不含模型投影和参数搬运。"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time


def golden(projected, weight, history, slots, layer):
    import torch
    result = []
    cache = history.reshape(32, 48, 4, 5120)
    for seq, slot in enumerate(slots):
        x = torch.cat((cache[slot, layer, 1:], projected[seq * 32:(seq + 1) * 32])).float()
        w = weight.float()
        pre = ((x[:32] * w[0] + x[1:33] * w[1]) +
               (x[2:34] * w[2] + x[3:35] * w[3])).bfloat16().float()
        result.append((pre / (1 + (-pre).exp())).bfloat16())
    return torch.cat(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profiles", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3 or args.profiles < 0:
        parser.error("a fresh path, 3+ repeats and nonnegative profile count are required")
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.tp2_gdn_convolution import make_entry
    from summarize_qwen38_pypto_profile import summarize

    torch.set_num_threads(4)
    torch.manual_seed(105)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    variants = {"sequential128": ("sequential", 128), "parallel128": ("parallel", 128),
                "parallel64": ("parallel", 64)}
    report = dict(passed=False, clean_shutdown=False, scope="isolated TP2 convolution, not model throughput", cases=[])
    worker, handles = None, []
    try:
        weight = torch.randn(4, 5120).bfloat16()
        history = torch.randn(32 * 48 * 4, 5120).bfloat16()
        programs = {}
        for batch in [1, 2]:
            for name, (backend, tile) in variants.items():
                meta = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
                programs[batch, name] = make_entry(batch, backend=backend, channel_tile=tile).compile(
                    meta((batch * 32, 5120)), meta(weight.shape), meta(history.shape),
                    meta((batch * 32, 5120)), 11, 3, 47, config=config)
        worker = DistributedWorker(list(programs.values()), config=config)
        def alloc(value):
            handle = worker.alloc_tensor(tuple(value.shape), value.dtype, init=value)
            handles.append(handle)
            return handle
        dw, dh = alloc(weight), alloc(history)
        for batch in [1, 2]:
            projected = torch.randn(batch * 32, 5120).bfloat16()
            output = torch.empty_like(projected)
            dx, dy = alloc(projected), alloc(torch.zeros_like(projected))
            for first_slot, second_slot, layer in [(0, 31, 0), (11, 3, 47), (7, 7, 12)]:
                slots = [first_slot] if batch == 1 else [first_slot, second_slot]
                expected = golden(projected, weight, history, slots, layer)
                reference = None
                samples = {name: [] for name in variants}
                call = [dx, dw, dh, dy, first_slot, second_slot, layer]
                for name in variants:
                    worker.run(programs[batch, name], *call, config=config)
                for repeat in range(args.repeats):
                    order = list(variants)
                    order = order[repeat % 3:] + order[:repeat % 3]
                    for name in order:
                        started = time.monotonic_ns()
                        worker.run(programs[batch, name], *call, config=config)
                        samples[name].append(time.monotonic_ns() - started)
                for name in variants:
                    program = programs[batch, name]
                    worker.run(program, *call, config=config)
                    worker.copy_from(output.data_ptr(), dy.data_ptr, output.nbytes)
                    torch.testing.assert_close(output.float(), expected.float(), atol=.002, rtol=.02)
                    if reference is None:
                        reference = output.clone()
                    if not torch.equal(output, reference):
                        raise AssertionError(f"{batch}/{name} changed rounded convolution: "
                                             f"max_abs={float((output.float()-reference.float()).abs().max())}")
                    returned = torch.empty_like(history)
                    worker.copy_from(returned.data_ptr(), dh.data_ptr, returned.nbytes)
                    assert torch.equal(returned, history), "target convolution cache was modified"
                    profiles = []
                    if layer == 47:
                        for sample in range(args.profiles):
                            worker.run(program, *call, config=replace(config, enable_chip_swimlane=1))
                            source = max(Path(program.output_dir).rglob("merged_swimlane*.json"),
                                         key=lambda p: p.stat().st_mtime_ns)
                            target = args.output.parent / "profiles" / f"b{batch}-{name}-{sample}.json"
                            target.parent.mkdir(exist_ok=True)
                            shutil.copyfile(source, target)
                            data = target.read_bytes()
                            profiles.append(dict(path=str(target), sha256=hashlib.sha256(data).hexdigest(),
                                                 **summarize(json.loads(data))))
                    row = dict(batch=batch, variant=name, slots=slots, layer=layer, bitwise_equal=True,
                               history_unchanged=True, golden_max_abs=float((output.float()-expected.float()).abs().max()),
                               call_median_ns=statistics.median(samples[name]), call_samples_ns=samples[name],
                               device_profiles=profiles)
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            while handles:
                worker.free_tensor(handles.pop())
            worker.close()
            report["clean_shutdown"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
