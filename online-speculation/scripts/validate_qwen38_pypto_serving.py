#!/usr/bin/env python3
"""独立验证设备端 token 选择；不加载 27B 权重。"""

import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    a = p.parse_args()
    if a.output.exists() or a.repeats < 3:
        p.error("new output and at least three repetitions required")
    import torch
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig, ExecutionMode
    from blockspec.pypto_qwen38.serving_programs import make_greedy

    torch.set_num_threads(4)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    x = torch.empty((128, 248320), dtype=torch.bfloat16).share_memory_()
    result = torch.empty((2, 32), dtype=torch.int32).share_memory_()
    program = make_greedy(source_rows=128)[1].compile(x, result, 0, 32, config=config)
    report = dict(passed=False, npu_execution=True, cases=[], artifact_directory=str(program.output_dir))
    worker, handles = None, []
    try:
        worker = DistributedWorker([program], config=config, inherited_host_tensors=[x, result])
        dx = worker.alloc_tensor(tuple(x.shape), x.dtype)
        handles.append(dx)
        dy = worker.alloc_tensor(tuple(result.shape), result.dtype)
        handles.append(dy)
        gen = torch.Generator().manual_seed(715)
        for name, begin, width in [("random", 0, n) for n in (1, 7, 16, 17, 31, 32)] + [
                ("ties", 96, 32), ("negative", 112, 16), ("zero", 112, 1),
                ("nan", 0, 31), ("inf", 0, 32), ("negative_inf", 112, 16)]:
            x.copy_(torch.randn(x.shape, generator=gen))
            active = x[begin:begin + width]
            if name == "ties":
                active.fill_(-100)
                active[:, 11] = active[:, 1029] = active[:, -1] = 100
            elif name == "negative":
                active.fill_(-100)
                active[:, -1] = -1
            elif name == "zero":
                active.zero_()
            elif name in {"nan", "inf", "negative_inf"}:
                active[-1, -1] = {"nan": float("nan"), "inf": float("inf"), "negative_inf": -float("inf")}[name]
            if begin:
                x[:begin].fill_(float("nan"))
            finite = active.isfinite().all(-1)
            expected = active.max(-1).indices
            worker.copy_to(dx.data_ptr, x.data_ptr(), x.numel() * x.element_size())
            worker.run(program, dx, dy, begin, width, config=config)
            worker.copy_from(result.data_ptr(), dy.data_ptr, result.numel() * result.element_size())
            if not torch.equal(result[1, :width].bool(), finite):
                raise AssertionError((name, "finite flags", result[1, :width], finite))
            if not torch.equal(result[0, :width][finite].long(), expected[finite]):
                raise AssertionError((name, "token indices", result[0, :width][finite], expected[finite]))
            report["cases"].append(dict(name=name, begin=begin, width=width, passed=True))
            print(name, begin, width, "passed", flush=True)
        x.normal_(generator=gen)
        worker.copy_to(dx.data_ptr, x.data_ptr(), x.numel() * x.element_size())
        for _ in range(3):
            worker.run(program, dx, dy, 0, 31, config=config)
        samples = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            worker.run(program, dx, dy, 0, 31, config=config)
            worker.copy_from(result.data_ptr(), dy.data_ptr, result.numel() * result.element_size())
            samples.append(time.perf_counter() - start)
        report.update(passed=True, wall_samples_seconds=samples,
                      wall_median_seconds=statistics.median(samples), output_bytes=result.numel() * result.element_size())
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if worker is not None:
            for handle in reversed(handles):
                worker.free_tensor(handle)
            worker.close()
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
