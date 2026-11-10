"""测双 NPU 共享缓冲的分块大小；只测传输，逐字节检查结果。"""

import argparse
import json
from pathlib import Path
import statistics
import time


def run(a):
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.online_bridge import TensorBridge
    from blockspec.pypto_qwen38.online_feedback_programs import make_copy

    torch.set_num_threads(4)
    runtime_identity(pypto)
    bridge = TensorBridge()
    storage = torch.empty(max(a.chunks) << 20, dtype=torch.uint8).share_memory_()
    workers = []
    report = dict(completed=False, chunks_mib=a.chunks, repetitions=a.repeats, results={})
    try:
        for device in (a.device, a.learner_device):
            config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                distributed_config=DistributedConfig(device_ids=[device], num_sub_workers=0))
            meta = torch.empty((32, 64), dtype=torch.bfloat16, device="meta")
            program = make_copy(32, 64).compile(meta, meta, config=config)
            workers.append(DistributedWorker([program], config=config, inherited_host_tensors=[storage]))
        source_worker, target_worker = workers
        for size_mib in a.sizes:
            count = size_mib << 20
            source = torch.randint(0, 256, (count,), dtype=torch.uint8,
                                   generator=torch.Generator().manual_seed(991 + size_mib))
            source_device = source_worker.alloc_tensor((count,), torch.uint8, init=source)
            target_device = target_worker.alloc_tensor((count,), torch.uint8)
            actual = torch.empty_like(source)
            entries = {str(c): [] for c in a.chunks}
            try:
                for chunk in a.chunks:
                    bridge.scratch = storage[:chunk << 20]
                    bridge.copy(source_worker, source_device, target_worker, target_device)
                for repeat in range(a.repeats):
                    for chunk in (a.chunks if repeat % 2 == 0 else a.chunks[::-1]):
                        bridge.scratch = storage[:chunk << 20]
                        start = time.perf_counter()
                        bridge.copy(source_worker, source_device, target_worker, target_device)
                        entries[str(chunk)].append(time.perf_counter() - start)
                        target_worker.copy_from(actual.data_ptr(), target_device.data_ptr, count)
                        if not torch.equal(actual, source):
                            raise AssertionError("cross-device bytes changed")
                report["results"][str(size_mib)] = {k: dict(seconds=v, median_seconds=statistics.median(v))
                                                    for k, v in entries.items()}
                print(json.dumps(dict(event="bridge_profile", size_mib=size_mib,
                                      results=report["results"][str(size_mib)])), flush=True)
            finally:
                target_worker.free_tensor(target_device)
                source_worker.free_tensor(source_device)
        report["completed"] = True
    finally:
        for worker in reversed(workers):
            worker.close()
        report["clean_shutdown"] = True
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--learner-device", type=int, default=1)
    p.add_argument("--source", type=Path)
    p.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 120, 800])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if (args.device == args.learner_device or args.repeats < 3
            or min(args.chunks) < 1 or max(args.chunks) > 128 or min(args.sizes) < 1):
        p.error("distinct devices, positive sizes, 1..128 MiB chunks and >=3 repeats required")
    run(args)
