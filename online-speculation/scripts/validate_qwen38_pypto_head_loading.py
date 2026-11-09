#!/usr/bin/env python3
"""真实 64 层 head：NPU 整理与独立 CPU 路径逐字节对比，并检查导出/重载。"""

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--head-dir", type=Path, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.prepared_dir.exists():
        p.error("new output paths required")
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import native_module, runtime_identity
    from blockspec.pypto_qwen38.head import HeadSource
    from blockspec.pypto_qwen38.prepared_head import save
    torch.set_num_threads(4)
    report = dict(passed=False, npu_execution=True, runtime=runtime_identity(pypto), cases=[])
    try:
        native_module(a.source, "weight").verify_model_identity(a.model)
        reference, candidate = HeadSource(a.model, a.head_dir), HeadSource(a.model, a.head_dir)
        start = time.perf_counter()
        reference.prepare()
        report["host_seconds"] = time.perf_counter() - start
        print("host reference prepared", flush=True)
        config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                           distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
        candidate.prepare_on_device(config)
        report["pypto_seconds"] = candidate.preparation_seconds
        for layer in range(64):
            for index, (expected, actual) in enumerate(zip(reference.banks(layer), candidate.banks(layer), strict=True)):
                if not torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)):
                    raise AssertionError(f"head bytes differ: layer {layer}, bank {index}")
                report["cases"].append(dict(layer=layer, bank=index, byte_exact=True,
                    sha256=hashlib.sha256(memoryview(actual.view(torch.uint8).numpy())).hexdigest()))
        report["manifest"] = save(candidate, a.prepared_dir)
        reloaded = HeadSource(a.model, a.prepared_dir)
        start = time.perf_counter()
        reloaded.prepare()
        report["prepared_load_seconds"] = time.perf_counter() - start
        assert reloaded.prepared_manifest is not None and reloaded.preparation_backend == "pypto"
        assert torch.equal(candidate.shared_weights, reloaded.shared_weights), "prepared file did not round-trip"
        report.update(passed=True, prepared_reload_byte_exact=True, prepared_directory=str(a.prepared_dir))
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
