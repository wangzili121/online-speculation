#!/usr/bin/env python3
"""独立核对加载阶段的 PyPTO 转换、拼接、补零、有限值及副本一致性。"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("new output required")
    import torch
    import pypto
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity
    from blockspec.pypto_qwen38.loading import LoadingPrograms, DeviceLoader

    torch.set_num_threads(4)
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=[a.device], num_sub_workers=0))
    registry = LoadingPrograms(config)
    cases = [
        dict(name="qkv_section", shape=(2560, 5120), target=(10240, 5120), size=(512, 5120), src=(512, 0), dst=(2048, 0)),
        dict(name="column_shard", shape=(5120, 1536), target=(5120, 6144), size=(5120, 1536), src=(0, 0), dst=(0, 1536)),
        dict(name="gate_padding", shape=(12, 5120), target=(128, 5120), size=(12, 5120), src=(0, 0), dst=(24, 0)),
        dict(name="conv_transpose", shape=(2560, 4), target=(4, 10240), size=(4, 512), src=(512, 0), dst=(0, 2048), conv=True),
        dict(name="base_conv_transpose", shape=(10240, 4), target=(192, 10240), size=(4, 10240), src=(0, 0), dst=(20, 0), conv=True),
        dict(name="norm_gamma", shape=(1, 256), target=(1, 256), size=(1, 256), src=(0, 0), dst=(0, 0), gamma=True, dtype=torch.float32),
        dict(name="base_gamma", shape=(1, 5120), target=(64, 5120), size=(1, 5120), src=(0, 0), dst=(17, 0), gamma=True, dtype=torch.float32),
        dict(name="a_log_cast", shape=(1, 64), target=(1, 64), size=(1, 64), src=(0, 0), dst=(0, 0), dtype=torch.float32),
        dict(name="a_log_fp32", shape=(1, 64), target=(1, 64), size=(1, 64), src=(0, 0), dst=(0, 0), dtype=torch.float32, source_dtype=torch.float32),
    ]
    for case in cases:
        case["dtype"] = case.get("dtype", torch.bfloat16)
        case["source_dtype"] = case.get("source_dtype", torch.bfloat16)
        case["key"] = registry.copy(case["shape"], case["target"], case["source_dtype"], case["dtype"],
                                     conv=case.get("conv", False), gamma=case.get("gamma", False), size=case["size"])
        registry.zero(case["target"], case["dtype"])
    registry.equal(128)
    registry.equal(256)
    worker = loader = None
    report = dict(passed=False, npu_execution=True, runtime=runtime_identity(pypto), cases=[])
    try:
        worker = DistributedWorker(list(registry.programs.values()), config=config)
        loader = DeviceLoader(registry, worker)
        generator = torch.Generator().manual_seed(554)
        for case in cases:
            x = torch.randn(case["shape"], generator=generator).to(case["source_dtype"])
            if case["name"].startswith("a_log"):
                x[0, 48:] = 0
            source = loader.allocate(case["shape"], case["source_dtype"], init=x)
            target = loader.allocate(case["target"], case["dtype"])
            loader.zero(target)
            loader.copy(case["key"], source, target, source_offset=case["src"], target_offset=case["dst"])
            actual = torch.empty(case["target"], dtype=case["dtype"])
            worker.copy_from(actual.data_ptr(), target.data_ptr, actual.numel() * actual.element_size())
            expected = torch.zeros_like(actual)
            sr, sc = case["src"]
            dr, dc = case["dst"]
            rows, cols = case["size"]
            piece = x[sr:sr + cols].T if case.get("conv") else x[sr:sr + rows, sc:sc + cols]
            piece = piece.float() + 1 if case.get("gamma") else piece
            expected[dr:dr + rows, dc:dc + cols] = piece.to(case["dtype"])
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), case["name"]
            if case["name"] == "gate_padding":
                x[0, 0] = float("nan")
                worker.copy_to(source.data_ptr, x.data_ptr(), x.numel() * x.element_size())
                try:
                    loader.copy(case["key"], source, target, source_offset=case["src"], target_offset=case["dst"])
                except ValueError as error:
                    assert "nonfinite" in str(error)
                else:
                    raise AssertionError("nonfinite weight was accepted")
            loader.free(target)
            loader.free(source)
            report["cases"].append(dict(name=case["name"], passed=True, byte_exact=True))
            print(case["name"], "passed", flush=True)
        for n in (128, 256):
            x = torch.zeros((1, n), dtype=torch.bfloat16)
            y = x.clone()
            y[0, 0] = -0.
            dx = loader.allocate(tuple(x.shape), x.dtype, init=x)
            dy = loader.allocate(tuple(y.shape), y.dtype, init=y)
            loader.equal(dx, dy)
            for bad in (1., float("nan"), float("inf")):
                y[0, -1] = bad
                worker.copy_to(dy.data_ptr, y.data_ptr(), y.numel() * y.element_size())
                try:
                    loader.equal(dx, dy)
                except ValueError:
                    pass
                else:
                    raise AssertionError("mismatched replicated norm accepted")
            loader.free(dy)
            loader.free(dx)
            report["cases"].append(dict(name=f"replicated_{n}", passed=True))
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if loader is not None:
                loader.close()
        finally:
            if worker is not None:
                worker.close()
            a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
