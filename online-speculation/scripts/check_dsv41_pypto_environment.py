#!/usr/bin/env python3
"""Record the installed A3 environment and optionally execute one NPU kernel."""

import argparse
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def report():
    if Path(sys.prefix) != Path("/opt/pypto/venv"):
        raise RuntimeError(f"Use the PyPTO application environment, got {sys.executable}")
    import pypto
    import pypto.runtime
    import simpler.task_interface
    from pypto_serving_a3.source import KernelSource

    source = KernelSource.resolve()
    packages = {name: version(name) for name in (
        "torch", "numpy", "transformers", "safetensors", "tokenizers", "pypto-serving", "online-speculation",
    )}
    if packages["torch"].split("+")[0] != "2.10.0":
        raise RuntimeError(f"unexpected Torch version: {packages['torch']}")
    core = Path(pypto.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(core), "rev-parse", "HEAD"], text=True).strip()
    if commit != "72a62d13efabbb25d240e35944f68ba5bcc22b7f":
        raise RuntimeError(f"unexpected PyPTO revision: {commit}")
    runtime = pypto.runtime.check_runtime_pin(checkout_root=core)
    if runtime.installed != "6b6b88f2353cca12cfafe809750b50b59931b0d1" or not runtime.matches:
        raise RuntimeError(f"unexpected Simpler build: {runtime}")
    patch = Path(__file__).parents[1] / "runtime/pypto_dsv41_a3/runtime-release.patch"
    subprocess.run(["git", "-C", str(core / "runtime"), "apply", "--reverse", "--check", str(patch)],
                   check=True)
    result = dict(architecture=platform.machine(), python=sys.executable, packages=packages, pypto=commit,
                  simpler=runtime.installed, runtime_compatibility_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                  serving_lib=source.audit(), kernel_compiled=False, kernel_executed=False)
    cann_home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    cann_version = next((file for file in (cann_home / "version.cfg", cann_home / "compiler/version.info")
                         if file.is_file()), cann_home / "version.cfg")
    result["cann_home"] = str(cann_home)
    for name, path in (
        ("driver", "/usr/local/Ascend/driver/version.info"),
        ("cann", str(cann_version)),
    ):
        file = Path(path)
        result[name] = file.read_text() if file.is_file() else "not exposed in this container"
    return result


def device_probe(*, device=0, compile_only=False):
    import pypto.language as pl
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    import torch

    @pl.jit
    def add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            left = pl.load(a, [0, 0], [128, 128])
            right = pl.load(b, [0, 0], [128, 128])
            pl.store(pl.add(left, right), [0, 0], c)
        return c

    a = torch.full((128, 128), 2.0)
    b = torch.full((128, 128), 3.0)
    c = torch.zeros_like(a)
    config = RunConfig(
        backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=[device], num_sub_workers=0),
    )
    if config.execution_mode != ExecutionMode.ONBOARD:
        raise RuntimeError("device probe must execute on the NPU")
    program = add.compile(a, b, c, config=config)
    if compile_only:
        return
    program(a, b, c, config=config)
    torch.testing.assert_close(c, a + b, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--device-probe", action="store_true")
    mode.add_argument("--compile-probe", action="store_true", help="Compile the device kernel without using an NPU")
    parser.add_argument("--device", type=int, default=0, help="Device ID passed to the pinned runtime")
    args = parser.parse_args()
    result = report()
    if args.device_probe or args.compile_probe:
        device_probe(device=args.device, compile_only=args.compile_probe)
        result.update(kernel_compiled=True, kernel_executed=args.device_probe)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
