#!/usr/bin/env python3
"""Record the installed A3 environment and optionally execute one NPU kernel."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess


def report():
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
    result = dict(architecture=platform.machine(), packages=packages, pypto=commit,
                  serving_lib=source.audit(), kernel_executed=False)
    for name, path in (
        ("driver", "/usr/local/Ascend/driver/version.info"),
        ("cann", "/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
    ):
        file = Path(path)
        result[name] = file.read_text() if file.is_file() else "not exposed in this container"
    return result


def device_probe():
    import pypto.language as pl
    from pypto.backend import BackendType
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
    add(a, b, c, config=RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=0))
    torch.testing.assert_close(c, a + b, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-probe", action="store_true")
    args = parser.parse_args()
    result = report()
    if args.device_probe:
        device_probe()
        result["kernel_executed"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
