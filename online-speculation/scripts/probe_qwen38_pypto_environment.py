#!/usr/bin/env python3
"""Verify the pinned toolchain and execute a small, exact NPU computation."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--platform", choices=("a2a3", "a2a3sim"), default="a2a3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0 or args.output.exists():
        parser.error("require an assigned nonnegative device and a new output file")
    import pypto
    import pypto.language as pl
    import torch
    from pypto.runtime import RunConfig
    from blockspec.pypto_qwen38.baseline import runtime_identity

    report = dict(passed=False, npu_execution=False, platform=args.platform,
                  scope="pinned_runtime_vector_probe")
    start = time.monotonic()
    try:
        report["runtime"] = runtime_identity(pypto)

        @pl.jit
        def add(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP):
                x = pl.load(a, [0, 0], [128, 128])
                y = pl.load(b, [0, 0], [128, 128])
                output = pl.store(pl.add(x, y), [0, 0], output)
            return output

        a = torch.arange(16384, dtype=torch.float32).reshape(128, 128)
        b, output = torch.full_like(a, 3), torch.zeros_like(a)
        report["npu_execution_attempted"] = args.platform == "a2a3"
        add(a, b, output, config=RunConfig(platform=args.platform, device_id=args.device))
        report.update(npu_execution=args.platform == "a2a3", passed=torch.equal(output, a + b),
                      max_abs_error=float((output - a - b).abs().max()))
        if not report["passed"]:
            raise AssertionError("NPU vector result differs from the exact expected output")
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
