#!/usr/bin/env python3
"""Build OneServing's pinned stack inside an isolated PyPTO development container.

Sources and wheels are supplied explicitly. No git reset, driver installation,
or dependency-pin bypass is performed. Run with the container's target Python.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PINS = {
    "pypto": "98752784548b777b2377a9084411c3716eb341d6",
    "simpler": "c1f89ea431f5306f78c0e4ab9573c829756d7f1e",
    "pto_isa": "418de18dde8fa33773dc5c384fdfe5506463b628",
    "oneserving": "7aa40ba5238e3a83bb5f25e18817497c181e3e9b",
}
PTOAS_SHA256 = "aea205361edd3d4e916b2c1353112fbcf2d1588c62cdf905c9f30856d4ecb744"


def revision(path):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto-source", type=Path, required=True)
    parser.add_argument("--oneserving-source", type=Path, required=True)
    parser.add_argument("--ptoas-wheel", type=Path, required=True,
                        help="official v0.65 cp310 aarch64 wheel, verified by SHA256")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for build logs")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists() or args.jobs < 1:
        parser.error("require a new output directory and positive jobs")
    core, serving = args.pypto_source.resolve(), args.oneserving_source.resolve()
    sources = dict(pypto=core, simpler=core / "runtime",
                   pto_isa=core / "runtime/build/pto-isa", oneserving=serving)
    args.output_dir.mkdir(parents=True)
    report = dict(pins=PINS, stages=[], npu_execution=False, state="checking")
    env = dict(os.environ, PYTHONPATH=str(core / "python"),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
               CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), OMP_NUM_THREADS=str(args.jobs))
    env.pop("PTOAS_ROOT", None)
    try:
        if sys.version_info[:2] != (3, 10):
            raise ValueError("the verified toolchain wheel requires Python 3.10")
        if hashlib.sha256(args.ptoas_wheel.read_bytes()).hexdigest() != PTOAS_SHA256:
            raise ValueError("PTOAS wheel SHA256 does not match the pinned toolchain")
        for name, source in sources.items():
            if revision(source) != PINS[name]:
                raise ValueError(f"wrong {name} source revision")
            # This script runs in an isolated container; preserve host git config.
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", str(source)], check=True)
            dirty = subprocess.check_output(
                ["git", "-C", str(source), "diff", "HEAD", "--ignore-submodules=none", "--name-only"],
                text=True).strip()
            if dirty:
                raise ValueError(f"modified pinned source: {name}: {dirty}")
        if (serving / "pypto-serving-qwen3.8-27B/pypto.pin").read_text().strip() != PINS["pypto"]:
            raise ValueError("OneServing dependency pin differs")
        if (core / "runtime/pto_isa.pin").read_text().strip() != PINS["pto_isa"]:
            raise ValueError("Simpler PTO-ISA pin differs")
        tasks = [
            ("ptoas", ["--no-deps", str(args.ptoas_wheel.resolve())]),
            ("simpler", ["--no-build-isolation", "--no-deps", "-e", str(core / "runtime")]),
            ("pypto", ["--no-build-isolation", "--no-deps", "-e", str(core)]),
            ("serving", ["-r", str(serving / "pypto-serving-qwen3.8-27B/requirements.txt")]),
        ]
        for name, options in tasks:
            report.update(state="building", stage=name)
            (args.output_dir / "status.json").write_text(json.dumps(report, indent=2) + "\n")
            start = time.monotonic()
            print(f"building {name}; log: {args.output_dir / (name + '.log')}", flush=True)
            with (args.output_dir / f"{name}.log").open("wb") as log:
                subprocess.run(["uv", "pip", "install", "--python", sys.executable, *options],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report["stages"].append(dict(stage=name, seconds=time.monotonic() - start))
        # A fresh interpreter must load the rebuilt native bindings with pin checks enabled.
        subprocess.run([sys.executable, "-c",
                        "from pypto.runtime.runtime_pin import check_runtime_pin; "
                        "s = check_runtime_pin(); "
                        "assert s.matches and not s.off_pin and s.installed == '" + PINS["simpler"] + "'"],
                       env=env, check=True)
        report["state"] = "built"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        (args.output_dir / "status.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
