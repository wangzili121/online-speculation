#!/usr/bin/env python3
"""Run a repository script while preserving the image's CANN Python paths."""

import os
from pathlib import Path
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_in_runtime.py SCRIPT [ARGS ...]")
    source = str(Path(__file__).resolve().parents[1] / "src")
    # worker 子进程也需要项目源码，但镜像原有的 acl/TBE 路径不能被覆盖。
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = source + (os.pathsep + previous if previous else "")
    sys.path.insert(0, source)
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
