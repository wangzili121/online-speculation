"""Private experiment manifests for reproducible Orthrus serving trials."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(command, cwd=None):
    try:
        return subprocess.run(
            command, cwd=cwd, check=True, capture_output=True, text=True,
            timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def capture_manifest(*, repo, workload, model, runtime, trial):
    repo = Path(repo).resolve()
    workload = Path(workload).resolve()
    diff = _command(["git", "diff", "--binary", "HEAD"], cwd=repo)
    return {
        "schema_version": 1,
        "created_unix_seconds": time.time(),
        "trial": trial,
        "source": {
            "repository": str(repo),
            "head": _command(["git", "rev-parse", "HEAD"], cwd=repo),
            "status": _command(["git", "status", "--porcelain"], cwd=repo),
            "working_tree_diff_sha256": hashlib.sha256(
                diff.encode()).hexdigest(),
        },
        "inputs": {
            "workload": str(workload),
            "workload_sha256": sha256_file(workload),
            "model": str(Path(model).resolve()),
        },
        "runtime": runtime,
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "platform": platform.platform(),
            "container_id": _command([
                "sh", "-c", "cat /proc/self/cgroup | tail -1"
            ]),
            "pip_versions": _command([
                sys.executable, "-m", "pip", "show", "vllm",
                "vllm-ascend", "torch", "torch-npu"
            ]),
            "npu_smi": _command(["npu-smi", "info"]),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        },
    }


def write_manifest(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def encode_private_json(value, **kwargs):
    """Encode protocol or reproducibility data without publication filtering."""
    return json.dumps(value, **kwargs)


def write_private_json(path, value, *, exclusive=False, **kwargs):
    """Write a local artifact that must retain paths, hashes, or token data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    options = {"ensure_ascii": True, "indent": 2, "sort_keys": True, **kwargs}
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, **options)
        handle.write("\n")
