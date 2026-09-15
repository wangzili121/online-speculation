#!/usr/bin/env python3
"""Capture immutable image/runtime revisions before applying the overlay."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path


def command(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit(path: Path) -> str | None:
    return command("git", "rev-parse", "HEAD", cwd=path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    return None


def image_digest(image: str) -> str | None:
    value = command(
        "docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image
    )
    if value and "@sha256:" in value:
        return value.split("@", 1)[1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, default=Path("/vllm-workspace/vllm"))
    parser.add_argument(
        "--ascend-root", type=Path, default=Path("/vllm-workspace/vllm-ascend")
    )
    parser.add_argument(
        "--image-ref",
        default="quay.io/ascend/vllm-ascend:nightly-releases-v0.26.0rc",
    )
    parser.add_argument("--image-digest")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = args.image_digest or image_digest(args.image_ref)
    if not digest or not digest.startswith("sha256:"):
        raise SystemExit(
            "image digest unavailable; pass --image-digest sha256:... from the host"
        )
    lock = {
        "image_ref": args.image_ref,
        "image_digest": digest,
        "vllm_version": package_version("vllm"),
        "vllm_ascend_version": package_version("vllm-ascend"),
        "vllm_commit": git_commit(args.vllm_root.resolve()),
        "vllm_ascend_commit": git_commit(args.ascend_root.resolve()),
        "cann": first_existing(
            [
                Path("/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info"),
                Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"),
                Path("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
            ]
        ),
        "driver": first_existing(
            [
                Path("/usr/local/Ascend/driver/version.info"),
                Path("/etc/ascend_install.info"),
            ]
        )
        or command("npu-smi", "info", "-t", "board", "-i", "0"),
    }
    missing = [key for key, value in lock.items() if value in (None, "")]
    if missing:
        raise SystemExit(f"refusing an incomplete runtime lock; missing {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.JSONEncoder(indent=2, sort_keys=True).encode(lock) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
