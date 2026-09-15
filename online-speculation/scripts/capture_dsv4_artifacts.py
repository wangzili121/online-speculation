#!/usr/bin/env python3
"""Hash code, runtime, model, and head inputs for an offline DSV4 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.vllm_ascend.artifacts import write_private_json  # noqa: E402
from blockspec.vllm_ascend.head_contract import sha256_file  # noqa: E402


CODE_DIRS = ("benchmarks", "configs", "docs", "runtime", "scripts", "src", "tests")
CODE_FILES = ("README.md", "pyproject.toml", ".gitignore")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_NAMES = {"runtime-lock.local.json"}


def _record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _tree(directory: Path, skip_weights: bool = False) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if skip_weights and path.suffix in {".safetensors", ".bin", ".pt"}:
            continue
        records.append(_record(path, directory))
    return records


def _code_tree(root: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for name in CODE_FILES:
        path = root / name
        if path.is_file():
            paths.append(path)
    for name in CODE_DIRS:
        directory = root / name
        if directory.is_dir():
            paths.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.name not in SKIP_NAMES
                and not any(part in SKIP_PARTS for part in path.parts)
            )
    return [_record(path, root) for path in sorted(set(paths))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-weight-shards",
        action="store_true",
        help="development-only fast mode; production manifests must hash every shard",
    )
    args = parser.parse_args()
    model = args.model.resolve()
    head = args.head.resolve()
    lock = args.runtime_lock.resolve()
    for name, path in (("model", model), ("head", head)):
        if not path.is_dir():
            raise SystemExit(f"{name} directory does not exist: {path}")
    if not lock.is_file():
        raise SystemExit(f"runtime lock does not exist: {lock}")
    lock_data = json.loads(lock.read_text(encoding="utf-8"))
    if not str(lock_data.get("image_digest", "")).startswith("sha256:"):
        raise SystemExit("runtime lock has no immutable image digest")
    manifest = {
        "schema": "dsv4-orthrus-offline-release-v1",
        "complete_weight_hashes": not args.skip_weight_shards,
        "runtime_lock": {
            "name": lock.name,
            "sha256": sha256_file(lock),
            "content": lock_data,
        },
        "code": _code_tree(ROOT),
        "model": {"path": str(model), "files": _tree(model, args.skip_weight_shards)},
        "head": {"path": str(head), "files": _tree(head, args.skip_weight_shards)},
    }
    write_private_json(args.output, manifest)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
