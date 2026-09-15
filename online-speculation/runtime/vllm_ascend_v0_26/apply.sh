#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
vllm_root="${VLLM_ROOT:-${1:-/vllm-workspace/vllm}}"
ascend_root="${VLLM_ASCEND_ROOT:-${2:-/vllm-workspace/vllm-ascend}}"
lock="${RUNTIME_LOCK:-}"
python_bin="${PYTHON_BIN:-python3}"

if [[ -z "$lock" && "${ORTHRUS_ALLOW_UNPINNED:-0}" != "1" ]]; then
  echo "Set RUNTIME_LOCK to a file produced by capture_dsv4_runtime_lock.py" >&2
  exit 1
fi

"$python_bin" - "$vllm_root" "$ascend_root" "$lock" <<'PY'
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys

vllm_root = pathlib.Path(sys.argv[1]).resolve()
ascend_root = pathlib.Path(sys.argv[2]).resolve()
lock_path = pathlib.Path(sys.argv[3]).resolve() if sys.argv[3] else None

def commit(root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

def command(*args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def first_existing(paths):
    for path in paths:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    return None

for path in (
    vllm_root / "vllm/config/speculative.py",
    ascend_root / "vllm_ascend/spec_decode/llm_base_proposer.py",
    ascend_root / "vllm_ascend/attention/dsa_v1.py",
):
    if not path.is_file():
        raise SystemExit(f"runtime source file not found: {path}")

if lock_path is not None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = (
        "image_digest",
        "vllm_version",
        "vllm_ascend_version",
        "vllm_commit",
        "vllm_ascend_commit",
        "cann",
        "driver",
    )
    missing = [key for key in required if not lock.get(key)]
    if missing:
        raise SystemExit(f"runtime lock is incomplete: {missing}")
    actual = {
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_ascend_version": importlib.metadata.version("vllm-ascend"),
        "vllm_commit": commit(vllm_root),
        "vllm_ascend_commit": commit(ascend_root),
        "image_digest": os.environ.get("ORTHRUS_IMAGE_DIGEST"),
        "cann": first_existing(
            [
                pathlib.Path(
                    "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/"
                    "ascend_toolkit_install.info"
                ),
                pathlib.Path(
                    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/"
                    "ascend_toolkit_install.info"
                ),
                pathlib.Path("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
            ]
        ),
        "driver": first_existing(
            [
                pathlib.Path("/usr/local/Ascend/driver/version.info"),
                pathlib.Path("/etc/ascend_install.info"),
            ]
        )
        or command("npu-smi", "info", "-t", "board", "-i", "0"),
    }
    if not actual["image_digest"]:
        raise SystemExit(
            "ORTHRUS_IMAGE_DIGEST is missing; create the container with "
            "create_dsv4_container.sh or export the pinned sha256 digest"
        )
    image_ref = os.environ.get("ORTHRUS_IMAGE_REF")
    if image_ref and image_ref != lock.get("image_ref"):
        raise SystemExit(
            f"runtime image ref mismatch: lock={lock.get('image_ref')!r}, "
            f"container={image_ref!r}"
        )
    mismatch = {
        key: (lock[key], value)
        for key, value in actual.items()
        if lock[key] != value
    }
    if mismatch:
        raise SystemExit(f"runtime lock mismatch: {mismatch}")
else:
    vllm = importlib.metadata.version("vllm").split("+", 1)[0]
    ascend = importlib.metadata.version("vllm-ascend")
    if vllm != "0.26.0" or not ascend.startswith("0.26.0rc"):
        raise SystemExit(f"expected vLLM 0.26.0 / Ascend 0.26.0rc, got {vllm} / {ascend}")
PY

"$python_bin" "$here/patch_runtime.py" \
  --vllm-root "$vllm_root" \
  --ascend-root "$ascend_root" \
  --dry-run
"$python_bin" "$here/patch_runtime.py" \
  --vllm-root "$vllm_root" \
  --ascend-root "$ascend_root"

"$python_bin" -m pip install --no-deps --no-build-isolation -e "$repo_root"
"$python_bin" - <<'PY'
import vllm_ascend.ops

from blockspec.vllm_ascend.plugin import register
from vllm_ascend.spec_decode import AscendOrthrusProposer

assert AscendOrthrusProposer.__name__ == "AscendOrthrusProposer"
register()
print("Orthrus package import and model registration passed")
PY
