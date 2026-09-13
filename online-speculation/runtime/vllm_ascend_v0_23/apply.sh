#!/usr/bin/env bash
set -euo pipefail

root="${1:-/vllm-workspace/vllm-ascend}"
python - "$root" <<'PY'
import importlib.metadata
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
if importlib.metadata.version("vllm").split("+", 1)[0] != "0.23.0":
    raise SystemExit("Orthrus overlay requires vLLM 0.23.0")
if importlib.metadata.version("vllm-ascend") != "0.23.0":
    raise SystemExit("Orthrus overlay requires vLLM-Ascend 0.23.0")
if not (root / "vllm_ascend/worker/model_runner_v1.py").is_file():
    raise SystemExit("vLLM-Ascend source tree not found")
PY
here="$(cd "$(dirname "$0")" && pwd)"
runner="$root/vllm_ascend/worker/model_runner_v1.py"
apply_one() {
    local patch_file="$1"
    local marker="$2"
    if grep -Fq "$marker" "$runner"; then
        echo "$(basename "$patch_file") is already applied"
        return
    fi
    if patch --dry-run --forward -d "$root" -p1 < "$patch_file" >/dev/null 2>&1; then
        patch --forward -d "$root" -p1 < "$patch_file"
    else
        echo "vLLM-Ascend source differs from $(basename "$patch_file")" >&2
        exit 1
    fi
}
apply_one "$here/orthrus.patch" "self._draft_prob_req_ids = None"
apply_one "$here/instrumentation_draft.patch" 'record_function_or_nullcontext("orthrus.draft.run")'
apply_one "$here/instrumentation.patch" 'record_function_or_nullcontext("orthrus.rejection_sampling")'
