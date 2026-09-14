#!/usr/bin/env bash
set -euo pipefail

root="${1:-/vllm-workspace/vllm-ascend}"
vllm_root="${2:-/vllm-workspace/vllm}"
python - "$root" "$vllm_root" <<'PY'
import importlib.metadata
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
vllm_root = pathlib.Path(sys.argv[2]).resolve()
if importlib.metadata.version("vllm").split("+", 1)[0] != "0.23.0":
    raise SystemExit("Orthrus overlay requires vLLM 0.23.0")
if importlib.metadata.version("vllm-ascend") != "0.23.0":
    raise SystemExit("Orthrus overlay requires vLLM-Ascend 0.23.0")
if not (root / "vllm_ascend/worker/model_runner_v1.py").is_file():
    raise SystemExit("vLLM-Ascend source tree not found")
if not (vllm_root / "vllm/config/vllm.py").is_file():
    raise SystemExit("vLLM source tree not found")
PY
here="$(cd "$(dirname "$0")" && pwd)"
runner="$root/vllm_ascend/worker/model_runner_v1.py"
apply_one() {
    local patch_root="$1"
    local patch_file="$2"
    local marker_file="$3"
    local marker="$4"
    if grep -Fq "$marker" "$marker_file"; then
        echo "$(basename "$patch_file") is already applied"
        return
    fi
    if patch --dry-run --forward -d "$patch_root" -p1 < "$patch_file" >/dev/null 2>&1; then
        patch --forward -d "$patch_root" -p1 < "$patch_file"
    else
        echo "vLLM-Ascend source differs from $(basename "$patch_file")" >&2
        exit 1
    fi
}
apply_one "$root" "$here/orthrus.patch" "$runner" "self._draft_prob_req_ids = None"
apply_one "$root" "$here/instrumentation_draft.patch" "$runner" 'record_function_or_nullcontext("orthrus.draft.run")'
apply_one "$root" "$here/instrumentation.patch" "$runner" 'record_function_or_nullcontext("orthrus.rejection_sampling")'
apply_one "$root" "$here/instrumentation_state.patch" "$runner" 'record_function_or_nullcontext("orthrus.state_commit")'
apply_one "$root" "$here/orthrus_async.patch" "$runner" 'take_last_async_state()'
apply_one "$root" "$here/orthrus_async_gate.patch" "$runner" \
    'one-cycle asynchronous correction pipeline'
apply_one "$vllm_root" "$here/../vllm_v0_23/orthrus_async.patch" \
    "$vllm_root/vllm/config/vllm.py" \
    'self.speculative_config.method != "custom_class"'
