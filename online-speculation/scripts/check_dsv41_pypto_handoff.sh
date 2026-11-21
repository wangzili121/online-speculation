#!/usr/bin/env bash
# Run the offline environment, checkpoint and optional NPU checks with one set of logs.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
cd /results

check() {
    local stage="$1" timeout_seconds="$2"
    shift 2
    local code=0
    timeout -k 5s "$timeout_seconds" "$@" >"$stage.json" 2>"$stage.log" || code=$?
    STAGE="$stage" CODE="$code" python - <<'PY'
import json
import os
from pathlib import Path

stage, code = os.environ["STAGE"], int(os.environ["CODE"])
if code:
    log = Path(f"{stage}.log").read_text(errors="replace").splitlines()
    print(json.dumps(dict(stage=stage, passed=False, exit_code=code, error=log[-1][:400] if log else None)))
else:
    report = json.loads(Path(f"{stage}.json").read_text())
    fields = (("pypto", "simpler") if stage == "environment"
              else ("identity", "engram_host_table_bytes", "head", "head_artifact_role"))
    print(json.dumps(dict(stage=stage, passed=True, **{key: report[key] for key in fields if key in report})))
PY
    return "$code"
}

check environment 60 python "$root/scripts/check_dsv41_pypto_environment.py"
head=()
if [[ -n "${DRAFT_HEAD:-}" ]]; then head=(--head "$DRAFT_HEAD"); fi
check checkpoint 120 python "$root/scripts/preflight_dsv41_pypto.py" "${head[@]}"
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    bash "$root/scripts/compare_dsv41_pypto_swa.sh" "$@"
fi
