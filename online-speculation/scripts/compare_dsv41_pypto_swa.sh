#!/usr/bin/env bash
# Validate staged SWA, with an optional same-environment comparison against the old kernel.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
cd /results

modes=(staged)
if [[ "${1:-}" == --compare-baseline ]]; then
    modes=(fused staged)
    shift
fi

status=0
for mode in "${modes[@]}"; do
    flags=(--device 0 --heads 16 --serving-layout --compressed --iterations 30 --check-every-run)
    if [[ "$mode" == fused ]]; then flags+=(--fused-swa); fi
    code=0
    timeout -k 5s 180s python "$root/scripts/validate_dsv41_pypto_swa.py" \
        "${flags[@]}" "$@" --output "$mode.json" >"$mode.log" 2>&1 || code=$?
    if [[ "$mode" == staged ]]; then status="$code"; fi
    python - "$mode" "$code" <<'PY'
import json
from pathlib import Path
import sys

mode, code = sys.argv[1], int(sys.argv[2])
report = Path(f"{mode}.json")
summary = dict(mode=mode, exit_code=code)
if report.is_file():
    result = json.loads(report.read_text())
    summary.update(passed=result["passed"], npu_execution=result["npu_execution"], cases=len(result["cases"]))
    if result["cases"]:
        case = result["cases"][-1]
        for field in ("anchor", "ratio", "runs_checked"):
            summary[field] = case[field]
        for field in ("swa_output_max_abs", "swa_lse_max_abs", "output_max_abs", "lse_max_abs"):
            summary[field] = max(row[field] for row in result["cases"])
    if "error" in result:
        summary["error"] = result["error"].split(r"\n", 1)[0][:200]
else:
    summary["log"] = f"{mode}.log"
print(json.dumps(summary))
PY
done
exit "$status"
