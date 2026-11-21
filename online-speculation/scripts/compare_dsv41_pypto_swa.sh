#!/usr/bin/env bash
# Run the two SWA paths within the same PyPTO environment and preserve both reports.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
cd /results

status=0
for mode in accumulated separate; do
    flags=(--device 0 --compressed --iterations 30 --check-every-run)
    if [[ "$mode" == separate ]]; then flags+=(--separate-swa-pv); fi
    code=0
    timeout -k 5s 180s python "$root/scripts/validate_dsv41_pypto_swa.py" \
        "${flags[@]}" "$@" --output "$mode.json" >"$mode.log" 2>&1 || code=$?
    if (( code != 0 )); then status=1; fi
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
        for field in ("anchor", "ratio", "runs_checked", "swa_output_max_abs", "swa_lse_max_abs",
                      "output_max_abs", "lse_max_abs"):
            summary[field] = case[field]
    if "error" in result:
        summary["error"] = result["error"].split(r"\n", 1)[0][:200]
else:
    summary["log"] = f"{mode}.log"
print(json.dumps(summary))
PY
done
exit "$status"
