#!/usr/bin/env bash
# Single-node gate before the dual-node resident online-update acceptance.
set -euo pipefail

device="${1:?device is required}"
output="${2:?output directory is required}"
root=/workspace/online-speculation
mkdir -p "$output"

python=/opt/pypto/venv/bin/python
# Keep the image's pinned PyPTO/PTOAS/CANN search path and prepend only this
# checkout.  Replacing PYTHONPATH would make the CPU tests pass but hide the
# compiler from the subsequent real-device gate.
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python" -m pytest -q "$root/tests" -k dsv41 > "$output/tests.log" 2>&1

serving="$(mktemp -d /tmp/orthrus-serving-online.XXXXXX)"
trap 'rm -r -- "$serving"' EXIT
cp -a /opt/orthrus/pypto-serving/. "$serving/"
git -C "$serving" apply --check "$root/runtime/pypto_dsv41_a3/serving-cluster.patch"
git -C "$serving" apply "$root/runtime/pypto_dsv41_a3/serving-cluster.patch"
git -C "$serving" apply --check "$root/runtime/pypto_dsv41_a3/serving-online.patch"
git -C "$serving" apply "$root/runtime/pypto_dsv41_a3/serving-online.patch"
"$python" -m py_compile "$serving/models/deepseek_v41_a3/session.py"

bash "$root/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh"
"$python" "$root/scripts/validate_dsv41_pypto_online_programs.py" \
  --device "$device" --vocabulary 129280 --output "$output/online-kernels.json"
"$python" "$root/scripts/validate_dsv41_resident_publication.py" \
  --device "$device" --output "$output/resident-publication.json"

"$python" - "$output" <<'PY'
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
text = (root / "tests.log").read_text()
match = re.search(r"(\d+) passed, (\d+) skipped.*?(\d+) subtests passed", text, re.S)
if match is None:
    match = re.search(r"(\d+) passed", text)
    tests = {"passed": int(match.group(1)) if match else None}
else:
    tests = {
        "passed": int(match.group(1)),
        "skipped": int(match.group(2)),
        "subtests_passed": int(match.group(3)),
    }
kernels = json.loads((root / "online-kernels.json").read_text())
publication = json.loads((root / "resident-publication.json").read_text())
report = {
    "format": "orthrus-dsv41-pypto-online-readiness-v1",
    "passed": bool(kernels.get("passed") and publication.get("passed")),
    "scope": "all single-node gates; dual-node resident update not included",
    "tests": tests,
    "serving_patch": {"applied": True, "python_compiled": True},
    "kernels": {
        "compiled": len(kernels.get("compiled", [])),
        "executed": kernels.get("executed", False),
        "numerical_cases": len(kernels.get("cases", [])),
    },
    "resident_publication": {
        "passed": publication.get("passed", False),
        "host_round_trip": publication.get("host_round_trip"),
        "d2d_bytes": publication.get("d2d_bytes"),
    },
}
(root / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

# The container runs as root while the report directory belongs to the host
# caller. Keep the evidence readable without a second privileged container.
chmod a+r "$output/tests.log" "$output/online-kernels.json" \
  "$output/resident-publication.json" "$output/readiness.json"
