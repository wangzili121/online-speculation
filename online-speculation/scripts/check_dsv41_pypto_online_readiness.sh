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
cleanup() {
  rm -r -- "$serving"
  chmod -R a+rX "$output" 2>/dev/null || true
}
trap cleanup EXIT
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

# Exercise every production reverse boundary on a real device before the
# factory is allowed to advertise online-update readiness. These checks cover
# the exact K32/hidden=5120 geometry; compact numerical fixtures are used only
# where allocating a second full golden tensor would obscure the device gate.
retry_gate() {
  local name="$1"
  shift
  local report="$output/$name.json"
  local attempt
  for attempt in 1 2 3; do
    rm -f -- "$report"
    if "$@"; then
      return 0
    fi
    if [[ -f "$report" ]]; then
      mv -- "$report" "$output/$name.attempt-$attempt.json"
    fi
    sleep $((attempt * 3))
  done
  echo "$name failed after three isolated device attempts" >&2
  return 1
}

retry_gate hc-collapse-vjp \
  "$python" "$root/scripts/validate_dsv41_pypto_hc_collapse_vjp.py" \
  --device "$device" --production --output "$output/hc-collapse-vjp.json"
retry_gate hc-pre-vjp \
  "$python" "$root/scripts/validate_dsv41_pypto_hc_pre_vjp.py" \
  --device "$device" --output "$output/hc-pre-vjp.json"
retry_gate c1a-vjp \
  "$python" "$root/scripts/validate_dsv41_pypto_c1a_vjp.py" \
  --device "$device" --output "$output/c1a-vjp.json"
retry_gate expert-capture \
  "$python" "$root/scripts/validate_dsv41_pypto_expert_capture.py" \
  --device "$device" --output "$output/expert-capture.json"
retry_gate expert-input-vjp \
  "$python" "$root/scripts/validate_dsv41_pypto_expert_input_vjp.py" \
  --device "$device" --output "$output/expert-input-vjp.json"
retry_gate ep-vjp \
  "$python" "$root/scripts/validate_dsv41_pypto_ep_vjp.py" \
  --device "$device" --output "$output/ep-vjp.json"

# Keep this as the final gate: the device kernels and D2D publication above
# still produce useful evidence while the full resident reverse is being
# developed, but a readiness report must never pass a partial factory.
"$python" "$root/scripts/check_dsv41_online_factory.py" \
  > "$output/online-factory.json"

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
reverse_names = (
    "hc-collapse-vjp", "hc-pre-vjp", "c1a-vjp", "expert-capture",
    "expert-input-vjp", "ep-vjp",
)
reverse = {
    name: json.loads((root / f"{name}.json").read_text())
    for name in reverse_names
}
factory = json.loads((root / "online-factory.json").read_text())
report = {
    "format": "orthrus-dsv41-pypto-online-readiness-v1",
    "passed": bool(
        kernels.get("passed") and publication.get("passed")
        and all(item.get("passed") for item in reverse.values())
        and factory.get("production_ready")
    ),
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
    "resident_reverse": {
        name: {
            "passed": item.get("passed", False),
            "compiled": item.get("compiled", False),
            "executed": item.get("executed", item.get("npu_execution", False)),
        }
        for name, item in reverse.items()
    },
    "factory": factory,
}
(root / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

# The container runs as root while the report directory belongs to the host
# caller. Keep the evidence readable without a second privileged container.
chmod a+r "$output/tests.log" "$output/online-kernels.json" \
  "$output/resident-publication.json" "$output/online-factory.json" \
  "$output/hc-collapse-vjp.json" "$output/hc-pre-vjp.json" \
  "$output/c1a-vjp.json" "$output/expert-capture.json" \
  "$output/expert-input-vjp.json" "$output/ep-vjp.json" \
  "$output/readiness.json"
