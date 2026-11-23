#!/usr/bin/env bash
# Launch the paired DSV4.1 resident online-update study with one stable command.
set -euo pipefail

node="${1:?usage: $0 {node7|node8} {smoke|tuning|eval} [last-N]}"
mode="${2:-smoke}"
last_layers="${3:-1}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
data_root="${DSV41_GSM8K_ROOT:-/workspace/data/orthrus-dsv41-gsm8k}"

case "$node" in node7|node8) ;; *) echo "node must be node7 or node8" >&2; exit 2 ;; esac
case "$last_layers" in 1|2|4|8) ;; *) echo "last-N must be one of 1, 2, 4 or 8" >&2; exit 2 ;; esac

case "$mode" in
  smoke) split=tuning; requests=4; tokens=32 ;;
  tuning) split=tuning; requests=32; tokens=192 ;;
  eval) split=online_eval; requests=384; tokens=192 ;;
  *) echo "mode must be smoke, tuning or eval" >&2; exit 2 ;;
esac

workload="$data_root/$split.jsonl"
study="$root/configs/dsv41/gsm8k-online.json"
factory="$root/src/blockspec/pypto_dsv41_resident_primitives.py"
for path in "$workload" "$study" "$factory"; do
  [[ -r "$path" ]] || { echo "required online-update input is missing: $path" >&2; exit 2; }
done

# Do not pay the multi-minute checkpoint/compile cost for a partial backend.
# Run the check in the pinned container so a clean serving host does not need
# the project's Python dependencies installed globally.
PYPTO_IMAGE="${PYPTO_IMAGE:-orthrus-dsv41-pypto:72a62d1-offline}" \
  bash "$root/scripts/dsv41_pypto_env.sh" online-factory-check

export ORTHRUS_WORKLOAD="$workload"
export CLUSTER_KEY_FILE="${CLUSTER_KEY_FILE:-/root/.orthrus-cluster.key}"
export NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"

exec bash "$root/scripts/dsv41_pypto_env.sh" cluster-accept "$node" \
  --max-requests "$requests" --tokens "$tokens" --no-trace-prefill \
  --online-study /workspace/online-speculation/configs/dsv41/gsm8k-online.json \
  --online-primitives blockspec.pypto_dsv41_resident_primitives:create_primitives \
  --online-overrides "{\"last_layers\":$last_layers}"
