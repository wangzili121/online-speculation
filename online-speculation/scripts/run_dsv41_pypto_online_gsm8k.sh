#!/usr/bin/env bash
# Launch the paired DSV4.1 resident online-update study with one stable command.
set -euo pipefail

[[ $# -ge 1 ]] || { echo "usage: $0 <node7|node8> <capture|profile|smoke|tuning|benchmark|eval|report> [last-N]" >&2; exit 2; }
node="$1"
mode="${2:-smoke}"
last_layers="${3:-1}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
data_root="${DSV41_GSM8K_ROOT:-/workspace/data/orthrus-dsv41-gsm8k}"

case "$node" in node7|node8) ;; *) echo "node must be node7 or node8" >&2; exit 2 ;; esac
case "$last_layers" in 1|2|4|8) ;; *) echo "last-N must be one of 1, 2, 4 or 8" >&2; exit 2 ;; esac

case "$mode" in
  capture) split=tuning; requests=2; tokens=2 ;;
  profile) split=tuning; requests=5; tokens=8 ;;
  smoke) split=tuning; requests=4; tokens=32 ;;
  tuning) split=tuning; requests=32; tokens=192 ;;
  benchmark) split=online_eval; requests=33; tokens=96 ;;
  eval) split=online_eval; requests=384; tokens=192 ;;
  report)
    train_requests="${DSV41_REPORT_TRAIN:-8}"
    eval_requests="${DSV41_REPORT_EVAL:-4}"
    tokens="${DSV41_REPORT_TOKENS:-32}"
    batch_size="${DSV41_REPORT_BATCH:-4}"
    if [[ ! "$train_requests" =~ ^[1-9][0-9]*$ || ! "$eval_requests" =~ ^[1-9][0-9]*$ ||
          ! "$tokens" =~ ^[1-9][0-9]*$ || "$tokens" -lt 16 || ! "$batch_size" =~ ^[1-4]$ ]]; then
      echo "report requires positive train/eval counts, at least 16 tokens and batch size 1..4" >&2
      exit 2
    fi
    requests=$((train_requests + eval_requests))
    split=tuning
    ;;
  *) echo "mode must be capture, profile, smoke, tuning, benchmark, eval or report" >&2; exit 2 ;;
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

dispatch_option=--no-short-speculative-dispatch
if [[ "${DSV41_SHORT_SPECULATIVE_DISPATCH:-0}" == 1 ]]; then
  dispatch_option=--short-speculative-dispatch
fi

if [[ "$mode" == capture || "$mode" == profile ]]; then
  # Each host has a different set of historical result directories.  Reusing
  # the locally newest head can therefore make the cluster identities differ.
  # A capture gate starts from the deterministic target initialization instead.
  export PYPTO_REUSE_PREVIOUS_HEAD=0
  teacher_rows=1
  short_gate=(--orthrus-only --require-online-update)
  if [[ "$mode" == profile ]]; then
    teacher_rows=7
    short_gate+=(--online-profile)
  fi
  online_overrides="{\"last_layers\":$last_layers,\"teacher_rows\":$teacher_rows,\"warm_update_stride\":1,\"warm_update_steps\":1,\"steady_update_stride\":1,\"publish_every\":1}"
else
  online_overrides="{\"last_layers\":$last_layers}"
  short_gate=()
  if [[ "$mode" == benchmark ]]; then
    export PYPTO_REUSE_PREVIOUS_HEAD=0
    short_gate=(--online-benchmark --require-online-update)
  fi
fi

if [[ "$mode" == report ]]; then
  heldout="$data_root/quality_test.jsonl"
  if [[ ! -r "$heldout" ]] || (( $(wc -l <"$workload") < train_requests || $(wc -l <"$heldout") < eval_requests )); then
    echo "report needs $train_requests tuning rows and $eval_requests held-out GSM8K test rows" >&2
    exit 2
  fi
  report_workload="$(mktemp)"
  trap 'rm -f -- "$report_workload"' EXIT
  { head -n "$train_requests" "$workload"; head -n "$eval_requests" "$heldout"; } >"$report_workload"
  export ORTHRUS_WORKLOAD="$report_workload" PYPTO_REUSE_PREVIOUS_HEAD=0
  online_overrides="{\"last_layers\":$last_layers,\"teacher_rows\":7,\"warmup_updates\":2,\"warm_update_stride\":4,\"warm_update_steps\":4,\"steady_update_stride\":8,\"publish_every\":1}"
  bash "$root/scripts/dsv41_pypto_env.sh" cluster-accept "$node" \
    --max-requests "$requests" --tokens "$tokens" --no-trace-prefill --require-online-update \
    --report-train-requests "$train_requests" --report-batch-size "$batch_size" \
    --online-study /workspace/online-speculation/configs/dsv41/gsm8k-online.json \
    --online-primitives blockspec.pypto_dsv41_resident_primitives:create_primitives \
    --online-overrides "$online_overrides" "$dispatch_option"
  exit $?
fi

exec bash "$root/scripts/dsv41_pypto_env.sh" cluster-accept "$node" \
  --max-requests "$requests" --tokens "$tokens" --no-trace-prefill \
  --online-study /workspace/online-speculation/configs/dsv41/gsm8k-online.json \
  --online-primitives blockspec.pypto_dsv41_resident_primitives:create_primitives \
  --online-overrides "$online_overrides" "${short_gate[@]}" "$dispatch_option"
