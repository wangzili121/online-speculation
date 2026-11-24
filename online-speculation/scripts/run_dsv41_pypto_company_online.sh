#!/usr/bin/env bash
# Company two-node Orthrus-DSV4.1 online-update entrypoint.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

case "${1:-}" in
  node7) cluster_key=/root/.orthrus-cluster.key ;;
  node8) cluster_key=/home/wangzili/.orthrus-cluster.key ;;
  *) echo "usage: $0 <node7|node8> <capture|profile|smoke|tuning|benchmark|eval> [last-N]" >&2; exit 2 ;;
esac

export PYPTO_IMAGE="${PYPTO_IMAGE:-orthrus-dsv41-pypto:72a62d1-offline}"
export CLUSTER_KEY_FILE="$cluster_key"
export TARGET_WEIGHTS="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
export PYPTO_RESULTS_ROOT="${PYPTO_RESULTS_ROOT:-/workspace}"
export DSV41_GSM8K_ROOT="${DSV41_GSM8K_ROOT:-/workspace/data/orthrus-dsv41-gsm8k}"
export NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"

exec bash "$root/scripts/run_dsv41_pypto_online_gsm8k.sh" "$@"
