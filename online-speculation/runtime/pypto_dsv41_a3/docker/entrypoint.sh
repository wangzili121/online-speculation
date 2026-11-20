#!/usr/bin/env bash
set -eo pipefail
source /opt/pypto/bin/pypto-activate.sh
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
export TARGET_WEIGHTS="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
root=/opt/orthrus/online-speculation
case "${1:-doctor}" in
  doctor)
    shift || true
    exec python "$root/scripts/check_dsv41_pypto_environment.py" "$@"
    ;;
  preflight)
    shift
    exec python "$root/scripts/preflight_dsv41_pypto.py" "$@"
    ;;
  probe)
    shift
    exec python "$root/scripts/check_dsv41_pypto_environment.py" --device-probe "$@"
    ;;
  *) exec "$@" ;;
esac
