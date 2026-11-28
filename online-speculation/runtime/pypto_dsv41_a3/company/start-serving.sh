#!/usr/bin/env bash
set -eo pipefail
source /opt/pypto/bin/pypto-activate.sh
set -u
: "${CHECKPOINT:?Set CHECKPOINT to the mounted W8A8 checkpoint directory}"
: "${HEAD_DIR:?Set HEAD_DIR to the mounted K16 BF16 head directory}"
: "${CACHE_DIR:?Set CACHE_DIR to a writable cache directory}"
DEVICES="${TASK_DEVICE:-${DEVICES:-}}"
: "${DEVICES:?Set DEVICES to the 16 allocated device IDs if TASK_DEVICE is unset}"
test -r "$CHECKPOINT/config.json"
test -r "$HEAD_DIR/manifest.json"
mkdir -p "$CACHE_DIR"
export PYPTO_LIB_ROOT=/opt/orthrus-a3/pypto-lib PYTHONUNBUFFERED=1
cd /opt/orthrus-a3/DeepSeek-V4.1-Flash-A3/pypto-serving-deepseek-v4.1-flash-a3
exec /opt/pypto/venv/bin/python -m serving.api \
  --host 0.0.0.0 --port 8000 \
  --model-root "$CHECKPOINT" --devices "$DEVICES" --cache-dir "$CACHE_DIR" \
  --max-model-len 512 --chunk-size 128 --prefill-moe-slab-rows 64 \
  --orthrus-head "$HEAD_DIR" --draft-block 16 --verify-width 8 \
  --online-last-layers 1 --max-num-seqs 1 "$@"
