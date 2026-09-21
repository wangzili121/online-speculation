#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
PROFILE="${2:-smoke}"
if [[ "${MODE}" != "offline" && "${MODE}" != "continue" && "${MODE}" != "resume" ]]; then
  echo "usage: $0 {offline|continue|resume} [smoke|paper]" >&2
  exit 2
fi
if [[ "${PROFILE}" != "smoke" && "${PROFILE}" != "paper" ]]; then
  echo "profile must be smoke or paper" >&2
  exit 2
fi

: "${BF16_BASE:?set BF16_BASE to a dequantized BF16 DSV4 checkpoint}"
: "${TRAIN_DATA:?set TRAIN_DATA to train.tokens.jsonl}"
: "${MASK_TOKEN_ID:?set MASK_TOKEN_ID}"
: "${OUTPUT_CHECKPOINT:?set OUTPUT_CHECKPOINT to a new path}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
LAYERS_PER_NPU="${LAYERS_PER_NPU:-1}"

COMMON=(
  --bf16-base "${BF16_BASE}"
  --data "${TRAIN_DATA}"
  --mask-token-id "${MASK_TOKEN_ID}"
  --sequence-length "${SEQUENCE_LENGTH:-2048}"
  --block-size 32
  --npu-devices "${NPU_DEVICES}"
  --layers-per-npu "${LAYERS_PER_NPU}"
  --cpu-offload
  --output-checkpoint "${OUTPUT_CHECKPOINT}"
)

if [[ -n "${VALIDATION_DATA:-}" ]]; then
  COMMON+=(--validation-data "${VALIDATION_DATA}" --validation-batches "${VALIDATION_BATCHES:-4}")
fi
if [[ -n "${SERVING_TARGET:-}" && -z "${EXPORT_HEAD:-}" ]] || \
   [[ -z "${SERVING_TARGET:-}" && -n "${EXPORT_HEAD:-}" ]]; then
  echo "SERVING_TARGET and EXPORT_HEAD must be set together" >&2
  exit 2
fi
if [[ -n "${SERVING_TARGET:-}" && -n "${EXPORT_HEAD:-}" ]]; then
  COMMON+=(--serving-target "${SERVING_TARGET}" --export-head "${EXPORT_HEAD}")
fi
if [[ -n "${INITIAL_HEAD:-}" ]]; then
  COMMON+=(--initial-head "${INITIAL_HEAD}")
fi
if [[ -n "${INPUT_CHECKPOINT:-}" ]]; then
  COMMON+=(--checkpoint "${INPUT_CHECKPOINT}")
fi

if [[ "${PROFILE}" == "paper" ]]; then
  # One model-parallel replica needs accumulation 128 for global batch 128.
  SCHEDULE=(
    --epochs "${EPOCHS:-2}"
    --batch-size 1
    --accumulate "${ACCUMULATE:-128}"
    --anchors-per-sequence "${ANCHORS_PER_SEQUENCE:-256}"
    --learning-rate 2e-4
    --warmup-ratio 0.05
    --minimum-lr-ratio 0
    --clip-grad 1
  )
else
  SCHEDULE=(
    --steps "${STEPS:-1}"
    --batch-size 1
    --accumulate "${ACCUMULATE:-1}"
    --anchors-per-sequence "${ANCHORS_PER_SEQUENCE:-2}"
  )
fi

case "${MODE}" in
  offline)
    TRAIN_MODE=train
    ;;
  continue)
    if [[ -z "${INPUT_CHECKPOINT:-}" && -z "${INITIAL_HEAD:-}" ]]; then
      echo "continue requires INPUT_CHECKPOINT or INITIAL_HEAD" >&2
      exit 2
    fi
    TRAIN_MODE=continue
    ;;
  resume)
    if [[ -z "${INPUT_CHECKPOINT:-}" ]]; then
      echo "resume requires INPUT_CHECKPOINT" >&2
      exit 2
    fi
    TRAIN_MODE=resume
    SCHEDULE=()
    ;;
esac

exec python3 "${SCRIPT_DIR}/train_dsv4_orthrus_head.py" \
  "${TRAIN_MODE}" "${COMMON[@]}" "${SCHEDULE[@]}"
