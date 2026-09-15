#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
: "${MODEL:?set MODEL to the W8A8 DSV4 target directory}"
: "${BF16_HEAD:?set BF16_HEAD to the trained BF16 v2 head}"
: "${OUTPUT_HEAD:?set OUTPUT_HEAD to a new output directory}"
: "${CALIBRATION_DATA:?set CALIBRATION_DATA to representative JSONL data}"
: "${ORTHRUS_MODELSLIM_ADAPTER:?set ORTHRUS_MODELSLIM_ADAPTER to the head-only ModelSlim adapter executable}"

if [[ -e "$OUTPUT_HEAD" ]]; then
  echo "OUTPUT_HEAD already exists: $OUTPUT_HEAD" >&2
  exit 1
fi
if [[ ! -x "$ORTHRUS_MODELSLIM_ADAPTER" ]]; then
  echo "ModelSlim adapter is not executable: $ORTHRUS_MODELSLIM_ADAPTER" >&2
  exit 1
fi

calibration_manifest="${OUTPUT_HEAD}.calibration.json"
"$python_bin" "$repo_root/scripts/preflight_dsv4_orthrus.py" \
  --model "$MODEL" \
  --head "$BF16_HEAD" \
  --tensor-parallel-size 8 \
  --production \
  --require-trained \
  --allow-bf16

"$ORTHRUS_MODELSLIM_ADAPTER" \
  --base-model "$MODEL" \
  --head "$BF16_HEAD" \
  --calibration-data "$CALIBRATION_DATA" \
  --output "$OUTPUT_HEAD" \
  --calibration-manifest "$calibration_manifest"

"$python_bin" "$repo_root/scripts/finalize_dsv4_orthrus_w8a8.py" \
  --model "$MODEL" \
  --bf16-head "$BF16_HEAD" \
  --quantized-head "$OUTPUT_HEAD" \
  --calibration-manifest "$calibration_manifest" \
  --calibration-data "$CALIBRATION_DATA"

"$python_bin" "$repo_root/scripts/preflight_dsv4_orthrus.py" \
  --model "$MODEL" \
  --head "$OUTPUT_HEAD" \
  --tensor-parallel-size 8 \
  --production \
  --require-trained
