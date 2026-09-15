#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
: "${MODEL:=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8}"
: "${HEAD:=/data/disk/models/orthrus-dsv4-head-v2-w8a8}"
: "${RUNTIME_LOCK:?set RUNTIME_LOCK to runtime-lock.local.json}"
: "${OUTPUT_DIR:=/data/disk/packages/dsv4-orthrus}"

mkdir -p "$OUTPUT_DIR"
manifest="$OUTPUT_DIR/dsv4-artifacts.json"
archive="$OUTPUT_DIR/online-speculation-dsv4-orthrus.tar.gz"

"$python_bin" "$repo_root/scripts/capture_dsv4_artifacts.py" \
  --model "$MODEL" \
  --head "$HEAD" \
  --runtime-lock "$RUNTIME_LOCK" \
  --output "$manifest"
test -s "$manifest"

tar -czf "$archive" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='runtime/vllm_ascend_v0_26/runtime-lock.local.json' \
  -C "$repo_root" \
  README.md pyproject.toml .gitignore benchmarks configs docs runtime scripts src tests \
  -C "$(dirname "$manifest")" "$(basename "$manifest")" \
  -C "$(dirname "$RUNTIME_LOCK")" "$(basename "$RUNTIME_LOCK")"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$archive" > "$archive.sha256"
else
  shasum -a 256 "$archive" > "$archive.sha256"
fi
echo "$archive"
