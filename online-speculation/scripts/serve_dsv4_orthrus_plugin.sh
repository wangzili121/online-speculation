#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
source "$repo_root/scripts/dsv4_service_env.sh"

: "${MODEL:=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8}"
: "${HEAD:=/data/disk/models/orthrus-dsv4-head-v2-w8a8}"
PORT="${PORT:-8901}"
TP="${TP:-8}"
MEMORY="${MEMORY:-0.90}"
MODEL_LOAD_THREADS="${MODEL_LOAD_THREADS:-128}"
DRAFT_SAMPLE_METHOD="${ORTHRUS_DRAFT_SAMPLE_METHOD:-probabilistic}"
BASE_ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-{\"enable_cpu_binding\":true}}"

if [[ ! "$MODEL_LOAD_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_LOAD_THREADS must be a positive integer" >&2
  exit 1
fi
MODEL_LOADER_CONFIG="{\"enable_multithread_load\":true,\"num_threads\":$MODEL_LOAD_THREADS}"

case "$DRAFT_SAMPLE_METHOD" in
  greedy|probabilistic) ;;
  *)
    echo "ORTHRUS_DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2
    exit 1
    ;;
esac

configure_dsv4_profile
configure_dsv4_runtime
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

preflight=(
  "$python_bin" "$repo_root/scripts/preflight_dsv4_orthrus.py"
  --model "$MODEL"
  --head "$HEAD"
  --tensor-parallel-size "$TP"
  --production
)
if [[ "${REQUIRE_TRAINED_HEAD:-0}" == "1" ]]; then
  preflight+=(--require-trained)
fi
"${preflight[@]}"

SPECULATIVE_CONFIG="$($python_bin - "$DRAFT_SAMPLE_METHOD" <<'PY'
import json
import sys

print(json.dumps({
    "method": "custom_class",
    "model": "blockspec.vllm_ascend.dsv4_custom_proposer.OrthrusDSV4CustomProposer",
    "num_speculative_tokens": 31,
    "parallel_drafting": True,
    "draft_sample_method": sys.argv[1],
    "rejection_sample_method": "standard",
    "enforce_eager": True,
}))
PY
)"

MERGED_ADDITIONAL_CONFIG="$($python_bin - "$BASE_ADDITIONAL_CONFIG" "$HEAD" <<'PY'
import json
import sys

config = json.loads(sys.argv[1])
custom = config.setdefault("custom_proposer_config", {})
orthrus = custom.setdefault("orthrus", {})
configured = orthrus.setdefault("head_model", sys.argv[2])
if configured != sys.argv[2]:
    raise SystemExit(
        "custom_proposer_config.orthrus.head_model conflicts with HEAD"
    )
print(json.dumps(config, separators=(",", ":")))
PY
)"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name dsv4-orthrus \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$MEMORY" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --no-enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --quantization ascend \
  --block-size "$BLOCK_SIZE" \
  --model-loader-extra-config "$MODEL_LOADER_CONFIG" \
  --enforce-eager \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --additional-config "$MERGED_ADDITIONAL_CONFIG"
