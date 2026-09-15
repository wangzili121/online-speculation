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
DRAFT_SAMPLE_METHOD="${ORTHRUS_DRAFT_SAMPLE_METHOD:-greedy}"
DEFAULT_ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-$DEFAULT_ADDITIONAL_CONFIG}"

case "$DRAFT_SAMPLE_METHOD" in
  greedy|probabilistic) ;;
  *)
    echo "ORTHRUS_DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2
    exit 1
    ;;
esac

execution_args=()
case "${ORTHRUS_EXECUTION_MODE:-eager}" in
  eager)
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
    execution_args+=(--enforce-eager)
    ;;
  graph)
    if [[ "${ORTHRUS_GRAPH_VALIDATED:-0}" != "1" ]]; then
      echo "Graph mode is gated; set ORTHRUS_GRAPH_VALIDATED=1 only after the NPU correctness matrix passes" >&2
      exit 1
    fi
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ;;
  *)
    echo "ORTHRUS_EXECUTION_MODE must be eager or graph" >&2
    exit 1
    ;;
esac
COMPILATION_CONFIG="${COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"

profile_args=()
if [[ -n "${ORTHRUS_TORCH_PROFILE_DIR:-}" ]]; then
  profiler_config="$("$python_bin" - "$ORTHRUS_TORCH_PROFILE_DIR" <<'PY'
import json
import sys

print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": sys.argv[1],
    "torch_profiler_with_stack": False,
    "torch_profiler_with_memory": False,
    "ignore_frontend": False,
}))
PY
)"
  profile_args+=(--profiler-config "$profiler_config")
fi

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

SPECULATIVE_CONFIG="$("$python_bin" - "$HEAD" "$DRAFT_SAMPLE_METHOD" <<'PY'
import json
import sys

print(json.dumps({
    "method": "orthrus",
    "model": sys.argv[1],
    "num_speculative_tokens": 31,
    "parallel_drafting": True,
    "draft_sample_method": sys.argv[2],
    "rejection_sample_method": "standard",
    "quantization": "ascend",
    "enforce_eager": True,
}))
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
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  "${execution_args[@]}" \
  "${profile_args[@]}" \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --compilation-config "$COMPILATION_CONFIG" \
  --additional-config "$ADDITIONAL_CONFIG"
