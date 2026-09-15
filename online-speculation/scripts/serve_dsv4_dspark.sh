#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/dsv4_service_env.sh"

: "${MODEL:=/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8}"
PORT="${PORT:-8902}"
TP="${TP:-8}"
MEMORY="${MEMORY:-0.90}"
DSPARK_TOKENS="${DSPARK_TOKENS:-7}"
DEFAULT_ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-$DEFAULT_ADDITIONAL_CONFIG}"

execution_args=()
case "${DSPARK_EXECUTION_MODE:-eager}" in
  eager)
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
    execution_args+=(--enforce-eager)
    ;;
  graph)
    DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ;;
  *)
    echo "DSPARK_EXECUTION_MODE must be eager or graph" >&2
    exit 1
    ;;
esac
COMPILATION_CONFIG="${COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"

configure_dsv4_profile
configure_dsv4_runtime

speculative_config="$(printf \
  '{"method":"dspark","num_speculative_tokens":%d,"enforce_eager":true}' \
  "$DSPARK_TOKENS")"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name dsv4-dspark \
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
  --speculative-config "$speculative_config" \
  --compilation-config "$COMPILATION_CONFIG" \
  --additional-config "$ADDITIONAL_CONFIG"
