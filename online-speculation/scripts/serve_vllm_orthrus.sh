#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?set MODEL to the Orthrus-Qwen3-8B directory}"

PORT="${PORT:-8000}"
MNS="${MNS:-128}"
MBT="${MBT:-65536}"
MEMORY="${MEMORY:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GRAPH_MODE="${GRAPH_MODE:-FULL_AND_PIECEWISE}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
ORTHRUS_TRACE_FILE="${ORTHRUS_TRACE_FILE:-}"
MAX_GRAPH_CAPTURE_SIZE="${MAX_GRAPH_CAPTURE_SIZE:-}"

SPECULATIVE_CONFIG='{"method":"custom_class","model":"blockspec.vllm_ascend.proposer.OrthrusProposer","num_speculative_tokens":31,"draft_sample_method":"probabilistic"}'
COMPILATION_CONFIG="{\"cudagraph_mode\":\"$GRAPH_MODE\"}"
if [[ -n "$MAX_GRAPH_CAPTURE_SIZE" ]]; then
  COMPILATION_CONFIG="{\"cudagraph_mode\":\"$GRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_GRAPH_CAPTURE_SIZE}"
fi

args=(
  serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name orthrus
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MNS"
  --max-num-batched-tokens "$MBT"
  --gpu-memory-utilization "$MEMORY"
  --enable-prefix-caching
  --enable-chunked-prefill
  --speculative-config "$SPECULATIVE_CONFIG"
  --compilation-config "$COMPILATION_CONFIG"
)

if [[ "$GRAPH_MODE" == "eager" ]]; then
  args+=(--enforce-eager)
fi
if [[ "$ASYNC_SCHEDULING" == "1" ]]; then
  args+=(--async-scheduling)
fi
if [[ -n "$TORCH_PROFILE_DIR" ]]; then
  mkdir -p "$TORCH_PROFILE_DIR"
  args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TORCH_PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":false}")
fi
if [[ -n "$ORTHRUS_TRACE_FILE" ]]; then
  mkdir -p "$(dirname "$ORTHRUS_TRACE_FILE")"
  export ORTHRUS_TRACE_FILE
fi

exec vllm "${args[@]}"
