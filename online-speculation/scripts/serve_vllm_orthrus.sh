#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL:?set MODEL to the Orthrus-Qwen3-8B directory}"

PORT="${PORT:-8000}"
MNS="${MNS:-128}"
MBT="${MBT:-65536}"
MEMORY="${MEMORY:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GRAPH_MODE="${GRAPH_MODE:-FULL_AND_PIECEWISE}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
SERVICE_PROFILE_CONFIG="${SERVICE_PROFILE_CONFIG:-}"
PROFILE_SYMBOLS_PATH="${PROFILE_SYMBOLS_PATH:-}"
ORTHRUS_TRACE_FILE="${ORTHRUS_TRACE_FILE:-}"
ORTHRUS_DRAFT_ATTENTION="${ORTHRUS_DRAFT_ATTENTION:-native_gqa}"
ORTHRUS_DRAFT_SEGMENT_THRESHOLD="${ORTHRUS_DRAFT_SEGMENT_THRESHOLD:-24576}"
ORTHRUS_PILLAR_POLICY="${ORTHRUS_PILLAR_POLICY:-verifier}"
ORTHRUS_PILLAR_BUDGET_RATIO="${ORTHRUS_PILLAR_BUDGET_RATIO:-0.05}"
ORTHRUS_PILLAR_MIN_TOKENS="${ORTHRUS_PILLAR_MIN_TOKENS:-128}"
ORTHRUS_PILLAR_MAX_TOKENS="${ORTHRUS_PILLAR_MAX_TOKENS:-6144}"
ORTHRUS_PILLAR_SELECTOR_LAYER="${ORTHRUS_PILLAR_SELECTOR_LAYER:--1}"
ORTHRUS_PHASE_SCHEDULER="${ORTHRUS_PHASE_SCHEDULER:-off}"
ORTHRUS_DRAFT_MAX_BATCH="${ORTHRUS_DRAFT_MAX_BATCH:-8}"
ORTHRUS_DRAFT_LENGTH_BUCKET="${ORTHRUS_DRAFT_LENGTH_BUCKET:-8192}"
ORTHRUS_DRAFT_MAX_DELAY_STEPS="${ORTHRUS_DRAFT_MAX_DELAY_STEPS:-2}"
MAX_GRAPH_CAPTURE_SIZE="${MAX_GRAPH_CAPTURE_SIZE:-}"

case "$ORTHRUS_DRAFT_ATTENTION" in
  expanded|native_gqa|paged_segmented|paged_pillar) ;;
  *) echo "ORTHRUS_DRAFT_ATTENTION must be expanded, native_gqa, paged_segmented, or paged_pillar" >&2; exit 2 ;;
esac
if ! [[ "$ORTHRUS_DRAFT_SEGMENT_THRESHOLD" =~ ^[1-9][0-9]*$ ]]; then
  echo "ORTHRUS_DRAFT_SEGMENT_THRESHOLD must be a positive integer" >&2
  exit 2
fi
export ORTHRUS_DRAFT_ATTENTION ORTHRUS_DRAFT_SEGMENT_THRESHOLD
export ORTHRUS_PILLAR_POLICY ORTHRUS_PILLAR_BUDGET_RATIO
export ORTHRUS_PILLAR_MIN_TOKENS ORTHRUS_PILLAR_MAX_TOKENS
export ORTHRUS_PILLAR_SELECTOR_LAYER
export ORTHRUS_PHASE_SCHEDULER ORTHRUS_DRAFT_MAX_BATCH
export ORTHRUS_DRAFT_LENGTH_BUCKET ORTHRUS_DRAFT_MAX_DELAY_STEPS

if [[ -n "$TORCH_PROFILE_DIR" && -n "$SERVICE_PROFILE_CONFIG" ]]; then
  echo "torch and service profilers must run in separate passes" >&2
  exit 2
fi

SPECULATIVE_CONFIG='{"method":"custom_class","model":"blockspec.vllm_ascend.proposer.OrthrusProposer","num_speculative_tokens":31,"draft_sample_method":"probabilistic"}'
if [[ "$GRAPH_MODE" == "eager" ]]; then
  COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
else
  COMPILATION_CONFIG="{\"cudagraph_mode\":\"$GRAPH_MODE\"}"
fi
if [[ -n "$MAX_GRAPH_CAPTURE_SIZE" && "$GRAPH_MODE" != "eager" ]]; then
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
if [[ -n "$SERVICE_PROFILE_CONFIG" ]]; then
  test -f "$SERVICE_PROFILE_CONFIG"
  export SERVICE_PROF_CONFIG_PATH="$SERVICE_PROFILE_CONFIG"
fi
if [[ -n "$PROFILE_SYMBOLS_PATH" ]]; then
  test -f "$PROFILE_SYMBOLS_PATH"
  export PROFILING_SYMBOLS_PATH="$PROFILE_SYMBOLS_PATH"
fi
if [[ -n "$ORTHRUS_TRACE_FILE" ]]; then
  mkdir -p "$(dirname "$ORTHRUS_TRACE_FILE")"
  export ORTHRUS_TRACE_FILE
fi

exec vllm "${args[@]}"
