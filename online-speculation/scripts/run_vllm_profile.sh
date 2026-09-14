#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

: "${PROFILE_ID:?set a unique profile id}"
: "${PROFILE_KIND:?set PROFILE_KIND to torch or service}"
: "${MODEL:?set MODEL to the Orthrus-Qwen3-8B directory}"
: "${MANIFEST:?set MANIFEST to a frozen workload}"

case "$PROFILE_KIND" in
  torch|service) ;;
  *) echo "PROFILE_KIND must be torch or service" >&2; exit 2 ;;
esac

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/profiling}"
REQUESTS="${REQUESTS:-64}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-$REQUESTS}"
ARRIVAL_RATE="${ARRIVAL_RATE:-0}"
PORT="${PORT:-8000}"
MNS="${MNS:-64}"
MBT="${MBT:-65536}"
MEMORY="${MEMORY:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GRAPH_MODE="${GRAPH_MODE:-FULL_AND_PIECEWISE}"
MAX_GRAPH_CAPTURE_SIZE="${MAX_GRAPH_CAPTURE_SIZE:-1024}"
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
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
SOURCE_REVISION="${SOURCE_REVISION:-unknown}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
WARMUP_MANIFEST="${WARMUP_MANIFEST:-$MANIFEST}"
RAMP_UP_SECONDS="${RAMP_UP_SECONDS:-20}"
PROFILE_SECONDS="${PROFILE_SECONDS:-10}"
PROFILE_WAIT_TIMEOUT="${PROFILE_WAIT_TIMEOUT:-1200}"
MIN_SPEC_DRAFTS="${MIN_SPEC_DRAFTS:-32}"
PROFILE_SYMBOLS_PATH="${PROFILE_SYMBOLS_PATH:-}"
SERVICE_CAPTURE_MODE="${SERVICE_CAPTURE_MODE:-window}"

output_dir="$ARTIFACT_ROOT/$PROFILE_ID"
mkdir -p "$output_dir"
if [[ -e "$output_dir/COMPLETE" ]]; then
  echo "completed profile exists: $output_dir"
  exit 0
fi
if [[ -e "$output_dir/RUNNING" ]]; then
  echo "incomplete profile marker exists: $output_dir/RUNNING" >&2
  exit 2
fi
touch "$output_dir/RUNNING"

server_pid=""
cleanup() {
  status=$?
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -f "$output_dir/RUNNING"
  if [[ $status -ne 0 ]]; then
    touch "$output_dir/FAILED"
  fi
}
trap cleanup EXIT INT TERM

python scripts/preflight_vllm_orthrus.py \
  --model "$MODEL" --workload "$MANIFEST" \
  --max-model-len "$MAX_MODEL_LEN" --profiler "$PROFILE_KIND" \
  --output "$output_dir/preflight.json"

runtime_json=$(printf \
  '{"vllm":"0.23.0","vllm_ascend":"0.23.0","device":"Ascend910B3","profiler":"%s"}' \
  "$PROFILE_KIND")
profile_json=$(printf \
  '{"profile_id":"%s","source_revision":"%s","profiler":"%s","requests":%s,"client_concurrency":%s,"arrival_rate":%s,"max_num_seqs":%s,"max_num_batched_tokens":%s,"gpu_memory_utilization":%s,"max_model_len":%s,"graph_mode":"%s","async_scheduling":%s,"max_graph_capture_size":"%s","draft_attention":"%s","draft_segment_threshold":%s,"pillar_policy":"%s","pillar_budget_ratio":%s,"pillar_min_tokens":%s,"pillar_max_tokens":%s,"pillar_selector_layer":%s,"phase_scheduler":"%s","draft_max_batch":%s,"draft_length_bucket":%s,"draft_max_delay_steps":%s,"ramp_up_seconds":%s,"profile_seconds":%s}' \
  "$PROFILE_ID" "$SOURCE_REVISION" "$PROFILE_KIND" "$REQUESTS" \
  "$CLIENT_CONCURRENCY" "$ARRIVAL_RATE" "$MNS" "$MBT" "$MEMORY" \
  "$MAX_MODEL_LEN" "$GRAPH_MODE" "$ASYNC_SCHEDULING" \
  "$MAX_GRAPH_CAPTURE_SIZE" "$ORTHRUS_DRAFT_ATTENTION" \
  "$ORTHRUS_DRAFT_SEGMENT_THRESHOLD" "$ORTHRUS_PILLAR_POLICY" \
  "$ORTHRUS_PILLAR_BUDGET_RATIO" "$ORTHRUS_PILLAR_MIN_TOKENS" \
  "$ORTHRUS_PILLAR_MAX_TOKENS" "$ORTHRUS_PILLAR_SELECTOR_LAYER" \
  "$ORTHRUS_PHASE_SCHEDULER" \
  "$ORTHRUS_DRAFT_MAX_BATCH" "$ORTHRUS_DRAFT_LENGTH_BUCKET" \
  "$ORTHRUS_DRAFT_MAX_DELAY_STEPS" "$RAMP_UP_SECONDS" "$PROFILE_SECONDS")
python scripts/capture_trial_manifest.py \
  --repo . --workload "$MANIFEST" --model "$MODEL" \
  --runtime-json "$runtime_json" --trial-json "$profile_json" \
  --output "$output_dir/manifest.json"

export PORT MNS MBT MEMORY MAX_MODEL_LEN GRAPH_MODE ASYNC_SCHEDULING
export MAX_GRAPH_CAPTURE_SIZE PROFILE_SYMBOLS_PATH
export ORTHRUS_DRAFT_ATTENTION ORTHRUS_DRAFT_SEGMENT_THRESHOLD
export ORTHRUS_PILLAR_POLICY ORTHRUS_PILLAR_BUDGET_RATIO
export ORTHRUS_PILLAR_MIN_TOKENS ORTHRUS_PILLAR_MAX_TOKENS
export ORTHRUS_PILLAR_SELECTOR_LAYER
export ORTHRUS_PHASE_SCHEDULER ORTHRUS_DRAFT_MAX_BATCH
export ORTHRUS_DRAFT_LENGTH_BUCKET ORTHRUS_DRAFT_MAX_DELAY_STEPS
export ORTHRUS_TRACE_FILE="$output_dir/orthrus-events-{pid}.jsonl"
export PROMETHEUS_MULTIPROC_DIR="$output_dir/prometheus"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
if [[ "$PROFILE_KIND" == "torch" ]]; then
  export TORCH_PROFILE_DIR="$output_dir/torch-raw"
  unset SERVICE_PROFILE_CONFIG
else
  if [[ -z "$PROFILE_SYMBOLS_PATH" ]]; then
    PROFILE_SYMBOLS_PATH="$repo_root/runtime/vllm_ascend_v0_23/service_profiling_symbols.yaml"
    export PROFILE_SYMBOLS_PATH
  fi
  mkdir -p "$output_dir/service-raw"
  SERVICE_PROFILE_CONFIG="$output_dir/service-config.json"
  service_create_args=()
  if [[ "$SERVICE_CAPTURE_MODE" == "startup" ]]; then
    service_create_args+=(--enabled)
  fi
  python scripts/service_profile_vllm_orthrus.py create \
    --config "$SERVICE_PROFILE_CONFIG" \
    --profile-directory "$output_dir/service-raw" \
    --seconds "$PROFILE_SECONDS" \
    "${service_create_args[@]}" >"$output_dir/service-create.json"
  export SERVICE_PROFILE_CONFIG
  unset TORCH_PROFILE_DIR
fi

bash scripts/serve_vllm_orthrus.sh >"$output_dir/server.log" 2>&1 &
server_pid=$!
deadline=$((SECONDS + SERVER_TIMEOUT))
until curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited before readiness; see $output_dir/server.log" >&2
    exit 3
  fi
  if (( SECONDS >= deadline )); then
    echo "server readiness timed out after ${SERVER_TIMEOUT}s" >&2
    exit 4
  fi
  sleep 2
done

if (( WARMUP_REQUESTS > 0 )); then
  python scripts/benchmark_vllm_orthrus.py \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --metrics-url "http://127.0.0.1:$PORT/metrics" \
    --manifest "$WARMUP_MANIFEST" --requests "$WARMUP_REQUESTS" \
    --client-concurrency "$WARMUP_REQUESTS" --timeout "$REQUEST_TIMEOUT" \
    --output "$output_dir/warmup.json" >"$output_dir/warmup-summary.json"
fi

profile_args=()
if [[ "$PROFILE_KIND" == "service" && "$SERVICE_CAPTURE_MODE" == "startup" ]]; then
  profile_args+=(--service-startup-capture)
fi
python scripts/profile_vllm_orthrus.py \
  --profiler "$PROFILE_KIND" \
  --endpoint "http://127.0.0.1:$PORT" \
  --manifest "$MANIFEST" --requests "$REQUESTS" \
  --client-concurrency "$CLIENT_CONCURRENCY" \
  --arrival-rate "$ARRIVAL_RATE" --timeout "$REQUEST_TIMEOUT" \
  --ramp-up-seconds "$RAMP_UP_SECONDS" \
  --profile-seconds "$PROFILE_SECONDS" \
  --profile-wait-timeout "$PROFILE_WAIT_TIMEOUT" \
  --min-spec-drafts "$MIN_SPEC_DRAFTS" \
  --service-config "${SERVICE_PROFILE_CONFIG:-}" \
  --service-log "$output_dir/server.log" \
  "${profile_args[@]}" \
  --output "$output_dir/load.json" >"$output_dir/load-summary.json"

kill -TERM "$server_pid"
wait "$server_pid" || true
server_pid=""
python scripts/merge_orthrus_trace.py \
  --input "$output_dir/orthrus-events-*.jsonl" \
  --output "$output_dir/orthrus-trace.json"

if [[ "$PROFILE_KIND" == "service" ]]; then
  python scripts/service_profile_vllm_orthrus.py analyze \
    --input "$output_dir/service-raw" \
    --output "$output_dir/service-analysis" \
    >"$output_dir/service-analysis.json"
  python scripts/analyze_orthrus_service_profile.py \
    --profile "$output_dir" \
    --output "$output_dir/service-derived-summary.json" \
    >"$output_dir/service-derived-summary.stdout"
else
  python scripts/analyze_orthrus_profile.py \
    --profile "$output_dir" \
    --output "$output_dir/derived-summary.json" \
    >"$output_dir/derived-summary.stdout"
fi
touch "$output_dir/COMPLETE"
