#!/usr/bin/env bash
set -euo pipefail

: "${TRIAL_ID:?set a unique trial id}"
: "${MODEL:?set MODEL to the Orthrus-Qwen3-8B directory}"
: "${MANIFEST:?set MANIFEST to a frozen workload}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/tuning}"
REQUESTS="${REQUESTS:-64}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-$REQUESTS}"
ARRIVAL_RATE="${ARRIVAL_RATE:-0}"
PORT="${PORT:-8000}"
MNS="${MNS:-64}"
MBT="${MBT:-65536}"
MEMORY="${MEMORY:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GRAPH_MODE="${GRAPH_MODE:-FULL_AND_PIECEWISE}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
MAX_GRAPH_CAPTURE_SIZE="${MAX_GRAPH_CAPTURE_SIZE:-}"
SOURCE_REVISION="${SOURCE_REVISION:-unknown}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
WARMUP_MANIFEST="${WARMUP_MANIFEST:-$MANIFEST}"

output_dir="$ARTIFACT_ROOT/$TRIAL_ID"
result="$output_dir/result.json"
mkdir -p "$output_dir"
if [[ -e "$result" ]]; then
  echo "completed trial exists: $result"
  exit 0
fi
if [[ -e "$output_dir/RUNNING" ]]; then
  echo "incomplete trial marker exists: $output_dir/RUNNING" >&2
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
  --max-model-len "$MAX_MODEL_LEN" --output "$output_dir/preflight.json"

runtime_json=$(printf \
  '{"vllm":"0.23.0","vllm_ascend":"0.23.0","device":"Ascend910B3"}')
trial_json=$(printf \
  '{"trial_id":"%s","source_revision":"%s","requests":%s,"client_concurrency":%s,"arrival_rate":%s,"max_num_seqs":%s,"max_num_batched_tokens":%s,"gpu_memory_utilization":%s,"max_model_len":%s,"graph_mode":"%s","async_scheduling":%s,"max_graph_capture_size":"%s"}' \
  "$TRIAL_ID" "$SOURCE_REVISION" "$REQUESTS" "$CLIENT_CONCURRENCY" "$ARRIVAL_RATE" \
  "$MNS" "$MBT" "$MEMORY" "$MAX_MODEL_LEN" "$GRAPH_MODE" \
  "$ASYNC_SCHEDULING" "$MAX_GRAPH_CAPTURE_SIZE")
python scripts/capture_trial_manifest.py \
  --repo . --workload "$MANIFEST" --model "$MODEL" \
  --runtime-json "$runtime_json" --trial-json "$trial_json" \
  --output "$output_dir/manifest.json"

export PORT MNS MBT MEMORY MAX_MODEL_LEN GRAPH_MODE ASYNC_SCHEDULING
export MAX_GRAPH_CAPTURE_SIZE
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
curl -fsS "http://127.0.0.1:$PORT/metrics" >"$output_dir/metrics.before.prom"
python scripts/benchmark_vllm_orthrus.py \
  --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
  --metrics-url "http://127.0.0.1:$PORT/metrics" \
  --manifest "$MANIFEST" --requests "$REQUESTS" \
  --client-concurrency "$CLIENT_CONCURRENCY" --arrival-rate "$ARRIVAL_RATE" \
  --timeout "$REQUEST_TIMEOUT" --output "$result" \
  | tee "$output_dir/summary.json"
curl -fsS "http://127.0.0.1:$PORT/metrics" >"$output_dir/metrics.after.prom"
touch "$output_dir/COMPLETE"
