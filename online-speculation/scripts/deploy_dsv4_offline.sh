#!/usr/bin/env bash
# Host-side, offline deployment of one DSV4 service on an exclusive A2 node.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_HOST="${REPO_HOST:-$repo_root}"
MODELS_HOST="${MODELS_HOST:-/data/disk/models}"
MODEL_HOST="${MODEL_HOST:-$MODELS_HOST/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8}"
HEAD_HOST="${HEAD_HOST:-$MODELS_HOST/orthrus-dsv4-head-v2-w8a8}"
MODE="${MODE:-orthrus}"
PROFILE="${PROFILE:-prod-128k}"
PORT="${PORT:-8901}"
TP=8
CONTAINER="${CONTAINER:-dsv4-orthrus-v026}"
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
IMAGE_PULL=never
IMAGE_REF="${IMAGE_REF:-quay.io/ascend/vllm-ascend:nightly-releases-v0.26.0rc}"
EXPECTED_IMAGE_DIGEST="${EXPECTED_IMAGE_DIGEST:-sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a}"
REQUIRE_TRAINED_HEAD="${REQUIRE_TRAINED_HEAD:-1}"
ORTHRUS_DRAFT_SAMPLE_METHOD="${ORTHRUS_DRAFT_SAMPLE_METHOD:-probabilistic}"
HEALTH_WAIT_SECONDS="${HEALTH_WAIT_SECONDS:-2400}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$MODE" in
  orthrus|target) ;;
  *) echo "MODE must be orthrus or target" >&2; exit 2 ;;
esac
if [[ "$IMAGE_REF" != "quay.io/ascend/vllm-ascend:nightly-releases-v0.26.0rc" ]]; then
  echo "IMAGE_REF must be the pinned v0.26.0rc container tag" >&2
  exit 2
fi
case "$PROFILE" in
  prod-128k|prod-1m) ;;
  *) echo "PROFILE must be prod-128k or prod-1m" >&2; exit 2 ;;
esac
case "$REQUIRE_TRAINED_HEAD" in
  0|1) ;;
  *) echo "REQUIRE_TRAINED_HEAD must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$MODE" == orthrus ]]; then
  case "$ORTHRUS_DRAFT_SAMPLE_METHOD" in
    greedy|probabilistic) ;;
    *) echo "ORTHRUS_DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2; exit 2 ;;
  esac
  if [[ "$REQUIRE_TRAINED_HEAD" == 1 ]]; then
    if [[ "$ORTHRUS_DRAFT_SAMPLE_METHOD" != probabilistic ]]; then
      echo "trained Chang-Bo deployment requires probabilistic draft proposals" >&2
      exit 2
    fi
    if [[ "${ORTHRUS_PROBABILISTIC_VALIDATED:-0}" != 1 ]]; then
      echo "set ORTHRUS_PROBABILISTIC_VALIDATED=1 only after multi-request RNG/correctness validation" >&2
      exit 2
    fi
  fi
fi
if [[ "${ACK_EXCLUSIVE_NODE:-0}" != 1 ]]; then
  echo "set ACK_EXCLUSIVE_NODE=1 only after reserving all eight A2 cards" >&2
  exit 2
fi
if [[ "$NPU_DEVICES" != "0,1,2,3,4,5,6,7" ]]; then
  echo "full DSV4-Flash deployment requires NPU_DEVICES=0,1,2,3,4,5,6,7" >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "PORT must be an integer from 1 to 65535" >&2
  exit 2
fi
for command_name in docker readlink "$PYTHON_BIN"; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing command: $command_name" >&2
    exit 2
  fi
done
if [[ ! -d "$REPO_HOST" || ! -d "$MODELS_HOST" ]]; then
  echo "REPO_HOST and MODELS_HOST must be existing directories" >&2
  exit 2
fi
if [[ ! -f "$MODEL_HOST/config.json" ]]; then
  echo "model config is missing: $MODEL_HOST/config.json" >&2
  exit 2
fi
if [[ "$MODE" == orthrus && ! -f "$HEAD_HOST/orthrus_head_manifest.json" ]]; then
  echo "Orthrus v2 head is missing: $HEAD_HOST/orthrus_head_manifest.json" >&2
  exit 2
fi

models_root="$(readlink -f "$MODELS_HOST")"
model_root="$(readlink -f "$MODEL_HOST")"
case "$model_root" in
  "$models_root"/*) MODEL="/data/disk/models/${model_root#"$models_root"/}" ;;
  *) echo "MODEL_HOST must be under MODELS_HOST" >&2; exit 2 ;;
esac
if [[ "$MODE" == orthrus ]]; then
  head_root="$(readlink -f "$HEAD_HOST")"
  case "$head_root" in
    "$models_root"/*) HEAD="/data/disk/models/${head_root#"$models_root"/}" ;;
    *) echo "HEAD_HOST must be under MODELS_HOST" >&2; exit 2 ;;
  esac
  preflight=("$PYTHON_BIN" "$REPO_HOST/scripts/preflight_dsv4_orthrus.py"
    --model "$model_root" --head "$head_root" --tensor-parallel-size "$TP" --production)
  if [[ "$REQUIRE_TRAINED_HEAD" == 1 ]]; then
    preflight+=(--require-trained)
  fi
  "${preflight[@]}"
else
  "$PYTHON_BIN" - "$model_root/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
ratios = config.get("compress_ratios", [])
if (config.get("model_type") != "deepseek_v4"
        or "DeepseekV4ForCausalLM" not in config.get("architectures", [])
        or config.get("num_hidden_layers") != 43
        or len(ratios) < 43
        or 4 not in ratios[:43] or 128 not in ratios[:43]):
    raise SystemExit("target-only preflight: not the 43-layer hybrid DSV4-Flash model")
PY
fi

if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
  echo "pinned v0.26.0rc image is not preloaded; offline deployment will not pull" >&2
  exit 2
fi
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container already exists: $CONTAINER; inspect it before choosing another name" >&2
  exit 2
fi
"$PYTHON_BIN" - "$PORT" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("0.0.0.0", int(sys.argv[1])))
    except OSError as error:
        raise SystemExit(f"PORT is unavailable: {error}") from error
PY

logs="$REPO_HOST/local/dsv4-deploy/$CONTAINER"
mkdir -p "$logs"
CONTAINER="$CONTAINER" REPO_HOST="$REPO_HOST" MODELS_HOST="$MODELS_HOST" \
  NPU_DEVICES="$NPU_DEVICES" IMAGE_PULL="$IMAGE_PULL" IMAGE_REF="$IMAGE_REF" \
  EXPECTED_IMAGE_DIGEST="$EXPECTED_IMAGE_DIGEST" \
  PYTHON_BIN="$PYTHON_BIN" "$REPO_HOST/scripts/create_dsv4_container.sh"

launcher="scripts/serve_dsv4_${MODE}.sh"
model_name="dsv4-${MODE}"
docker exec -d -w /workspace/online-speculation \
  -e MODEL="$MODEL" -e HEAD="${HEAD:-}" -e PORT="$PORT" \
  -e TP="$TP" -e PROFILE="$PROFILE" \
  -e REQUIRE_TRAINED_HEAD="$REQUIRE_TRAINED_HEAD" \
  -e ORTHRUS_DRAFT_SAMPLE_METHOD="$ORTHRUS_DRAFT_SAMPLE_METHOD" \
  "$CONTAINER" bash -lc "exec bash $launcher > /workspace/online-speculation/local/dsv4-deploy/$CONTAINER/service.log 2>&1"

if ! "$PYTHON_BIN" "$REPO_HOST/scripts/healthcheck_dsv4.py" \
  --base-url "http://127.0.0.1:$PORT" --model "$model_name" \
  --wait-seconds "$HEALTH_WAIT_SECONDS"; then
  echo "service smoke failed; inspect $logs/service.log and container $CONTAINER" >&2
  echo "target-only rollback: stop this owned service, then run MODE=target with a new CONTAINER" >&2
  exit 1
fi
echo "service ready: http://127.0.0.1:$PORT/v1 (model $model_name)"
echo "runtime lock: $REPO_HOST/runtime/vllm_ascend_v0_26/runtime-lock.local.json"
echo "service log: $logs/service.log"
