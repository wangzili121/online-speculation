#!/usr/bin/env bash
# Host-side launcher for one node of the qualified A2 DP2/TP8 topology.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${NODE0_IP:?set NODE0_IP to the API node address}"
: "${LOCAL_IP:?set LOCAL_IP to this node's HCCL address}"
: "${NIC_NAME:?set NIC_NAME to this node's HCCL interface}"

MODE="${MODE:-orthrus}"
PROFILE="${PROFILE:-smoke-16k}"
PORT="${PORT:-8901}"
DP_RPC_PORT="${DP_RPC_PORT:-13400}"
MODELS_HOST="${MODELS_HOST:-/data/disk/models}"
MODEL_HOST="${MODEL_HOST:-$MODELS_HOST/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8}"
HEAD_HOST="${HEAD_HOST:-$MODELS_HOST/orthrus-dsv41-head-v1}"
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-quay.io/ascend/vllm-ascend}"
IMAGE_TAG="${IMAGE_TAG:-nightly-releases-v0.32.0rc}"
EXPECTED_IMAGE_DIGEST="${EXPECTED_IMAGE_DIGEST:-sha256:f040ca190fe5c5864a5491b2cbe19e687966ae8609b8a68fccc2c1ad1d260cbe}"
CONTAINER="${CONTAINER:-dsv41-${MODE}-node${NODE_RANK}}"

case "$NODE_RANK" in 0|1) ;; *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;; esac
case "$MODE" in orthrus|target) ;; *) echo "MODE must be orthrus or target" >&2; exit 2 ;; esac
case "$PROFILE" in smoke-16k|prod-128k|prod-1m) ;; *) echo "invalid PROFILE" >&2; exit 2 ;; esac
if [[ ! "$CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]; then
  echo "CONTAINER contains unsupported characters" >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" != "1" && "${ACK_EXCLUSIVE_NODE:-0}" != "1" ]]; then
  echo "set ACK_EXCLUSIVE_NODE=1 only after reserving all eight cards on this node" >&2
  exit 2
fi

image="$IMAGE_REPOSITORY@$EXPECTED_IMAGE_DIGEST"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for command_name in docker readlink; do
    command -v "$command_name" >/dev/null 2>&1 || {
      echo "missing command: $command_name" >&2
      exit 2
    }
  done
  tag="$IMAGE_REPOSITORY:$IMAGE_TAG"
  docker image inspect "$tag" >/dev/null 2>&1 || {
    echo "pinned image is not available locally: $tag" >&2
    exit 2
  }
  repo_digests="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$tag")"
  if ! grep -Fxq "$image" <<<"$repo_digests"; then
    echo "image digest mismatch: expected $image" >&2
    exit 2
  fi
  docker container inspect "$CONTAINER" >/dev/null 2>&1 && {
    echo "container already exists: $CONTAINER" >&2
    exit 2
  }
  [[ -f "$MODEL_HOST/config.json" ]] || { echo "missing model: $MODEL_HOST" >&2; exit 2; }
  if [[ "$MODE" == "orthrus" ]]; then
    [[ -f "$HEAD_HOST/config.json" ]] || { echo "missing head: $HEAD_HOST" >&2; exit 2; }
  fi
  for device in {0..7}; do
    [[ -e "/dev/davinci$device" ]] || { echo "missing /dev/davinci$device" >&2; exit 2; }
  done
  model_root="$(readlink -f "$MODEL_HOST")"
  models_root="$(readlink -f "$MODELS_HOST")"
  case "$model_root" in
    "$models_root"/*) MODEL="/data/disk/models/${model_root#"$models_root"/}" ;;
    *) echo "MODEL_HOST must be below MODELS_HOST" >&2; exit 2 ;;
  esac
  if [[ "$MODE" == "orthrus" ]]; then
    head_root="$(readlink -f "$HEAD_HOST")"
    case "$head_root" in
      "$models_root"/*) HEAD="/data/disk/models/${head_root#"$models_root"/}" ;;
      *) echo "HEAD_HOST must be below MODELS_HOST" >&2; exit 2 ;;
    esac
  fi
else
  MODEL="/data/disk/models/${MODEL_HOST#"$MODELS_HOST"/}"
  HEAD="/data/disk/models/${HEAD_HOST#"$MODELS_HOST"/}"
fi

devices=(
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)
for device in {0..7}; do
  devices+=(--device "/dev/davinci$device:/dev/davinci$device")
done

docker_run=(
  docker run -d
  --name "$CONTAINER"
  --network host
  --shm-size 512g
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  "${devices[@]}"
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /etc/hccn.conf:/etc/hccn.conf:ro
  -v "$repo_root:/workspace/online-speculation"
  -v "$MODELS_HOST:/data/disk/models"
  "$image" bash -lc 'sleep infinity'
)

launcher="scripts/serve_dsv41_${MODE}.sh"
service_log="/workspace/online-speculation/local/dsv41-deploy/$CONTAINER/service.log"
docker_exec=(
  docker exec -d
  -w /workspace/online-speculation
  -e NODE_RANK="$NODE_RANK"
  -e NODE0_IP="$NODE0_IP"
  -e LOCAL_IP="$LOCAL_IP"
  -e NIC_NAME="$NIC_NAME"
  -e MODEL="$MODEL"
  -e HEAD="${HEAD:-}"
  -e PROFILE="$PROFILE"
  -e PORT="$PORT"
  -e DP_RPC_PORT="$DP_RPC_PORT"
  -e REQUIRE_TRAINED_HEAD="${REQUIRE_TRAINED_HEAD:-1}"
  -e ORTHRUS_DRAFT_SAMPLE_METHOD="${ORTHRUS_DRAFT_SAMPLE_METHOD:-greedy}"
  "$CONTAINER" bash -lc "mkdir -p \"$(dirname "$service_log")\"; exec bash \"$launcher\" >\"$service_log\" 2>&1"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'command:'; printf ' %q' "${docker_run[@]}"; printf '\n'
  printf 'command:'; printf ' %q' "${docker_exec[@]}"; printf '\n'
  exit 0
fi

"${docker_run[@]}" >/dev/null
"${docker_exec[@]}"
echo "started $CONTAINER as DSV4.1 $MODE node $NODE_RANK"
echo "service log: $repo_root/local/dsv41-deploy/$CONTAINER/service.log"
if [[ "$NODE_RANK" == "0" ]]; then
  echo "start node 1, then check http://$NODE0_IP:$PORT/health"
fi
