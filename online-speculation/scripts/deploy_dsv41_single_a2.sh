#!/usr/bin/env bash
# Host launcher for the experimental one-node, eight-card A2 smoke.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LOCAL_IP:?set LOCAL_IP to this node's HCCL address}"
: "${NIC_NAME:?set NIC_NAME to this node's HCCL interface}"

MODE="${MODE:-target}"
PORT="${PORT:-8900}"
MODELS_HOST="${MODELS_HOST:-/data/disk/wangzili/models}"
MODEL_HOST="${MODEL_HOST:-$MODELS_HOST/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8}"
HEAD_HOST="${HEAD_HOST:-$MODELS_HOST/orthrus-dsv41-head-v1}"
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-quay.io/ascend/vllm-ascend}"
EXPECTED_IMAGE_DIGEST="${EXPECTED_IMAGE_DIGEST:-sha256:f040ca190fe5c5864a5491b2cbe19e687966ae8609b8a68fccc2c1ad1d260cbe}"
CONTAINER="${CONTAINER:-dsv41-${MODE}-single-a2}"

case "$MODE" in target|orthrus) ;; *) echo "MODE must be target or orthrus" >&2; exit 2 ;; esac
case "$MODEL_HOST" in
  "$MODELS_HOST"/*) ;;
  *) echo "MODEL_HOST must be below MODELS_HOST" >&2; exit 2 ;;
esac
case "$HEAD_HOST" in
  "$MODELS_HOST"/*) ;;
  *) echo "HEAD_HOST must be below MODELS_HOST" >&2; exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" != "1" && "${ACK_EXCLUSIVE_NODE:-0}" != "1" ]]; then
  echo "set ACK_EXCLUSIVE_NODE=1 only after reserving all eight cards" >&2
  exit 2
fi

image="$IMAGE_REPOSITORY@$EXPECTED_IMAGE_DIGEST"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  command -v docker >/dev/null 2>&1 || { echo "missing command: docker" >&2; exit 2; }
  [[ -f "$MODEL_HOST/config.json" ]] || { echo "missing model: $MODEL_HOST" >&2; exit 2; }
  if [[ "$MODE" == "orthrus" ]]; then
    [[ -f "$HEAD_HOST/config.json" ]] || { echo "missing head: $HEAD_HOST" >&2; exit 2; }
  fi
  docker image inspect "$image" >/dev/null 2>&1 || {
    echo "pinned image is not available locally: $image" >&2
    exit 2
  }
  docker container inspect "$CONTAINER" >/dev/null 2>&1 && {
    echo "container already exists: $CONTAINER" >&2
    exit 2
  }
  for device in {0..7}; do
    [[ -e "/dev/davinci$device" ]] || { echo "missing /dev/davinci$device" >&2; exit 2; }
  done
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
  --ipc host
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

model="/data/disk/models/${MODEL_HOST#"$MODELS_HOST"/}"
head="/data/disk/models/${HEAD_HOST#"$MODELS_HOST"/}"
service_log="/workspace/online-speculation/local/dsv41-deploy/$CONTAINER/service.log"
docker_exec=(
  docker exec -d
  -w /workspace/online-speculation
  -e MODE="$MODE"
  -e MODEL="$model"
  -e HEAD="$head"
  -e PORT="$PORT"
  -e LOCAL_IP="$LOCAL_IP"
  -e NIC_NAME="$NIC_NAME"
  -e REQUIRE_TRAINED_HEAD="${REQUIRE_TRAINED_HEAD:-0}"
  "$CONTAINER" bash -lc "mkdir -p \"$(dirname "$service_log")\"; exec bash scripts/serve_dsv41_single_a2.sh >\"$service_log\" 2>&1"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'command:'; printf ' %q' "${docker_run[@]}"; printf '\n'
  printf 'command:'; printf ' %q' "${docker_exec[@]}"; printf '\n'
  exit 0
fi

"${docker_run[@]}" >/dev/null
"${docker_exec[@]}"
echo "started experimental single-node A2 $MODE smoke in $CONTAINER"
echo "service log: $repo_root/local/dsv41-deploy/$CONTAINER/service.log"
