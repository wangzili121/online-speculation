#!/usr/bin/env bash
# Launch the W4A8 service or the HCCL probe on one A2 node.
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
HCCL_PROBE_PORT="${HCCL_PROBE_PORT:-13401}"
MODELS_HOST="${MODELS_HOST:-/workspace/models}"
MODEL_HOST="${MODEL_HOST:-$MODELS_HOST/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8}"
HEAD_HOST="${HEAD_HOST:-$MODELS_HOST/orthrus-dsv41-head-v1}"
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-quay.io/ascend/vllm-ascend}"
EXPECTED_IMAGE_DIGEST="${EXPECTED_IMAGE_DIGEST:-sha256:f040ca190fe5c5864a5491b2cbe19e687966ae8609b8a68fccc2c1ad1d260cbe}"
CONTAINER="${CONTAINER:-dsv41-${MODE}-node${NODE_RANK}}"
INIT_HEAD="${INIT_HEAD:-0}"

fail() { echo "$*" >&2; exit 2; }

case "$NODE_RANK" in 0|1) ;; *) fail "NODE_RANK must be 0 or 1" ;; esac
case "$MODE" in orthrus|target|probe) ;; *) fail "MODE must be orthrus, target or probe" ;; esac
case "$PROFILE" in smoke-16k|prod-128k|prod-1m) ;; *) fail "invalid PROFILE" ;; esac
[[ "$CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || fail "invalid container name"
[[ "$MODEL_HOST" == /* && "$HEAD_HOST" == /* ]] || fail "MODEL_HOST and HEAD_HOST must be absolute paths"
if [[ "${DRY_RUN:-0}" != 1 && "${ACK_EXCLUSIVE_NODE:-0}" != 1 ]]; then
  fail "set ACK_EXCLUSIVE_NODE=1 after reserving all eight cards on this node"
fi

image="$IMAGE_REPOSITORY@$EXPECTED_IMAGE_DIGEST"
if [[ -n "${PINNED_LOCAL_IMAGE_ID:-}" ]]; then
  [[ "$PINNED_LOCAL_IMAGE_ID" == "sha256:5a70e7cdbe81b2b2addad33cf31db6f8e8163fa4b510ff307a8864d9db36b3a2" ]] ||
    fail "PINNED_LOCAL_IMAGE_ID must match the tested v0.32.0rc image"
  image="$PINNED_LOCAL_IMAGE_ID"
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  command -v docker >/dev/null || fail "docker missing on this host"
  docker image inspect "$image" >/dev/null 2>&1 || fail "load the pinned image on this node: $image"
  if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    fail "container already exists: $CONTAINER. Inspect its logs and remove this container before retrying."
  fi
  for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc /dev/davinci{0..7}; do
    [[ -e "$device" ]] || fail "missing $device"
  done
  if [[ "$MODE" != probe ]]; then
    [[ -f "$MODEL_HOST/config.json" && -f "$MODEL_HOST/tokenizer.json" ]] || fail "missing model at $MODEL_HOST"
  fi
  if [[ "$MODE" == orthrus ]]; then
    if [[ -e "$HEAD_HOST" || "$INIT_HEAD" != 1 ]]; then
      [[ -f "$HEAD_HOST/manifest.json" && -f "$HEAD_HOST/config.json" ]] || fail "missing Orthrus head at $HEAD_HOST"
    else
      mkdir -p "$(dirname "$HEAD_HOST")"
    fi
  fi
fi

devices=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
for device in {0..7}; do
  devices+=(--device "/dev/davinci$device:/dev/davinci$device")
done

mounts=(-v "$repo_root:/workspace/online-speculation")
if [[ "$MODE" != probe ]]; then
  mounts+=(-v "$MODEL_HOST:/workspace/target:ro")
fi
head_dir="/workspace/heads/$(basename "$HEAD_HOST")"
if [[ "$MODE" == orthrus ]]; then
  if [[ "$INIT_HEAD" == 1 && ! -e "$HEAD_HOST" ]]; then
    mounts+=(-v "$(dirname "$HEAD_HOST"):/workspace/heads")
  else
    mounts+=(-v "$HEAD_HOST:$head_dir:ro")
  fi
fi

docker_run=(
  docker run --pull never -d --name "$CONTAINER" --network host --shm-size "${SHM_SIZE:-512g}"
  -w /workspace/online-speculation
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  --label org.online-speculation.service=dsv41-a2
  -e "NODE_RANK=$NODE_RANK" -e "NODE0_IP=$NODE0_IP" -e "LOCAL_IP=$LOCAL_IP" -e "NIC_NAME=$NIC_NAME"
  -e "PORT=$PORT" -e "DP_RPC_PORT=$DP_RPC_PORT" -e "PROFILE=$PROFILE"
  "${devices[@]}"
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /etc/hccn.conf:/etc/hccn.conf:ro
  "${mounts[@]}"
)
if [[ "$MODE" == probe ]]; then
  docker_run+=(
    "$image" bash -c 'source scripts/dsv41_service_env.sh
configure_dsv41_network
export HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=180 HCCL_INTRA_ROCE_ENABLE=1 OMP_NUM_THREADS=1
exec python3 -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" --master-addr="$NODE0_IP" --master-port="$1" scripts/probe_dsv41_cluster.py' bash "$HCCL_PROBE_PORT"
  )
else
  docker_run+=(
    -e MODEL=/workspace/target -e "HEAD=$head_dir" -e "INIT_HEAD=$INIT_HEAD"
    -e REQUIRE_TRAINED_HEAD="${REQUIRE_TRAINED_HEAD:-1}" -e MASK_TOKEN_ID="${MASK_TOKEN_ID:-128799}"
    -e ORTHRUS_DRAFT_SAMPLE_METHOD="${ORTHRUS_DRAFT_SAMPLE_METHOD:-greedy}"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PIP_NO_INDEX=1
    "$image" bash "scripts/serve_dsv41_${MODE}.sh"
  )
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'command:'; printf ' %q' "${docker_run[@]}"; printf '\n'
  exit 0
fi
"${docker_run[@]}" >/dev/null
echo "Started $CONTAINER. Logs: docker logs -f $CONTAINER"
