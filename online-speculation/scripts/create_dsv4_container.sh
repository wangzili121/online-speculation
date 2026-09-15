#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER:?set a unique container name}"
: "${REPO_HOST:?set the online-speculation repository directory}"
MODELS_HOST="${MODELS_HOST:-/data/disk/models}"
IMAGE_REF="${IMAGE_REF:-quay.io/ascend/vllm-ascend:nightly-releases-v0.26.0rc}"
IMAGE_PULL="${IMAGE_PULL:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
DOCKER_PRIVILEGED="${DOCKER_PRIVILEGED:-0}"

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container already exists: $CONTAINER" >&2
  exit 1
fi
if [[ ! -d "$MODELS_HOST" ]]; then
  echo "model directory does not exist: $MODELS_HOST" >&2
  exit 1
fi

case "$IMAGE_PULL" in
  auto)
    if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
      docker pull "$IMAGE_REF"
    fi
    ;;
  always)
    docker pull "$IMAGE_REF"
    ;;
  never)
    if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
      echo "image is not available locally and IMAGE_PULL=never: $IMAGE_REF" >&2
      exit 1
    fi
    ;;
  *)
    echo "IMAGE_PULL must be auto, always, or never" >&2
    exit 1
    ;;
esac
repo_digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE_REF")"
if [[ "$repo_digest" != *@sha256:* ]]; then
  echo "could not resolve immutable digest for $IMAGE_REF" >&2
  exit 1
fi
image_digest="${repo_digest#*@}"
image="$repo_digest"
if [[ -n "${EXPECTED_IMAGE_DIGEST:-}" && "$image_digest" != "$EXPECTED_IMAGE_DIGEST" ]]; then
  echo "image digest mismatch: expected $EXPECTED_IMAGE_DIGEST, got $image_digest" >&2
  exit 1
fi

IFS=',' read -r -a requested_npus <<<"$NPU_DEVICES"
if (( ${#requested_npus[@]} == 0 )); then
  echo "NPU_DEVICES must contain at least one device" >&2
  exit 1
fi

devices=()
logical_npus=()
declare -A seen_npus=()
for logical_npu in "${!requested_npus[@]}"; do
  npu="${requested_npus[$logical_npu]}"
  if [[ ! "$npu" =~ ^[0-7]$ ]]; then
    echo "invalid NPU id in NPU_DEVICES: $npu" >&2
    exit 1
  fi
  if [[ -n "${seen_npus[$npu]:-}" ]]; then
    echo "duplicate NPU id in NPU_DEVICES: $npu" >&2
    exit 1
  fi
  if [[ ! -e "/dev/davinci${npu}" ]]; then
    echo "NPU device does not exist: /dev/davinci${npu}" >&2
    exit 1
  fi
  seen_npus[$npu]=1
  devices+=(--device "/dev/davinci${npu}:/dev/davinci${logical_npu}")
  logical_npus+=("$logical_npu")
done
visible_devices="$(IFS=,; echo "${logical_npus[*]}")"

privileged=()
if [[ "$DOCKER_PRIVILEGED" == "1" ]]; then
  privileged+=(--privileged=true)
elif [[ "$DOCKER_PRIVILEGED" != "0" ]]; then
  echo "DOCKER_PRIVILEGED must be 0 or 1" >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER" \
  --entrypoint bash \
  --network host \
  --shm-size 512g \
  "${privileged[@]}" \
  -e ORTHRUS_IMAGE_REF="$IMAGE_REF" \
  -e ORTHRUS_IMAGE_DIGEST="$image_digest" \
  -e ASCEND_RT_VISIBLE_DEVICES="$visible_devices" \
  -e ORTHRUS_PHYSICAL_NPU_DEVICES="$NPU_DEVICES" \
  "${devices[@]}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "$REPO_HOST:/workspace/online-speculation" \
  -v "$MODELS_HOST:/data/disk/models" \
  "$image" -lc 'sleep infinity'

lock=/workspace/online-speculation/runtime/vllm_ascend_v0_26/runtime-lock.local.json
docker exec -w /workspace/online-speculation "$CONTAINER" \
  "$PYTHON_BIN" scripts/capture_dsv4_runtime_lock.py \
    --image-ref "$IMAGE_REF" \
    --image-digest "$image_digest" \
    --output "$lock"
docker exec -e RUNTIME_LOCK="$lock" -w /workspace/online-speculation "$CONTAINER" \
  bash runtime/vllm_ascend_v0_26/apply.sh

echo "created $CONTAINER with $image"
echo "physical NPU devices: $NPU_DEVICES (container-visible: $visible_devices)"
echo "runtime lock: $REPO_HOST/runtime/vllm_ascend_v0_26/runtime-lock.local.json"
