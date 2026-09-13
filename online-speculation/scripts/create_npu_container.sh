#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER:?set a unique experiment container name}"
: "${NPU_ID:?set an idle physical NPU id}"
: "${WORKSPACE_HOST:?set the host directory mounted at /workspace}"
: "${MODELS_HOST:?set the host model parent mounted at /models}"

IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0}"
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container already exists: $CONTAINER" >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER" \
  --entrypoint bash \
  --network host \
  --shm-size 16g \
  --device "/dev/davinci${NPU_ID}:/dev/davinci0" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v "$WORKSPACE_HOST:/workspace" \
  -v "$MODELS_HOST:/models:ro" \
  "$IMAGE" -lc 'sleep infinity'

docker exec -w /workspace/online-speculation "$CONTAINER" \
  python -m pip install --no-deps -e .
docker exec -w /workspace/online-speculation "$CONTAINER" \
  runtime/vllm_ascend_v0_23/apply.sh /vllm-workspace/vllm-ascend
