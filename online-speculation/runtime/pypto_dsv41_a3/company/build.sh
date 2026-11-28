#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
source "$here/sources/commits.env"
pin_file="$here/sources/serving/pypto-serving-deepseek-v4.1-flash-a3/pypto.pin"
pypto_pin="$(tr -d '[:space:]' < "$pin_file")"
image="${ORTHRUS_IMAGE:-orthrus-dsv41-a3:${LIB_COMMIT:0:7}-${SERVING_COMMIT:0:7}-${pypto_pin:0:7}}"
swr_base='swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev@sha256:7712c0bae0f8a3c46cae127be4802324682e33ad1c49436e17db7bd60975dbee'
known_layer='sha256:ccfb03958f342a432b1e8360d1be7e7088a7e2a48e0a645c9ea29444f75911e8'

if [[ "$(uname -m)" != aarch64 ]]; then
    echo 'Build this image on an ARM64 machine.' >&2
    exit 2
fi
docker info >/dev/null
mapfile -t local_images < <(docker image ls orthrus-dsv41-pypto --format '{{.Repository}}:{{.Tag}}')
base=''
for candidate in "$swr_base" orthrus-dsv41-pypto:72a62d1-offline "${local_images[@]}"; do
    if layer="$(docker image inspect "$candidate" --format '{{index .RootFS.Layers 26}}' 2>/dev/null)" &&
       [[ "$layer" == "$known_layer" ]]; then
        base="$candidate"
        break
    fi
done
if [[ -z "$base" ]]; then
    docker pull "$swr_base"
    base="$swr_base"
fi
printf 'Base image: %s\n' "$base"

work="$(mktemp -d "${TMPDIR:-/tmp}/orthrus-a3-build.XXXXXX")"
printf 'Build directory: %s\n' "$work"
cd "$work"
cp "$here/Dockerfile" "$here/start-serving.sh" "$here/../runtime-release.patch" .
cp -R "$here/sources" .
mkdir -p study/online-speculation/scripts
for script in run_dsv41_pypto_a3_study.py prepare_dsv41_gsm8k.py; do
    cp "$repo/scripts/$script" study/online-speculation/scripts/
done

# The pinned PyPTO build scripts need Git metadata for the checkout and runtime.
git init -q pypto
git -C pypto fetch --depth 1 https://github.com/hw-native-sys/pypto.git "$pypto_pin"
git -C pypto checkout -q --detach FETCH_HEAD
git -C pypto submodule update --init --recursive --depth 1
printf 'PYPTO_BUILD_JOBS=8\nPYPTO_TEST_JOBS=2\n' > pypto/.claude/skills/testing/testing.env

DOCKER_BUILDKIT=0 docker build --network host --build-arg BASE_IMAGE="$base" \
    --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
    -t "$image" . 2>&1 | tee build.log

docker run --rm --network none --entrypoint bash "$image" -ec '
    source /opt/pypto/bin/pypto-activate.sh
    cd /opt/orthrus-a3/DeepSeek-V4.1-Flash-A3/pypto-serving-deepseek-v4.1-flash-a3
    python -c '"'"'
import pypto.language
import simpler.task_interface
import serving.api
from models.deepseek_v4_1_flash_a3.packed import PackedSession
from models.deepseek_v4_1_flash_a3.orthrus_draft.resident import ResidentDraft
from models.deepseek_v4_1_flash_a3.orthrus_draft.online.learner import ResidentLearner
assert pypto.language.__file__.startswith("/opt/orthrus-a3/pypto/")
    '"'"'
    ptoas --version
    python -m serving.api --help >/dev/null
' 2>&1 | tee check.log
printf '\nImage ready: %s\nBuild log: %s/build.log\nCheck log: %s/check.log\n' "$image" "$work" "$work"
