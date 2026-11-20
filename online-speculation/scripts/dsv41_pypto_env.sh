#!/usr/bin/env bash
# Build once with internet access, then export the image for offline deployment.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${PYPTO_IMAGE:-orthrus-dsv41-pypto:72a62d1}"
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  build)
    [[ "$(uname -s)/$(uname -m)" == Linux/aarch64 ]] || {
      echo "Build this ARM64 image on a Linux aarch64 Docker host." >&2; exit 2;
    }
    exec docker build --build-arg "BUILD_JOBS=${BUILD_JOBS:-4}" \
      -t "$image" -f "$root/runtime/pypto_dsv41_a3/docker/Dockerfile" "$root/.."
    ;;
  save)
    destination="${1:?Usage: dsv41_pypto_env.sh save /path/to/image.tar}"
    if [[ "$destination" == *.tar.gz || "$destination" == *.tgz ]]; then
      docker save "$image" | gzip -1 > "$destination"
    else
      docker save -o "$destination" "$image"
    fi
    (cd "$(dirname "$destination")" && sha256sum "$(basename "$destination")") > "$destination.sha256"
    ;;
  load)
    archive="${1:?Usage: dsv41_pypto_env.sh load /path/to/image.tar}"
    if [[ -f "$archive.sha256" ]]; then
      (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$archive").sha256")
    fi
    exec docker load -i "$archive"
    ;;
  host-info)
    uname -m
    cat /usr/local/Ascend/driver/version.info
    docker version --format '{{.Server.Version}}'
    docker info --format '{{json .Runtimes}}'
    ;;
  doctor|preflight|compile-probe|probe|shell)
    docker image inspect "$image" >/dev/null 2>&1 || {
      echo "Image missing. Build it on a networked host or load the transferred image.tar." >&2; exit 2;
    }
    options=(--rm --network host --shm-size "${SHM_SIZE:-16g}")
    if [[ "$action" == preflight ]]; then
      model="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
      [[ -f "$model/config.json" ]] || { echo "Missing checkpoint $model" >&2; exit 2; }
      options+=(-v "$model:/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8:ro")
    fi
    if [[ "$action" == probe || "$action" == shell ]]; then
      : "${NPU_DEVICE:?Set NPU_DEVICE to an available physical device number}"
      [[ "$NPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "NPU_DEVICE must be a number" >&2; exit 2; }
      options+=(
        --device "/dev/davinci$NPU_DEVICE:/dev/davinci$NPU_DEVICE"
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
        -e "ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICE"
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
      )
    else
      options+=(-e ASCEND_RT_VISIBLE_DEVICES=)
      if [[ -d /usr/local/Ascend/driver ]]; then
        options+=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro)
      fi
    fi
    if [[ "$action" == shell ]]; then
      options+=(-it -v "$root:/workspace/online-speculation:ro")
      exec docker run "${options[@]}" "$image" bash
    fi
    exec docker run "${options[@]}" "$image" "$action" "$@"
    ;;
  *)
    echo "Usage: $0 {build|save FILE|load FILE|host-info|doctor|preflight|compile-probe|probe|shell}"
    ;;
esac
