#!/usr/bin/env bash
# Build once with internet access, then export the image for offline deployment.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
default_image=orthrus-dsv41-pypto:72a62d1
if [[ "$action" == fabric-check && "${1:-}" == node* ]]; then
  if [[ $# -ne 1 || ( "$1" != node7 && "$1" != node8 ) ]]; then
    echo "Usage: $0 fabric-check {node7|node8}" >&2
    exit 2
  fi
  master_address=7.150.6.33
  case "$1" in
    node7)
      rank=0
      HCCL_IF_IP="$master_address"
      NPU_DEVICE="${NPU_DEVICE:-1}"
      ;;
    node8)
      rank=1
      HCCL_IF_IP=7.150.6.37
      NPU_DEVICE="${NPU_DEVICE:-0}"
      ;;
  esac
  default_image+=-offline
  set -- --rank "$rank" --master-address "$master_address"
fi
image="${PYPTO_IMAGE:-$default_image}"
case "$action" in
  build)
    [[ "$(uname -s)/$(uname -m)" == Linux/aarch64 ]] || {
      echo "Build this ARM64 image on a Linux aarch64 Docker host." >&2; exit 2;
    }
    recipe="$root/runtime/pypto_dsv41_a3/docker/Dockerfile"
    build_options=(--build-arg "BUILD_JOBS=${BUILD_JOBS:-4}")
    if [[ -n "${PYPTO_BUILD_NETWORK:-}" ]]; then
      build_options+=(--network "$PYPTO_BUILD_NETWORK")
    fi
    if [[ -n "${PYPTO_BUILD_CA:-}" ]]; then
      [[ -f "$PYPTO_BUILD_CA" && -r "$PYPTO_BUILD_CA" && -s "$PYPTO_BUILD_CA" ]] || {
        echo 'PYPTO_BUILD_CA must point to a readable, non-empty PEM CA file.' >&2; exit 2;
      }
      build_options+=(--secret "id=pypto_ca,src=$PYPTO_BUILD_CA")
    fi
    if ! docker buildx version >/dev/null 2>&1; then
      if [[ -n "${PYPTO_BUILD_PROXY:-}${PYPTO_BUILD_CA:-}${PYPTO_BUILD_PYPI:-}" ]]; then
        echo 'Builds with a proxy, custom CA or Python index require Docker Buildx.' >&2
        exit 2
      fi
      sed -E 's/^RUN( --mount=type=secret,id=[a-z_]+)+ /RUN /; s/^RUN --network=none /RUN /' "$recipe" |
        DOCKER_BUILDKIT=0 docker build "${build_options[@]}" -t "$image" -f - "$root/.."
      exit
    fi
    if [[ -n "${PYPTO_BUILD_PROXY:-}" ]]; then
      export PYPTO_BUILD_PROXY
      build_options+=(--secret id=pypto_proxy,env=PYPTO_BUILD_PROXY)
    fi
    if [[ -n "${PYPTO_BUILD_PYPI:-}" ]]; then
      export PYPTO_BUILD_PYPI
      build_options+=(--secret id=pypto_pypi,env=PYPTO_BUILD_PYPI)
    fi
    export DOCKER_BUILDKIT=1
    exec docker build "${build_options[@]}" \
      -t "$image" -f "$recipe" "$root/.."
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
  doctor|preflight|compile-probe|probe|device-map|shell|attention-check|handoff-check|fabric-check|init-head)
    docker image inspect "$image" >/dev/null 2>&1 || {
      echo "Image missing. Build it on a networked host or load the transferred image.tar." >&2; exit 2;
    }
    network=host
    if [[ "$action" == attention-check || "$action" == handoff-check || "$action" == init-head || "$action" == device-map ]]; then network=none; fi
    options=(--rm --network "$network" --shm-size "${SHM_SIZE:-16g}")
    command_env=(env)
    if [[ "$action" == preflight || "$action" == handoff-check || "$action" == init-head ]]; then
      model="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
      [[ -f "$model/config.json" ]] || { echo "Missing checkpoint $model" >&2; exit 2; }
      options+=(-v "$model:/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8:ro")
      if [[ "$action" != init-head && -n "${DRAFT_HEAD:-}" ]]; then
        [[ -f "$DRAFT_HEAD/manifest.json" ]] || { echo "Missing head manifest in $DRAFT_HEAD" >&2; exit 2; }
        options+=(-v "$DRAFT_HEAD:/workspace/draft-head:ro" -e DRAFT_HEAD=/workspace/draft-head)
        if [[ "$action" == preflight ]]; then set -- --head /workspace/draft-head "$@"; fi
      fi
    fi
    if [[ "$action" == probe || "$action" == device-map || "$action" == shell || "$action" == attention-check || "$action" == fabric-check ||
          ( "$action" == handoff-check && -n "${NPU_DEVICE:-}" ) ]]; then
      : "${NPU_DEVICE:?Set NPU_DEVICE to an available physical device number}"
      [[ "$NPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "NPU_DEVICE must be a number" >&2; exit 2; }
      options+=(
        --device "/dev/davinci$NPU_DEVICE:/dev/davinci$NPU_DEVICE"
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
      )
      if [[ "$action" == device-map ]]; then
        options+=(-v "$root:/workspace/online-speculation:ro")
        status=1
        filters=(unset 0)
        if [[ "$NPU_DEVICE" != 0 ]]; then filters+=("$NPU_DEVICE"); fi
        for filter in "${filters[@]}"; do
          if [[ "$filter" == unset ]]; then
            command_env=(env -u ASCEND_RT_VISIBLE_DEVICES)
          else
            command_env=(env "ASCEND_RT_VISIBLE_DEVICES=$filter")
          fi
          if docker run "${options[@]}" "$image" "${command_env[@]}" timeout -k 2s 20s python \
            /workspace/online-speculation/scripts/probe_dsv41_pypto_device_map.py "$NPU_DEVICE"; then status=0; fi
        done
        exit "$status"
      fi
      visible_device="${PYPTO_VISIBLE_DEVICE:-0}"
      runtime_device="${PYPTO_RUNTIME_DEVICE:-0}"
      [[ "$runtime_device" =~ ^[0-9]+$ ]] || { echo "PYPTO_RUNTIME_DEVICE must be a number" >&2; exit 2; }
      options+=(-e "PYPTO_RUNTIME_DEVICE=$runtime_device")
      if [[ "$visible_device" == unset ]]; then
        command_env=(env -u ASCEND_RT_VISIBLE_DEVICES)
      else
        [[ "$visible_device" =~ ^[0-9]+$ ]] || { echo "PYPTO_VISIBLE_DEVICE must be a number or unset" >&2; exit 2; }
        options+=(-e "ASCEND_RT_VISIBLE_DEVICES=$visible_device")
      fi
    else
      options+=(-e ASCEND_RT_VISIBLE_DEVICES=)
      if [[ -d /usr/local/Ascend/driver ]]; then
        options+=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro)
      fi
    fi
    if [[ "$action" == probe ]]; then
      options+=(-v "$root:/workspace/online-speculation:ro")
      exec docker run "${options[@]}" "$image" "${command_env[@]}" python \
        /workspace/online-speculation/scripts/check_dsv41_pypto_environment.py --device-probe "$@" --device "$runtime_device"
    fi
    if [[ "$action" == shell ]]; then
      options+=(-it -v "$root:/workspace/online-speculation:ro")
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash
    fi
    if [[ "$action" == init-head ]]; then
      : "${DRAFT_HEAD:?Set DRAFT_HEAD to a new output directory on the host}"
      [[ ! -e "$DRAFT_HEAD" ]] || { echo "Head output already exists: $DRAFT_HEAD" >&2; exit 2; }
      mkdir -p "$(dirname "$DRAFT_HEAD")"
      head_parent="$(cd "$(dirname "$DRAFT_HEAD")" && pwd)"
      options+=(--user "$(id -u):$(id -g)" --entrypoint /opt/pypto/venv/bin/python
                -e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}" -e PYTHONPATH=/workspace/online-speculation/src
                -v "$root:/workspace/online-speculation:ro" -v "$head_parent:/results")
      for group in $(id -G); do options+=(--group-add "$group"); done
      exec docker run "${options[@]}" "$image" /workspace/online-speculation/scripts/export_dsv41_orthrus_head.py \
        --model /workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8 \
        --output "/results/$(basename "$DRAFT_HEAD")" --from-target "$@"
    fi
    if [[ "$action" == fabric-check ]]; then
      if [[ -n "${HCCL_IF_IP:-}" ]]; then options+=(-e "HCCL_IF_IP=$HCCL_IF_IP"); fi
      if [[ -f /etc/hccn.conf ]]; then options+=(-v /etc/hccn.conf:/etc/hccn.conf:ro); fi
      log_dir="$(mktemp -d "$PWD/orthrus-fabric.XXXXXX")"
      echo "CANN logs: $log_dir" >&2
      options+=(-v "$root:/workspace/online-speculation:ro"
                -v "$log_dir:/cann-logs" -e ASCEND_PROCESS_LOG_PATH=/cann-logs)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" timeout -k 5s 180s python \
        /workspace/online-speculation/scripts/probe_dsv41_pypto_fabric.py "$@" --device "$runtime_device"
    fi
    if [[ "$action" == attention-check || "$action" == handoff-check ]]; then
      kind=handoff
      if [[ "$action" == attention-check ]]; then kind=attention; fi
      report_dir="$(mktemp -d "$PWD/orthrus-$kind.XXXXXX")"
      echo "Reports: $report_dir"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results")
      if [[ -n "${runtime_device:-}" ]]; then set -- "$@" --device "$runtime_device"; fi
      if [[ "$action" == attention-check ]]; then
        exec docker run "${options[@]}" "$image" "${command_env[@]}" bash /workspace/online-speculation/scripts/compare_dsv41_pypto_swa.sh "$@"
      fi
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash /workspace/online-speculation/scripts/check_dsv41_pypto_handoff.sh "$@"
    fi
    exec docker run "${options[@]}" "$image" "$action" "$@"
    ;;
  *)
    echo "Usage: $0 {build|save FILE|load FILE|host-info|doctor|preflight|compile-probe|probe|device-map|shell|attention-check|handoff-check|fabric-check|init-head}"
    echo "Company fabric probe: $0 fabric-check {node7|node8}"
    ;;
esac
