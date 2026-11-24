#!/usr/bin/env bash
# Build once with internet access, then export the image for offline deployment.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
default_image=orthrus-dsv41-pypto:72a62d1
if [[ "$action" == cluster-check || "$action" == cluster-accept ||
      ( "$action" == fabric-check && "${1:-}" == node* ) ]]; then
  node="${1:-}"
  if [[ "$node" != node7 && "$node" != node8 ]]; then
    echo "Specify node7 or node8 for $action" >&2
    exit 2
  fi
  shift
  if [[ "$action" == fabric-check && $# -ne 0 ]]; then
    echo "Usage: $0 fabric-check {node7|node8}" >&2
    exit 2
  fi
  master_address=7.150.6.33
  case "$node" in
    node7)
      rank=0
      HCCL_IF_IP="$master_address"
      NPU_DEVICE="${NPU_DEVICE:-1}"
      ;;
    node8)
      rank=1
      if ! route="$(ip -4 route get "$master_address")"; then
        echo "Cannot resolve the node8 route to $master_address" >&2
        exit 2
      fi
      HCCL_IF_IP="$(awk '{for (i=1; i<NF; i++) if ($i == "src") {print $(i+1); exit}}' <<< "$route")"
      if [[ -z "$HCCL_IF_IP" || "$HCCL_IF_IP" == "$master_address" ]]; then
        echo "Run the node8 probe on the peer host with an IPv4 source route to $master_address" >&2
        exit 2
      fi
      NPU_DEVICE="${NPU_DEVICE:-0}"
      ;;
  esac
  echo "Rank $rank HCCL host IP: $HCCL_IF_IP" >&2
  default_image+=-offline
  if [[ "$action" == fabric-check ]]; then
    set -- --rank "$rank" --master-address "$master_address"
  fi
fi
image="${PYPTO_IMAGE:-$default_image}"
case "$action" in
  online-abi-check)
    image="${PYPTO_IMAGE:-$default_image-offline}"
    report_dir="$(mktemp -d "$PWD/orthrus-online-abi.XXXXXX")"
    echo "Report: $report_dir/resident-online-abi.json"
    exec docker run --rm --network none \
      -v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results" \
      -e PYTHONPATH=/workspace/online-speculation/src \
      "$image" bash /workspace/online-speculation/scripts/run_dsv41_pypto_cluster.sh \
      --online-abi-only
    ;;
  gsm8k-prepare)
    destination="${1:-/workspace/data/orthrus-dsv41-gsm8k}"
    arguments=(--output "$destination")
    if [[ -n "${2:-}" ]]; then arguments+=(--source-directory "$2"); fi
    exec python3 "$root/scripts/prepare_dsv41_gsm8k.py" "${arguments[@]}"
    ;;
  gsm8k-summary)
    [[ $# -gt 0 ]] || { echo "Usage: $0 gsm8k-summary RESULT..." >&2; exit 2; }
    exec python3 "$root/scripts/summarize_dsv41_gsm8k.py" "$@"
    ;;
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
  refresh-image)
    docker image inspect "$image" >/dev/null 2>&1 || {
      echo "Base image is missing: $image" >&2; exit 2;
    }
    image_id="$(docker image inspect --format '{{.Id}}' "$image")"
    image_hex="${image_id#sha256:}"
    base_image="${image%:*}:base-${image_hex:0:12}"
    docker tag "$image" "$base_image"
    refresh_context="$(mktemp -d /tmp/orthrus-dsv41-refresh.XXXXXX)"
    trap 'rm -rf -- "$refresh_context"' EXIT
    # Build from tracked files only. PyPTO compilation may leave root-owned
    # build_output directories in the checkout, which Docker cannot stat and
    # which are neither inputs nor safe to copy into the refreshed image.
    git -C "$root/.." archive --format=tar HEAD online-speculation | \
      tar -C "$refresh_context" -xf -
    recipe="$refresh_context/online-speculation/runtime/pypto_dsv41_a3/docker/Dockerfile.refresh"
    # This refresh is deliberately offline: it only layers the current
    # repository's model patch and Python sources over the pinned image.
    DOCKER_BUILDKIT=0 docker build \
      --build-arg "BASE_IMAGE=$base_image" \
      -t "$image" -f "$recipe" "$refresh_context"
    docker run --rm --network none --entrypoint /bin/bash "$image" -lc \
      'git -C /opt/orthrus/pypto-lib apply --reverse --check \
        /opt/orthrus/online-speculation/runtime/pypto_dsv41_a3/lib-online-expert-capture.patch'
    echo "Refreshed $image from pinned base $base_image"
    rm -rf -- "$refresh_context"
    trap - EXIT
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
  doctor|preflight|compile-probe|probe|device-map|shell|attention-check|online-kernel-check|hc-collapse-vjp-check|hc-pre-vjp-check|resident-publish-check|online-factory-check|online-ready|multicard-check|c1a-vjp-check|expert-capture-check|expert-vjp-check|expert-input-vjp-check|ep-vjp-check|handoff-check|fabric-check|cluster-check|cluster-accept|init-head)
    docker image inspect "$image" >/dev/null 2>&1 || {
      echo "Image missing. Build it on a networked host or load the transferred image.tar." >&2; exit 2;
    }
    network=host
    if [[ "$action" == attention-check || "$action" == online-kernel-check || "$action" == hc-collapse-vjp-check || "$action" == hc-pre-vjp-check || "$action" == resident-publish-check || "$action" == online-factory-check || "$action" == online-ready || "$action" == multicard-check || "$action" == c1a-vjp-check || "$action" == expert-capture-check || "$action" == expert-vjp-check || "$action" == expert-input-vjp-check || "$action" == ep-vjp-check || "$action" == handoff-check || "$action" == init-head || "$action" == device-map ]]; then network=none; fi
    options=(--rm --network "$network" --shm-size "${SHM_SIZE:-16g}")
    command_env=(env)
    if [[ "$action" == preflight || "$action" == handoff-check || "$action" == init-head ||
          "$action" == cluster-accept ]]; then
      model="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
      [[ -f "$model/config.json" ]] || { echo "Missing checkpoint $model" >&2; exit 2; }
      options+=(-v "$model:/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8:ro")
      if [[ "$action" != init-head && -n "${DRAFT_HEAD:-}" ]]; then
        [[ -f "$DRAFT_HEAD/manifest.json" ]] || { echo "Missing head manifest in $DRAFT_HEAD" >&2; exit 2; }
        options+=(-v "$DRAFT_HEAD:/workspace/draft-head:ro" -e DRAFT_HEAD=/workspace/draft-head)
        if [[ "$action" == preflight ]]; then set -- --head /workspace/draft-head "$@"; fi
      fi
    fi
    if [[ "$action" == multicard-check ]]; then
      : "${NPU_DEVICES:?Set NPU_DEVICES to four reserved physical cards, e.g. 4,5,6,7}"
      IFS=, read -ra cards <<< "$NPU_DEVICES"
      [[ "${#cards[@]}" == 4 ]] || { echo 'multicard-check requires exactly four cards.' >&2; exit 2; }
      visible=""
      seen=","
      for index in "${!cards[@]}"; do
        card="${cards[$index]}"
        [[ "$card" =~ ^[0-9]+$ && "$seen" != *",$card,"* ]] || {
          echo 'NPU_DEVICES must contain four distinct device numbers.' >&2; exit 2;
        }
        seen+="$card,"
        options+=(--device "/dev/davinci$card:/dev/davinci$card")
        visible="${visible:+$visible,}$index"
      done
      options+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
                -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
                -e "ASCEND_RT_VISIBLE_DEVICES=$visible")
    elif [[ "$action" == probe || "$action" == device-map || "$action" == shell || "$action" == attention-check || "$action" == online-kernel-check || "$action" == hc-collapse-vjp-check || "$action" == hc-pre-vjp-check || "$action" == resident-publish-check || "$action" == online-ready || "$action" == c1a-vjp-check || "$action" == expert-capture-check || "$action" == expert-vjp-check || "$action" == expert-input-vjp-check || "$action" == ep-vjp-check || "$action" == fabric-check ||
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
      if [[ "$action" != cluster-check && "$action" != cluster-accept ]]; then options+=(-e ASCEND_RT_VISIBLE_DEVICES=); fi
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
    if [[ "$action" == online-kernel-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-online-kernels.XXXXXX")"
      echo "Report: $report_dir/online-kernels.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      # Older transferred images may contain the pinned PTO-ISA checkout only
      # in the PyPTO source tree. Seed Simpler's managed path locally before
      # assembly; this is offline and never substitutes a different revision.
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && exec /opt/pypto/venv/bin/python \
          /workspace/online-speculation/scripts/validate_dsv41_pypto_online_programs.py \
          --device "$@"' _ "$runtime_device" --vocabulary 129280 --output /results/online-kernels.json "$@"
    fi
    if [[ "$action" == hc-collapse-vjp-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-hc-collapse-vjp.XXXXXX")"
      echo "Report: $report_dir/hc-collapse-vjp.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && \
          /opt/pypto/venv/bin/python /workspace/online-speculation/scripts/validate_dsv41_pypto_hc_collapse_vjp.py \
          --device "$@" --production | tee /results/hc-collapse-vjp.json' _ "$runtime_device" "$@"
    fi
    if [[ "$action" == hc-pre-vjp-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-hc-pre-vjp.XXXXXX")"
      echo "Report: $report_dir/hc-pre-vjp.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && \
          /opt/pypto/venv/bin/python /workspace/online-speculation/scripts/validate_dsv41_pypto_hc_pre_vjp.py \
          --device "$@" --output /results/hc-pre-vjp.json; status=$?; \
          cat /results/hc-pre-vjp.json 2>/dev/null || true; exit $status' _ "$runtime_device" "$@"
    fi
    if [[ "$action" == resident-publish-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-resident-publish.XXXXXX")"
      echo "Report: $report_dir/resident-publication.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && exec /opt/pypto/venv/bin/python \
          /workspace/online-speculation/scripts/validate_dsv41_resident_publication.py \
          --device "$@" --output /results/resident-publication.json' _ "$runtime_device"
    fi
    if [[ "$action" == online-factory-check ]]; then
      options+=(-v "$root:/workspace/online-speculation:ro"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" /opt/pypto/venv/bin/python \
        /workspace/online-speculation/scripts/check_dsv41_online_factory.py
    fi
    if [[ "$action" == online-ready ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-online-readiness.XXXXXX")"
      echo "Reports: $report_dir" >&2
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results")
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash \
        /workspace/online-speculation/scripts/check_dsv41_pypto_online_readiness.sh \
        "$runtime_device" /results
    fi
    if [[ "$action" == multicard-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-multicard.XXXXXX")"
      echo "Reports: $report_dir" >&2
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" bash -lc \
        '/opt/pypto/venv/bin/python /workspace/online-speculation/scripts/validate_dsv41_pypto_multicard.py \
           --lib /opt/orthrus/pypto-lib --cards 4 --tp 4 --output /results/tp4.json && \
         /opt/pypto/venv/bin/python /workspace/online-speculation/scripts/validate_dsv41_pypto_multicard.py \
           --lib /opt/orthrus/pypto-lib --cards 4 --tp 1 --output /results/dp4.json'
    fi
    if [[ "$action" == c1a-vjp-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-c1a-vjp.XXXXXX")"
      echo "Report: $report_dir/c1a-vjp.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && /opt/pypto/venv/bin/python \
          /workspace/online-speculation/scripts/validate_dsv41_pypto_c1a_vjp.py \
          --device "$@" --output /results/c1a-vjp.json; status=$?; \
          cat /results/c1a-vjp.json 2>/dev/null || true; exit $status' _ "$runtime_device" "$@"
    fi
    if [[ "$action" == expert-capture-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-expert-capture.XXXXXX")"
      echo "Report: $report_dir/expert-capture.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results")
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && /opt/pypto/venv/bin/python \
          /workspace/online-speculation/scripts/validate_dsv41_pypto_expert_capture.py \
          --device "$@" --output /results/expert-capture.json; status=$?; \
          cat /results/expert-capture.json 2>/dev/null || true; exit $status' _ "$runtime_device" "$@"
    fi
    if [[ "$action" == expert-vjp-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-expert-vjp.XXXXXX")"
      echo "Report: $report_dir/expert-vjp.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && /opt/pypto/venv/bin/python \
          /workspace/online-speculation/scripts/validate_dsv41_pypto_expert_vjp.py \
          --device "$@" --output /results/expert-vjp.json; status=$?; \
          cat /results/expert-vjp.json 2>/dev/null || true; exit $status' _ "$runtime_device" "$@"
    fi
    if [[ "$action" == expert-input-vjp-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-expert-input-vjp.XXXXXX")"
      echo "Report: $report_dir/expert-input-vjp.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && /opt/pypto/venv/bin/python \
          /workspace/online-speculation/scripts/validate_dsv41_pypto_expert_input_vjp.py \
          --device "$@" --output /results/expert-input-vjp.json; status=$?; \
          cat /results/expert-input-vjp.json 2>/dev/null || true; exit $status' _ "$runtime_device" "$@"
    fi
    if [[ "$action" == ep-vjp-check ]]; then
      report_dir="$(mktemp -d "$PWD/orthrus-ep-vjp.XXXXXX")"
      echo "Report: $report_dir/ep-vjp.json"
      options+=(-v "$root:/workspace/online-speculation:ro" -v "$report_dir:/results"
                -e PYTHONPATH=/workspace/online-speculation/src)
      exec docker run "${options[@]}" "$image" "${command_env[@]}" bash -lc \
        'bash /workspace/online-speculation/runtime/pypto_dsv41_a3/docker/cache-pto-isa.sh && /opt/pypto/venv/bin/python \
          /workspace/online-speculation/scripts/validate_dsv41_pypto_ep_vjp.py \
          --device "$@" --output /results/ep-vjp.json; status=$?; \
          cat /results/ep-vjp.json 2>/dev/null || true; exit $status' _ "$runtime_device" "$@"
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
    if [[ "$action" == fabric-check || "$action" == cluster-check || "$action" == cluster-accept ]]; then
      for name in HCCL_IF_IP HCCL_SOCKET_IFNAME HCCL_IF_BASE_PORT HCCL_HOST_SOCKET_PORT_RANGE; do
        if [[ -n "${!name:-}" ]]; then options+=(-e "$name=${!name}"); fi
      done
      if [[ -f /etc/hccn.conf ]]; then options+=(-v /etc/hccn.conf:/etc/hccn.conf:ro); fi
      logs_root="$PWD"
      if [[ "$action" == cluster-accept ]]; then logs_root="${PYPTO_RESULTS_ROOT:-/workspace}"; fi
      mkdir -p "$logs_root"
      log_dir="$(mktemp -d "$logs_root/orthrus-${action%-check}.XXXXXX")"
      echo "CANN logs: $log_dir" >&2
      options+=(-v "$root:/workspace/online-speculation:ro"
                -v "$log_dir:/cann-logs" -e ASCEND_PROCESS_LOG_PATH=/cann-logs -e ASCEND_MODULE_LOG_LEVEL=HCCL=1)
    fi
    if [[ "$action" == fabric-check ]]; then
      exec docker run "${options[@]}" "$image" "${command_env[@]}" timeout -k 5s 180s python \
        /workspace/online-speculation/scripts/probe_dsv41_pypto_fabric.py "$@" --device "$runtime_device"
    fi
    if [[ "$action" == cluster-check || "$action" == cluster-accept ]]; then
      key_file="${CLUSTER_KEY_FILE:-$HOME/.orthrus-cluster.key}"
      [[ -f "$key_file" && -r "$key_file" ]] || { echo "Missing cluster key file: $key_file" >&2; exit 2; }
      key_dir="$(cd "$(dirname "$key_file")" && pwd -P)"
      options+=(-v "$key_dir/$(basename "$key_file"):/run/orthrus-cluster.key:ro"
                -e PYTHONPATH=/workspace/online-speculation/src)
      if [[ -n "${PYPTO_REUSE_PREVIOUS_HEAD:-}" ]]; then
        options+=(-e "PYPTO_REUSE_PREVIOUS_HEAD=$PYPTO_REUSE_PREVIOUS_HEAD")
      fi
      control_only=false
      for arg in "$@"; do
        if [[ "$arg" == --control-only ]]; then control_only=true; fi
      done
      if [[ "$action" == cluster-accept ]]; then
        if "$control_only"; then echo 'cluster-accept requires real devices.' >&2; exit 2; fi
        export NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
      fi
      if "$control_only"; then
        options+=(-e ASCEND_RT_VISIBLE_DEVICES=)
      else
        : "${NPU_DEVICES:?Set NPU_DEVICES to the reserved physical cards, e.g. 0,1,2,3,4,5,6,7}"
        [[ "$NPU_DEVICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || { echo "Invalid NPU_DEVICES list" >&2; exit 2; }
        IFS=, read -ra cards <<< "$NPU_DEVICES"
        if [[ "$action" == cluster-accept && "${#cards[@]}" != 8 ]]; then
          echo 'cluster-accept requires exactly eight reserved cards on each host.' >&2
          exit 2
        fi
        visible=""
        seen=","
        index=0
        for card in "${cards[@]}"; do
          if [[ "$seen" == *",$card,"* ]]; then
            echo "NPU_DEVICES contains repeated cards" >&2
            exit 2
          fi
          seen+="$card,"
          options+=(--device "/dev/davinci$card:/dev/davinci$card")
          visible="${visible:+$visible,}$index"
          index=$((index + 1))
        done
        options+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
                  -e "ASCEND_RT_VISIBLE_DEVICES=$visible"
                  -e "ORTHRUS_CLUSTER_PHYSICAL_CARDS=$NPU_DEVICES")
      fi
      if [[ "$action" == cluster-accept ]]; then
        echo "Acceptance reports: $log_dir/acceptance/acceptance.json" >&2
        state_dir="${PYPTO_CLUSTER_STATE_DIR:-$logs_root/.orthrus-dsv41-state}"
        mkdir -p "$state_dir"
        state_dir="$(cd "$state_dir" && pwd -P)"
        options+=(-v "$log_dir:/results" -v "$state_dir:/cluster-state" -e PYPTO_A3_STATE_DIR=/cluster-state)
        if [[ "${PYPTO_DSV41_JIT_CACHE:-1}" == 1 ]]; then
          options+=(-e PYPTO_DSV41_JIT_CACHE_DIR=/cluster-state/orthrus-jit)
        fi
        if [[ -n "${ORTHRUS_WORKLOAD:-}" ]]; then
          workload="$(cd "$(dirname "$ORTHRUS_WORKLOAD")" && pwd -P)/$(basename "$ORTHRUS_WORKLOAD")"
          [[ -f "$workload" && -r "$workload" ]] || {
            echo "Unreadable ORTHRUS_WORKLOAD: $workload" >&2
            exit 2
          }
          options+=(-v "$workload:/run/orthrus-workload.jsonl:ro")
          set -- "$@" --workload /run/orthrus-workload.jsonl
        fi
        if [[ -z "${DRAFT_HEAD:-}" ]]; then
          previous_head=""
          for previous in "$logs_root"/orthrus-cluster-accept.*/initialized-head; do
            [[ -f "$previous/manifest.json" ]] || continue
            if [[ -z "$previous_head" || "$previous/manifest.json" -nt "$previous_head/manifest.json" ]]; then
              previous_head="$previous"
            fi
          done
          if [[ -n "$previous_head" ]]; then
            previous_head="$(cd "$previous_head" && pwd -P)"
            options+=(-v "$previous_head:/previous-head:ro" -e PYPTO_PREVIOUS_HEAD=/previous-head)
          fi
        fi
        set +e
        docker run "${options[@]}" "$image" bash \
          /workspace/online-speculation/scripts/run_dsv41_pypto_cluster.sh "$@" \
          --node "$rank" --master-address "$master_address" --key-file /run/orthrus-cluster.key \
          2>&1 | tee "$log_dir/cluster-accept.log"
        acceptance_status="${PIPESTATUS[0]}"
        set -e
        if [[ -s "$log_dir/acceptance/online-profile-dispatches.jsonl" ]]; then
          python3 "$root/scripts/summarize_dsv41_pypto_cluster.py" "$log_dir/acceptance" --online-kernels \
            --output "$log_dir/acceptance/online-profile-kernels.json" \
            || echo "Operator trace summary unavailable. Raw dispatch timings are saved." >&2
        fi
        if [[ -f "$log_dir/acceptance/acceptance.json" ]] && \
           grep -q '"divergence"' "$log_dir/acceptance/acceptance.json"; then
          echo "Automatic failure diagnosis:" >&2
          docker run --rm --network none \
            -v "$log_dir/acceptance:/acceptance:ro" \
            -v "$root/scripts:/tools:ro" \
            "$image" /opt/pypto/venv/bin/python \
            /tools/summarize_dsv41_pypto_cluster.py /acceptance --failure-diagnosis || true
        fi
        exit "$acceptance_status"
      fi
      docker run "${options[@]}" "$image" timeout -k 5s 15m python \
        /workspace/online-speculation/scripts/validate_dsv41_pypto_cluster.py --timeout 120 --port 29641 "$@" \
        --node "$rank" --master-address "$master_address" --key-file /run/orthrus-cluster.key \
        2>&1 | tee "$log_dir/probe.log"
      exit
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
    echo "Usage: $0 {build|refresh-image|save FILE|load FILE|host-info|doctor|preflight|compile-probe|probe|device-map|shell|attention-check|online-kernel-check|hc-collapse-vjp-check|hc-pre-vjp-check|resident-publish-check|online-factory-check|online-ready|multicard-check|c1a-vjp-check|expert-capture-check|expert-vjp-check|expert-input-vjp-check|ep-vjp-check|handoff-check|fabric-check|cluster-check|cluster-accept|init-head|gsm8k-prepare|gsm8k-summary}"
    echo "Company checks: $0 {fabric-check|cluster-check|cluster-accept} {node7|node8}"
    echo "Offline online-update compilation check: $0 online-abi-check"
    ;;
esac
