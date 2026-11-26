#!/usr/bin/env bash
# Run the pinned Simpler device-to-device TLOAD probe from company node7.
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
checkout="$(cd "$scripts/../.." && pwd -P)"
peer_ip="${NODE8_HCCL_IP:-7.150.0.37}"
peer="${NODE8_SSH:-root@$peer_ip}"
peer_checkout="${NODE8_CHECKOUT:-$checkout}"
image="${PYPTO_IMAGE:-orthrus-dsv41-pypto:1122-949a18d}"
local_card="${NODE7_FABRIC_NPU:-1}"
peer_card="${NODE8_FABRIC_NPU:-0}"
port="${PYPTO_NATIVE_FABRIC_PORT:-29652}"
local_ip="${RANK0_IP:-7.150.6.33}"
ssh_peer=(ssh -o BatchMode=yes -o ConnectTimeout=8 -- "$peer")

[[ "$local_card" =~ ^[0-9]+$ && "$peer_card" =~ ^[0-9]+$ && "$port" =~ ^[0-9]{1,5}$ ]] &&
    ((10#$port >= 1024 && 10#$port <= 65535)) || {
    echo "Select one physical NPU per host and a TCP port between 1024 and 65535." >&2
    exit 2
}
root="${PYPTO_STAGED_WORKSPACE:-/workspace/orthrus-dsv41-1122}"
mkdir -p "$root"
reports="$(mktemp -d "$root/native-fabric.XXXXXX")"
echo "Native Fabric reports: $reports"

printf -v remote '%q ' env "PYPTO_IMAGE=$image" "NPU_DEVICE=$peer_card" bash \
    "$peer_checkout/online-speculation/scripts/dsv41_pypto_env.sh" global-window-check node8
"${ssh_peer[@]}" "$remote" >"$reports/node8-window.log" 2>&1 &
window_pid=$!
local_status=0
peer_status=0
PYPTO_IMAGE="$image" NPU_DEVICE="$local_card" bash "$scripts/dsv41_pypto_env.sh" \
    global-window-check node7 >"$reports/node7-window.log" 2>&1 || local_status=$?
wait "$window_pid" || peer_status=$?
for node in node7 node8; do
    echo "$node window export"
    tail -n 4 "$reports/$node-window.log"
done
if ((local_status || peer_status)); then
    echo "One or both hosts failed the device-window export required by this Simpler Fabric path." >&2
    exit 1
fi

tag="orthrus-native-fabric-${reports##*.}"
runtime=/opt/dsv41-1122/pypto/runtime
container=(docker run --network host --shm-size 1g --init --entrypoint /bin/bash
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -w "$runtime"
    -e ASCEND_RT_VISIBLE_DEVICES=0)

cleanup() {
    docker stop -t 2 "$tag-parent" >/dev/null 2>&1 || true
    "${ssh_peer[@]}" "docker logs $tag-peer" >"$reports/node8-peer.log" 2>&1 || true
    "${ssh_peer[@]}" "docker rm -f $tag-peer" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf -v remote '%q ' "${container[@]}" --detach --name "$tag-peer" \
    --device "/dev/davinci$peer_card" -e "HCCL_IF_IP=$peer_ip" "$image" -lc \
    'source /opt/pypto/bin/pypto-activate.sh >/dev/null; exec timeout -k 5s 300s python -m simpler.remote_l3_worker --host "$1" --port "$2"' \
    remote-worker "$peer_ip" "$port"
"${ssh_peer[@]}" "$remote" >/dev/null

# The daemon and its session listeners use node8's interconnect address.
if "${container[@]}" --rm --name "$tag-parent" --device "/dev/davinci$local_card" \
    -e "HCCL_IF_IP=$local_ip" "$image" -lc '
        set -e
        source /opt/pypto/bin/pypto-activate.sh >/dev/null
        python - "$1" "$2" <<'"'"'PY'"'"'
import socket
import sys
import time

address = (sys.argv[1], int(sys.argv[2]))
deadline = time.monotonic() + 20
while True:
    try:
        with socket.create_connection(address, timeout=1):
            break
    except OSError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(0.25)
PY
        exec timeout -k 5s 240s python -m examples.workers.l4.global_tload_mixed_l3.main \
            --remote "$1:$2" --local-devices 0 --remote-devices 0 \
            --session-listen-host "$1" --comm-profile a3-fabric-v1 --session-timeout 60
    ' parent "$peer_ip" "$port" >"$reports/node7-peer.log" 2>&1; then
    grep -qF 'global_tload_mixed_l3 passed' "$reports/node7-peer.log"
    grep -E 'max_diff=|global_tload_mixed_l3 passed' "$reports/node7-peer.log"
else
    tail -n 18 "$reports/node7-peer.log" >&2
    exit 1
fi
