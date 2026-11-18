#!/usr/bin/env bash
# Run on node 0. Transfer this checkout to node 1 over an existing SSH login.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  up|probe|status|down) action="$1" ;;
  *) echo "Usage: $0 {up|probe|status|down}  # set NODE0_IP, NODE1_IP and NODE1_SSH" >&2; exit 2 ;;
esac
: "${NODE0_IP:?set the internal IPv4 address of node 0}"
: "${NODE1_IP:?set the internal IPv4 address of node 1}"
: "${NODE1_SSH:?set the SSH login for node 1, e.g. myuser@10.0.0.2}"
[[ "$NODE0_IP" != "$NODE1_IP" ]] || { echo "Node IPs must differ" >&2; exit 2; }
mode="${MODE:-orthrus}"
case "$action:$mode" in
  *:orthrus|*:target|status:probe|down:probe) ;;
  *) echo "MODE must be orthrus or target (probe also allowed for status/down)" >&2; exit 2 ;;
esac
[[ "$action" != probe ]] || mode=probe
run_name="${RUN_NAME:-dsv41-$mode}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_NAME" >&2; exit 2; }

remote() { ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$NODE1_SSH" "$@"; }
quote() { printf '%q ' "$@"; }

if [[ "$action" == status || "$action" == down ]]; then
  operation='
name="$1"; action="$2"
if ! docker container inspect "$name" >/dev/null 2>&1; then
  echo "$name absent"
  exit 0
fi
if [[ "$(docker inspect --format '"'"'{{index .Config.Labels "org.online-speculation.service"}}'"'"' "$name")" != dsv41-a2 ]]; then
  echo "$name belongs to another deployment" >&2
  exit 2
fi
if [[ "$action" == down ]]; then
  docker rm -f "$name"
else
  docker inspect --format "{{.Name}} {{.State.Status}} exit={{.State.ExitCode}}" "$name"
fi'
  # Check both even if one node is unreachable.
  status=0
  bash -c "$operation" bash "$run_name-node0" "$action" || status=1
  remote "$(quote bash -c "$operation" bash "$run_name-node1" "$action")" || status=1
  if [[ "$action" == status && "$mode" != probe ]] && command -v curl >/dev/null; then
    curl -fsS --max-time 3 "http://$NODE0_IP:${PORT:-8901}/health" && echo "API ready" || echo "API not ready"
  fi
  exit "$status"
fi

nic_script='ip -4 route get "$1" | awk '"'"'{for (i=1; i<NF; i++) if ($i=="dev") {print $(i+1); exit}}'"'"
nic0="${NIC0:-}"
nic1="${NIC1:-}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  nic0="${nic0:-NIC0}"
  nic1="${nic1:-NIC1}"
else
  : "${ACK_EXCLUSIVE_NODE:?set ACK_EXCLUSIVE_NODE=1 after reserving both nodes}"
  [[ "$ACK_EXCLUSIVE_NODE" == 1 ]] || { echo "ACK_EXCLUSIVE_NODE must be 1" >&2; exit 2; }
  [[ -n "$nic0" ]] || nic0="$(bash -c "$nic_script" bash "$NODE1_IP")"
  [[ -n "$nic1" ]] || nic1="$(remote "$(quote bash -c "$nic_script" bash "$NODE0_IP")")"
  [[ -n "$nic0" && -n "$nic1" ]] || { echo "could not infer an interface. Set NIC0 and NIC1." >&2; exit 2; }
fi
defaults=(
  "MODE=$mode" "NODE0_IP=$NODE0_IP" "DRY_RUN=${DRY_RUN:-0}"
  "ACK_EXCLUSIVE_NODE=${ACK_EXCLUSIVE_NODE:-0}"
  "MODEL_HOST=${MODEL_HOST:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8}"
  "HEAD_HOST=${HEAD_HOST:-/workspace/models/orthrus-dsv41-head-v1}"
  "INIT_HEAD=${INIT_HEAD:-1}" "REQUIRE_TRAINED_HEAD=${REQUIRE_TRAINED_HEAD:-0}"
  "PROFILE=${PROFILE:-smoke-16k}" "PORT=${PORT:-8901}" "DP_RPC_PORT=${DP_RPC_PORT:-13400}"
  "HCCL_PROBE_PORT=${HCCL_PROBE_PORT:-13401}"
  "PINNED_LOCAL_IMAGE_ID=${PINNED_LOCAL_IMAGE_ID:-}"
)
for option in IMAGE_REPOSITORY EXPECTED_IMAGE_DIGEST SHM_SIZE MASK_TOKEN_ID; do
  [[ -z "${!option:-}" ]] || defaults+=("$option=${!option}")
done
node0=(env "${defaults[@]}" NODE_RANK=0 "LOCAL_IP=$NODE0_IP" "NIC_NAME=$nic0" "CONTAINER=$run_name-node0")
node1=(env "${defaults[@]}" NODE_RANK=1 "LOCAL_IP=$NODE1_IP" "NIC_NAME=$nic1" "CONTAINER=$run_name-node1")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  "${node0[@]}" bash "$root/scripts/deploy_dsv41_node.sh"
  "${node1[@]}" bash "$root/scripts/deploy_dsv41_node.sh"
  exit 0
fi

# Stage source only. Both nodes use their existing checkpoint and Docker image.
remote_stage="$(remote 'mkdir -p "$HOME/.local/share/orthrus-dsv41" && mktemp -d "$HOME/.local/share/orthrus-dsv41/run.XXXXXXXX"')"
[[ "$remote_stage" == /* ]] || { echo "invalid staging path received from node 1" >&2; exit 2; }
COPYFILE_DISABLE=1 tar -C "$root" -czf - scripts src README.md pyproject.toml |
  remote "$(quote tar -xzf - -C "$remote_stage")"

"${node0[@]}" bash "$root/scripts/deploy_dsv41_node.sh"
if ! remote "$(quote "${node1[@]}" bash "$remote_stage/scripts/deploy_dsv41_node.sh")"; then
  docker stop "$run_name-node0" >/dev/null
  echo "Node 1 failed to start. Node 0 was stopped. Logs are available via docker logs." >&2
  exit 1
fi
if [[ "$mode" == probe ]]; then
  remote_wait="$(quote docker wait "$run_name-node1")"
  left="$(docker wait "$run_name-node0")"
  right="$(remote "$remote_wait")"
  docker logs "$run_name-node0"
  remote "$(quote docker logs "$run_name-node1")"
  if [[ "$left" != 0 || "$right" != 0 ]]; then
    echo "HCCL probe failed on node 0 ($left) or node 1 ($right)" >&2
    exit 1
  fi
  docker rm "$run_name-node0" >/dev/null
  remote "$(quote docker rm "$run_name-node1")" >/dev/null
  echo "HCCL passed across all 16 ranks"
else
  echo "Check API readiness at http://$NODE0_IP:${PORT:-8901}/health"
fi
