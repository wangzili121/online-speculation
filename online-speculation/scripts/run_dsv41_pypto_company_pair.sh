#!/usr/bin/env bash
# Launch the company pair from node7 over an existing SSH connection to node8.
set -euo pipefail

mode="${1:-}"
last="${2:-1}"
case "$mode" in
  prepare|smoke|study) ;;
  *) echo "Usage: $0 {prepare|smoke|study} [last-N]" >&2; exit 2 ;;
esac
[[ "$last" =~ ^([1-9]|1[0-9]|2[0-6])$ ]] || { echo 'last-N must be 1–26.' >&2; exit 2; }

checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner=online-speculation/scripts/run_dsv41_pypto_company_staged.sh
peer="${NODE8_SSH:-root@7.150.0.37}"
peer_checkout="${NODE8_CHECKOUT:-$checkout}"
ssh_peer=(ssh -o BatchMode=yes -o ConnectTimeout=8 -- "$peer")

printf -v peer_git '%q ' git -C "$peer_checkout" rev-parse HEAD
peer_commit="$("${ssh_peer[@]}" "$peer_git")"
commit="$(git -C "$checkout" rev-parse HEAD)"
[[ "$commit" == "$peer_commit" ]] || {
  echo "Checkouts differ: node7=$commit, node8=$peer_commit. Update both before launching." >&2
  exit 2
}

shared_env=("PYPTO_PACK_DECODE=${PYPTO_PACK_DECODE:-0}")
for key in PYPTO_IMAGE PYPTO_FORWARD_PROBE PYPTO_TEACHER_PROBE RANK0_IP ORTHRUS_CLUSTER_PORT HCCL_CONNECT_TIMEOUT \
    HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE; do
  if [[ -n "${!key:-}" ]]; then shared_env+=("$key=${!key}"); fi
done
if [[ -n "${NODE8_NPU_DEVICES:-}" ]]; then shared_env+=("NPU_DEVICES=$NODE8_NPU_DEVICES"); fi

node8_command() {
  printf -v peer_command '%q ' env "${shared_env[@]}" "$@" bash "$peer_checkout/$runner" node8 "$mode" "$last"
}

if [[ "$mode" == prepare ]]; then
  bash "$checkout/$runner" node7 prepare "$last"
  node8_command
  exec "${ssh_peer[@]}" "$peer_command"
fi

node8_command PYPTO_CHECK_ONLY=1
PYPTO_CHECK_ONLY=1 bash "$checkout/$runner" node7 "$mode" "$last"
"${ssh_peer[@]}" "$peer_command"
if [[ "${PYPTO_CHECK_ONLY:-0}" == 1 ]]; then exit; fi

tag="orthrus-dsv41-$mode-$(date +%s)-$$"
node7_pid=
node8_pid=
cleanup() {
  trap - EXIT INT TERM
  for pid in "$node7_pid" "$node8_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  docker stop -t 3 "$tag" >/dev/null 2>&1 || true
  "${ssh_peer[@]}" "docker stop -t 3 $tag" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

node8_command PYPTO_CHECK_ONLY=0 "PYPTO_RUN_NAME=$tag"
(
  exec "${ssh_peer[@]}" "$peer_command"
) > >(awk '{print "[node8] " $0; fflush()}') 2>&1 &
node8_pid=$!
PYPTO_CHECK_ONLY=0 PYPTO_RUN_NAME="$tag" bash "$checkout/$runner" node7 "$mode" "$last" \
  > >(awk '{print "[node7] " $0; fflush()}') 2>&1 &
node7_pid=$!

# Catch either startup failure even if the other worker is waiting at rendezvous.
while [[ -n "$node7_pid" || -n "$node8_pid" ]]; do
  if [[ -n "$node7_pid" ]] && ! kill -0 "$node7_pid" 2>/dev/null; then
    if wait "$node7_pid"; then node7_pid=; else echo 'node7 failed.' >&2; exit 1; fi
  fi
  if [[ -n "$node8_pid" ]] && ! kill -0 "$node8_pid" 2>/dev/null; then
    if wait "$node8_pid"; then node8_pid=; else echo 'node8 failed.' >&2; exit 1; fi
  fi
  if [[ -n "$node7_pid" || -n "$node8_pid" ]]; then sleep 1; fi
done
trap - EXIT INT TERM
