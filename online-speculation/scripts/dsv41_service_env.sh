#!/usr/bin/env bash

configure_dsv41_profile() {
  case "${PROFILE:-smoke-16k}" in
    smoke-16k)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
      ;;
    prod-128k)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-133120}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
      ;;
    prod-1m)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
      ;;
    *)
      echo "PROFILE must be smoke-16k, prod-128k or prod-1m" >&2
      return 1
      ;;
  esac
  BLOCK_SIZE="${BLOCK_SIZE:-128}"
  export MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS BLOCK_SIZE
}

configure_dsv41_runtime() {
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-36000}"
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}"
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
  export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
  export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
}

configure_dsv41_network() {
  : "${NODE_RANK:?set NODE_RANK to 0 or 1}"
  : "${NODE0_IP:?set NODE0_IP to the API node address}"
  : "${LOCAL_IP:?set LOCAL_IP to this node's HCCL address}"
  : "${NIC_NAME:?set NIC_NAME to this node's HCCL interface}"
  if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
    echo "NODE_RANK must be 0 or 1 for the qualified A2 DP2/TP8 topology" >&2
    return 1
  fi
  export HCCL_IF_IP="$LOCAL_IP"
  export GLOO_SOCKET_IFNAME="$NIC_NAME"
  export TP_SOCKET_IFNAME="$NIC_NAME"
  export HCCL_SOCKET_IFNAME="$NIC_NAME"
}

run_dsv41_command() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  exec "$@"
}
