#!/usr/bin/env bash

configure_dsv4_profile() {
  case "${PROFILE:-prod-128k}" in
    prod-128k)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-133120}"
      BLOCK_SIZE="${BLOCK_SIZE:-128}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
      ;;
    prod-1m)
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
      BLOCK_SIZE="${BLOCK_SIZE:-32}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
      MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-10240}"
      export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-4096}"
      ;;
    *)
      echo "PROFILE must be prod-128k or prod-1m" >&2
      return 1
      ;;
  esac
  export MAX_MODEL_LEN BLOCK_SIZE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
}

configure_dsv4_runtime() {
  if [[ "${VLLM_BATCH_INVARIANT:-0}" == "1" ]]; then
    # The pinned multi-stream prolog uses different norm/quant paths in
    # prefill and decode. Use its existing single-stream path for exact checks.
    ADDITIONAL_CONFIG="$("${PYTHON_BIN:-python3}" - "${ADDITIONAL_CONFIG:-}" <<'PY'
import json
import sys
config = json.loads(sys.argv[1] or "{}")
config["multistream_dsv4_dsa_overlap"] = False
print(json.dumps(config))
PY
)"
    export ADDITIONAL_CONFIG
  fi
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
  export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
  export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}"
  if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
}
