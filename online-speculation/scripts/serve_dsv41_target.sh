#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/dsv41_service_env.sh"

: "${MODEL:=/data/disk/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8}"
PORT="${PORT:-8900}"
DP_RPC_PORT="${DP_RPC_PORT:-13399}"
MEMORY="${MEMORY:-0.90}"
default_additional_config='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}'
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-$default_additional_config}"

configure_dsv41_profile
configure_dsv41_runtime
configure_dsv41_network

execution_args=()
case "${TARGET_EXECUTION_MODE:-eager}" in
  eager)
    execution_args+=(--enforce-eager)
    compilation_config='{"cudagraph_mode":"NONE"}'
    ;;
  graph)
    compilation_config='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ;;
  *)
    echo "TARGET_EXECUTION_MODE must be eager or graph" >&2
    exit 1
    ;;
esac

command=(
  vllm serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
)
if [[ "$NODE_RANK" != "0" ]]; then
  command+=(--headless --data-parallel-start-rank "$NODE_RANK")
fi
command+=(
  --data-parallel-address "$NODE0_IP"
  --data-parallel-rpc-port "$DP_RPC_PORT"
  --data-parallel-size 2
  --data-parallel-size-local 1
  --tensor-parallel-size 8
  --enable-expert-parallel
  --seed 1024
  --served-model-name dsv41-target
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --no-enable-prefix-caching
  --trust-remote-code
  --tokenizer-mode deepseek_v41
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --safetensors-load-strategy lazy
  --gpu-memory-utilization "$MEMORY"
  --quantization ascend
  --no-swa-bounded-replay
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}'
  "${execution_args[@]}"
  --compilation-config "$compilation_config"
  --additional-config "$ADDITIONAL_CONFIG"
)
run_dsv41_command "${command[@]}"
