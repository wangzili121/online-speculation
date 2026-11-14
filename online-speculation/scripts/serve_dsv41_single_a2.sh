#!/usr/bin/env bash
# Experimental one-node A2 smoke. The qualified production topology remains two nodes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
source "$repo_root/scripts/dsv41_service_env.sh"

: "${MODEL:=/data/disk/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8}"
MODE="${MODE:-target}"
HEAD="${HEAD:-/data/disk/models/orthrus-dsv41-head-v1}"
PORT="${PORT:-8900}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MEMORY="${MEMORY:-0.88}"
DRAFT_SAMPLE_METHOD="${ORTHRUS_DRAFT_SAMPLE_METHOD:-greedy}"
default_additional_config='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_flashcomm1":false,"enable_shared_expert_dp":false}'
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-$default_additional_config}"

case "$MODE" in
  target|orthrus) ;;
  *) echo "MODE must be target or orthrus" >&2; exit 1 ;;
esac
case "$DRAFT_SAMPLE_METHOD" in
  greedy|probabilistic) ;;
  *) echo "ORTHRUS_DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2; exit 1 ;;
esac

configure_dsv41_runtime
if [[ -n "${LOCAL_IP:-}" ]]; then
  export HCCL_IF_IP="$LOCAL_IP"
fi
if [[ -n "${NIC_NAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME="$NIC_NAME"
  export TP_SOCKET_IFNAME="$NIC_NAME"
  export HCCL_SOCKET_IFNAME="$NIC_NAME"
fi

command=(
  vllm serve "$MODEL"
  --host "${HOST:-0.0.0.0}"
  --port "$PORT"
  --tensor-parallel-size 8
  --enable-expert-parallel
  --seed 1024
  --served-model-name "dsv41-$MODE-single-a2"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size 128
  --no-enable-prefix-caching
  --trust-remote-code
  --language-model-only
  --tokenizer-mode deepseek_v41
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --safetensors-load-strategy lazy
  --gpu-memory-utilization "$MEMORY"
  --quantization ascend
  --no-swa-bounded-replay
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
  --enforce-eager
  --compilation-config '{"cudagraph_mode":"NONE"}'
  --additional-config "$ADDITIONAL_CONFIG"
)

if [[ "$MODE" == "orthrus" ]]; then
  if [[ "${DRY_RUN:-0}" != "1" && "${AUTO_INSTALL_PLUGIN:-1}" == "1" ]]; then
    "$python_bin" -m pip install --no-deps --no-build-isolation --disable-pip-version-check -e "$repo_root"
  fi
  export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,ascend_model,ascend_model_loader,blockspec_orthrus}"
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
  preflight=(
    "$python_bin" "$repo_root/scripts/preflight_dsv41_orthrus.py"
    --model "$MODEL"
    --head "$HEAD"
    --require-plugin
  )
  if [[ "${REQUIRE_TRAINED_HEAD:-0}" == "1" ]]; then
    preflight+=(--require-trained)
  fi
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${preflight[@]}"
  fi
  speculative_config="$($python_bin - "$HEAD" "$DRAFT_SAMPLE_METHOD" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "method": "dflash",
    "num_speculative_tokens": 31,
    "draft_sample_method": sys.argv[2],
    "rejection_sample_method": "standard",
    "enforce_eager": True,
}, separators=(",", ":")))
PY
)"
  command+=(--speculative-config "$speculative_config")
fi

run_dsv41_command "${command[@]}"
