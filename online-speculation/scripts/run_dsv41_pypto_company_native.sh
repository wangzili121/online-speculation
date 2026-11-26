#!/usr/bin/env bash
# Run the company 8+8 native whole-graph job from node7.
set -euo pipefail

mode="$1"
last="$2"
scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
checkout="$(cd "$scripts/../.." && pwd -P)"
peer_ip="${NODE8_HCCL_IP:-7.150.0.37}"
node7_ip="${RANK0_IP:-7.150.6.33}"
port="${PYPTO_NATIVE_PORT:-29653}"
image="${PYPTO_IMAGE:-orthrus-dsv41-pypto:1122-949a18d}"
peer_checkout="${NODE8_CHECKOUT:-$checkout}"
model="${DSV41_MODEL_DIR:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
workspace="${PYPTO_NATIVE_WORKSPACE:-/workspace/orthrus-dsv41-1122/native}"
loss="${PYPTO_NATIVE_LOSS:-tv}"
ssh_peer=(ssh -o BatchMode=yes -o ConnectTimeout=8 -- "${NODE8_SSH:-root@$peer_ip}")
[[ "$last" =~ ^([1-9]|1[0-9]|2[0-6])$ ]] || { echo 'last-N must be 1–26.' >&2; exit 2; }
[[ "$loss" == tv || "$loss" == forward_kl ]] || { echo 'Use tv or forward_kl.' >&2; exit 2; }
bundle="$workspace/last-$last-$loss"

printf -v remote '%q ' git -C "$peer_checkout" rev-parse HEAD
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$("${ssh_peer[@]}" "$remote")" ]] ||
  { echo "Update both checkouts to the same commit." >&2; exit 2; }
# Reject an older image with the same tag before compiling or loading weights.
image_check='
    cd /opt/dsv41-1122
    for script in run_dsv41_pypto_native_ar.py compile_dsv41_pypto_native_online.py validate_dsv41_pypto_packed_generation.py; do
        cmp "/checkout/scripts/$script" "$script"
    done
    for source in /checkout/src/blockspec/{dsv41_native_*,dsv41_packed_*,pypto_dsv41_packed_*}.py; do
        cmp "$source" "orthrus/blockspec/${source##*/}"
    done
    /opt/pypto/venv/bin/python -c "from _task_interface import MAX_REGISTERED_CALLABLE_IDS as limit; assert limit >= 256"
'
check=(docker run --rm --network none --entrypoint /bin/bash)
if ! "${check[@]}" -v "$checkout/online-speculation:/checkout:ro" "$image" -ec "$image_check"; then
  echo "Refresh the 1122 image on node7 before running $mode." >&2
  exit 2
fi
printf -v remote '%q ' "${check[@]}" -v "$peer_checkout/online-speculation:/checkout:ro" "$image" -ec "$image_check"
if ! "${ssh_peer[@]}" "$remote"; then
  echo "Refresh the 1122 image on node8 before running $mode." >&2
  exit 2
fi
mkdir -p "$workspace"

base=(docker run --rm --network none -v "$model:$model:ro")
if [[ "$mode" == native-prepare ]]; then
  if [[ ! -d "$bundle" ]]; then
    tmp="$(mktemp -d "$workspace/.prepare.XXXXXX")"
    echo "Compile log: $tmp/build.log"
    "${base[@]}" -v "$tmp:/bundle" --entrypoint /bin/bash "$image" -lc '
      set -e
      source /opt/pypto/bin/pypto-activate.sh
      export PYTHONPATH="/opt/dsv41-1122/pypto-lib:/opt/dsv41-1122/orthrus:$PYTHONPATH"
      python -c "from models.deepseek_v4_1_flash_a3.driver import compile_generation_graphs as compile; compile(
          \"/bundle/graphs/prefill\", \"/bundle/graphs/decode\", devices=\",\".join(map(str, range(16))),
          context=512, prefill_capacity=272, prefill_slab_rows=64)"
      python /opt/dsv41-1122/compile_dsv41_pypto_packed_draft.py --block 16 --context 512 \
        --last-layers "$1" --decode-capacity 16 --artifact-dir /bundle/draft
      python /opt/dsv41-1122/compile_dsv41_pypto_native_online.py \
        --draft-dir /bundle/draft --artifact-dir /bundle/online --loss "$2"
      python /opt/dsv41-1122/export_dsv41_orthrus_head.py --model "$3" \
        --from-target --mask-token-id 128803 --output /bundle/head
    ' compile "$last" "$loss" "$model" >"$tmp/build.log" 2>&1 || { tail -n 25 "$tmp/build.log" >&2; exit 1; }
    mv "$tmp" "$bundle"
  fi
  # Distribute the compiler artifacts before node7 adds its binary assembly cache.
  printf -v remote '%q ' bash -c '
    set -euo pipefail
    bundle="$1"
    if [[ -d "$bundle" ]]; then
      echo "Peer bundle already exists at $bundle" >&2
      exit 0
    fi
    mkdir -p "$(dirname "$bundle")"
    tmp="$(mktemp -d "$(dirname "$bundle")/.receive.XXXXXX")"
    tar -C "$tmp" -xf -
    mv "$tmp" "$bundle"
  ' stage "$bundle"
  printf -v has_bundle '%q ' test -d "$bundle"
  if ! "${ssh_peer[@]}" "$has_bundle"; then
    tar -C "$bundle" -cf - graphs draft online head | "${ssh_peer[@]}" "$remote"
  fi
  echo "CPU assembly log: $bundle/callable-assembly.log"
  "${base[@]}" -v "$bundle:$bundle" "$image" \
    /opt/dsv41-1122/run_dsv41_pypto_native_ar.py --assemble-only \
    --graphs "$bundle/graphs" --draft-graph "$bundle/draft" --head "$bundle/head" \
    --online-artifacts "$bundle/online" --checkpoint "$model" --loss "$loss" \
    --output "$bundle/callable-assembly.json" >"$bundle/callable-assembly.log" 2>&1 ||
    { tail -n 25 "$bundle/callable-assembly.log" >&2; exit 1; }
  echo "Native last-$last artifacts: $bundle"
  exit
fi

[[ -r "$bundle/online/online_manifest.json" ]] ||
  { echo "Prepare native last-$last artifacts first." >&2; exit 2; }
run="$(mktemp -d "$workspace/$mode.XXXXXX")"
echo "Reports: $run"
tokens=8
width="${PYPTO_NATIVE_WIDTH:-8}"
if [[ "$mode" == native-study ]]; then
  data="${PYPTO_NATIVE_GSM8K_DIR:-/workspace/data/orthrus-dsv41-gsm8k}"
  if [[ ! -f "$data/manifest.json" ]]; then
    prepare_data=(python3 "$scripts/prepare_dsv41_gsm8k.py" --output "$data")
    if [[ -n "${PYPTO_NATIVE_GSM8K_RAW:-}" ]]; then
      prepare_data+=(--source-directory "$PYPTO_NATIVE_GSM8K_RAW")
    fi
    "${prepare_data[@]}" >"$run/dataset.log"
  fi
  head -n "${PYPTO_NATIVE_TRAIN_ROWS:-64}" "$data/tuning.jsonl" >"$run/requests.jsonl"
  head -n "${PYPTO_NATIVE_EVAL_ROWS:-16}" "$data/quality_test.jsonl" >"$run/heldout.jsonl"
  tokens="${PYPTO_NATIVE_TOKENS:-64}"
else
  printf '%s\n' '{"id":"dp0","prompt":"What is 2 plus 3? Answer in one line."}' \
    '{"id":"dp1","prompt":"A box holds four apples. If I add three, how many are there?"}' \
    '{"id":"dp2","prompt":"Solve 12 divided by 3. State the result."}' \
    '{"id":"dp3","prompt":"Compute 14 minus 5."}' >"$run/requests.jsonl"
fi

flags=(--graphs "$bundle/graphs" --draft-graph "$bundle/draft" --head "$bundle/head"
  --online-artifacts "$bundle/online" --checkpoint "$model" --loss "$loss"
  --workload /run-data/requests.jsonl --tokens "$tokens"
  --width "$width" --teacher-rows "${PYPTO_NATIVE_TEACHER_ROWS:-7}"
  --update-every "${PYPTO_NATIVE_UPDATE_EVERY:-16}" --learning-rate "${PYPTO_NATIVE_LR:-1e-5}"
  --warmup-updates "${PYPTO_NATIVE_WARMUP_UPDATES:-8}")
if [[ "$mode" == native-study ]]; then
  flags+=(--study --heldout-workload /run-data/heldout.jsonl --concurrency 4
          --warmup-waves "${PYPTO_NATIVE_WARMUP_WAVES:-1}")
fi

# Check the local workload before either host starts a device worker.
"${base[@]}" -v "$bundle:$bundle:ro" -v "$run:/run-data" "$image" \
  /opt/dsv41-1122/run_dsv41_pypto_native_ar.py "${flags[@]}" --contract-only \
  --output /run-data/contract.json >"$run/contract.log"
identity="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["identity"])' "$run/contract.json")"
printf -v remote '%q ' "${base[@]}" -v "$bundle:$bundle:ro" --entrypoint /bin/bash "$image" -lc '
    set -e
    source /opt/pypto/bin/pypto-activate.sh
    exec python -c "
from pathlib import Path
import sys
from blockspec.dsv41_native_forward import ForwardArtifacts

root, model, loss, expected = sys.argv[1:]
bundle = Path(root)
artifacts = ForwardArtifacts(bundle / \"graphs\", model, 1,
                             draft_directory=bundle / \"draft\", head=bundle / \"head\",
                             online_directory=bundle / \"online\", loss=loss)
actual = artifacts.identity.hex()
if actual != expected:
    raise SystemExit(f\"Native artifacts differ: node7={expected}, node8={actual}\")
print(actual)
    " "$@"
' verify "$bundle" "$model" "$loss" "$identity"
if ! "${ssh_peer[@]}" "$remote" >"$run/node8-contract.log" 2>&1; then
    tail -n 16 "$run/node8-contract.log" >&2
    exit 2
fi

if [[ "$mode" == native-windows ]]; then
  flags+=(--windows-only)
fi

check_npus='
    set -e
    smi="$(command -v npu-smi || printf %s /usr/local/bin/npu-smi)"
    table="$("$smi" info)"
    for dev in {0..7}; do
        [[ "$table" == *"No running processes found in NPU $dev "* ]] || {
            echo "NPU $dev must be idle for the native EP16 job." >&2; exit 1;
        }
    done
'
bash -c "$check_npus"
printf -v remote '%q ' bash -c "$check_npus"
"${ssh_peer[@]}" "$remote"
container=(docker run --network host --shm-size 4g --init
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v "$model:$model:ro")
for dev in {0..7}; do container+=(--device "/dev/davinci$dev"); done
tag="orthrus-dsv41-${run##*/}"
cleanup() {
  docker stop -t 3 "$tag-node7" >/dev/null 2>&1 || true
  "${ssh_peer[@]}" "docker logs $tag-node8" >"$run/node8.log" 2>&1 || true
  "${ssh_peer[@]}" "docker rm -f $tag-node8" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf -v remote '%q ' "${container[@]}" --detach --name "$tag-node8" -e "HCCL_IF_IP=$peer_ip" \
  -v "$bundle:$bundle:ro" \
  --entrypoint /bin/bash "$image" -lc \
  'source /opt/pypto/bin/pypto-activate.sh; exec python -m simpler.remote_l3_worker --host "$1" --port "$2"' \
  daemon "$peer_ip" "$port"
"${ssh_peer[@]}" "$remote" >"$run/node8-container.txt"
python3 - "$peer_ip" "$port" <<'PY'
import socket
import sys
import time

address = (sys.argv[1], int(sys.argv[2]))
until = time.monotonic() + 20
while True:
    try:
        with socket.create_connection(address, timeout=1):
            break
    except OSError:
        if time.monotonic() >= until:
            raise
        time.sleep(.25)
PY
# PyPTO assembles chip callables on node7 and takes a write lock in the compiled graph cache.
"${container[@]}" --rm --name "$tag-node7" -e "HCCL_IF_IP=$node7_ip" -v "$bundle:$bundle" -v "$run:/run-data" "$image" \
  /opt/dsv41-1122/run_dsv41_pypto_native_ar.py "${flags[@]}" --output /run-data/result.json --endpoint "$peer_ip:$port" \
  --peer-listen-host "$peer_ip" 2>&1 | tee "$run/node7.log" | awk '/^\{"stage"/ {print; fflush()}' ||
  { tail -n 26 "$run/node7.log" >&2; exit 1; }
