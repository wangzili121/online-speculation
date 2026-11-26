#!/usr/bin/env bash
# Company entrypoint for the pinned 1122 dual-node W8A8 runner.
set -euo pipefail

case "${1:-}" in
  node7) rank=0; key=/root/.orthrus-cluster.key ;;
  node8) rank=1; key=/home/wangzili/.orthrus-cluster.key ;;
  *) printf 'Usage: %s {node7|node8} {prepare|smoke|study} [last-N]\n' "$0" >&2; exit 2 ;;
esac
mode="${2:-}"
last="${3:-1}"
[[ "$last" =~ ^([1-9]|1[0-9]|2[0-6])$ ]] || { echo 'last-N must be 1–26.' >&2; exit 2; }

image="${PYPTO_IMAGE:-orthrus-dsv41-pypto:1122-949a18d}"
model="${TARGET_WEIGHTS:-/workspace/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8}"
root="${PYPTO_STAGED_WORKSPACE:-/workspace/orthrus-dsv41-1122}"
dataset="${DSV41_GSM8K_ROOT:-/workspace/data/orthrus-dsv41-gsm8k}"
head="${DRAFT_HEAD:-$root/initialized-head}"
rank0_ip="${RANK0_IP:-7.150.6.33}"
port="${ORTHRUS_CLUSTER_PORT:-29641}"
key="${CLUSTER_KEY_FILE:-$key}"
geometry=(--context 512 --slots-per-dp 2)
docker image inspect "$image" >/dev/null 2>&1 || {
  echo "Missing $image. Build it with dsv41_pypto_env.sh build-1122 or load the image archive." >&2
  exit 2
}
[[ -r "$model/config.json" ]] || { echo "Missing checkpoint at $model." >&2; exit 2; }
mkdir -p "$root"
base=(--rm --shm-size "${SHM_SIZE:-16g}" -v "$model:/model:ro" -v "$root:/results")

if [[ "$mode" == prepare ]]; then
  if [[ ! -e "$head" ]]; then
    mkdir -p "$(dirname "$head")"
    docker run "${base[@]}" --network none -e OMP_NUM_THREADS=4 -v "$(dirname "$head"):/head-output" "$image" \
      /opt/dsv41-1122/export_dsv41_orthrus_head.py \
      --model /model --from-target --mask-token-id 129279 --output "/head-output/$(basename "$head")"
  fi
  [[ -r "$dataset/tuning.jsonl" ]] || { echo "Missing prepared GSM8K splits at $dataset." >&2; exit 2; }
  docker run "${base[@]}" --network none -v "$dataset:/gsm8k:ro" --entrypoint /bin/bash "$image" -lc '
    set -euo pipefail
    source /opt/pypto/bin/pypto-activate.sh >/dev/null
    last="$1"
    tmp="$(mktemp -d /results/.prepare.XXXXXX)"
    trap '\''rm -rf -- "$tmp"'\'' EXIT
    python -c '\''
from pathlib import Path
import sys
from blockspec.dsv41_packed_proposer import load_staged_draft_graphs
from blockspec.dsv41_staged_adam import load_staged_adam_programs
from blockspec.dsv41_staged_reverse import load_staged_reverse_programs
from blockspec.dsv41_staged_target import load_staged_target_programs
from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE

root, last = Path("/results"), int(sys.argv[1])
draft = root / f"draft-last{last}"
checks = (
    (("prefill", "decode"), lambda: load_staged_target_programs(
        root / "prefill", root / "decode", (272, 16), (64, 16), context=512, slots=2, local_ranks=8)),
    ((draft.name,), lambda: (load_staged_draft_graphs(
        draft, context=512, block=16, local_ranks=8, request_slots=2, last_layers=last),
        load_staged_reverse_programs(draft, last))),
    (("adam",), lambda: load_staged_adam_programs(root / "adam", FLASH, TP_SIZE)),
)
for names, check in checks:
    try:
        check()
    except (OSError, ValueError) as error:
        joined = ", ".join(names)
        print(f"Rebuild {joined}: {error}", file=sys.stderr)
        print(*names, sep="\n")
'\'' "$last" > "$tmp/rebuild"
    mapfile -t paths < "$tmp/rebuild"
    for name in "${paths[@]}"; do
      case "$name" in
        prefill) python /opt/dsv41-1122/validate_dsv41_pypto_packed_prefill.py --capacity 272 \
                   --prefill-moe-slab-rows 64 --context 512 --slots-per-dp 2 --output "$tmp/$name" ;;
        decode) python /opt/dsv41-1122/validate_dsv41_pypto_packed_decode_attention.py --decode-capacity 16 \
                  --context 512 --slots-per-dp 2 --output "$tmp/$name" ;;
        draft-*) python /opt/dsv41-1122/compile_dsv41_pypto_packed_draft.py --local-ranks 8 --block 16 \
                   --decode-capacity 16 --context 512 --slots-per-dp 2 --last-layers "$last" --artifact-dir "$tmp/$name" ;;
        adam) python /opt/dsv41-1122/compile_dsv41_pypto_staged_adam.py --artifact-dir "$tmp/$name" ;;
      esac
      if [[ -d "/results/$name" ]]; then
        previous="$(mktemp -d "/results/.superseded.$name.XXXXXX")"
        mv "/results/$name" "$previous/"
      fi
      mv "$tmp/$name" "/results/$name"
    done
    python /opt/dsv41-1122/prepare_dsv41_pypto_staged_gsm8k.py \
      --dataset /gsm8k --checkpoint /model --output /results/gsm8k
    python -c '\''import json; from pathlib import Path
p = Path("/results/gsm8k")
warm = json.loads((p / "warmup.json").read_text())
(p / "smoke.json").write_text(json.dumps([dict(warm[i], new_tokens=24) for i in (0, 2)]) + "\n")
(p / "smoke-packed.json").write_text(json.dumps([dict(row, dp=2 * (i % 2), new_tokens=24)
                                               for i, row in enumerate(warm)]) + "\n")'\''
  ' prepare "$last"
  exit
fi

case "$mode" in
  smoke)
    smoke=/results/gsm8k/smoke.json
    if [[ "${PYPTO_PACK_DECODE:-0}" == 1 ]]; then smoke=/results/gsm8k/smoke-packed.json; fi
    workload=(--workload "$smoke" --teacher-rows 7 --update-every 2 --max-updates 2 --verify-width 8)
    ;;
  study) workload=(--workload /results/gsm8k/online.json --warmup-workload /results/gsm8k/warmup.json
          --eval-workload /results/gsm8k/heldout.json --teacher-rows 7 --update-every 16 --max-updates 128 --verify-width 8) ;;
  *) echo 'Choose prepare, smoke or study.' >&2; exit 2 ;;
esac
if [[ "${PYPTO_PACK_DECODE:-0}" == 1 ]]; then workload+=(--pack-decode); fi
[[ -r "$key" && -s "$key" ]] || { echo "Missing cluster key at $key." >&2; exit 2; }
[[ -r "$head/manifest.json" ]] || { echo "Missing draft head at $head. Run prepare first." >&2; exit 2; }
for entry in prefill decode "draft-last$last" adam gsm8k; do
  [[ -d "$root/$entry" ]] || { echo "Missing $root/$entry. Run prepare first." >&2; exit 2; }
done
for arg in "${workload[@]}"; do
  if [[ "$arg" == /results/gsm8k/* && ! -r "$root/${arg#/results/}" ]]; then
    echo "Missing $root/${arg#/results/}. Run prepare first." >&2
    exit 2
  fi
done
if [[ "$rank" == 0 ]]; then
  local_ip="$rank0_ip"
  nic="$(ip -o -4 addr show | awk -v ip="$local_ip" '{split($4, address, "/"); if (address[1] == ip) {print $2; exit}}')"
else
  route="$(ip -4 route get "$rank0_ip")"
  local_ip="$(awk '{for(i=1;i<NF;i++) if ($i == "src") {print $(i+1); exit}}' <<< "$route")"
  nic="$(awk '{for(i=1;i<NF;i++) if ($i == "dev") {print $(i+1); exit}}' <<< "$route")"
fi
[[ -n "$nic" && -n "$local_ip" ]] || { echo 'Cannot resolve the HCCL interface.' >&2; exit 2; }

IFS=, read -r -a physical <<< "${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
[[ "${#physical[@]}" == 8 && "$(printf '%s\n' "${physical[@]}" | sort -u | wc -l)" -eq 8 ]] || {
  echo 'Reserve eight distinct physical NPU devices per node.' >&2; exit 2;
}
cards=()
for card in "${physical[@]}"; do
  [[ "$card" =~ ^[0-9]+$ && -c "/dev/davinci$card" ]] || { echo "Unavailable NPU device $card." >&2; exit 2; }
  cards+=(--device "/dev/davinci$card")
done
if [[ "${PYPTO_CHECK_ONLY:-0}" == 1 ]]; then
  for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [[ -c "$device" ]] || { echo "Missing $device." >&2; exit 2; }
  done
  python3 - "$model" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
index = model / "quant_model_weights.safetensors.index.json"
shards = set(json.loads(index.read_text())["weight_map"].values())
for name in sorted(shards):
    shard = model / name
    if not shard.is_file() or not shard.stat().st_size:
        sys.exit(f"Missing or empty checkpoint shard: {shard}")
PY
  printf '%s preflight passed for %s on %s\n' "$1" "$mode" "$local_ip"
  exit
fi
if [[ -n "${PYPTO_RUN_NAME:-}" ]]; then base+=(--name "$PYPTO_RUN_NAME"); fi
run="$(mktemp -d "$root/$mode-$1.XXXXXX")"
echo "Report: $run/result.json"
docker run "${base[@]}" --network host "${cards[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v "$head:/head:ro" -v "$key:/cluster.key:ro" \
  -v "$run:/run-results" -e ASCEND_PROCESS_LOG_PATH=/run-results -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e "HCCL_IF_IP=$local_ip" -e "HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$nic}" \
  -e "HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1200}" "$image" \
  /opt/dsv41-1122/run_dsv41_pypto_staged_target.py --node "$rank" --rank0-ip "$rank0_ip" --port "$port" \
  --key-file /cluster.key --checkpoint /model --prefill-graph /results/prefill --decode-graph /results/decode \
  --prefill-capacity 272 --prefill-moe-tokens 64 --decode-capacity 16 "${geometry[@]}" \
  --draft-head /head --draft-graph "/results/draft-last$last" --adam-graph /results/adam \
  --draft-block 16 --capture-last-layers "$last" --output /run-results/result.json "${workload[@]}" \
  2>&1 | tee "$run/run.log"
