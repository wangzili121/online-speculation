"""Run continuous learning against matched AR and frozen baselines on idle cards."""
import argparse
from itertools import combinations
from pathlib import Path
import subprocess
import sys
import time

from watch_glm53_validation import acquire_devices, idle_devices, release_devices

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
p.add_argument("--ring-heap-mib", type=int, default=512)
p.add_argument("--publication-backend", choices=("auto","cpu","device"), default="auto")
p.add_argument("--job-timeout",type=int,default=21600)
p.add_argument("--prefill-chunk",type=int,default=0)
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--temperature", type=float, choices=(0,1), default=0)
p.add_argument("--strides", default="16")
p.add_argument("--head", default="step160-suffix-tp2")
p.add_argument("--wait", type=int, default=45)
p.add_argument("--learner-device", type=int, choices=(0,2), default=2)
p.add_argument("--last-layers", type=int, choices=(1,4), default=1)
p.add_argument("--parameter-scope", choices=("full","norms","output_norms"), default="full")
p.add_argument("--replay-blocks", default=4, type=int)
p.add_argument("--publish-every", default=4, type=int)
p.add_argument("--request-file", default="long-gsm8k.json")
p.add_argument("--ar-check-requests", default=8, type=int)
p.add_argument("--holdout-file", default="mbpp-holdout.json")
p.add_argument("--learning-rate", default=1e-5, type=float)
p.add_argument("--objective", choices=("tv", "kl"), default="tv")
p.add_argument("--focus-weight", type=int, default=1)
p.add_argument("--candidates", type=int, default=31)
p.add_argument("--stop-on-eos", action="store_true")
p.add_argument("--background", action="store_true")
p.add_argument("--compile-only", action="store_true")
p.add_argument("--with-cold-fallback", action="store_true")
p.add_argument("--feedback-mode", choices=("reached","delayed"), default="delayed")
p.add_argument("--teacher-tokens", type=int, default=31)
p.add_argument("--audit-replay", action="store_true")
p.add_argument("--lazy-sampling", action="store_true")
p.add_argument("--teacher-temperature", type=float, default=1.0)
p.add_argument("--skip-seen-replay", action="store_true")
p.add_argument("--warm-update-stride", type=int)
p.add_argument("--warm-update-steps", type=int, default=0)
p.add_argument("--warmup-updates", type=int, default=0)
p.add_argument("--seed", type=int, default=743)
p.add_argument("--lr-schedule", choices=("constant","inverse_sqrt"), default="constant")
a = p.parse_args()
if not a.tag.replace("-", "").isalnum():
    p.error("invalid experiment tag")

root = Path(__file__).resolve().parent
project = root.parent.parent
sys.path[0:0] = [str(project)]
from host_runtime import PYTHON, environment
model = project / "inference-tuning-20261116/model-view"
if not model.is_dir():
    model = Path("/data/disk/wangzili/models/Qwen3.8-27B")
env = environment()
env["PYTHONPATH"] = ":".join(map(str, [
    root.parent / "pypto-serving-qwen3.8-27B", (root / "src-dense-v15"), env["PYTHONPATH"], root]))
env["PYPTO_CACHE_DIR"] = str(root / "jit")
env["PYPTO_CACHE"] = "1"

card_count = 2 if a.learner_device == 0 else 3
allowed = set(map(int, a.devices.split(",")))
if not allowed <= set(range(8)) or len(allowed) < card_count:
    p.error("not enough valid allowed physical devices")
deadline = time.monotonic() + a.wait
while True:
    if a.compile_only:
        cards, locks = tuple(range(card_count)), ()
        break
    idle = idle_devices(subprocess.check_output(["/usr/local/bin/npu-smi", "info"], text=True))
    for cards in combinations([i for i in idle if i in allowed], card_count):
        locks = acquire_devices(project.parent / "orthrus-glm53-flash-20261101", cards)
        if locks is not None:
            break
    else:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no {card_count} idle A2 cards")
        time.sleep(1)
        continue
    break

env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, cards))
try:
    command = [str(PYTHON), "-c",
               "from diagnose_prefill_native import main; from guarded_entry import Guard;"
               + ("main()" if a.compile_only else "g=Guard();\ntry: main(guard=g)\nfinally: g.close()"),
               "--model", str(model), "--head", str(root.parent / a.head),
               "--prompts", str(root / "assets" / a.request_file), "--holdout", str(root / "assets" / a.holdout_file),
               "--output", str(root / (a.tag + ".json")), "--export-heads", str(root / (a.tag + "-heads")), "--tokens", str(a.tokens),
               "--strides", a.strides, "--last-layers", str(a.last_layers), "--learner-device", str(a.learner_device),
               "--replay-blocks", str(a.replay_blocks), "--publish-every", str(a.publish_every),
               "--ar-check-requests", str(a.ar_check_requests)]
    command += ["--parameter-scope", a.parameter_scope, "--ring-heap-mib", str(a.ring_heap_mib), "--publication-backend", a.publication_backend, "--prefill-chunk", str(a.prefill_chunk)]
    command += ["--gather-feedback", "--kv-pages", "32", "--learning-rate", str(a.learning_rate), "--objective", a.objective, "--candidates", str(a.candidates)]
    if a.stop_on_eos:
        command.append("--stop-on-eos")
    command.extend(["--feedback-mode", a.feedback_mode, "--teacher-tokens", str(a.teacher_tokens)])
    if a.warm_update_stride is not None:
        command.extend(["--warm-update-stride", str(a.warm_update_stride)])
    command.extend(["--warm-update-steps", str(a.warm_update_steps)])
    if a.lazy_sampling:
        command.append("--lazy-sampling")
    if a.audit_replay:
        command.append("--audit-replay")
    if not a.skip_seen_replay:
        command.append("--replay-learned-head")
    command.extend(["--lr-schedule", a.lr_schedule])
    command.extend(["--warmup-updates", str(a.warmup_updates), "--seed", str(a.seed)])
    command.extend(["--teacher-temperature", str(a.teacher_temperature), "--continuous-updates", "--temperature", str(a.temperature), "--focus-weight", str(a.focus_weight)])
    if a.background:
        command.append("--background")
    if a.compile_only:
        command.append("--compile-only")
    if a.with_cold_fallback:
        command += ["--cold-head", str(root / "assets/step0-suffix-tp2")]
    print("TP2 ranks and learner", "compile only" if a.compile_only else cards, flush=True)
    with (root / (a.tag + ".log")).open("xb") as log:
        raise SystemExit(subprocess.call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=a.job_timeout))
finally:
    release_devices(locks)
