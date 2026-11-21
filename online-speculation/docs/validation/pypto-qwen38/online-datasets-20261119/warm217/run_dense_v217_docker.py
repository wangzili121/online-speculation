"""Run continuous learning against matched AR and frozen baselines on idle cards."""
import argparse
import os
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
p.add_argument("--interleaved",action="store_true")
p.add_argument("--interleave-ar",action="store_true")
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--temperature", type=float, choices=(0,1), default=0)
p.add_argument("--strides", default="16")
p.add_argument("--head", default="step160-suffix-tp2")
p.add_argument("--wait", type=int, default=45)
p.add_argument("--learner-device", type=int, choices=(0,2), default=2)
p.add_argument("--last-layers", type=int, choices=(1,4), default=1)
p.add_argument("--parameter-scope", choices=("full","norms","output_norms"), default="full")
p.add_argument("--replay-blocks", default=4, type=int)
p.add_argument("--overlap-sampling",action="store_true")
p.add_argument("--verify-chunk-size",type=int,choices=(1,2,4),default=4)
p.add_argument("--inner-steps",type=int,default=1)
p.add_argument("--publish-every", default=4, type=int)
p.add_argument("--request-file", default="long-gsm8k.json")
p.add_argument("--ar-check-requests", default=8, type=int)
p.add_argument("--holdout-file", default="mbpp-holdout.json")
p.add_argument("--learning-rate", default=1e-5, type=float)
p.add_argument("--objective", choices=("tv", "kl", "log_overlap"), default="tv")
p.add_argument("--focus-weight", type=int, default=1)
p.add_argument("--candidates", type=int, default=31)
p.add_argument("--stop-on-eos", action="store_true")
p.add_argument("--background", action="store_true")
p.add_argument("--compile-only", action="store_true")
p.add_argument("--with-cold-fallback", action="store_true")
p.add_argument("--feedback-mode", choices=("reached","delayed"), default="delayed")
p.add_argument("--teacher-tokens", type=int, default=31)
p.add_argument("--audit-replay", action="store_true")
p.add_argument("--audit-convolution", action="store_true")
p.add_argument("--draft-convolution", choices=("sequential","parallel64","parallel128"), default="sequential")
p.add_argument("--position-weighting", choices=("uniform","rejection_decay","first_rejection"), default="uniform")
p.add_argument("--lazy-sampling", action="store_true")
p.add_argument("--teacher-temperature", type=float, default=1.0)
p.add_argument("--teacher-scaling", choices=("host","device"), default="host")
p.add_argument("--teacher-targets", choices=("distribution","greedy"), default="distribution")
p.add_argument("--skip-seen-replay", action="store_true")
p.add_argument("--warm-update-stride", type=int)
p.add_argument("--warm-update-steps", type=int, default=0)
p.add_argument("--warmup-updates", type=int, default=0)
p.add_argument("--seed", type=int, default=743)
p.add_argument("--lr-schedule", choices=("constant","inverse_sqrt"), default="constant")
p.add_argument('--parameterization',choices=('dense','lora_merged'),default='dense')
p.add_argument('--adapter-rank',type=int,default=64)
p.add_argument('--adapter-seed',type=int,default=743)
p.add_argument('--draft-width',type=int,default=32)
p.add_argument('--gdn-valid-rows',action='store_true')
p.add_argument('--adapter-merge-backend',choices=('chained','fused'),default='chained')
p.add_argument('--feedback-timing',choices=('early','near_update'),default='early')
p.add_argument('--stream-window',type=int,default=2)
p.add_argument('--audit-admission',action='store_true')
a = p.parse_args()
if not a.tag.replace("-", "").isalnum():
    p.error("invalid experiment tag")

root = Path(__file__).resolve().parent
project = root.parent.parent
sys.path[0:0] = [str(project)]
from host_runtime import PYTHON, environment
model = Path('/data/disk/models/Qwen3.8-27B')
if not all((model/n).is_file() for n in ['config.json','tokenizer.json','model.safetensors.index.json']):
    raise RuntimeError('incomplete public model directory')
env = environment()
env["PYTHONPATH"] = ":".join(map(str, [
    root.parent / "pypto-serving-qwen3.8-27B", (root / "src-dense-v217"), env["PYTHONPATH"], root]))
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
    main_module = "orthrus_bench_dense_v217"
    command = [str(PYTHON), "-c",
               f"from {main_module} import main; from guarded_entry_v28 import Guard;"
               + ("main()" if a.compile_only else "g=Guard();\ntry: main(guard=g)\nfinally: g.close()"),
               "--model", str(model), "--head", str(root.parent / a.head),
               "--prompts", str(root / "assets" / a.request_file), "--holdout", str(root / "assets" / a.holdout_file),
               "--output", str(root / (a.tag + ".json")), "--export-heads", str(root / (a.tag + "-heads")), "--tokens", str(a.tokens),
               "--strides", a.strides, "--last-layers", str(a.last_layers), "--learner-device", str(a.learner_device),
               "--verify-chunk-size", str(a.verify_chunk_size), "--inner-steps", str(a.inner_steps), "--replay-blocks", str(a.replay_blocks), "--publish-every", str(a.publish_every),
               "--ar-check-requests", str(a.ar_check_requests)]
    command += ["--parameter-scope", a.parameter_scope, "--ring-heap-mib", str(a.ring_heap_mib), "--publication-backend", a.publication_backend]
    command += ["--gather-feedback", "--kv-pages", "32", "--learning-rate", str(a.learning_rate), "--objective", a.objective, "--candidates", str(a.candidates)]
    command.extend(['--parameterization',a.parameterization,'--adapter-rank',str(a.adapter_rank),'--adapter-seed',str(a.adapter_seed)])
    command.extend(['--draft-width',str(a.draft_width),'--adapter-merge-backend',a.adapter_merge_backend])
    if a.gdn_valid_rows: command.append('--gdn-valid-rows')
    command.extend(['--feedback-timing',a.feedback_timing])
    command.extend(['--stream-window',str(a.stream_window)])
    if a.audit_admission:command.append('--audit-admission')
    if a.stop_on_eos:
        command.append("--stop-on-eos")
    command.extend(["--feedback-mode", a.feedback_mode, "--teacher-tokens", str(a.teacher_tokens)])
    if a.warm_update_stride is not None:
        command.extend(["--warm-update-stride", str(a.warm_update_stride)])
    command.extend(["--warm-update-steps", str(a.warm_update_steps)])
    if a.lazy_sampling:
        command.append("--lazy-sampling")
    command.extend(["--draft-convolution", a.draft_convolution, "--position-weighting", a.position_weighting])
    command.extend(["--teacher-scaling", a.teacher_scaling])
    command.extend(["--teacher-targets", a.teacher_targets])
    if a.audit_convolution:
        command.append("--audit-convolution")
    if a.audit_replay:
        command.append("--audit-replay")
    if not a.skip_seen_replay:
        command.append("--replay-learned-head")
    command.extend(["--lr-schedule", a.lr_schedule])
    command.extend(["--warmup-updates", str(a.warmup_updates), "--seed", str(a.seed)])
    command.extend(["--teacher-temperature", str(a.teacher_temperature), "--continuous-updates", "--temperature", str(a.temperature), "--focus-weight", str(a.focus_weight)])
    if a.interleaved:
        command.append("--interleaved")
    if a.interleave_ar:
        command.append("--interleave-ar")
    if a.overlap_sampling:
        command.append("--overlap-sampling")
    if a.background:
        command.append("--background")
    if a.compile_only:
        command.append("--compile-only")
    if a.with_cold_fallback:
        command += ["--cold-head", str(root / "assets/step0-suffix-tp2")]
    if a.compile_only:
        raise RuntimeError("use the pinned runtime for CPU compilation")
    name = "orthrus-online-111-" + a.tag
    command = ["docker", "run", "--rm", "--name", name, "--privileged", "--network", "host", "--shm-size", "128g",
               "-e", "ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.1",
               "-v", str(project)+":"+str(project), "-v", str(model)+":"+str(model)+":ro",
               "-v", str(project/"cann-9.1.1")+":/usr/local/Ascend/cann-9.1.1:ro",
               "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
               "--entrypoint", "/usr/bin/python3", "quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc",
               str(root/"docker_entry_v217.py"), ",".join(map(str,cards)), *command[1:]]
    env = dict(os.environ)
    print("TP2 ranks and learner", "compile only" if a.compile_only else cards, flush=True)
    with (root / (a.tag + ".log")).open("xb") as log:
        raise SystemExit(subprocess.call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=a.job_timeout))
finally:
    subprocess.run(["docker","rm","-f","orthrus-online-111-"+a.tag],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    release_devices(locks)
