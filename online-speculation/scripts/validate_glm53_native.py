#!/usr/bin/env python3
"""Run native GLM target/draft/reload checks on explicitly selected idle cards."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time

from watch_glm53_validation import (
    acquire_devices, check_reloaded, idle_devices, release_devices, run_job, runtime_ready, stages, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--job-timeout", type=int, default=1200)
    parser.add_argument("--wait-hours", type=float, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--long-prompt-repeats", type=int)
    parser.add_argument("--block-size", type=int, choices=range(2, 9), default=8)
    parser.add_argument("--privileged-runtime", action="store_true",
                        help="explicit trusted-node workaround; use the same mode in runtime preflight")
    parser.add_argument("--stages", nargs="+", choices=[label for _, label, _ in stages()])
    args = parser.parse_args()
    if (len(set(args.devices)) != len(args.devices) or not set(args.devices) <= set(range(8))
            or "@sha256:" not in args.image):
        parser.error("unique physical cards 0..7 and a pinned runtime image are required")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "native-validation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status = dict(state="starting", model=str(args.model), devices=args.devices, completed=[],
                  privileged_runtime=args.privileged_runtime)
    status_path = args.output / "native-status.json"
    locks = None
    try:
        if not runtime_ready(args.source / "validation" / "runtime-ready.json", args.source, args.image,
                             8 if len(args.devices) == 8 else 1, privileged_runtime=args.privileged_runtime):
            raise RuntimeError("validate the current native GLM runtime before loading model weights")
        deadline = time.monotonic() + args.wait_hours * 3600
        while (locks := acquire_devices(args.source, args.devices)) is None:
            status.update(state="waiting_test_lease")
            write_json(status_path, status)
            if time.monotonic() >= deadline:
                raise TimeoutError("another validation still owns the requested test devices")
            time.sleep(30)
        for mode, label, extra in stages():
            if args.stages and label not in args.stages:
                continue
            if not runtime_ready(args.source / "validation" / "runtime-ready.json", args.source, args.image,
                                 8 if len(args.devices) == 8 else 1, privileged_runtime=args.privileged_runtime):
                raise RuntimeError("native runtime or source changed; repeat preflight")
            extra += ("--block-size", str(args.block_size))
            for option in ("max_model_len", "long_prompt_repeats"):
                value = getattr(args, option)
                if value is not None:
                    if value < 1:
                        raise ValueError(f"{option} must be positive")
                    extra += ("--" + option.replace("_", "-"), str(value))
            deadline = time.monotonic() + args.wait_hours * 3600
            while True:
                snapshot = subprocess.check_output(["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)
                if set(args.devices) <= set(idle_devices(snapshot)):
                    break
                status.update(state="waiting_cards", stage=label)
                write_json(status_path, status)
                if time.monotonic() >= deadline:
                    raise TimeoutError("requested cards did not become idle")
                time.sleep(30)
            status.update(state="running", stage=label)
            write_json(status_path, status)
            run_job(args, mode, label=label, extra=extra)
            if not runtime_ready(args.source / "validation" / "runtime-ready.json", args.source, args.image,
                                 8 if len(args.devices) == 8 else 1, privileged_runtime=args.privileged_runtime):
                raise RuntimeError("source changed during native validation")
            report = json.loads((args.output / f"{label}.json").read_text())
            if report.get("status") != "passed":
                raise RuntimeError(f"native validation stage did not pass: {label}")
            check_reloaded(args.output, label)
            status["completed"].append(label)
            print(json.dumps(dict(stage=label, state="passed")), flush=True)
        status["state"] = "passed"
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        release_devices(locks or [])
        write_json(status_path, status)


if __name__ == "__main__":
    main()
