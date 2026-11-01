#!/usr/bin/env python3
"""Run native GLM target/draft/reload checks on explicitly selected idle cards."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time

from watch_glm53_validation import idle_devices, run_job, write_json


def stages():
    common = ("--max-output-tokens", "32", "--ignore-eos")
    audit = common + ("--audit-cache",)
    return [
        ("baseline", "baseline", common),
        ("orthrus", "orthrus", audit + ("--save-head", "/results/head-causal")),
        ("orthrus", "orthrus-bidir", audit + ("--kda-mode", "bidirectional", "--kda-backward-weight", "0.25",
                                             "--save-head", "/results/head-bidir")),
        ("orthrus", "reload-causal", audit + ("--head-dir", "/results/head-causal")),
        ("orthrus", "reload-bidir", audit + ("--head-dir", "/results/head-bidir", "--kda-mode", "bidirectional")),
        ("orthrus", "sampling-bidir", audit + ("--head-dir", "/results/head-bidir", "--kda-mode", "bidirectional",
                                             "--temperature", "0.7")),
    ]


def check_reloaded(output, label):
    if label not in {"reload-causal", "reload-bidir", "sampling-bidir"}:
        return
    original = "orthrus" if label == "reload-causal" else "orthrus-bidir"
    expected = json.loads((output / f"{original}.json").read_text())["head_exports"]
    loaded = json.loads((output / f"{label}.json").read_text())["head_digests"]
    if sorted(expected, key=lambda x: x["rank"]) != sorted(loaded, key=lambda x: x["rank"]):
        raise RuntimeError("reloaded GLM head tensors differ from the exported parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--job-timeout", type=int, default=1200)
    parser.add_argument("--wait-hours", type=float, default=1)
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
    status = dict(state="starting", model=str(args.model), devices=args.devices, completed=[])
    status_path = args.output / "native-status.json"
    try:
        for mode, label, extra in stages():
            if args.stages and label not in args.stages:
                continue
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
        write_json(status_path, status)


if __name__ == "__main__":
    main()
