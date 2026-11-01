#!/usr/bin/env python3
"""Run a bounded Qwen acceptance matrix on an idle pair, using shared local leases."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from watch_glm53_validation import (
    acquire_devices, idle_devices, release_devices, runtime_network_environment, write_json,
)


def eight_card_job_waiting(root, idle):
    if set(idle) != set(range(8)):
        return False
    try:
        state = json.loads((root / "validation" / "watch-status.json").read_text())
        os.kill(state["pid"], 0)
        return state["state"] == "waiting" and state.get("weights_verified") is True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def container_command(args, name):
    command = ["docker", "run", "-d", "--name", name, "--label", "owner=wangzili",
               "--label", "experiment=orthrus-qwen38", "--network=host", "--shm-size=32g"]
    for device in ("davinci_manager", "devmm_svm", "hisi_hdc"):
        command.extend(["--device", f"/dev/{device}"])
    for logical, physical in enumerate(args.devices):
        command.extend(["--device", f"/dev/davinci{physical}:/dev/davinci{logical}"])
    environment = dict(ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(len(args.devices)))),
                       VLLM_USE_V2_MODEL_RUNNER="0", VLLM_ASCEND_ENABLE_FLASHCOMM="0",
                       VLLM_WORKER_MULTIPROC_METHOD="spawn", HCCL_BUFFSIZE="200",
                       PYTORCH_NPU_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="4",
                       TORCH_EXTENSIONS_DIR="/results/runtime-cache",
                       TRITON_CACHE_DIR="/results/runtime-cache/triton") | runtime_network_environment()
    for key, value in environment.items():
        command.extend(["-e", f"{key}={value}"])
    for host, target in (("/usr/local/Ascend/driver", "/usr/local/Ascend/driver"),
                         ("/usr/local/dcmi", "/usr/local/dcmi"),
                         ("/etc/ascend_install.info", "/etc/ascend_install.info"),
                         (args.source, "/work"), (args.model, "/models/qwen38")):
        command.extend(["-v", f"{host}:{target}:ro"])
    command.extend(["-v", f"{args.output}:/results", "-w", "/results", "--entrypoint", "python3",
                    args.image, "/work/scripts/run_in_runtime.py", "/work/scripts/accept_qwen38_vllm.py",
                    "--model", "/models/qwen38", "--mask-token-id", "1", "--tp", str(len(args.devices)),
                    "--output-dir", "/results/matrix", "--max-model-len", str(args.max_model_len), *args.acceptance_args])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--yield-to-eight-card", action="store_true")
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("acceptance_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (len(args.devices) not in {2, 4} or len(set(args.devices)) != len(args.devices)
            or not set(args.devices) <= set(range(8)) or "@sha256:" not in args.image
            or args.wait_hours <= 0 or args.timeout <= 0 or args.max_model_len < 2048):
        parser.error("require 2/4 distinct devices, a pinned image and positive time limits")
    if args.acceptance_args[:1] == ["--"]:
        args.acceptance_args = args.acceptance_args[1:]
    args.output.mkdir(parents=True, exist_ok=False)
    name = f"wangzili-qwen38-validation-{os.getpid()}-{time.time_ns()}"
    state = dict(state="waiting", devices=args.devices, container=name,
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status = args.output / "status.json"

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    locks = None
    started = False
    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        while time.monotonic() < deadline:
            snapshot = subprocess.check_output(["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)
            idle = idle_devices(snapshot)
            state.update(idle=idle)
            write_json(status, state)
            defer = args.yield_to_eight_card and eight_card_job_waiting(args.lease_root, idle)
            if set(args.devices) <= set(idle) and not defer:
                locks = acquire_devices(args.lease_root, args.devices)
                if locks is not None:
                    break
            time.sleep(3)
        if locks is None:
            raise TimeoutError("requested devices did not become available")
        # Recheck after the local lease; Docker device mappings are not exclusive.
        snapshot = subprocess.check_output(["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)
        if not set(args.devices) <= set(idle_devices(snapshot)):
            raise RuntimeError("another job claimed a device before launch")
        command = container_command(args, name)
        state.update(state="starting", command=command)
        write_json(status, state)
        started = True
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        state.update(state="running")
        write_json(status, state)
        result = subprocess.run(["docker", "wait", name], capture_output=True, text=True, timeout=args.timeout)
        if result.returncode or result.stdout.strip() != "0":
            raise RuntimeError(f"Qwen acceptance failed: {result.stdout.strip()} {result.stderr.strip()}")
        state.update(state="finished")
    except BaseException as error:
        state.update(state="stopped" if isinstance(error, SystemExit) else "failed", error=repr(error))
        raise
    finally:
        if started:
            subprocess.run(["docker", "stop", "--time", "15", name], capture_output=True, timeout=60)
            with (args.output / "container.log").open("wb") as stream:
                subprocess.run(["docker", "logs", name], stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
        release_devices(locks or [])
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(status, state)


if __name__ == "__main__":
    main()
