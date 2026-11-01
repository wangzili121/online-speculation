#!/usr/bin/env python3
"""Validate a pulled GLM runtime on idle NPUs, then release them."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from watch_glm53_validation import (
    acquire_devices, container_access_args, idle_devices, release_devices, runtime_network_environment, write_json,
)


def candidate_groups(idle, *, all_devices=False, allowed=None):
    if all_devices:
        return [list(range(8))] if set(idle) == set(range(8)) else []
    return [[card] for card in idle if allowed is None or card in allowed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pull-image", action="store_true")
    parser.add_argument("--devices", type=int, nargs="+", choices=range(8))
    parser.add_argument("--all-devices", action="store_true")
    parser.add_argument("--retry-seconds", type=int, default=0)
    parser.add_argument("--wait-hours", type=float, default=48)
    parser.add_argument("--privileged-runtime", action="store_true",
                        help="explicit trusted-node workaround; disables device isolation, never enabled automatically")
    args = parser.parse_args()
    if args.retry_seconds < 0 or args.wait_hours <= 0 or args.all_devices and args.devices is not None:
        parser.error("positive wait, nonnegative retry, and either --devices or --all-devices required")
    if args.privileged_runtime and not args.all_devices and args.devices != [0]:
        parser.error("privileged preflight requires --devices 0 or --all-devices")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ("prepare-all.lock" if args.all_devices else "prepare.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state_path = args.output / ("prepare-all-status.json" if args.all_devices else "prepare-status.json")
    deadline = time.monotonic() + args.wait_hours * 3600
    while time.monotonic() < deadline:
        available = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True, timeout=30)
        if available.returncode:
            write_json(state_path, dict(state="waiting_image", image=args.image))
            # 已有同一镜像的 pull 时等待它；失败退出后才接着重试。
            pulling = False
            for path in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    if path.stat().st_uid != os.getuid():
                        continue
                    argv = path.read_bytes().decode().rstrip("\0").split("\0")
                    pulling |= len(argv) == 3 and Path(argv[0]).name == "docker" and argv[1:] == ["pull", args.image]
                except (OSError, UnicodeError):
                    continue
            if args.pull_image and not pulling:
                with (args.output / "image-pull-retry.log").open("ab") as log:
                    try:
                        subprocess.run(["docker", "pull", args.image], stdout=log, stderr=subprocess.STDOUT,
                                       timeout=7200, check=False)
                    except subprocess.TimeoutExpired:
                        write_json(state_path, dict(state="image_pull_timeout", image=args.image))
            time.sleep(30)
            continue
        snapshot = subprocess.check_output(["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)
        idle = idle_devices(snapshot)
        locks = None
        for devices in candidate_groups(idle, all_devices=args.all_devices, allowed=args.devices):
            locks = acquire_devices(args.source, devices)
            if locks is not None:
                break
        if locks is None:
            write_json(state_path, dict(state="waiting_eight_idle_cards" if args.all_devices else "waiting_one_idle_card"))
            time.sleep(30)
            continue
        name = f"wangzili-glm53-preflight-{os.getpid()}"
        runtime_cache = args.source / "validation" / "runtime-cache" / args.image.rsplit(":", 1)[-1][:12]
        command = ["docker", "run", "-d", "--name", name, "--network=host",
                   "--label", "owner=wangzili", "--label", "experiment=orthrus-glm53",
                   "--device", "/dev/davinci_manager", "--device", "/dev/devmm_svm",
                   "--device", "/dev/hisi_hdc", "-e", "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, range(len(devices)))),
                   "-e", "VLLM_USE_V2_MODEL_RUNNER=0",
                   "-e", "PYTHONFAULTHANDLER=1",
                   "-e", "TORCH_EXTENSIONS_DIR=/runtime-cache",
                   "-e", "TRITON_CACHE_DIR=/runtime-cache/triton",
                   "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
                   "-v", "/usr/local/dcmi:/usr/local/dcmi:ro",
                   "-v", "/etc/ascend_install.info:/etc/ascend_install.info:ro",
                   "-v", f"{args.source}:/work:ro", "-v", f"{args.output}:/results",
                   "-v", f"{runtime_cache}:/runtime-cache"]
        command.extend(container_access_args(devices, args.privileged_runtime))
        for key, value in runtime_network_environment().items():
            command.extend(["-e", f"{key}={value}"])
        for logical, physical in enumerate(devices):
            command.extend(["--device", f"/dev/davinci{physical}:/dev/davinci{logical}"])
        command.extend(["-w", "/runtime-cache", "--entrypoint", "python3", args.image,
                   "/work/scripts/run_in_runtime.py", "/work/scripts/preflight_glm53_runtime.py",
                   "--output", "/results/runtime-ready.json",
                   "--image", args.image, "--num-devices", str(len(devices))])
        if args.privileged_runtime:
            command.append("--privileged-runtime")
        write_json(state_path, dict(state="running", devices=devices, container=name,
                                   privileged_runtime=args.privileged_runtime))
        passed = False
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            result = subprocess.run(["docker", "wait", name], text=True, capture_output=True, timeout=900)
            if result.returncode or result.stdout.strip() != "0":
                raise RuntimeError(f"GLM runtime preflight failed: {result.stdout.strip()}")
            passed = True
            write_json(state_path, dict(state="passed", devices=devices))
        except Exception as error:
            write_json(state_path, dict(state="failed", error=repr(error), devices=devices))
            if not args.retry_seconds:
                raise
        finally:
            subprocess.run(["docker", "stop", "--time", "15", name], capture_output=True, timeout=60)
            with (args.output / ("prepare-all-runtime.log" if args.all_devices else "prepare-runtime.log")).open("wb") as log:
                subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT, timeout=60)
            subprocess.run(["docker", "rm", name], capture_output=True, timeout=60)
            release_devices(locks)
        if passed:
            return
        time.sleep(args.retry_seconds)
    write_json(state_path, dict(state="expired"))


if __name__ == "__main__":
    main()
