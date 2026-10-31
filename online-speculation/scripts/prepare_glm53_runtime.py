#!/usr/bin/env python3
"""Validate a pulled GLM runtime on one idle NPU, then release it."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from watch_glm53_validation import idle_devices, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pull-image", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state_path = args.output / "prepare-status.json"
    deadline = time.monotonic() + 24 * 3600
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
        if not idle:
            write_json(state_path, dict(state="waiting_one_idle_card"))
            time.sleep(30)
            continue
        card = idle[0]
        name = f"wangzili-glm53-preflight-{os.getpid()}"
        command = ["docker", "run", "-d", "--name", name, "--network=none",
                   "--label", "owner=wangzili", "--label", "experiment=orthrus-glm53",
                   "--device", f"/dev/davinci{card}:/dev/davinci0",
                   "--device", "/dev/davinci_manager", "--device", "/dev/devmm_svm",
                   "--device", "/dev/hisi_hdc", "-e", "ASCEND_RT_VISIBLE_DEVICES=0",
                   "-e", "PYTHONPATH=/work/src", "-e", "VLLM_USE_V2_MODEL_RUNNER=0",
                   "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
                   "-v", "/usr/local/dcmi:/usr/local/dcmi:ro",
                   "-v", "/etc/ascend_install.info:/etc/ascend_install.info:ro",
                   "-v", f"{args.source}:/work:ro", "-v", f"{args.output}:/results",
                   "-w", "/work", "--entrypoint", "python3", args.image,
                   "scripts/preflight_glm53_runtime.py", "--output", "/results/runtime-ready.json",
                   "--image", args.image]
        write_json(state_path, dict(state="running", card=card, container=name))
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            result = subprocess.run(["docker", "wait", name], text=True, capture_output=True, timeout=900)
            if result.returncode or result.stdout.strip() != "0":
                raise RuntimeError(f"GLM runtime preflight failed: {result.stdout.strip()}")
            write_json(state_path, dict(state="passed", card=card))
        except Exception as error:
            write_json(state_path, dict(state="failed", error=repr(error), card=card))
            raise
        finally:
            subprocess.run(["docker", "stop", "--time", "15", name], capture_output=True, timeout=60)
            with (args.output / "prepare-runtime.log").open("wb") as log:
                subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT, timeout=60)
            subprocess.run(["docker", "rm", name], capture_output=True, timeout=60)
        return
    write_json(state_path, dict(state="expired"))


if __name__ == "__main__":
    main()
