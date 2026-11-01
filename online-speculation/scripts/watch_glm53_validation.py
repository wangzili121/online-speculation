#!/usr/bin/env python3
"""Wait for verified weights, a validated runtime and eight idle NPUs.

No reservation kernels. Only this monitor's uniquely named containers are
stopped. Jobs have a wall-time bound and release devices on success or failure.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def idle_devices(snapshot):
    memories = {int(i): int(m) for i, m in re.findall(
        r"^\|\s*(\d+)\s+910\w*\s*\|[^\n]*\n\|\s*\d+\s*\|[^\n]*?\b(\d+)\s*/\s*65536\s*\|",
        snapshot, re.M)}
    empty = set(map(int, re.findall(r"No running processes found in NPU (\d+)\b", snapshot)))
    if set(memories) != set(range(8)):
        raise ValueError("incomplete eight-card NPU snapshot")
    return sorted(i for i in empty if i in memories and memories[i] < 8192)


def weights_ready(path, directory):
    state = json.loads(path.read_text())
    if state.get("state") in {"failed", "cancelled"}:
        raise RuntimeError(f"checkpoint download is {state['state']}: {state.get('error', '')}")
    if state.get("state") != "complete":
        return False
    if (state.get("model_id") != "Eco-Tech/GLM-5.3-Flash-w8a8"
            or Path(state["directory"]).resolve() != directory.resolve()
            or state.get("verified_files") != state["files"]
            or state.get("verified_bytes") != state["total_bytes"]):
        raise ValueError("download completion is not a fully verified GLM-Flash snapshot")
    manifest = path.with_name("source_manifest.json")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != state["source_manifest_sha256"]:
        raise ValueError("download manifest has changed")
    return True


def runtime_sources(source):
    paths = set((source / "src" / "blockspec").rglob("*.py"))
    paths.update((source / "scripts").glob("*glm53*.py"))
    paths.add(source / "scripts" / "run_in_runtime.py")
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def runtime_ready(path, source, image):
    if not path.is_file():
        return False
    state = json.loads(path.read_text())
    if state.get("status") != "passed" or state.get("image") != image:
        return False
    try:
        return state.get("sha256") == runtime_sources(source)
    except OSError:
        return False


def acquire_devices(source, devices):
    # 仅协调本项目的测试，不能代替 npu-smi，也不预留其他用户的设备。
    directory = source / "validation" / "device-locks"
    directory.mkdir(parents=True, exist_ok=True)
    locks = []
    try:
        for card in sorted(devices):
            lock = (directory / f"npu{card}.lock").open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locks
    except BlockingIOError:
        release_devices(locks)
        return None
    except BaseException:
        release_devices(locks)
        raise


def release_devices(locks):
    for lock in locks:
        lock.close()


def run_job(args, mode, *, label=None, extra=()):
    label = label or mode
    if not re.fullmatch(r"[a-z0-9-]+", label):
        raise ValueError("invalid validation stage label")
    devices = list(getattr(args, "devices", range(8)))
    runtime_cache = args.source / "validation" / "runtime-cache" / args.image.rsplit(":", 1)[-1][:12]
    name = f"wangzili-glm53-{label}-{os.getpid()}-{int(time.time())}"
    log = args.output / f"{label}.log"
    command = ["docker", "run", "-d", "--name", name, "--label", "owner=wangzili",
               "--label", "experiment=orthrus-glm53", "--network=host", "--shm-size=32g",
               "--device", "/dev/davinci_manager", "--device", "/dev/devmm_svm",
               "--device", "/dev/hisi_hdc",
               "-e", "VLLM_USE_V2_MODEL_RUNNER=0", "-e", "VLLM_ASCEND_ENABLE_FLASHCOMM=0",
               "-e", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
               "-e", "TORCH_EXTENSIONS_DIR=/runtime-cache",
               "-e", "TRITON_CACHE_DIR=/runtime-cache/triton",
               "-e", "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, range(len(devices)))),
               "-e", "HCCL_BUFFSIZE=200", "-e", "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
               "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
               "-v", "/usr/local/dcmi:/usr/local/dcmi:ro",
               "-v", "/etc/ascend_install.info:/etc/ascend_install.info:ro",
               "-v", f"{args.source}:/work:ro", "-v", f"{args.output}:/results",
               "-v", f"{runtime_cache}:/runtime-cache",
               "-v", f"{args.model}:/models/glm53:ro", "-w", "/work"]
    for logical, physical in enumerate(devices):
        command.extend(["--device", f"/dev/davinci{physical}:/dev/davinci{logical}"])
    command.extend(["--entrypoint", "python3", args.image, "scripts/run_in_runtime.py", "scripts/run_glm53_vllm.py",
                    "--model", "/models/glm53", "--output", f"/results/{label}.json", "--mode", mode,
                    "--tp", str(len(devices))])
    if mode == "orthrus":
        command.extend(["--mask-token-id", "154820", "--experimental-hybrid-policy"])
        if "--head-dir" not in extra:
            command.append("--initialize-from-target")
    command.extend(extra)
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        result = subprocess.run(["docker", "wait", name], text=True, capture_output=True, timeout=args.job_timeout)
        if result.returncode or result.stdout.strip() != "0":
            raise RuntimeError(f"{mode} container failed: {result.stdout.strip()} {result.stderr.strip()}")
    finally:
        # Bound to this exact container; never stop a competing user's worker.
        subprocess.run(["docker", "stop", "--time", "15", name], capture_output=True, timeout=60)
        with log.open("wb") as stream:
            subprocess.run(["docker", "logs", name], stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        subprocess.run(["docker", "rm", name], capture_output=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--download-status", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-hours", type=float, default=48)
    parser.add_argument("--job-timeout", type=int, default=2400)
    args = parser.parse_args()
    if "@sha256:" not in args.image:
        parser.error("pin the validated runtime image by digest")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status_path = args.output / "watch-status.json"
    state = dict(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    deadline = time.monotonic() + args.wait_hours * 3600
    previous = None
    completed_modes = set()
    try:
        while time.monotonic() < deadline:
            try:
                snapshot = subprocess.check_output(["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)
                idle = idle_devices(snapshot)
            except (subprocess.SubprocessError, ValueError, OSError) as error:
                state.update(state="waiting_resource_probe", error=repr(error))
                write_json(status_path, state)
                time.sleep(args.poll_seconds)
                continue
            weights = weights_ready(args.download_status, args.model)
            ready = runtime_ready(args.output / "runtime-ready.json", args.source, args.image)
            state.update(state="waiting", idle=idle, weights_verified=weights, runtime_validated=ready)
            summary = (idle, weights, ready)
            if summary != previous:
                write_json(status_path, state)
                print(json.dumps(state), flush=True)
                previous = summary
            if weights and ready and idle == list(range(8)):
                locks = acquire_devices(args.source, range(8))
                if locks is None:
                    time.sleep(args.poll_seconds)
                    continue
                try:
                    for mode, label, extra in (("baseline", "baseline", ()),
                            ("orthrus", "orthrus", ("--audit-cache",)),
                            ("orthrus", "orthrus-bidir", ("--audit-cache", "--kda-mode", "bidirectional",
                                                           "--kda-backward-weight", "0.25"))):
                        if label in completed_modes:
                            continue
                        # Baseline exits before Orthrus starts; recheck that no one claimed a card.
                        if idle_devices(subprocess.check_output(
                                ["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)) != list(range(8)):
                            break
                        state.update(state="running", mode=label)
                        write_json(status_path, state)
                        run_job(args, mode, label=label, extra=extra)
                        completed_modes.add(label)
                    else:
                        state.update(state="smoke_complete", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                        write_json(status_path, state)
                        return
                finally:
                    release_devices(locks)
            time.sleep(args.poll_seconds)
        state.update(state="expired")
        write_json(status_path, state)
    except BaseException as error:
        state.update(state="stopped" if isinstance(error, SystemExit) else "failed", error=repr(error))
        write_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
