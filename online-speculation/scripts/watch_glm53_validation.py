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


def runtime_network_environment():
    names = ("HCCL_SOCKET_IFNAME", "HCCL_IF_IP", "GLOO_SOCKET_IFNAME", "VLLM_HOST_IP", "TP_SOCKET_IFNAME")
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def runtime_ready(path, source, image, min_devices=1, *, privileged_runtime=False):
    if not path.is_file():
        return False
    state = json.loads(path.read_text())
    if (state.get("status") != "passed" or state.get("image") != image
            or state.get("privileged_runtime", False) != privileged_runtime
            or state.get("network_environment", {}) != runtime_network_environment()
            or state.get("native_worker_imported") is not True
            or state.get("checked_devices", [])[:min_devices] != list(range(min_devices))):
        return False
    try:
        return state.get("sha256") == runtime_sources(source)
    except OSError:
        return False


def container_access_args(devices, privileged_runtime=False):
    if not privileged_runtime:
        return []
    # 特权模式可见全部物理设备，不允许继续按局部容器序号重排。
    if list(devices) != list(range(len(devices))):
        raise ValueError("privileged runtime requires identity-mapped cards starting at physical 0")
    return ["--privileged"]


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
        ("orthrus", "sampling-causal", audit + ("--head-dir", "/results/head-causal", "--temperature", "0.7")),
        ("orthrus", "sampling-bidir", audit + ("--head-dir", "/results/head-bidir", "--kda-mode", "bidirectional",
                                             "--temperature", "0.7")),
    ]


def check_reloaded(output, label):
    if label not in {"reload-causal", "reload-bidir", "sampling-causal", "sampling-bidir"}:
        return
    original = "orthrus" if label.endswith("-causal") else "orthrus-bidir"
    expected = json.loads((output / f"{original}.json").read_text())["head_exports"]
    loaded = json.loads((output / f"{label}.json").read_text())["head_digests"]
    if sorted(expected, key=lambda x: x["rank"]) != sorted(loaded, key=lambda x: x["rank"]):
        raise RuntimeError("reloaded GLM head tensors differ from the exported parameters")


def retry_attempt(state, output, *, max_attempts):
    """Keep failed evidence separate; a retry must rerun the baseline and every head stage."""
    failed = dict(attempt=state["attempt"], results=state["results"],
                  completed=list(state["completed"]), error=state["error"])
    state.setdefault("failed_attempts", []).append(failed)
    if state["attempt"] >= max_attempts:
        return None
    directory = output / f"full-tp8-retry-{time.time_ns()}"
    directory.mkdir()
    state.update(state="retry_wait", attempt=state["attempt"] + 1, completed=[], results=str(directory))
    state.pop("error", None)
    return directory


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
               "-e", "PYTHONFAULTHANDLER=1", "-e", "TORCH_SHOW_CPP_STACKTRACES=1",
               "-e", "TORCH_EXTENSIONS_DIR=/runtime-cache",
               "-e", "TRITON_CACHE_DIR=/runtime-cache/triton",
               "-e", "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, range(len(devices)))),
               "-e", "HCCL_BUFFSIZE=200", "-e", "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
               "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
               "-v", "/usr/local/dcmi:/usr/local/dcmi:ro",
               "-v", "/etc/ascend_install.info:/etc/ascend_install.info:ro",
               "-v", f"{args.source}:/work:ro", "-v", f"{args.output}:/results",
               "-v", f"{runtime_cache}:/runtime-cache",
               "-v", f"{args.model}:/models/glm53:ro", "-w", "/runtime-cache"]
    command.extend(container_access_args(devices, getattr(args, "privileged_runtime", False)))
    for key, value in runtime_network_environment().items():
        command.extend(["-e", f"{key}={value}"])
    for logical, physical in enumerate(devices):
        command.extend(["--device", f"/dev/davinci{physical}:/dev/davinci{logical}"])
    command.extend(["--entrypoint", "python3", args.image, "/work/scripts/run_in_runtime.py", "/work/scripts/run_glm53_vllm.py",
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
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--retry-seconds", type=float, default=300)
    parser.add_argument("--privileged-runtime", action="store_true",
                        help="explicit trusted-node workaround for driver container-access conflicts; exposes all devices")
    args = parser.parse_args()
    if "@sha256:" not in args.image:
        parser.error("pin the validated runtime image by digest")
    if (args.poll_seconds <= 0 or args.wait_hours <= 0 or args.job_timeout <= 0
            or not 1 <= args.max_attempts <= 3 or args.retry_seconds < 0):
        parser.error("positive time limits, 1..3 attempts and nonnegative retry delay required")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status_path = args.output / "watch-status.json"
    job_output = args.output / f"full-tp8-{os.getpid()}-{time.time_ns()}"
    job_output.mkdir()
    job_args = argparse.Namespace(**(vars(args) | {"output": job_output}))
    state = dict(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 results=str(job_output), completed=[], attempt=1, privileged_runtime=args.privileged_runtime)
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
            ready = runtime_ready(args.output / "runtime-ready.json", args.source, args.image, 8,
                                  privileged_runtime=args.privileged_runtime)
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
                    for mode, label, extra in stages():
                        if label in completed_modes:
                            continue
                        if not runtime_ready(args.output / "runtime-ready.json", args.source, args.image, 8,
                                             privileged_runtime=args.privileged_runtime):
                            raise RuntimeError("native runtime or source changed; restart validation after preflight")
                        # Baseline exits before Orthrus starts; recheck that no one claimed a card.
                        if idle_devices(subprocess.check_output(
                                ["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)) != list(range(8)):
                            break
                        state.update(state="running", mode=label)
                        write_json(status_path, state)
                        run_job(job_args, mode, label=label, extra=extra)
                        if not runtime_ready(args.output / "runtime-ready.json", args.source, args.image, 8,
                                             privileged_runtime=args.privileged_runtime):
                            raise RuntimeError("source changed during full-model validation")
                        check_reloaded(job_output, label)
                        completed_modes.add(label)
                        state["completed"].append(label)
                    else:
                        state.update(state="validation_complete", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                        write_json(status_path, state)
                        return
                except Exception as error:
                    state.update(state="attempt_failed", error=repr(error))
                    next_output = retry_attempt(state, args.output, max_attempts=args.max_attempts)
                    write_json(status_path, state)
                    if next_output is None:
                        raise
                    job_output = next_output
                    job_args.output = job_output
                    completed_modes.clear()
                    previous = None
                finally:
                    release_devices(locks)
                if state["state"] == "retry_wait":
                    time.sleep(args.retry_seconds)
            time.sleep(args.poll_seconds)
        state.update(state="expired")
        write_json(status_path, state)
    except BaseException as error:
        state.update(state="stopped" if isinstance(error, SystemExit) else "failed", error=repr(error))
        write_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
