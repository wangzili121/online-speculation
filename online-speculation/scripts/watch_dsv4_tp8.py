#!/usr/bin/env python3
"""One-shot host-side TP8 validation window. Never stop another user's work.

Requires an explicitly dedicated, already prepared Orthrus container. Container
restart is intentional: it cleans up all of this test's workers between services.
Docker device mappings are NOT an exclusive resource reservation.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def idle_devices(snapshot):
    """Fail closed on partial/error output, residual HBM, or any process row."""
    cards = set(map(int, re.findall(r"^\|\s*(\d+)\s+910\w*\s*\|", snapshot, re.M)))
    idle = set(map(int, re.findall(r"No running processes found in NPU (\d+)\b", snapshot)))
    processes = re.findall(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|", snapshot, re.M)
    hbm = [int(value) for value in re.findall(r"\b(\d+)\s*/\s*65536\b", snapshot)]
    return (cards == idle == set(range(8)) and not processes
            and len(hbm) == 8 and all(value < 8192 for value in hbm))


def container_is_idle(table):
    rows = table.strip().splitlines()[1:]
    return bool(rows) and all(re.search(r"\bsleep infinity$", row) for row in rows)


class Window:
    def __init__(self, args):
        self.args = args
        self.started = False
        self.report = {"state": "waiting", "checks": {}, "full_acceptance": False,
                       "scope": "TP8 eager API checks; not graph, 1M or trained-head performance",
                       "container": args.container, "source": args.root,
                       "watch_hours": args.watch_hours}

    def save(self, state=None, **fields):
        if state:
            self.report["state"] = state
        self.report.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
        path = self.args.output / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps({"state": self.report["state"], **fields}), flush=True)

    def snapshot(self):
        return subprocess.check_output([self.args.npu_smi, "info"], text=True, timeout=10)

    def wait_free(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                snapshot = self.snapshot()
                if idle_devices(snapshot):
                    (self.args.output / "free-snapshot.txt").write_text(snapshot)
                    return
            except (OSError, subprocess.SubprocessError) as error:
                self.save(last_monitor_error=str(error))
            time.sleep(3)
        raise TimeoutError("eight idle devices were not observed within the window")

    def exec_command(self, command, extra_env=None):
        env = {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
               "VLLM_BATCH_INVARIANT": "1", "TP": "8",
               "MODEL": self.args.model, "HEAD": self.args.head,
               "PORT": str(self.args.port), "MAX_MODEL_LEN": "133120",
               "MAX_NUM_SEQS": "4", "MEMORY": "0.90",
               "ORTHRUS_PREFLIGHT_MODE": "production",
               "ORTHRUS_EXECUTION_MODE": "eager", "TARGET_EXECUTION_MODE": "eager",
               "ORTHRUS_DRAFT_SAMPLE_METHOD": "probabilistic",
               "ORTHRUS_VERIFY_CACHE_CHECKSUM": "1"}
        env.update(extra_env or {})
        return ["docker", "exec", "-w", self.args.root,
                *[part for key, value in env.items() for part in ("-e", f"{key}={value}")],
                self.args.container, *command]

    def check(self, name, command, timeout=7200):
        with (self.args.output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(self.exec_command(command), stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                process.terminate()
                process.wait(timeout=10)
                raise
        self.report["checks"][name] = {"returncode": code, "command": command}
        self.save()
        return code

    def restart(self):
        subprocess.run(["docker", "restart", "-t", "20", self.args.container],
                       check=True, timeout=90, stdout=subprocess.DEVNULL)
        # Restart kills only our dedicated container's workers. If somebody
        # acquired a device meanwhile, do not start the next service over them.
        self.wait_free(60)

    def service(self, kind, suffix, *, extended=False):
        name = f"{kind}-{suffix}"
        self.save("running", phase=name)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.args.port))
        command = self.exec_command(["bash", f"scripts/serve_dsv4_{kind}.sh"])
        with (self.args.output / f"{name}-server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            self.started = True
            try:
                deadline = time.monotonic() + 2400
                url = f"http://127.0.0.1:{self.args.port}"
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited before readiness: {process.returncode}")
                    try:
                        with urllib.request.urlopen(url + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{name} did not become ready")
                    time.sleep(2)
                model_name = f"dsv4-{kind}"
                if self.check(name + "-health", ["python3", "scripts/healthcheck_dsv4.py",
                              "--base-url", url, "--model", model_name], timeout=400):
                    raise RuntimeError(f"{name} decode smoke failed")
                self.check(name, ["python3", "scripts/validate_dsv4_readiness.py", "--stage", "service",
                           "--base-url", url, "--model", model_name, "--eos-token-id", "1",
                           "--output", f"{self.args.container_output}/{name}"])
                if extended:
                    self.check(name + "-long", ["python3", "scripts/validate_dsv4_service.py", "snapshot",
                               "--base-url", url, "--model", model_name,
                               "--cases", "configs/dsv4/correctness-extended.json", "--include-long",
                               "--output", f"{self.args.container_output}/{name}-long.json"])
            finally:
                self.restart()
                process.wait(timeout=30)

    def run(self):
        table = subprocess.check_output(["docker", "top", self.args.container, "-eo", "pid,args"], text=True)
        if not container_is_idle(table):
            raise RuntimeError("dedicated container has work other than sleep infinity")
        self.save()
        self.wait_free(self.args.watch_hours * 3600)
        # Recheck our own container once before taking responsibility for it.
        table = subprocess.check_output(["docker", "top", self.args.container, "-eo", "pid,args"], text=True)
        if not container_is_idle(table):
            raise RuntimeError("container acquired another task while waiting")
        self.service("target", "initial", extended=True)
        self.service("orthrus", "initial", extended=True)
        self.check("comparison", ["python3", "scripts/validate_dsv4_service.py", "compare",
                   "--baseline", f"{self.args.container_output}/target-initial/snapshot.json",
                   "--candidate", f"{self.args.container_output}/orthrus-initial/snapshot.json",
                   "--output", f"{self.args.container_output}/comparison.json"], timeout=60)
        self.service("orthrus", "restart")
        self.check("restart-comparison", ["python3", "scripts/validate_dsv4_service.py", "compare",
                   "--baseline", f"{self.args.container_output}/orthrus-initial/snapshot.json",
                   "--candidate", f"{self.args.container_output}/orthrus-restart/snapshot.json",
                   "--output", f"{self.args.container_output}/restart-comparison.json"], timeout=60)
        self.save("finished", checks_passed=all(c["returncode"] == 0 for c in self.report["checks"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--root", required=True, help="prepared source root inside the container")
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new host-side log directory")
    parser.add_argument("--container-output", required=True, help="new persistent path inside the container")
    parser.add_argument("--npu-smi", default="/usr/local/sbin/npu-smi")
    parser.add_argument("--port", type=int, default=19871)
    parser.add_argument("--watch-hours", type=float, default=12)
    parser.add_argument("--allow-restart-container", action="store_true")
    args = parser.parse_args()
    if not args.allow_restart_container or not args.container.startswith("wangzili-orthrus-"):
        parser.error("requires an explicitly dedicated wangzili-orthrus-* container and restart consent")
    if not 0 < args.watch_hours <= 24:
        parser.error("watch-hours must be in (0, 24]")
    args.output.mkdir(parents=True, exist_ok=False)
    # One watcher per container, including processes launched in other sessions.
    lock = Path(f"/tmp/{args.container}-tp8-window.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    window = Window(args)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        window.run()
    except BaseException as error:
        window.save("failed" if window.started else "stopped", error=repr(error))
        return 1
    finally:
        if window.started:
            subprocess.run(["docker", "stop", "-t", "20", args.container], check=True, timeout=90)
            window.save(released_own_container=True)
    return 0 if window.report.get("checks_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
