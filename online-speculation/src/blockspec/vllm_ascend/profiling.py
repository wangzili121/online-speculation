"""MS Service Profiler configuration and artifact validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def write_service_config(path, profile_directory, duration, *, enabled=False):
    payload = {
        "enable": int(enabled),
        "prof_dir": str(Path(profile_directory).resolve()),
        "profiler_level": "INFO",
        "host_system_usage_freq": -1,
        "npu_memory_usage_freq": -1,
        "acl_task_time": 0,
        "acl_prof_task_time_level": "",
        # The runner stops collection explicitly. Keep a documented 120-second
        # safety limit so CANN cannot race a short requested window.
        "timelimit": 0 if enabled else max(120, int(duration) + 60),
        # Keep all domains. ms_service_profiler 26.0.0's vLLM handlers emit
        # Engine/Schedule/Schedule.KVCache/Execute, while the public guide
        # documents a different set of domain labels. A whitelist would
        # silently produce empty trace databases.
        "domain": "",
    }
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def set_service_enabled(path, enabled):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["enable"] = int(enabled)
    # CANN watches this exact path. Preserve the inode and force the update to
    # disk so its polling thread cannot miss a short profiling window.
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def analyze_service(input_directory, output_directory, timeout=1800):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        [sys.executable, "-m", "ms_service_profiler", "analyze",
         f"--input-path={Path(input_directory).resolve()}",
         f"--output-path={output_directory.resolve()}"],
        text=True, capture_output=True, timeout=timeout)
    produced = {path.name for path in output_directory.rglob("*") if path.is_file()}
    required = {"batch.csv", "chrome_tracing.json", "profiler.db"}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "produced": sorted(produced),
        "missing": sorted(required - produced),
        "valid": completed.returncode == 0 and required <= produced,
    }
