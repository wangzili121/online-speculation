#!/usr/bin/env python3
"""汇总独立采集的设备时间线，避免把并行 block 的耗时相加。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(trace):
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "event"]
    if not events or any(not math.isfinite(e[k]) for e in events for k in ("ts", "dur")):
        raise ValueError("finite device task events are required")
    if any(e["dur"] < 0 for e in events):
        raise ValueError("negative task duration")
    epochs = {e.get("args", {}).get("run_epoch") for e in events}
    if len(epochs) != 1:
        raise ValueError("profile contains multiple dispatch epochs")
    groups = {}
    for event in events:
        groups.setdefault(event["name"].split("(")[0], []).append(event["dur"])
    return dict(device_task_window_us=max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events),
                run_epoch=next(iter(epochs)), block_records=len(events),
                tasks={name: dict(blocks=len(values), min_us=min(values), median_us=statistics.median(values),
                                  max_us=max(values)) for name, values in groups.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-result", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    component = json.loads(args.component_result.read_text())
    if not component["passed"] or not component["npu_execution"]:
        parser.error("a passed NPU result is required")
    result = dict(scope="separate_instrumented_last_case_profiles", profiles={})
    for backend, directory in component["artifact_directories"].items():
        files = list((args.work_dir / directory).rglob("merged_swimlane*.json"))
        if not files:
            raise FileNotFoundError(f"no device profile for {backend}")
        path = max(files, key=lambda p: p.stat().st_mtime_ns)
        data = path.read_bytes()
        result["profiles"][backend] = dict(source=str(path), sha256=hashlib.sha256(data).hexdigest(),
                                           **summarize(json.loads(data)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
