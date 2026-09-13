"""Low-overhead request-aware JSONL events for bounded profile windows."""

from __future__ import annotations

from contextlib import contextmanager
import atexit
import glob
import json
import os
from pathlib import Path
import threading
import time


class TraceWriter:
    def __init__(self, path):
        path = Path(path.format(pid=os.getpid()))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "a", buffering=1, encoding="utf-8")
        self.lock = threading.Lock()
        atexit.register(self.handle.close)
        self.write(
            "orthrus.clock_sync", phase="i",
            timestamp_us=time.perf_counter_ns() / 1000,
            args={"wall_time_us": time.time_ns() / 1000})

    def write(self, name, *, phase, timestamp_us, duration_us=None, args=None):
        event = {
            "name": name,
            "cat": "orthrus",
            "ph": phase,
            "ts": timestamp_us,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "args": args or {},
        }
        if duration_us is not None:
            event["dur"] = duration_us
        if phase == "i":
            event["s"] = "t"
        with self.lock:
            self.handle.write(json.dumps(event, separators=(",", ":")) + "\n")


_WRITER = None


def writer():
    global _WRITER
    path = os.environ.get("ORTHRUS_TRACE_FILE")
    if not path:
        return None
    if _WRITER is None:
        _WRITER = TraceWriter(path)
    return _WRITER


def instant(name, args=None):
    output = writer()
    if output is not None:
        output.write(name, phase="i", timestamp_us=time.perf_counter_ns() / 1000,
                     args=args)


@contextmanager
def span(name, args=None):
    output = writer()
    if output is None:
        yield
        return
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        output.write(name, phase="X", timestamp_us=start / 1000,
                     duration_us=(end - start) / 1000, args=args)


def merge_events(patterns, output):
    events = []
    for pattern in patterns:
        for path in [Path(value) for value in sorted(glob.glob(pattern))]:
            with open(path, encoding="utf-8") as handle:
                events.extend(json.loads(line) for line in handle if line.strip())
    events.sort(key=lambda event: event["ts"])
    with open(output, "x", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle, separators=(",", ":"))
        handle.write("\n")
    return len(events)
