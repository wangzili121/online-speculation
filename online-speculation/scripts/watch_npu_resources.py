#!/usr/bin/env python3
"""Record local eight-card availability without allocating devices or stopping jobs."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from watch_glm53_validation import idle_devices, write_json


def probe(host):
    snapshot = subprocess.check_output(["/usr/local/bin/npu-smi", "info"], text=True, timeout=20)
    idle = idle_devices(snapshot)
    return dict(host=host, state="eight_card_window" if len(idle) == 8 else "monitoring",
                idle=idle, unavailable=sorted(set(range(8)) - set(idle)),
                eight_cards_idle=len(idle) == 8, action="observe_only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default=socket.gethostname())
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = None
    while not stopped:
        try:
            state = probe(args.host)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            state = dict(host=args.host, state="probe_failed", error=repr(error), action="observe_only")
        # 每轮更新心跳；只有资源状态变化时才追加日志。
        if state != previous:
            print(json.dumps(state), flush=True)
            previous = state
        write_json(args.output, dict(state, pid=os.getpid(),
                                    observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        time.sleep(args.poll_seconds)
    write_json(args.output, dict(host=args.host, state="stopped", pid=os.getpid(),
                                observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
