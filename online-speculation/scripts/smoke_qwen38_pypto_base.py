#!/usr/bin/env python3
"""Check OneServing HTTP and shutdown, optionally loading an Orthrus head."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def exchange(base, path, payload=None, *, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--head-dir", type=Path, help="enable the Orthrus runner instead of target-only")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args()
    if (args.device < 0 or args.output_dir.exists() or args.startup_timeout <= 0
            or not 3 <= args.tokens <= 64):
        parser.error("require an assigned device, new output directory, positive timeout and 3..64 tokens")
    args.output_dir.mkdir(parents=True)
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    script = "serve_qwen38_pypto.py" if args.head_dir else "serve_qwen38_pypto_base.py"
    command = [sys.executable, str(Path(__file__).with_name(script)),
               "--source", str(args.source), "--model", str(args.model),
               "--port", str(port)]
    if args.head_dir:
        command += ["--device", str(args.device), "--head-dir", str(args.head_dir)]
    else:
        command += ["--devices", str(args.device), "--tp", "1"]
    scope = "orthrus_qwen38_http" if args.head_dir else "native_qwen38_target_only_http"
    report = dict(scope=scope, orthrus_enabled=bool(args.head_dir), passed=False)
    server = None
    start = time.monotonic()
    with (args.output_dir / "server.log").open("xb") as log:
        try:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (args.output_dir / "server.json").write_text(json.dumps(dict(pid=server.pid, port=port)) + "\n")
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"native server exited during startup: {server.returncode}; see server.log")
                try:
                    report["health"] = exchange(base, "/health", timeout=2)
                    break
                except (URLError, TimeoutError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native server did not become ready")
                    time.sleep(.5)
            report["startup_seconds"] = time.monotonic() - start
            print(scope + " server ready", flush=True)

            completion = exchange(base, "/v1/completions", dict(prompt="Hello", max_tokens=args.tokens, temperature=0))
            report["completion"] = completion
            choice = completion["choices"][0]
            if not choice["token_ids"] or not choice["text"].strip():
                raise AssertionError("native completion returned no visible output")
            print("completion: " + choice["text"], flush=True)

            chat = exchange(base, "/v1/chat/completions", dict(
                messages=[dict(role="user", content="Reply with just the number: 1+1=?")],
                max_tokens=args.tokens, temperature=0))
            report["chat"] = chat
            choice = chat["choices"][0]
            if not choice["token_ids"] or not choice["message"]["content"].strip():
                raise AssertionError("native chat returned no visible output")
            print("chat: " + choice["message"]["content"], flush=True)

            request = Request(base + "/v1/chat/completions", data=json.dumps(dict(
                messages=[dict(role="user", content="Say hello briefly.")], max_tokens=min(args.tokens, 4),
                temperature=0, stream=True)).encode(), headers={"Content-Type": "application/json"})
            events, done = [], False
            with urlopen(request, timeout=300) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    value = line[6:]
                    if value == "[DONE]":
                        done = True
                        break
                    event = json.loads(value)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    events.append(event)
            streamed = [token for event in events for row in event.get("choices", [])
                        for token in row.get("token_ids", [])]
            report["sse"] = dict(done=done, token_ids=streamed, events=events)
            if not done or not streamed:
                raise AssertionError("native SSE did not stream tokens and finish")

            report["metrics"] = exchange(base, "/metrics")
            if report["metrics"]["requests"]["errors"]:
                raise AssertionError("native service recorded a request error")
            report["passed"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            if server is not None and server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=30)
            report["server_returncode"] = None if server is None else server.returncode
            report["clean_shutdown"] = server is not None and server.returncode in (0, -signal.SIGTERM)
            shutdown_failed = report["passed"] and not report["clean_shutdown"]
            if shutdown_failed:
                report.update(passed=False, error="native service required a forced shutdown")
            report["seconds"] = time.monotonic() - start
            (args.output_dir / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            if shutdown_failed:
                raise RuntimeError(report["error"])


if __name__ == "__main__":
    main()
