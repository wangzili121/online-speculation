#!/usr/bin/env python3
"""Check a DSV4 OpenAI-compatible service and execute one decode."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.reporting import dumps  # noqa: E402
from blockspec.vllm_ascend.artifacts import encode_private_json  # noqa: E402


def request(url: str, payload: dict | None, timeout: float) -> dict | str:
    data = None if payload is None else encode_private_json(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=timeout
    ) as response:
        body = response.read().decode("utf-8")
    if not body:
        return ""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8901")
    parser.add_argument("--model", default="dsv4-orthrus")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            request(base + "/health", None, args.timeout)
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(2.0, max(deadline - time.monotonic(), 0.0)))

    models = request(base + "/v1/models", None, args.timeout)
    if not isinstance(models, dict):
        raise SystemExit("/v1/models did not return JSON")
    available = {
        item.get("id") for item in models.get("data", []) if isinstance(item, dict)
    }
    if args.model not in available:
        raise SystemExit(f"served model {args.model!r} is absent; available={sorted(available)}")

    completion = request(
        base + "/v1/completions",
        {
            "model": args.model,
            "prompt": "Reply with OK.",
            "max_tokens": 2,
            "temperature": 0,
        },
        args.timeout,
    )
    if not isinstance(completion, dict) or not completion.get("choices"):
        raise SystemExit("completion smoke request returned no choices")
    print(
        dumps(
            {
                "status": "ok",
                "base_url": base,
                "model": args.model,
                "completion_id": completion.get("id"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
