#!/usr/bin/env python3
"""Check mixed-length batching, stopping and cancellation on a live service."""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from validate_dsv4_service import _completion, _json_request, _signature
from blockspec.vllm_ascend.artifacts import write_private_json


def wait_idle(base_url):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
            lines = response.read().decode().splitlines()
        counts = [float(line.rsplit(" ", 1)[-1]) for line in lines
                  if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
        if counts and all(value == 0 for value in counts):
            return
        time.sleep(1)
    raise AssertionError("service did not become idle after cancellation")


def cancel_stream(base_url, model, prompt):
    payload = dict(model=model, prompt=prompt, max_tokens=512, temperature=0.0,
                   stream=True, ignore_eos=True, return_token_ids=True)
    request = urllib.request.Request(
        base_url + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    saw_token = False
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            item = json.loads(line[6:])
            if any(choice.get("token_ids") or choice.get("text") for choice in item["choices"]):
                saw_token = True
                break  # Closing the SSE connection must abort the active request.
    assert saw_token, "stream ended before emitting a token"
    wait_idle(base_url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    args = parser.parse_args()
    url = args.base_url.rstrip("/")
    baseline = json.loads(args.baseline.read_text())
    # Different lengths force requests to finish and leave the batch independently.
    cases = baseline["greedy"][::3]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda old: _completion(url, args.model, old["case"]), cases))
    concurrent_exact = all(_signature(old) == _signature(new)
                           for old, new in zip(cases, records, strict=True))

    first = baseline["greedy"][0]
    payload = dict(model=args.model, prompt=first["case"]["prompt"],
                   max_tokens=32, temperature=0.0, return_token_ids=True)
    forced_eos = _json_request(url + "/v1/completions", {
        **payload, "allowed_token_ids": [args.eos_token_id],
    })
    assert forced_eos["choices"][0]["finish_reason"] == "stop"
    assert forced_eos["usage"]["completion_tokens"] == 1
    normal = _json_request(url + "/v1/completions", payload)
    stop_id = normal["choices"][0]["token_ids"][0]
    stopped = _json_request(url + "/v1/completions", {**payload, "stop_token_ids": [stop_id]})
    assert stopped["choices"][0]["finish_reason"] == "stop"
    assert stopped["usage"]["completion_tokens"] == 1

    cancel_stream(url, args.model, first["case"]["prompt"])
    replay = _completion(url, args.model, first["case"])
    replay_exact = _signature(replay) == _signature(first)
    report = dict(passed=concurrent_exact and replay_exact,
                  concurrent_greedy_exact=concurrent_exact,
                  concurrent_prompt_lengths=[r["usage"]["prompt_tokens"] for r in records],
                  eos_stop=True, explicit_token_stop=True, cancelled_stream_drained=True,
                  post_cancel_replay_exact=replay_exact)
    write_private_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
