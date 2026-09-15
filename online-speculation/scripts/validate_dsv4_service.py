#!/usr/bin/env python3
"""Snapshot and compare target-only and Orthrus OpenAI-compatible services."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.reporting import dumps  # noqa: E402
from blockspec.vllm_ascend.artifacts import (  # noqa: E402
    encode_private_json,
    write_private_json,
)


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else encode_private_json(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def _choice_record(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or choice.get("token_ids") or []
    return {
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text", ""),
        "tokens": tokens,
        "usage": response.get("usage", {}),
    }


def _completion(base_url: str, model: str, case: dict[str, Any]) -> dict[str, Any]:
    prompt = case["prompt"] * int(case.get("prompt_repeat", 1))
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(case["max_tokens"]),
        "temperature": float(case.get("temperature", 0.0)),
        "logprobs": 1,
    }
    for key in ("seed", "top_p"):
        if key in case:
            payload[key] = case[key]
    started = time.perf_counter()
    response = _json_request(base_url.rstrip("/") + "/v1/completions", payload)
    record = _choice_record(response)
    record["elapsed_seconds"] = time.perf_counter() - started
    record["case"] = case
    return record


def snapshot(args: argparse.Namespace) -> int:
    config = json.loads(args.cases.read_text(encoding="utf-8"))
    models = _json_request(args.base_url.rstrip("/") + "/v1/models")
    available = {item["id"] for item in models.get("data", [])}
    if args.model not in available:
        raise SystemExit(f"model {args.model!r} is not served; available={sorted(available)}")

    greedy_cases = list(config["greedy"]) + list(config.get("boundary_greedy", []))
    if args.include_long:
        greedy_cases.extend(config.get("long_greedy", []))
    greedy = [_completion(args.base_url, args.model, case) for case in greedy_cases]
    repeated_case = dict(config["greedy"][0])
    repeated = [
        _completion(args.base_url, args.model, repeated_case)
        for _ in range(int(config.get("cache_repetitions", 4)))
    ]
    sampling = config["sampling"]
    sampled_cases = [
        {**sampling, "seed": seed}
        for seed in range(int(sampling.get("seeds", 64)))
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        sampled = list(
            pool.map(lambda case: _completion(args.base_url, args.model, case), sampled_cases)
        )
    replay_count = min(int(sampling.get("fixed_seed_replays", 8)), len(sampled_cases))
    sampled_replay = [
        _completion(args.base_url, args.model, case)
        for case in sampled_cases[:replay_count]
    ]
    output = {
        "schema": "dsv4-service-snapshot-v1",
        "base_url": args.base_url,
        "model": args.model,
        "greedy": greedy,
        "cache_repetitions": repeated,
        "sampling": sampled,
        "sampling_replay": sampled_replay,
    }
    write_private_json(args.output, output)
    print(args.output.resolve())
    return 0


def _signature(record: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    return record["text"], tuple(record.get("tokens", []))


def _first_token(record: dict[str, Any]) -> str:
    tokens = record.get("tokens") or []
    return str(tokens[0]) if tokens else record.get("text", "")[:1]


def _empirical_tv(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_counts = Counter(_first_token(item) for item in left)
    right_counts = Counter(_first_token(item) for item in right)
    keys = left_counts.keys() | right_counts.keys()
    left_n = max(len(left), 1)
    right_n = max(len(right), 1)
    return 0.5 * sum(
        abs(left_counts[key] / left_n - right_counts[key] / right_n) for key in keys
    )


def compare(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if baseline.get("schema") != "dsv4-service-snapshot-v1" or candidate.get("schema") != baseline.get("schema"):
        raise SystemExit("unsupported or mismatched snapshot schema")
    greedy_equal = len(baseline["greedy"]) == len(candidate["greedy"]) and all(
        _signature(left) == _signature(right)
        for left, right in zip(baseline["greedy"], candidate["greedy"], strict=True)
    )
    baseline_repeat = {_signature(item) for item in baseline["cache_repetitions"]}
    candidate_repeat = {_signature(item) for item in candidate["cache_repetitions"]}
    cache_stable = len(baseline_repeat) == 1 and len(candidate_repeat) == 1 and baseline_repeat == candidate_repeat
    baseline_seed_replay = all(
        _signature(original) == _signature(replay)
        for original, replay in zip(
            baseline["sampling"], baseline.get("sampling_replay", []), strict=False
        )
    ) and bool(baseline.get("sampling_replay"))
    candidate_seed_replay = all(
        _signature(original) == _signature(replay)
        for original, replay in zip(
            candidate["sampling"], candidate.get("sampling_replay", []), strict=False
        )
    ) and bool(candidate.get("sampling_replay"))
    pairs = list(zip(baseline["sampling"], candidate["sampling"], strict=True))
    fixed_seed_equal = sum(_signature(left) == _signature(right) for left, right in pairs)
    fixed_seed_rate = fixed_seed_equal / max(len(pairs), 1)
    tv = _empirical_tv(baseline["sampling"], candidate["sampling"])
    passed = (
        greedy_equal
        and cache_stable
        and baseline_seed_replay
        and candidate_seed_replay
        and fixed_seed_rate >= args.min_fixed_seed_match
        and tv <= args.max_first_token_tv
        and math.isfinite(tv)
    )
    report = {
        "schema": "dsv4-correctness-report-v1",
        "passed": passed,
        "greedy_exact": greedy_equal,
        "cache_repetition_exact": cache_stable,
        "baseline_fixed_seed_reproducible": baseline_seed_replay,
        "candidate_fixed_seed_reproducible": candidate_seed_replay,
        "cross_service_fixed_seed_match_rate_diagnostic": fixed_seed_rate,
        "first_token_empirical_tv": tv,
        "thresholds": {
            "min_fixed_seed_match": args.min_fixed_seed_match,
            "max_first_token_tv": args.max_first_token_tv,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    snap = subparsers.add_parser("snapshot")
    snap.add_argument("--base-url", required=True)
    snap.add_argument("--model", required=True)
    snap.add_argument("--cases", type=Path, required=True)
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--concurrency", type=int, default=4)
    snap.add_argument("--include-long", action="store_true")
    snap.set_defaults(func=snapshot)
    check = subparsers.add_parser("compare")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument(
        "--min-fixed-seed-match",
        type=float,
        default=0.0,
        help="optional cross-service diagnostic; distributional correctness does not require it",
    )
    check.add_argument("--max-first-token-tv", type=float, default=0.10)
    check.set_defaults(func=compare)
    args = parser.parse_args()
    if getattr(args, "concurrency", 1) < 1:
        parser.error("--concurrency must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
