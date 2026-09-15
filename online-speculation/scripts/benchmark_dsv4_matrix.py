#!/usr/bin/env python3
"""Run the versioned DSV4 random-token vLLM service benchmark matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.vllm_ascend.artifacts import write_private_json  # noqa: E402


MAX_LENGTH = {"prod-128k": 133120, "prod-1m": 1048576}
SPEC_COUNTERS = {
    "spec_decode_num_drafts": "vllm:spec_decode_num_drafts",
    "spec_decode_draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "spec_decode_accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def _case_id(case: dict[str, Any]) -> str:
    rate = str(case["request_rate"]).replace(".", "p")
    return (
        f"in{case['input_len']}-out{case['output_len']}-"
        f"c{case['concurrency']}-r{rate}"
    )


def _parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def _spec_metrics(base_url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        metrics = _parse_prometheus(response.read().decode("utf-8"))
    values = {}
    for output_name, metric_name in SPEC_COUNTERS.items():
        values[output_name] = metrics.get(
            metric_name + "_total", metrics.get(metric_name, 0.0)
        )
    return values


def _attach_spec_delta(
    result_path: Path,
    before: dict[str, float],
    after: dict[str, float],
) -> None:
    if not result_path.is_file():
        raise FileNotFoundError(f"vLLM benchmark result was not created: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for name in SPEC_COUNTERS:
        result[name] = max(after[name] - before[name], 0.0)
    drafts = result["spec_decode_num_drafts"]
    accepted = result["spec_decode_accepted_tokens"]
    result["spec_decode_acceptance_length"] = (
        1.0 + accepted / drafts if drafts > 0 else None
    )
    write_private_json(result_path, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer",
        default="/data/disk/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8",
        help="local target tokenizer path; --model is the served API name",
    )
    parser.add_argument("--profile", choices=tuple(MAX_LENGTH), required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label", choices=("target", "dspark", "orthrus"), required=True
    )
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--metrics-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.matrix.read_text(encoding="utf-8"))
    cases = config["profiles"][args.profile]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if int(case["input_len"]) + int(case["output_len"]) > MAX_LENGTH[args.profile]:
            raise SystemExit(f"case exceeds {args.profile}: {case}")
        identifier = _case_id(case)
        result_name = f"{args.label}-{args.profile}-{identifier}.json"
        command = [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--base-url", args.base_url,
            "--model", args.model,
            "--tokenizer", args.tokenizer,
            "--tokenizer-mode", "deepseek_v4",
            "--dataset-name", "random",
            "--random-input-len", str(case["input_len"]),
            "--random-output-len", str(case["output_len"]),
            "--num-prompts", str(case["num_prompts"]),
            "--max-concurrency", str(case["concurrency"]),
            "--request-rate", str(case["request_rate"]),
            "--seed", str(args.seed),
            "--ignore-eos",
            "--save-result",
            "--result-dir", str(args.output_dir),
            "--result-filename", result_name,
            "--percentile-metrics", "ttft,tpot,itl,e2el",
        ]
        slos = config.get("slo", [])
        if slos:
            command.append("--goodput")
            command.extend(slos)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            before = _spec_metrics(args.base_url, args.metrics_timeout)
            subprocess.run(command, check=True)
            after = _spec_metrics(args.base_url, args.metrics_timeout)
            _attach_spec_delta(args.output_dir / result_name, before, after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
