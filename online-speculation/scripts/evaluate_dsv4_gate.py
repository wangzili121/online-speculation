#!/usr/bin/env python3
"""Evaluate the 5% Orthrus goodput gate over paired vLLM result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.reporting import dumps  # noqa: E402


def _key(path: Path, label: str) -> str:
    prefix = label + "-"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def _load(directory: Path, label: str) -> dict[str, dict[str, Any]]:
    return {
        _key(path, label): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


def _goodput(result: dict[str, Any]) -> float:
    for key in ("request_goodput", "request_goodput_rps", "goodput"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError("benchmark result has no request goodput; run with --goodput SLOs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--orthrus-dir", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-improvement", type=float, default=0.05)
    args = parser.parse_args()
    correctness = json.loads(args.correctness.read_text(encoding="utf-8"))
    target = _load(args.target_dir, "target")
    orthrus = _load(args.orthrus_dir, "orthrus")
    if not target or target.keys() != orthrus.keys():
        raise SystemExit("target and Orthrus benchmark result sets do not match")
    rows = []
    for key in target:
        baseline = _goodput(target[key])
        candidate = _goodput(orthrus[key])
        improvement = candidate / baseline - 1 if baseline > 0 else float("-inf")
        drafts = orthrus[key].get("spec_decode_num_drafts")
        draft_tokens = orthrus[key].get("spec_decode_draft_tokens")
        accepted_tokens = orthrus[key].get("spec_decode_accepted_tokens")
        acceptance_length = orthrus[key].get("spec_decode_acceptance_length")
        has_spec_metrics = all(
            isinstance(value, (int, float))
            for value in (drafts, draft_tokens, accepted_tokens, acceptance_length)
        ) and float(drafts) > 0
        rejection_rate = (
            1.0 - float(accepted_tokens) / float(draft_tokens)
            if has_spec_metrics and float(draft_tokens) > 0
            else None
        )
        rows.append(
            {
                "case": key,
                "target_goodput": baseline,
                "orthrus_goodput": candidate,
                "improvement": improvement,
                "acceptance_length": acceptance_length,
                "rejection_rate": rejection_rate,
                "spec_metrics_present": has_spec_metrics,
                "passed": has_spec_metrics and improvement >= args.minimum_improvement,
            }
        )
    passed = bool(correctness.get("passed")) and all(row["passed"] for row in rows)
    report = {
        "schema": "dsv4-orthrus-gate-v1",
        "passed": passed,
        "correctness_passed": bool(correctness.get("passed")),
        "minimum_improvement": args.minimum_improvement,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
