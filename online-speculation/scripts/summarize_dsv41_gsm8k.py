#!/usr/bin/env python3
"""Summarize one or more DSV4.1 GSM8K target/Orthrus result directories."""

import argparse
import json
from pathlib import Path
import re


ANSWER = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def numeric(text):
    matches = ANSWER.findall(text or "")
    return matches[-1].replace(",", "") if matches else None


def load_run(path):
    path = Path(path)
    if (path / "result.json").is_file():
        result = json.loads((path / "result.json").read_text())
    elif (path / "acceptance" / "orthrus" / "result.json").is_file():
        result = json.loads((path / "acceptance" / "orthrus" / "result.json").read_text())
    else:
        raise ValueError(f"Cannot find a result.json below {path}")
    requests = result["requests"]
    scored = [row for row in requests if row.get("reference_answer") and row.get("output_text") is not None]
    correct = sum(numeric(row["output_text"]) == numeric(row["reference_answer"]) for row in scored)
    speculation = result.get("speculation", {})
    proposed = speculation.get("proposed_candidates", 0)
    accepted = speculation.get("accepted_candidates", 0)
    online = result.get("online") or {}
    return {
        "path": str(path),
        "requests": len(requests),
        "output_tokens": result.get("output_tokens", sum(len(row["token_ids"]) for row in requests)),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "output_tokens_per_second": result.get("output_tokens_per_second"),
        "gsm8k_scored": len(scored),
        "gsm8k_exact_match": correct / len(scored) if scored else None,
        "draft_rounds": speculation.get("draft_rounds"),
        "proposed_candidates": proposed,
        "accepted_candidates": accepted,
        "candidate_acceptance_rate": accepted / proposed if proposed else None,
        "mean_accepted_per_round": (
            accepted / speculation["draft_rounds"] if speculation.get("draft_rounds") else None
        ),
        "updates": online.get("updates"),
        "publications": online.get("publications"),
        "update_seconds": online.get("update_seconds"),
        "publication_seconds": online.get("publication_seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [load_run(path) for path in args.runs]
    report = {"format": "orthrus-dsv41-gsm8k-summary-v1", "runs": rows}
    if len(rows) > 1:
        target = rows[0].get("output_tokens_per_second")
        for row in rows[1:]:
            throughput = row.get("output_tokens_per_second")
            row["speedup_vs_first"] = throughput / target if target and throughput else None
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
