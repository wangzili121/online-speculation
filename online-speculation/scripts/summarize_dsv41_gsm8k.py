#!/usr/bin/env python3
"""Summarize one or more DSV4.1 GSM8K target/Orthrus result directories."""

import argparse
import json
from pathlib import Path
import re
from statistics import median


ANSWER = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def numeric(text):
    matches = ANSWER.findall(text or "")
    return matches[-1].replace(",", "") if matches else None


def staged_update_summary(online):
    updates = online["updates"]
    first_by_dp, warm = {}, []
    for update in updates:
        dp = str(update["dp"])
        elapsed = update["update_seconds"]
        if dp in first_by_dp:
            warm.append(elapsed)
        else:
            first_by_dp[dp] = elapsed
    phases = ("reverse_seconds", "stage_seconds", "optimizer_seconds", "publication_seconds")
    return {
        "updates": len(updates),
        "publications": sum(update["applied"] for update in updates),
        "head_version": online["version"],
        "rounds_after_publication": online["rounds_after_publication"],
        "update_seconds": online["update_seconds"],
        "first_update_seconds_by_dp": first_by_dp,
        "warm_updates": len(warm),
        "median_warm_update_seconds": median(warm) if warm else None,
        "decode_tok_per_s_excluding_updates": online["decode_tok_per_s_excluding_updates"],
        "wall_tok_per_s_excluding_updates": online.get("wall_tok_per_s_excluding_updates"),
        **{key: sum(update[key] for update in updates) for key in phases},
    }


def staged_run(path, result):
    """Compare only arms that served identical prompts in this node's report."""
    report = {key: result.get(key) for key in ("node", "passed", "parity", "online_passed",
                                              "frozen_eval_parity", "divergence", "initialization_seconds",
                                              "warmup_seconds", "configuration")}
    report.update(path=str(path), backend="staged-1122", arms={})
    arms = result.get("arms", {"ar": result} if "generated" in result else {})
    baselines = {"orthrus": "ar", "eval_initial": "eval_ar", "eval_frozen": "eval_ar"}
    for name, arm in arms.items():
        generated = arm["generated"]
        summary = {key: arm.get(key) for key in ("wall_tok_per_s", "decode_tok_per_s", "decode_steps",
                                                  "prefill_seconds", "decode_seconds", "decode_tokens",
                                                  "steady_decode", "max_live_requests", "stage_seconds",
                                                  "speculation", "acceptance")}
        summary.update(requests=len(generated), output_tokens=sum(map(len, generated.values())))
        baseline = arms.get(baselines.get(name))
        if baseline is not None and baseline["generated"] == generated and baseline["decode_tok_per_s"]:
            summary["decode_speedup"] = arm["decode_tok_per_s"] / baseline["decode_tok_per_s"]
        if "online" in arm:
            summary["online"] = staged_update_summary(arm["online"])
            if "decode_speedup" in summary:
                rate = summary["online"]["decode_tok_per_s_excluding_updates"]
                summary["online"]["decode_speedup_excluding_updates"] = rate / baseline["decode_tok_per_s"]
        report["arms"][name] = summary
    return report


def load_run(path):
    path = Path(path)
    if path.is_file():
        result = json.loads(path.read_text())
    elif (path / "result.json").is_file():
        result = json.loads((path / "result.json").read_text())
    elif (path / "acceptance" / "orthrus" / "result.json").is_file():
        result = json.loads((path / "acceptance" / "orthrus" / "result.json").read_text())
    else:
        raise ValueError(f"Cannot find a result.json below {path}")
    if "generated" in result or "divergence" in result:
        return staged_run(path, result)
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
    if any("arms" in row for row in rows):
        report["format"] = "orthrus-dsv41-gsm8k-summary-v2"
    elif len(rows) > 1:
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
