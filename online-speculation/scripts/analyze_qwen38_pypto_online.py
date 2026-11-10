#!/usr/bin/env python3
"""汇总真实在线学习对照，按问题配对估计接受长度变化。"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics


def totals(rows):
    return {key: sum(row.get(key, 0) for row in rows)
            for key in ("rounds", "accepted", "proposed", "output_tokens", "seconds")}


def prefix_length(rows):
    values = totals(rows)
    return values["accepted"] / values["rounds"] if values["rounds"] else None


def paired_rows(before, after):
    left = {row["id"]: row for row in before}
    right = {row["id"]: row for row in after}
    if len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys() or not left:
        raise ValueError("distinct paired question IDs required")
    return list(left.values()), [right[key] for key in left]


def paired_prefix_change(before, after, *, samples=10000, seed=20261111):
    """按问题重采样后重新计算总接受数/总轮数，保留同题前后的相关性。"""
    before, after = paired_rows(before, after)
    initial, final = prefix_length(before), prefix_length(after)
    if initial is None or final is None:
        raise ValueError("nonempty speculative rounds required")
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        selected = rng.choices(range(len(before)), k=len(before))
        x = prefix_length([before[i] for i in selected])
        y = prefix_length([after[i] for i in selected])
        if x is not None and y is not None:
            differences.append(y - x)
    if not differences:
        raise ValueError("no resampled speculative rounds")
    differences.sort()
    return dict(questions=len(before), before=initial, after=final, delta=final-initial,
                ci95=[differences[int((len(differences)-1)*p)] for p in (.025, .975)],
                improved_questions=sum(prefix_length([y]) > prefix_length([x])
                    for x, y in zip(before, after) if x["rounds"] and y["rounds"]),
                bootstrap_samples=len(differences), bootstrap_seed=seed)


def validate_evaluation(entry):
    for name in ("dev", "test"):
        if not entry.get(name):
            continue
        group = entry[name]
        rows, refs = paired_rows(group["rows"], group["target_rows"])
        if any(row["tokens"] != ref["tokens"] for row, ref in zip(rows, refs)):
            raise ValueError("speculative output differs from target")
        for row in (*rows, *refs):
            if row["updates"] or row["head_version_before"] != row["head_version_after"]:
                raise ValueError("evaluation changed the head")


def analyze(report, *, allow_incomplete=False):
    if not report.get("completed") and not allow_incomplete:
        raise ValueError("complete run required")
    if report.get("error"):
        raise ValueError(f"experiment failed: {report['error']}")
    arguments = report["arguments"]
    updates = report["updates"]
    for i, row in enumerate(updates, 1):
        if row["step"] != i or row["version"] != i:
            raise ValueError("noncontiguous optimizer or published head versions")
        if any(not math.isfinite(row[k]) for k in ("loss", "gradient_norm", "clip_scale", "seconds")):
            raise ValueError("nonfinite update statistics")
        if sum(row["feedback_valid_rows"]) != row["reached_positions"]:
            raise ValueError("supervised row accounting mismatch")
        if len(row["feedback_valid_rows"]) != arguments["replay_blocks"]:
            raise ValueError("incomplete feedback batch")
    if report.get("completed") and (len(updates) != arguments["steps"] or not report.get("clean_shutdown")):
        raise ValueError("unfinished update budget or worker cleanup")
    splits = report["dataset"]["splits"]
    hashes = [row["question_sha256"] for values in splits.values() for row in values]
    if len(set(hashes)) != len(hashes):
        raise ValueError("questions overlap across partitions")
    evaluations = sorted(report["evaluations"], key=lambda row: row["step"])
    for entry in evaluations:
        validate_evaluation(entry)
    valid = Counter(n for row in updates for n in row["feedback_valid_rows"])
    result = dict(last_layers=arguments["last_layers"], completed=report.get("completed", False),
        updates=len(updates), training_questions=len(report["train"]),
        training=totals(report["train"]), supervised_positions=sum(row["reached_positions"] for row in updates),
        feedback_valid_row_histogram=dict(sorted(valid.items())),
        feedback_mean_valid_rows=statistics.mean(list(valid.elements())) if valid else None,
        update_seconds=sum(row["seconds"] for row in updates),
        median_update_seconds=statistics.median(row["seconds"] for row in updates) if updates else None,
        first8_mean_tv=statistics.mean(row["loss"] for row in updates[:8]) if updates else None,
        last8_mean_tv=statistics.mean(row["loss"] for row in updates[-8:]) if updates else None,
        dev_curve=[dict(step=row["step"], **row["dev"]["summary"]) for row in evaluations if row.get("dev")])
    if evaluations:
        first, final = evaluations[0], evaluations[-1]
        for split in ("dev", "test"):
            if first.get(split) and final.get(split) and first["step"] != final["step"]:
                result[split] = paired_prefix_change(first[split]["rows"], final[split]["rows"])
                result[split]["steps"] = [first["step"], final["step"]]
                result[split]["initial_summary"] = first[split]["summary"]
                result[split]["final_summary"] = final[split]["summary"]
    if report.get("online_net"):
        result["online_net"] = report["online_net"]
        result["online_net"] = {k: v for k, v in result["online_net"].items() if k != "target_rows"}
    if report.get("frozen_stream"):
        result["frozen_stream"] = report["frozen_stream"]["summary"]
        if not result["frozen_stream"]["greedy_equal_all"] or not result["online_net"]["summary"]["greedy_equal_all"]:
            raise ValueError("unpaired or non-lossless stream comparison")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    a = p.parse_args()
    reports = [json.loads(path.read_text()) for path in a.reports]
    common = ("steps", "width", "tokens", "stride", "replay_blocks", "learning_rate", "warmup_steps",
              "seed", "train_count", "dev_count", "test_count", "max_prompt", "frozen_stream_control")
    first = reports[0]
    for report in reports[1:]:
        if (any(report["arguments"][k] != first["arguments"][k] for k in common)
                or report["dataset"]["tokenized_sha256"] != first["dataset"]["tokenized_sha256"]):
            raise ValueError("controlled runs changed the data or hyperparameters")
        if (report.get("initial_head_sha256") and first.get("initial_head_sha256")
                and report["initial_head_sha256"] != first["initial_head_sha256"]):
            raise ValueError("controlled runs did not start from the same head")
    result = dict(runs=[analyze(report, allow_incomplete=a.allow_incomplete) for report in reports])
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
