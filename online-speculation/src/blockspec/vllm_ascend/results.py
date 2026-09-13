"""Comparable summaries for Orthrus serving trials."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _counter(engine, suffix):
    for name, value in engine.get("counter_delta", {}).items():
        if name.endswith(suffix):
            return value
    return 0.0


def summarize_trial(directory):
    directory = Path(directory)
    with open(directory / "result.json", encoding="utf-8") as handle:
        result = json.load(handle)
    with open(directory / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    summary = result["summary"]
    engine = result.get("engine", {})
    accepted = _counter(engine, "spec_decode_num_accepted_tokens_total")
    drafted = _counter(engine, "spec_decode_num_draft_tokens_total")
    preemptions = _counter(engine, "num_preemptions_total")
    trial = manifest["trial"]
    return {
        "trial_id": trial["trial_id"],
        "requests": trial["requests"],
        "mns": trial["max_num_seqs"],
        "mbt": trial["max_num_batched_tokens"],
        "memory": trial["gpu_memory_utilization"],
        "graph": trial["graph_mode"],
        "async": trial["async_scheduling"],
        "success_rate": summary["success_rate"],
        "request_throughput": summary["request_throughput"],
        "output_token_throughput": summary["output_token_throughput"],
        "slo_goodput": summary["slo_goodput"],
        "p95_ttft": summary["ttft_seconds"]["p95"],
        "p95_tpot": summary["tpot_seconds"]["p95"],
        "max_running": engine.get("max_running", 0.0),
        "max_waiting": engine.get("max_waiting", 0.0),
        "max_kv_cache_usage": engine.get("max_kv_cache_usage", 0.0),
        "preemptions": preemptions,
        "acceptance_rate": accepted / drafted if drafted else None,
        "workload_sha256": manifest["inputs"]["workload_sha256"],
    }


def load_trials(root):
    rows = []
    for result in sorted(Path(root).glob("*/result.json")):
        manifest = result.parent / "manifest.json"
        if manifest.exists():
            rows.append(summarize_trial(result.parent))
    return rows


def rank_trials(rows, *, require_no_preemption=False):
    valid = [row for row in rows if row["success_rate"] == 1.0
             and (not require_no_preemption or row["preemptions"] == 0)]
    if not valid:
        return []
    if len({row["workload_sha256"] for row in valid}) != 1:
        raise ValueError("trials with different workloads cannot be ranked together")
    metric = ("slo_goodput" if any(row["slo_goodput"] > 0 for row in valid)
              else "request_throughput")
    return sorted(valid, key=lambda row: row[metric], reverse=True)


def write_csv(path, rows):
    if not rows:
        raise ValueError("at least one completed trial is required")
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_trials(rows):
    return json.dumps(rows, ensure_ascii=True, indent=2)
