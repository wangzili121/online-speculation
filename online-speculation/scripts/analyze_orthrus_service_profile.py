"""Summarize scheduler, KV and speculative batches from MS Service Profiler."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import statistics

from blockspec.reporting import dump, dumps


def number(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=None),
    }


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_batches(rows):
    engine = [row for row in rows if row["name"] == "specDecoding"]
    framework = [row for row in rows
                 if row["name"] == "batchFrameworkProcessing"]
    batch_types = {}
    for row in engine:
        name = row["batch_type"] or "unknown"
        batch_types[name] = batch_types.get(name, 0) + 1
    request_widths = []
    for row in engine:
        resources = ast.literal_eval(row["res_list"])
        request_widths.append(len(resources))
    return {
        "engine_batches": len(engine),
        "batch_types": batch_types,
        "request_width": distribution(request_widths),
        "prefill_requests": distribution([
            number(row["prefill_batch_size"]) for row in engine]),
        "decode_requests": distribution([
            number(row["decode_batch_size"]) for row in engine]),
        "scheduled_tokens": distribution([
            number(row["total_scheduled_tokens"]) for row in engine]),
        "batch_duration_ms": distribution([
            number(row["during_time(ms)"]) for row in engine]),
        "framework_duration_ms": distribution([
            number(row["during_time(ms)"]) for row in framework]),
    }


def summarize_spec_decode(rows):
    groups = {}
    accepted = 0.0
    drafted = 0.0
    for row in rows:
        key = (row["start_datetime"], row["end_datetime"])
        item = groups.setdefault(key, {
            "duration_ms": number(row["draft_model_time_ms"]),
            "requests": 0,
        })
        item["requests"] += 1
        accepted += number(row["accepted_tokens"])
        drafted += number(row["spec_tokens"])
    return {
        "draft_batches": len(groups),
        "request_rows": len(rows),
        "batch_requests": distribution([
            item["requests"] for item in groups.values()]),
        "draft_duration_ms": distribution([
            item["duration_ms"] for item in groups.values()]),
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def summarize_kv(rows):
    usages = [number(row["kvcache_usage_rate"]) for row in rows
              if row["kvcache_usage_rate"]]
    return {"usage": distribution(usages),
            "samples_at_capacity": sum(value >= 0.999 for value in usages)}


def findings(summary):
    result = []
    draft = summary["spec_decode"]["draft_duration_ms"]
    framework = summary["batches"]["framework_duration_ms"]
    if draft["p50"] and framework["p50"]:
        ratio = framework["p50"] / draft["p50"]
        result.append({
            "id": "framework-versus-draft",
            "evidence": (f"P50 scheduler framework time is {framework['p50']:.3f} ms "
                         f"versus {draft['p50']:.3f} ms for draft execution "
                         f"({ratio * 100:.2f}%)"),
            "implication": "draft execution shape dominates generic scheduler bookkeeping",
        })
    kv = summary["kv_cache"]["usage"]
    if kv["p99"] and kv["p99"] >= 0.99:
        result.append({
            "id": "kv-at-capacity",
            "evidence": (f"KV usage P99 is {kv['p99'] * 100:.2f}% and "
                         f"{summary['kv_cache']['samples_at_capacity']} samples "
                         "reached 99.9%"),
            "implication": "temporary draft-state reductions can also improve tail progress",
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = Path(args.profile).resolve()
    analysis = profile / "service-analysis"
    result = {
        "schema_version": 1,
        "profile": str(profile),
        "batches": summarize_batches(read_rows(analysis / "batch.csv")),
        "spec_decode": summarize_spec_decode(
            read_rows(analysis / "spec_decode.csv")),
        "kv_cache": summarize_kv(read_rows(analysis / "kvcache.csv")),
    }
    result["findings"] = findings(result)
    with open(args.output, "x", encoding="utf-8") as handle:
        dump(result, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
