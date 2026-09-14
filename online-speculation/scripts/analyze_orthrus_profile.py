"""Derive reproducible Orthrus bottleneck evidence from raw Ascend artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from blockspec.reporting import dump, dumps


MATERIALIZATION_OPS = {"Transpose", "RepeatInterleaveV2", "GatherV3", "ConcatD"}
DRAFT_ATTENTION_OPS = {"FlashAttentionScore"}
VERIFY_ATTENTION_OPS = {"FusedInferAttentionScore"}


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


def find_output(profile):
    matches = list(profile.glob("torch-raw/*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one Ascend output under {profile}, found {matches}")
    return matches[0]


def op_summary(output):
    with (output / "op_statistic.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = [{
        "op": row["OP Type"],
        "count": int(row["Count"]),
        "total_us": number(row["Total Time(us)"]),
        "ratio_percent": number(row["Ratio(%)"]),
    } for row in rows]
    total = sum(row["total_us"] for row in parsed)

    def group(names):
        duration = sum(row["total_us"] for row in parsed if row["op"] in names)
        return {"total_us": duration,
                "ratio_percent": 100 * duration / total if total else 0}

    return {
        "total_kernel_us": total,
        "top_ops": parsed[:20],
        "draft_kv_materialization": group(MATERIALIZATION_OPS),
        "draft_attention": group(DRAFT_ATTENTION_OPS),
        "verify_attention": group(VERIFY_ATTENTION_OPS),
        "draft_materialization_plus_attention": group(
            MATERIALIZATION_OPS | DRAFT_ATTENTION_OPS),
    }


def utilization(output):
    with (output / "step_trace_time.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    computing = sum(number(row["Computing"]) for row in rows)
    free = sum(number(row["Free"]) for row in rows)
    stage = sum(number(row["Stage"]) for row in rows)
    return {
        "computing_us": computing,
        "free_us": free,
        "stage_us": stage,
        "device_busy_percent": 100 * computing / stage if stage else None,
        "device_free_percent": 100 * free / stage if stage else None,
    }


def framework_spans(output):
    aggregate = defaultdict(lambda: {"count": 0, "host_self_us": 0.0,
                                     "host_total_us": 0.0,
                                     "device_total_us": 0.0})
    with (output / "operator_details.csv").open(
            encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            name = row["Name"]
            if not name.startswith("orthrus."):
                continue
            item = aggregate[name]
            item["count"] += 1
            item["host_self_us"] += number(row["Host Self Duration(us)"])
            item["host_total_us"] += number(row["Host Total Duration(us)"])
            item["device_total_us"] += number(row["Device Total Duration(us)"])
    return dict(sorted(aggregate.items(),
                       key=lambda item: item[1]["device_total_us"], reverse=True))


def algorithm_events(profile):
    batches = []
    admissions = []
    histories = []
    request_ids = set()
    for path in profile.glob("orthrus-events-*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("name") == "orthrus.phase.admission":
                    admissions.append(event.get("args", {}))
                    continue
                if event.get("name") != "orthrus.draft.batch" or event.get("ph") != "X":
                    continue
                args = event.get("args", {})
                batches.append({
                    "duration_us": number(event.get("dur")),
                    "requests": int(args.get(
                        "draft_batch_requests", args.get("batch_requests", 0))),
                    "scheduled_tokens": int(args.get("scheduled_tokens", 0)),
                })
                histories.extend(int(value) for value in args.get("history_tokens", []))
                request_ids.update(args.get("request_ids", []))
    durations = [row["duration_us"] for row in batches]
    widths = [row["requests"] for row in batches]
    result = {
        "draft_batches": len(batches),
        "distinct_requests": len(request_ids),
        "batch_requests": {
            "p50": percentile(widths, 0.5), "p95": percentile(widths, 0.95),
            "max": max(widths, default=None),
        },
        "history_tokens": {
            "p50": percentile(histories, 0.5),
            "p95": percentile(histories, 0.95),
            "max": max(histories, default=None),
        },
        "draft_batch_ms": {
            "mean": statistics.fmean(durations) / 1000 if durations else None,
            "p50": (percentile(durations, 0.5) / 1000 if durations else None),
            "p95": (percentile(durations, 0.95) / 1000 if durations else None),
        },
    }
    if admissions:
        ready = [int(row.get("draft_ready", 0)) for row in admissions]
        admitted = [int(row.get("draft_admitted", 0)) for row in admissions]
        deferred = [int(row.get("draft_deferred", 0)) for row in admissions]
        ages = [int(row.get("max_queue_age", 0)) for row in admissions]
        result["phase_admission"] = {
            "steps": len(admissions),
            "ready_p50": percentile(ready, 0.5),
            "ready_p95": percentile(ready, 0.95),
            "admitted_p50": percentile(admitted, 0.5),
            "admitted_p95": percentile(admitted, 0.95),
            "deferred_p50": percentile(deferred, 0.5),
            "deferred_p95": percentile(deferred, 0.95),
            "max_queue_age": max(ages, default=0),
        }
    return result


def load_result(profile):
    path = profile / "load.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    engine = value.get("engine", {}).copy()
    engine.pop("telemetry", None)
    return {"config": value.get("config"), "summary": value.get("summary"),
            "profile_window": value.get("profile_window"), "engine": engine}


def execution_shape(spans):
    layers = spans.get("orthrus.draft.layer", {}).get("count", 0)
    native_calls = spans.get(
        "orthrus.draft.attention.native_gqa", {}).get("count", 0)
    paged_calls = spans.get(
        "orthrus.draft.attention.paged_history", {}).get("count", 0)
    local_calls = spans.get(
        "orthrus.draft.attention.local", {}).get("count", 0)
    return {
        "draft_layer_invocations": layers,
        "native_attention_calls": native_calls,
        "paged_history_calls": paged_calls,
        "local_attention_calls": local_calls,
        "history_attention_calls_per_layer": (
            (native_calls + paged_calls) / layers if layers else None),
        "all_attention_calls_per_layer": (
            (native_calls + paged_calls + local_calls) / layers
            if layers else None),
    }


def path_attribution(operators, spans):
    total = operators["total_kernel_us"]

    def device(name):
        return spans.get(name, {}).get("device_total_us", 0.0)

    segmented_attention = (
        device("orthrus.draft.attention.paged_history")
        + device("orthrus.draft.attention.local"))
    segmented_merge = device("orthrus.draft.attention.merge")
    fused_total = operators["verify_attention"]["total_us"]
    verify_estimate = max(0.0, fused_total - segmented_attention)

    def item(duration):
        return {"total_us": duration,
                "ratio_percent": 100 * duration / total if total else 0.0}

    return {
        "segmented_draft_attention": item(segmented_attention),
        "segmented_merge": item(segmented_merge),
        "verifier_fused_attention_estimate": item(verify_estimate),
    }


def findings(summary):
    result = []
    use = summary["utilization"]
    ops = summary["operators"]
    if use["device_busy_percent"] and use["device_busy_percent"] > 90:
        result.append({
            "id": "device-saturated",
            "evidence": f"NPU computing occupies {use['device_busy_percent']:.1f}% of the captured stage",
            "implication": "admission-only scheduling is not the first-order bottleneck",
        })
    elif use["device_busy_percent"] and use["device_busy_percent"] < 75:
        result.append({
            "id": "device-underfed-inside-draft",
            "evidence": (f"NPU is busy for only {use['device_busy_percent']:.1f}% "
                         "of the captured stage"),
            "implication": ("batch Orthrus draft attention across requests before "
                            "adding more external admission"),
        })
    shape = summary["execution_shape"]
    history_calls = shape["history_attention_calls_per_layer"]
    if history_calls is not None and history_calls > 2:
        result.append({
            "id": "per-request-draft-launches",
            "evidence": (
                f"{shape['native_attention_calls']} native draft attention calls "
                f"were issued for {shape['draft_layer_invocations']} layer "
                f"invocations ({history_calls:.1f} per layer)"),
            "implication": ("replace the Python request loop with one variable-length "
                            "TND attention call per segment and layer"),
        })
    elif shape["paged_history_calls"] and history_calls is not None:
        result.append({
            "id": "draft-attention-batched",
            "evidence": (f"paged history attention is issued {history_calls:.1f} "
                         "time per draft layer"),
            "implication": "the per-request draft launch fragmentation is removed",
        })
    material = ops["draft_kv_materialization"]
    if material["ratio_percent"] > 10:
        result.append({
            "id": "draft-gqa-materialization",
            "evidence": ("Transpose/RepeatInterleave/Gather/Concat consume "
                         f"{material['ratio_percent']:.1f}% of NPU kernel time"),
            "implication": "use native GQA and paged/segmented KV without head expansion",
        })
    if not shape["paged_history_calls"]:
        draft = ops["draft_materialization_plus_attention"]
        verify = ops["verify_attention"]
        result.append({
            "id": "draft-dominates-verify-attention",
            "evidence": (
                f"draft materialization+attention is {draft['ratio_percent']:.1f}% "
                f"versus verifier attention {verify['ratio_percent']:.1f}%"),
            "implication": "optimize Orthrus draft attention before generic verifier kernels",
        })
    else:
        paths = summary["path_attribution"]
        draft = paths["segmented_draft_attention"]["ratio_percent"]
        verify = paths["verifier_fused_attention_estimate"]["ratio_percent"]
        result.append({
            "id": "attention-after-segmentation",
            "evidence": (f"segmented draft attention accounts for about {draft:.1f}% "
                         f"and residual verifier FIA about {verify:.1f}% of kernel time"),
            "implication": "future attention work must distinguish draft from verifier FIA",
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = Path(args.profile).resolve()
    output = find_output(profile)
    result = {
        "schema_version": 1,
        "profile": str(profile),
        "ascend_output": str(output),
        "operators": op_summary(output),
        "utilization": utilization(output),
        "framework_spans": framework_spans(output),
        "algorithm": algorithm_events(profile),
        "load": load_result(profile),
    }
    result["execution_shape"] = execution_shape(result["framework_spans"])
    result["path_attribution"] = path_attribution(
        result["operators"], result["framework_spans"])
    result["findings"] = findings(result)
    with open(args.output, "x", encoding="utf-8") as handle:
        dump(result, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
