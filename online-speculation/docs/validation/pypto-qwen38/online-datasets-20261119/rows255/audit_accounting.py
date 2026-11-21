"""只读检查完整 AR / 固定起草 / 持续更新记录的计时与请求账目。

此检查不代替预留题隔离、源码/权重哈希和硬件资源审核。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        isinstance(actual, (int, float))
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
        f"{label}: {actual!r} != {expected!r}",
    )


def audit(report, stride):
    require(report.get("completed") and report.get("clean_shutdown"), "run incomplete")
    require(not report.get("diagnostic_only") and not report.get("audit_replay"), "diagnostic run")
    require(report["requests"] == report["ar_checked_requests"], "AR does not cover full cohort")
    require(report["max_updates"] is None, "updates were capped")
    require(report["online_update_policy"] in {"continuous_periodic", "continuous_two_stage"},
            "not a continuous update policy")
    require(report["benchmark_layout"] == "interleaved_ar_frozen_online", "unmatched timing layout")
    require(report["stream_window"] == report["execution_max_num_seqs"] == 2, "different execution window")

    results = report["results"]
    online = results[str(stride)]
    arms = {"ar": results["ar"], "fixed": results["frozen"], "online": online}
    lengths = None
    rows = {}
    excluded_stats = ("updates", "publications", "feedback_ns", "update_ns", "publish_ns")
    for name, arm in arms.items():
        timeline = arm["request_timeline"]
        ids = [i for window in timeline for i in window["requests"]]
        require(ids == list(range(1, report["requests"] + 1)), f"{name}: request coverage")
        require(all(len(window["requests"]) == len(window["output_tokens"]) == 2
                    for window in timeline), f"{name}: malformed request window")
        current_lengths = [n for window in timeline for n in window["output_tokens"]]
        require(all(type(n) is int and n > 0 for n in current_lengths), f"{name}: invalid output length")
        if lengths is None:
            lengths = current_lengths
        require(current_lengths == lengths, f"{name}: per-request output lengths differ")

        stream = arm["stream"]
        require(sum(current_lengths) == stream["tokens"], f"{name}: output token sum")
        require(all(math.isfinite(window["seconds"]) and window["seconds"] > 0
                    for window in timeline), f"{name}: invalid request timing")
        tail = stream["tail_seconds"]
        require(math.isfinite(tail) and tail >= 0, f"{name}: invalid tail timing")
        close(stream["seconds"], sum(window["seconds"] for window in timeline) + tail,
              f"{name}: total time including tail")
        close(stream["tok_s"], stream["tokens"] / stream["seconds"], f"{name}: throughput")
        # tail 可包含最后一个异步更新；没有 tail 时，所有计数必须逐窗口守恒。
        for key, value in stream["stats"].items():
            require(type(value) is int and value >= 0, f"{name}: invalid counter {key}")
            subtotal = sum(window["stats"][key] for window in timeline)
            require(value >= subtotal, f"{name}: excluded counter {key}")
            if tail == 0:
                require(value == subtotal, f"{name}: unaccounted counter {key}")
        if name != "online":
            require(all(stream["stats"][key] == 0 for key in excluded_stats),
                    f"{name}: baseline performed learning")
        rows[name] = {key: stream[key] for key in ("tokens", "seconds", "tail_seconds", "tok_s")}

    updates = online["updates"]
    require(updates and len(updates) == online["stream"]["stats"]["updates"], "update count mismatch")
    require([u["version"] for u in updates] == list(range(1, len(updates) + 1)),
            "optimizer versions are not consecutive")
    require(all(math.isfinite(u["loss"]) and math.isfinite(u["gradient_norm"])
                and u["gradient_norm"] >= 0 for u in updates), "nonfinite training state")
    require(any(u["gradient_norm"] > 0 for u in updates), "all updates have zero gradient")
    require(all(a["round"] <= b["round"] for a, b in zip(updates, updates[1:])),
            "nonmonotonic update rounds")
    schedule = online["update_schedule"]
    require(not schedule["update_limit_reached"] and schedule["max_updates"] is None,
            "stream stopped updating at a cap")
    require(schedule["completed_updates"] == len(updates), "schedule count mismatch")
    require(schedule["last_update_round"] == updates[-1]["round"], "last update mismatch")
    require(schedule["rounds_after_last_update"] == schedule["rounds"] - updates[-1]["round"],
            "last update tail mismatch")
    close(online["net_increment"], rows["online"]["tok_s"] / rows["fixed"]["tok_s"] - 1,
          "gain versus fixed")
    close(online["net_increment_vs_ar"], rows["online"]["tok_s"] / rows["ar"]["tok_s"] - 1,
          "gain versus AR")
    require(online["exact_greedy"] is True, "greedy output mismatch")
    for label, stream in (("final seen", online["learned_replay"]["stream"]),
                          ("final holdout", online["holdout"])):
        require(all(stream["stats"][key] == 0 for key in excluded_stats),
                f"{label}: frozen evaluation performed learning")
    require(online["learned_replay"]["exact_greedy"] is True, "final seen output mismatch")
    return dict(
        accounting_passed=True, requests=report["requests"], arms=rows,
        updates=len(updates), publications=online["stream"]["stats"]["publications"],
        last_update_round=updates[-1]["round"], rounds_after_last_update=schedule["rounds_after_last_update"],
        online_vs_ar=online["net_increment_vs_ar"], online_vs_fixed=online["net_increment"],
        final_seen_tok_s=online["learned_replay"]["stream"]["tok_s"],
        independent_holdout_tok_s=online["holdout"]["tok_s"],
        scope="Request/timing/update accounting only; frozen recipe, data and source audited separately.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--stride", type=int, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = audit(json.loads(raw), args.stride)
    result["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
