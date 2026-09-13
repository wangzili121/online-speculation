"""Coverage and paired quality/cost summaries for reasoning comparisons."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable

from experiments.shared.statistics import (
    bootstrap_mean_interval, clustered_paired_binary_difference, wilson_interval,
)


def record_key(row: dict[str, Any]) -> tuple[str, int, str, str, int]:
    return (row["problem_id"], row["draw"], row["method"], row["reward"], row["budget_forward_tokens"])


def comparison_coverage(records: list[dict[str, Any]], *, problem_ids: Iterable[str],
                        budgets: Iterable[int], draws: int, methods: Iterable[str],
                        rewards: Iterable[str]) -> dict[str, Any]:
    problems, budgets, methods, rewards = tuple(problem_ids), tuple(budgets), tuple(methods), tuple(rewards)
    if draws <= 0 or not problems or not budgets:
        raise ValueError("comparison coverage requires problems, budgets and positive draws")
    conditions = [(method + "_" + mode, "none") for method in ("base", "budget_base") if method in methods
                  for mode in ("disabled", "enabled")]
    conditions += [("vote", "none")] if "vote" in methods else []
    conditions += [(method, reward) for method in methods if method in {"is", "mh"} for reward in rewards]
    expected = {(problem, draw, method, reward, budget) for problem in problems for draw in range(draws)
                for method, reward in conditions for budget in budgets}
    observed = {record_key(row) for row in records}
    if len(observed) != len(records):
        raise ValueError("duplicate comparison records")
    missing = expected - observed
    return {"expected_records": len(expected), "completed_records": len(expected & observed),
            "missing_records": len(missing), "unexpected_records": len(observed - expected),
            "completed_problems": sum(not any(key[0] == problem for key in missing) for problem in problems),
            "expected_problems": len(problems), "complete": not missing and not (observed - expected)}


def summarize_reasoning(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("no comparison records are available")
    if len({record_key(row) for row in records}) != len(records):
        raise ValueError("duplicate comparison records")
    fingerprints = {row.get("manifest_fingerprint") for row in records}
    if len(fingerprints) > 1:
        raise ValueError("comparison records mix experiment configurations")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        cost = row["cost"]
        used = cost.get("generation_forward_token_slots", 0) + cost.get("score_forward_token_slots", 0)
        if used < 0 or used != row["used_forward_tokens"] or used > row["budget_forward_tokens"]:
            raise ValueError("invalid comparison token accounting")
        if cost["estimated_dense_forward_flops"] != 2 * row["parameter_count"] * used:
            raise ValueError("comparison FLOPs do not match the declared dense-forward estimator")
        grouped[(row["method"], row["reward"], row["budget_forward_tokens"])].append(row)
    summary = []
    for (method, reward, budget), rows in sorted(grouped.items()):
        trials, correct = len(rows), sum(row["correct"] for row in rows)
        by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_problem[row["problem_id"]].append(row)
        counts = {len(values) for values in by_problem.values()}
        accuracy = fmean(fmean(float(row["correct"]) for row in values) for values in by_problem.values())
        interval = (wilson_interval(correct, trials) if counts == {1} else
                    bootstrap_mean_interval([fmean(float(row["correct"]) for row in values)
                                             for values in by_problem.values()], seed=20260911))
        item: dict[str, Any] = {"method": method, "reward": reward, "budget_forward_tokens": budget,
            "trials": trials, "problems": len(by_problem), "correct": correct, "accuracy": accuracy,
            "accuracy_95": interval, "interval_method": "wilson" if counts == {1} else "problem_bootstrap",
            "balanced_draws": len(counts) == 1,
            "mean_used_forward_tokens": fmean(row["used_forward_tokens"] for row in rows),
            "mean_budget_utilization": fmean(row["used_forward_tokens"] for row in rows) / budget,
            "mean_generated_tokens": fmean(row["cost"].get("generated_tokens", 0) for row in rows),
            "mean_pfLOPs": fmean(row["cost"]["estimated_dense_forward_flops"] for row in rows) / 1e15,
            "mean_selected_tokens": fmean(row["selected_tokens"] for row in rows),
            "unparseable": sum(not row["parseable"] for row in rows),
            "incomplete_thinking": sum(row["thinking_status"] not in {"complete", "disabled"} for row in rows),
            "mh_updates": sum(row.get("updates", 0) for row in rows),
            "changed_mh_updates": sum(row.get("changed_updates", 0) for row in rows),
            "accepted_changed_mh_updates": sum(row.get("accepted_changed_updates", 0) for row in rows),
        }
        if method == "is":
            item["mean_ess"] = fmean(row["ess"] for row in rows)
            item["mean_conditional_expected_correct"] = fmean(row["conditional_expected_correct"] for row in rows)
        keys = {(row["problem_id"], row["draw"]) for row in rows}
        for baseline_method, field in (("base_enabled", "paired_vs_thinking_base"),
                                       ("budget_base_enabled", "paired_vs_budget_thinking_base")):
            baseline = grouped.get((baseline_method, "none", budget), [])
            baseline_keys = {(row["problem_id"], row["draw"]) for row in baseline}
            if method != baseline_method and keys == baseline_keys:
                indices = {problem: index for index, problem in enumerate(sorted(by_problem))}
                item[field] = clustered_paired_binary_difference(
                    [dict(row, cluster=indices[row["problem_id"]]) for row in rows],
                    [dict(row, cluster=indices[row["problem_id"]]) for row in baseline],
                    cluster_key="cluster", outcome_key="correct", seed=20260911)
        summary.append(item)
    return summary
