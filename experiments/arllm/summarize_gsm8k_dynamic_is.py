"""Summarize the controlled GSM8K dynamic-candidate replay experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from inference_scaling.shared.rng import SeedStream
from experiments.shared.artifacts import load_jsonl as _load_jsonl
from experiments.shared.methods import AR_DYNAMIC_METHODS

METHODS = AR_DYNAMIC_METHODS


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_quality(
    records: Sequence[dict[str, Any]],
    reference: str,
    candidate: str,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    differences = [
        float(record["methods"][candidate]["correct"])
        - float(record["methods"][reference]["correct"])
        for record in records
    ]
    generator = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        indices = generator.integers(0, len(differences), size=len(differences))
        bootstrap.append(statistics.fmean(differences[index] for index in indices))
    return {
        "candidate_minus_reference_accuracy": statistics.fmean(differences),
        "paired_problem_bootstrap_95": [
            _quantile(bootstrap, 0.025),
            _quantile(bootstrap, 0.975),
        ],
        "candidate_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "reference_wins": sum(value < 0 for value in differences),
    }


def _sum(records: Sequence[dict[str, Any]], method: str, field: str) -> float:
    return float(sum(record["methods"][method][field] for record in records))


def _method_summary(records: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    correct = sum(bool(record["methods"][method]["correct"]) for record in records)
    history_used = _sum(records, method, "history_used")
    fresh_used = _sum(records, method, "fresh_used")
    history_generated = _sum(records, method, "history_generated")
    cache_hits = _sum(records, method, "candidate_cache_hits")
    nonterminal = _sum(records, method, "nonterminal_candidates")
    auxiliary = _sum(records, method, "auxiliary_candidates")
    candidates = _sum(records, method, "candidate_count")
    step_count = _sum(records, method, "steps")
    summary = {
        "correct": correct,
        "examples": len(records),
        "accuracy": correct / len(records),
        "steps": int(step_count),
        "history_generated": int(history_generated),
        "history_used": int(history_used),
        "fresh_used": int(fresh_used),
        "rollout_reuse_fraction": (
            history_used / (history_used + fresh_used)
            if history_used + fresh_used
            else 0.0
        ),
        "cache_record_utilization": (
            history_used / history_generated if history_generated else 0.0
        ),
        "candidate_replay_hit_rate": cache_hits / nonterminal if nonterminal else 0.0,
        "auxiliary_candidate_fraction": auxiliary / candidates if candidates else 0.0,
        "mean_outer_weight_ess": (
            _sum(records, method, "outer_weight_ess_sum") / step_count
            if step_count
            else 0.0
        ),
        "mean_final_weight_ess": (
            _sum(records, method, "final_weight_ess_sum") / step_count
            if step_count
            else 0.0
        ),
        "proxy_budget": _sum(records, method, "proxy_budget"),
        "proxy_cost_used": _sum(records, method, "proxy_cost_used"),
        "cache_build_seconds": _sum(records, method, "cache_build_seconds"),
        "design_seconds": _sum(records, method, "design_seconds"),
        "steady_online_seconds": _sum(records, method, "steady_online_seconds"),
        "online_total_seconds": _sum(records, method, "online_total_seconds"),
        "one_shot_seconds": _sum(records, method, "one_shot_seconds"),
        "cache_build_forward_token_slots": int(
            _sum(records, method, "cache_build_forward_token_slots")
        ),
        "design_forward_token_slots": int(
            _sum(records, method, "design_forward_token_slots")
        ),
        "steady_online_forward_token_slots": int(
            _sum(records, method, "steady_online_forward_token_slots")
        ),
        "online_total_forward_token_slots": int(
            _sum(records, method, "online_total_forward_token_slots")
        ),
        "cache_build_estimated_dense_forward_flops": int(
            _sum(records, method, "cache_build_estimated_dense_forward_flops")
        ),
        "design_estimated_dense_forward_flops": int(
            _sum(records, method, "design_estimated_dense_forward_flops")
        ),
        "steady_online_estimated_dense_forward_flops": int(
            _sum(records, method, "steady_online_estimated_dense_forward_flops")
        ),
        "online_total_estimated_dense_forward_flops": int(
            _sum(records, method, "online_total_estimated_dense_forward_flops")
        ),
        "one_shot_estimated_dense_forward_flops": int(
            _sum(records, method, "one_shot_estimated_dense_forward_flops")
        ),
        "candidate_reproduction_all": all(
            record["methods"][method]["candidate_reproduction_all"] for record in records
        ),
    }
    return summary


def _factor(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("inf")


def build_summary(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    expected = [int(value) for value in manifest["effective"]["problem_indices"]]
    by_index = {int(record["problem_index"]): record for record in records}
    if sorted(by_index) != sorted(expected):
        raise ValueError("dynamic-candidate records do not cover the manifest problem grid")
    ordered = [by_index[index] for index in expected]
    fingerprint = manifest["fingerprint"]
    if any(record["manifest_fingerprint"] != fingerprint for record in ordered):
        raise ValueError("record fingerprint does not match the dynamic-candidate manifest")
    for record in ordered:
        if tuple(record["methods"]) != METHODS:
            raise ValueError("record method order does not match the comparison contract")

    methods = {method: _method_summary(ordered, method) for method in METHODS}
    base = methods["base_candidate_fixed"]
    fixed = methods["replay_aware_fixed"]
    optimal = methods["replay_aware_optimal"]
    seed = int(manifest["effective"]["config"]["run"]["seed"])
    fixed_quality = _paired_quality(
        ordered,
        "base_candidate_fixed",
        "replay_aware_fixed",
        seed=SeedStream(seed).derive("dynamic-summary", "fixed"),
        replicates=bootstrap_replicates,
    )
    optimal_quality = _paired_quality(
        ordered,
        "replay_aware_fixed",
        "replay_aware_optimal",
        seed=SeedStream(seed).derive("dynamic-summary", "optimal"),
        replicates=bootstrap_replicates,
    )
    return {
        "schema_version": 1,
        "benchmark": "GSM8K replay-aware candidate proposal and allocation",
        "manifest_fingerprint": fingerprint,
        "profile": manifest["effective"]["config"]["run"]["name"],
        "problem_indices": expected,
        "examples": len(ordered),
        "settings": manifest["effective"]["settings"],
        "input_weight_sha256": manifest["effective"]["input_weight_sha256"],
        "implementation_sha256": manifest["effective"]["implementation_sha256"],
        "methods": methods,
        "comparisons": {
            "replay_aware_fixed_vs_base_candidate_fixed": {
                **fixed_quality,
                "base_over_replay_aware_steady_online_flop_factor": _factor(
                    base["steady_online_estimated_dense_forward_flops"],
                    fixed["steady_online_estimated_dense_forward_flops"],
                ),
                "base_over_replay_aware_steady_online_wall_time_factor": _factor(
                    base["steady_online_seconds"], fixed["steady_online_seconds"]
                ),
                "base_over_replay_aware_one_shot_flop_factor": _factor(
                    base["one_shot_estimated_dense_forward_flops"],
                    fixed["one_shot_estimated_dense_forward_flops"],
                ),
                "candidate_replay_hit_rate_difference": (
                    fixed["candidate_replay_hit_rate"]
                    - base["candidate_replay_hit_rate"]
                ),
            },
            "replay_aware_optimal_vs_replay_aware_fixed": {
                **optimal_quality,
                "fixed_over_optimal_steady_online_flop_factor": _factor(
                    fixed["steady_online_estimated_dense_forward_flops"],
                    optimal["steady_online_estimated_dense_forward_flops"],
                ),
                "fixed_over_optimal_steady_online_wall_time_factor": _factor(
                    fixed["steady_online_seconds"], optimal["steady_online_seconds"]
                ),
                "fixed_over_optimal_one_shot_flop_factor": _factor(
                    fixed["one_shot_estimated_dense_forward_flops"],
                    optimal["one_shot_estimated_dense_forward_flops"],
                ),
                "final_weight_ess_difference": (
                    optimal["mean_final_weight_ess"] - fixed["mean_final_weight_ess"]
                ),
            },
        },
        "comparison_contract": {
            "dynamic_increment": (
                "base_candidate_fixed samples candidates only from the base policy, "
                "uses three fresh rollouts per nonterminal candidate, and does not build "
                "a replay cache; "
                "replay_aware_fixed uses the defensive base/small-model candidate mixture "
                "and the exact outer p/q ratio. Both freeze three estimator rollouts per "
                "nonterminal candidate, replacing fresh rollouts by cache records on hits."
            ),
            "allocation_increment": (
                "replay_aware_optimal keeps the same defensive proposal and the same "
                "per-step proxy-cost budget as replay_aware_fixed, but estimates history "
                "and fresh standard deviations from an independent design pool before "
                "revealing evaluation records."
            ),
            "steady_online": (
                "candidate generation, probability scoring, selected replay reads, and "
                "fresh correction; excludes cache construction and independent design-pool "
                "construction"
            ),
            "one_shot": (
                "cache construction plus design-pool construction plus the online decision"
            ),
            "factor_direction": "values above one mean the denominator method uses less cost",
        },
        "compute_definition": (
            "2 * each model parameter count * observed forward token slots; base and "
            "small-model contributions are accumulated separately"
        ),
        "bootstrap": {
            "unit": "GSM8K problem",
            "replicates": bootstrap_replicates,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = build_summary(
        manifest,
        _load_jsonl(args.records),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
