"""Combine compatible GSM8K pass@k reports with paired comparisons."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

from experiments.arllm.gsm8k_passk import _estimated_pass_at_k
from experiments.shared.artifacts import file_sha256 as _file_sha256
from inference_scaling.shared.rng import SeedStream


def _paired_difference(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    draws: int,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    reference_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in reference["per_problem"]
    }
    candidate_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in candidate["per_problem"]
    }
    if reference_by_problem.keys() != candidate_by_problem.keys():
        raise ValueError("paired pass@k methods do not contain the same problem rows")

    problem_indices = tuple(reference_by_problem)
    comparisons: dict[str, Any] = {}
    for k_text in reference["estimated_pass_at_k"]:
        if k_text not in candidate["estimated_pass_at_k"]:
            raise ValueError("paired pass@k methods do not contain the same k values")
        k = int(k_text)
        differences = [
            _estimated_pass_at_k(candidate_by_problem[index], draws, k)
            - _estimated_pass_at_k(reference_by_problem[index], draws, k)
            for index in problem_indices
        ]
        rng = random.Random(SeedStream(seed).derive("paired", k))
        bootstrap = sorted(
            statistics.fmean(
                differences[rng.randrange(len(differences))] for _ in differences
            )
            for _ in range(replicates)
        )
        comparisons[k_text] = {
            "candidate_minus_reference": statistics.fmean(differences),
            "paired_problem_bootstrap_95": [
                bootstrap[int(0.025 * replicates)],
                bootstrap[int(0.975 * replicates)],
            ],
        }
    return comparisons


def _cost_comparison(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | str]:
    candidate_seconds = float(candidate["seconds_excluding_model_load"])
    candidate_flops = int(candidate["estimated_dense_forward_flops"])
    if candidate_seconds <= 0 or candidate_flops <= 0:
        raise ValueError("candidate pass@k cost must be positive")
    return {
        "reference_over_candidate_inference_wall_time": (
            float(reference["seconds_excluding_model_load"]) / candidate_seconds
        ),
        "reference_over_candidate_inference_flops": (
            int(reference["estimated_dense_forward_flops"]) / candidate_flops
        ),
        "interpretation": (
            "The pair name is candidate_minus_reference, while each cost ratio uses "
            "reference as numerator and candidate as denominator. A cost ratio above "
            "one means the candidate used less of that inference-only resource."
        ),
    }


def _concise_method(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "examples",
        "draws_per_example",
        "generated_answers",
        "single_draw_accuracy",
        "estimated_pass_at_k",
        "estimated_pass_at_k_problem_bootstrap_95",
        "mean_unique_parsed_answers_across_all_draws",
        "mean_unique_full_outputs_across_all_draws",
        "unparseable_fraction",
        "total_forward_token_slots",
        "estimated_dense_forward_flops",
        "estimated_dense_forward_petaflops",
        "seconds_excluding_model_load",
        "seconds_per_generated_answer",
        "continuous_batching",
        "compute_by_model",
        "continuous_batching_by_model",
    )
    return {key: summary[key] for key in keys if key in summary}


def summarize_is_raw_chunks(
    path: Path, reports: list[dict[str, Any]]
) -> dict[str, Any]:
    actual_sha256 = _file_sha256(path)
    matching = [
        report
        for report in reports
        if str(report.get("raw_chunks_sha256", "")) == actual_sha256
    ]
    if len(matching) != 1:
        raise ValueError("IS raw chunks must match exactly one input report SHA-256")
    report = matching[0]
    chunks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        raise ValueError("IS raw chunks are empty")
    if any(
        chunk.get("manifest_fingerprint") != report["manifest_fingerprint"]
        for chunk in chunks
    ):
        raise ValueError("IS raw chunks and final report use different manifests")

    records_by_method: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        records_by_method.setdefault(str(chunk["method"]), []).extend(chunk["records"])
    if set(records_by_method) != set(report["methods"]):
        raise ValueError("IS raw chunks and final report contain different methods")

    method_diagnostics: dict[str, Any] = {}
    for method, records in records_by_method.items():
        expected_records = int(report["methods"][method]["generated_answers"])
        if len(records) != expected_records:
            raise ValueError(f"{method} raw record count does not match its final report")
        diagnostics = [record["diagnostics"] for record in records]
        rollout_counts = [int(item["rollout_evaluations"]) for item in diagnostics]
        total_rollouts = sum(rollout_counts)
        if total_rollouts <= 0:
            raise ValueError(f"{method} has no rollout diagnostics")

        def weighted(
            field: str,
            diagnostics=diagnostics,
            rollout_counts=rollout_counts,
            total_rollouts=total_rollouts,
        ) -> float:
            return sum(
                float(item[field]) * count
                for item, count in zip(diagnostics, rollout_counts, strict=True)
            ) / total_rollouts

        clipped = sum(int(item["clipped_rollout_corrections"]) for item in diagnostics)
        method_diagnostics[method] = {
            "records": len(records),
            "rollout_evaluations": total_rollouts,
            "mean_guidance_steps_per_record": statistics.fmean(
                int(item["guidance_steps"]) for item in diagnostics
            ),
            "mean_rollout_ess_weighted_by_rollout_evaluations": weighted(
                "mean_rollout_ess"
            ),
            "mean_rollout_reward_weighted_by_rollout_evaluations": weighted(
                "mean_rollout_reward"
            ),
            "mean_absolute_raw_log_importance_correction": weighted(
                "mean_absolute_raw_log_importance_correction"
            ),
            "mean_absolute_applied_log_importance_correction": weighted(
                "mean_absolute_applied_log_importance_correction"
            ),
            "clipped_rollout_corrections": clipped,
            "clipped_rollout_correction_fraction": clipped / total_rollouts,
        }

    pairwise_agreement: dict[str, Any] = {}
    for left, right in combinations(records_by_method, 2):
        left_by_key = {
            (int(item["problem_index"]), int(item["draw_index"])): item
            for item in records_by_method[left]
        }
        right_by_key = {
            (int(item["problem_index"]), int(item["draw_index"])): item
            for item in records_by_method[right]
        }
        if left_by_key.keys() != right_by_key.keys():
            raise ValueError("IS methods do not contain the same raw draw/problem grid")
        keys = tuple(left_by_key)
        pairwise_agreement[f"{left}_vs_{right}"] = {
            "samples": len(keys),
            "exact_output_match_fraction": statistics.fmean(
                left_by_key[key]["output_sha256"]
                == right_by_key[key]["output_sha256"]
                for key in keys
            ),
            "parsed_answer_match_fraction": statistics.fmean(
                left_by_key[key]["prediction"] == right_by_key[key]["prediction"]
                for key in keys
            ),
            "correctness_match_fraction": statistics.fmean(
                bool(left_by_key[key]["correct"])
                == bool(right_by_key[key]["correct"])
                for key in keys
            ),
        }

    return {
        "source": {"path": str(path), "sha256": actual_sha256},
        "manifest_fingerprint": report["manifest_fingerprint"],
        "methods": method_diagnostics,
        "pairwise_agreement": pairwise_agreement,
        "weighting_note": (
            "Correction and reward means weight each per-record mean by that record's "
            "rollout evaluation count. ESS uses the same weighting; rollout count is "
            "fixed within this experiment, so it is also proportional to the number "
            "of candidate-step ESS values."
        ),
    }


def combine_reports(
    reports: list[dict[str, Any]],
    sources: list[dict[str, str]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if not reports or len(reports) != len(sources):
        raise ValueError("at least one report and one matching source are required")
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")

    draws = int(reports[0]["draws_per_problem"])
    problem_indices = tuple(int(value) for value in reports[0]["problem_indices"])
    methods: dict[str, dict[str, Any]] = {}
    source_by_method: dict[str, str] = {}
    full_summaries: dict[str, dict[str, Any]] = {}
    method_order: list[str] = []
    for report, source in zip(reports, sources, strict=True):
        if int(report["draws_per_problem"]) != draws:
            raise ValueError("pass@k reports use different draw counts")
        if tuple(int(value) for value in report["problem_indices"]) != problem_indices:
            raise ValueError("pass@k reports use different problem rows or order")
        for method, summary in report["methods"].items():
            if method in methods:
                raise ValueError(f"duplicate pass@k method across reports: {method}")
            per_problem = tuple(
                int(item["problem_index"]) for item in summary["per_problem"]
            )
            if per_problem != problem_indices:
                raise ValueError(f"{method} per-problem rows do not match the report")
            methods[method] = _concise_method(summary)
            full_summaries[method] = summary
            source_by_method[method] = source["path"]
            method_order.append(method)

    paired: dict[str, Any] = {}
    for reference, candidate in combinations(method_order, 2):
        comparison_seed = SeedStream(bootstrap_seed).derive(reference, candidate)
        paired[f"{candidate}_minus_{reference}"] = {
            "pass_at_k": _paired_difference(
                full_summaries[reference],
                full_summaries[candidate],
                draws=draws,
                seed=comparison_seed,
                replicates=bootstrap_replicates,
            ),
            "cost": _cost_comparison(
                full_summaries[reference], full_summaries[candidate]
            ),
        }

    return {
        "schema_version": 1,
        "benchmark": reports[0]["benchmark"],
        "problem_indices": list(problem_indices),
        "draws_per_problem": draws,
        "method_order": method_order,
        "methods": methods,
        "source_by_method": source_by_method,
        "paired_comparisons": paired,
        "source_reports": sources,
        "bootstrap": {
            "unit": "problem",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
        },
        "cost_scope": (
            "Inference only. GRPO training cost is intentionally absent and must be "
            "added separately for end-to-end or amortized comparisons. Each method's "
            "FLOPs already sum model-specific parameter_count * forward_token_slots."
        ),
        "limitations": (
            "Problem bootstrap intervals capture row sampling uncertainty only. They do "
            "not include model, prompt, hyperparameter, finite-chain mixing, or finite "
            "importance-weight approximation uncertainty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--is-raw-chunks", type=Path)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    sources = [
        {"path": str(path), "sha256": _file_sha256(path)} for path in args.inputs
    ]
    combined = combine_reports(
        reports,
        sources,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    if args.is_raw_chunks is not None:
        combined["is_raw_diagnostics"] = summarize_is_raw_chunks(
            args.is_raw_chunks, reports
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
