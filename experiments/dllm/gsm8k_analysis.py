"""Aggregate standardized dLLM draw records without loading a model."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.shared.statistics import (
    bootstrap_answer_distance,
    bootstrap_mean_interval,
    estimated_pass_at_k,
    jensen_shannon_bits,
    probability_distribution,
    total_variation_distance,
)
from experiments.shared.artifacts import load_jsonl


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing draw records: {path}")
    return load_jsonl(path)


def load_draw_grid(
    run_root: Path,
    methods: Sequence[str],
    draws: int,
) -> dict[str, list[dict[str, Any]]]:
    if not methods or draws <= 0:
        raise ValueError("methods and draws must be non-empty")
    result: dict[str, list[dict[str, Any]]] = {}
    expected_problems: set[int] | None = None
    for method in methods:
        records: list[dict[str, Any]] = []
        observed_keys: set[tuple[int, int]] = set()
        for draw_index in range(draws):
            draw_records = _load_jsonl(
                run_root / method / f"draw-{draw_index}" / "records.jsonl"
            )
            draw_problems = {int(record["problem_index"]) for record in draw_records}
            if expected_problems is None:
                expected_problems = draw_problems
            elif draw_problems != expected_problems:
                raise ValueError("draws do not share the same problem set")
            for record in draw_records:
                key = (int(record["draw_index"]), int(record["problem_index"]))
                if record["method"] != method or key[0] != draw_index:
                    raise ValueError(f"record metadata does not match {method} draw {draw_index}")
                if key in observed_keys:
                    raise ValueError(f"duplicate draw record: {method} {key}")
                observed_keys.add(key)
                records.append(record)
        if not records:
            raise ValueError(f"method {method!r} has no records")
        result[method] = records
    return result


def _compute_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    def total(role: str, field: str) -> float:
        return sum(float(record[role][field]) for record in records)

    main_flops = total("main_compute", "estimated_active_flops")
    proposal_flops = total("proposal_compute", "estimated_active_flops")
    return {
        "samples": len(records),
        "wall_clock_seconds": sum(float(record["elapsed_seconds"]) for record in records),
        "main_generation_forward_token_slots": total(
            "main_compute", "sample_model_token_slots"
        ),
        "main_exact_rescoring_forward_token_slots": total(
            "main_compute", "score_model_token_slots"
        ),
        "proposal_generation_forward_token_slots": total(
            "proposal_compute", "sample_model_token_slots"
        ),
        "proposal_exact_rescoring_forward_token_slots": total(
            "proposal_compute", "score_model_token_slots"
        ),
        "main_estimated_active_flops": main_flops,
        "proposal_estimated_active_flops": proposal_flops,
        "total_estimated_active_flops": main_flops + proposal_flops,
    }


def summarize_passk(
    records_by_method: Mapping[str, Sequence[dict[str, Any]]],
    *,
    draws: int,
    ks: Sequence[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if any(not 1 <= k <= draws for k in ks):
        raise ValueError("every k must lie in [1, draws]")
    methods: dict[str, Any] = {}
    problem_indices: list[int] | None = None
    for method, records in records_by_method.items():
        by_problem: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_problem.setdefault(int(record["problem_index"]), []).append(record)
        indices = sorted(by_problem)
        if problem_indices is None:
            problem_indices = indices
        elif indices != problem_indices:
            raise ValueError("methods do not share the same problem set")
        per_problem = []
        for problem_index in indices:
            problem_records = by_problem[problem_index]
            if len(problem_records) != draws:
                raise ValueError("pass@k requires every requested draw for every problem")
            correct = sum(bool(record["correct"]) for record in problem_records)
            per_problem.append(
                {
                    "problem_index": problem_index,
                    "correct_draws": correct,
                    "distinct_parsed_answers": len(
                        {record["prediction"] for record in problem_records}
                    ),
                    "pass_at_k": {
                        str(k): estimated_pass_at_k(correct, draws, k) for k in ks
                    },
                }
            )
        pass_at_k = {
            str(k): statistics.fmean(item["pass_at_k"][str(k)] for item in per_problem)
            for k in ks
        }
        intervals = {
            str(k): bootstrap_mean_interval(
                [item["pass_at_k"][str(k)] for item in per_problem],
                seed=bootstrap_seed + k,
                replicates=bootstrap_replicates,
            )
            for k in ks
        }
        methods[method] = {
            "pass_at_k": pass_at_k,
            "pass_at_k_bootstrap_95": intervals,
            "per_problem": per_problem,
            "compute": _compute_summary(records),
        }
    return {
        "schema_version": 1,
        "analysis": "pass_at_k",
        "draws_per_problem": draws,
        "k": list(ks),
        "problem_indices": problem_indices or [],
        "methods": methods,
    }


def summarize_distribution(
    records_by_method: Mapping[str, Sequence[dict[str, Any]]],
    *,
    draws: int,
    reference: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if reference not in records_by_method:
        raise ValueError(f"distribution reference is absent: {reference}")
    answer_draws: dict[str, dict[int, list[str]]] = {}
    method_summaries: dict[str, Any] = {}
    problem_indices: list[int] | None = None
    for method, records in records_by_method.items():
        grouped: dict[int, list[str]] = {}
        for record in records:
            answer = record["prediction"]
            grouped.setdefault(int(record["problem_index"]), []).append(
                "[invalid]" if answer is None else str(answer)
            )
        indices = sorted(grouped)
        if problem_indices is None:
            problem_indices = indices
        elif indices != problem_indices:
            raise ValueError("methods do not share the same problem set")
        if any(len(grouped[index]) != draws for index in indices):
            raise ValueError("distribution analysis requires a complete draw grid")
        answer_draws[method] = grouped
        method_summaries[method] = {
            "accuracy": sum(bool(record["correct"]) for record in records) / len(records),
            "per_problem_answer_distribution": {
                str(index): probability_distribution(Counter(grouped[index]))
                for index in indices
            },
            "compute": _compute_summary(records),
        }

    comparisons: dict[str, Any] = {}
    assert problem_indices is not None
    for method in records_by_method:
        if method == reference:
            continue
        per_problem = []
        for problem_index in problem_indices:
            left = probability_distribution(Counter(answer_draws[method][problem_index]))
            right = probability_distribution(
                Counter(answer_draws[reference][problem_index])
            )
            per_problem.append(
                {
                    "problem_index": problem_index,
                    "total_variation": total_variation_distance(left, right),
                    "jensen_shannon_bits": jensen_shannon_bits(left, right),
                }
            )
        comparisons[f"{method}_vs_{reference}"] = {
            "mean_total_variation": statistics.fmean(
                item["total_variation"] for item in per_problem
            ),
            "mean_jensen_shannon_bits": statistics.fmean(
                item["jensen_shannon_bits"] for item in per_problem
            ),
            "per_problem": per_problem,
            **bootstrap_answer_distance(
                answer_draws[method],
                answer_draws[reference],
                problem_indices,
                replicates=bootstrap_replicates,
            ),
        }
    return {
        "schema_version": 1,
        "analysis": "parsed_answer_distribution",
        "draws_per_problem": draws,
        "reference": reference,
        "problem_indices": problem_indices,
        "methods": method_summaries,
        "comparisons": comparisons,
    }


def collect_sweep(run_root: Path) -> dict[str, Any]:
    runs = []
    for summary_path in sorted(run_root.rglob("summary.json")):
        if summary_path == run_root / "summary.json":
            continue
        manifest_path = summary_path.with_name("manifest.json")
        if not manifest_path.is_file():
            continue
        summary = _load_json(summary_path)
        manifest = _load_json(manifest_path)
        config = manifest["config"]
        runs.append(
            {
                "run": summary_path.parent.relative_to(run_root).as_posix(),
                "method": manifest["method"],
                "draw_index": manifest["draw_index"],
                "parameters": {
                    key: config[key]
                    for key in (
                        "generation",
                        "exact_policy",
                        "search",
                        "best_of_n",
                        "mh",
                        "conditional_is",
                        "replay",
                        "dynamic_is",
                    )
                },
                "summary": summary,
            }
        )
    if not runs:
        raise ValueError(f"no completed sweep runs under {run_root}")
    return {"schema_version": 1, "analysis": "parameter_sweep", "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("passk", "distribution", "sweep"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--draws", type=int)
    parser.add_argument("--k", nargs="+", type=int)
    parser.add_argument("--reference")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / "summary.json"
    if args.kind == "sweep":
        report = collect_sweep(args.run_root)
    else:
        if not args.methods or args.draws is None:
            raise ValueError("passk and distribution require --methods and --draws")
        records = load_draw_grid(args.run_root, args.methods, args.draws)
        if args.kind == "passk":
            report = summarize_passk(
                records,
                draws=args.draws,
                ks=args.k or tuple(range(1, args.draws + 1)),
                bootstrap_seed=0,
                bootstrap_replicates=args.bootstrap_replicates,
            )
        else:
            if args.reference is None:
                raise ValueError("distribution analysis requires --reference")
            report = summarize_distribution(
                records,
                draws=args.draws,
                reference=args.reference,
                bootstrap_replicates=args.bootstrap_replicates,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
