"""Combine per-method GSM8K records and make comparison directions explicit."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tomllib
from pathlib import Path
from typing import Any, Sequence


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_difference(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    seed: int = 20260808,
    draws: int = 10_000,
) -> dict[str, Any]:
    left_by_index = {int(item["problem_index"]): bool(item["correct"]) for item in left}
    right_by_index = {int(item["problem_index"]): bool(item["correct"]) for item in right}
    if left_by_index.keys() != right_by_index.keys():
        raise ValueError("paired methods do not contain the same public benchmark rows")
    indices = sorted(left_by_index)
    differences = [float(left_by_index[index]) - float(right_by_index[index]) for index in indices]
    rng = random.Random(seed)
    bootstrap = [
        statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(draws)
    ]
    estimate = statistics.fmean(differences)
    return {
        "left_minus_right_accuracy": estimate,
        "paired_bootstrap_95": [_quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)],
        "within_five_percentage_points": abs(estimate) <= 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--tag", default="default")
    parser.add_argument("--results-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--output", type=Path, default=Path("results/gsm8k_comparison.json"))
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    profile_root = args.results_root / str(config["run"]["name"])
    methods = (
        "base",
        "beam",
        "best_of_n",
        "mh",
        "conditional_is",
        "conditional_is_small_proposal",
        "rl_sample",
        "rl_greedy",
    )
    summaries: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for method in methods:
        directory = profile_root / f"{method}-{args.tag}"
        summaries[method] = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        manifests[method] = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        records[method] = _records(directory / "records.jsonl")

    indices = {
        method: tuple(sorted(int(item["problem_index"]) for item in values))
        for method, values in records.items()
    }
    if len(set(indices.values())) != 1:
        raise ValueError("main comparison methods did not evaluate identical GSM8K rows")
    implementation_variants = {
        json.dumps(
            manifest["effective"]["implementation_sha256"], sort_keys=True
        )
        for manifest in manifests.values()
    }
    if len(implementation_variants) != 1:
        raise ValueError("main comparison methods used different implementation files")
    base_seconds = float(summaries["base"]["sum_example_seconds"])
    base_flops = int(summaries["base"]["estimated_dense_forward_flops"])
    table = []
    for method in methods:
        summary = summaries[method]
        seconds = float(summary["sum_example_seconds"])
        table.append(
            {
                "method": method,
                "examples": int(summary["examples"]),
                "accuracy": float(summary["accuracy"]),
                "accuracy_wilson_95": summary["accuracy_wilson_95"],
                "seconds_excluding_model_load": seconds,
                "runtime_multiple_vs_base": seconds / base_seconds,
                "base_over_method_wall_time": base_seconds / seconds,
                "mean_selected_output_tokens": float(
                    summary["mean_selected_output_tokens"]
                ),
                "forward_token_slots": int(summary["total_forward_token_slots"]),
                "shared_prefill_tokens_saved": int(
                    summary["total_shared_prefill_tokens_saved"]
                ),
                "estimated_dense_forward_flops": int(
                    summary["estimated_dense_forward_flops"]
                ),
                "estimated_dense_forward_petaflops": float(
                    summary["estimated_dense_forward_petaflops"]
                ),
                "compute_multiple_vs_base": (
                    int(summary["estimated_dense_forward_flops"]) / base_flops
                ),
            }
        )

    standard_conditional = summaries["conditional_is"]
    accelerated_conditional = summaries["conditional_is_small_proposal"]
    conditional_speedup = (
        float(standard_conditional["sum_example_seconds"])
        / float(accelerated_conditional["sum_example_seconds"])
    )
    conditional_compute_reduction = (
        int(standard_conditional["estimated_dense_forward_flops"])
        / int(accelerated_conditional["estimated_dense_forward_flops"])
    )
    verifier_methods = (
        "verifier_mh",
        "verifier_conditional_is",
        "verifier_conditional_is_small_proposal",
    )
    verifier_directories = {
        method: profile_root / f"{method}-{args.tag}" for method in verifier_methods
    }
    matched_target: dict[str, Any] | None = None
    if all((directory / "summary.json").is_file() for directory in verifier_directories.values()):
        verifier_summaries = {
            method: json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            for method, directory in verifier_directories.items()
        }
        verifier_manifests = {
            method: json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
            for method, directory in verifier_directories.items()
        }
        verifier_records = {
            method: _records(directory / "records.jsonl")
            for method, directory in verifier_directories.items()
        }
        verifier_indices = {
            method: tuple(
                sorted(int(item["problem_index"]) for item in method_records)
            )
            for method, method_records in verifier_records.items()
        }
        expected_indices = indices["base"]
        if any(value != expected_indices for value in verifier_indices.values()):
            raise ValueError(
                "matched-target methods did not evaluate the main comparison rows"
            )
        verifier_implementations = {
            json.dumps(
                manifest["effective"]["implementation_sha256"], sort_keys=True
            )
            for manifest in verifier_manifests.values()
        }
        if verifier_implementations != implementation_variants:
            raise ValueError(
                "matched-target methods used different implementation files"
            )
        matched_target = {
            "target_definition": (
                "base probability multiplied by exp(exact numeric verifier reward / beta)"
            ),
            "reward_temperature_beta": float(config["matched_target"]["reward_temperature"]),
            "same_problem_indices_as_main_table": True,
            "method_manifest_fingerprints": {
                method: manifest["fingerprint"]
                for method, manifest in verifier_manifests.items()
            },
            "table": [
                {
                    "method": method,
                    "examples": int(verifier_summaries[method]["examples"]),
                    "accuracy": float(verifier_summaries[method]["accuracy"]),
                    "accuracy_wilson_95": verifier_summaries[method][
                        "accuracy_wilson_95"
                    ],
                    "seconds_excluding_model_load": float(
                        verifier_summaries[method]["sum_example_seconds"]
                    ),
                    "forward_token_slots": int(
                        verifier_summaries[method]["total_forward_token_slots"]
                    ),
                    "estimated_dense_forward_flops": int(
                        verifier_summaries[method]["estimated_dense_forward_flops"]
                    ),
                }
                for method in verifier_methods
            ],
            "quality_comparisons": {
                "verifier_mh_minus_verifier_conditional_is": _paired_difference(
                    verifier_records["verifier_mh"],
                    verifier_records["verifier_conditional_is"],
                ),
                "verifier_mh_minus_rl_sample": _paired_difference(
                    verifier_records["verifier_mh"], records["rl_sample"]
                ),
                "verifier_conditional_is_minus_rl_sample": _paired_difference(
                    verifier_records["verifier_conditional_is"],
                    records["rl_sample"],
                ),
                "small_proposal_minus_standard_conditional_is": _paired_difference(
                    verifier_records["verifier_conditional_is_small_proposal"],
                    verifier_records["verifier_conditional_is"],
                ),
            },
        }
    report = {
        "schema_version": 4,
        "profile": str(config["run"]["name"]),
        "public_dataset": "GSM8K official test split",
        "same_problem_indices_for_every_method": True,
        "problem_indices": list(indices["base"]),
        "method_manifest_fingerprints": {
            method: manifest["fingerprint"]
            for method, manifest in manifests.items()
        },
        "input_weight_sha256_by_method": {
            method: manifest["effective"]["input_weight_sha256"]
            for method, manifest in manifests.items()
        },
        "implementation_sha256": manifests["base"]["effective"][
            "implementation_sha256"
        ],
        "table": table,
        "matched_target_comparison": matched_target,
        "quality_comparisons": {
            "mh_minus_rl_sample": _paired_difference(
                records["mh"], records["rl_sample"]
            ),
            "conditional_is_minus_rl_sample": _paired_difference(
                records["conditional_is"], records["rl_sample"]
            ),
            "conditional_is_minus_rl_greedy": _paired_difference(
                records["conditional_is"], records["rl_greedy"]
            ),
            "accelerated_conditional_is_minus_standard_conditional_is": _paired_difference(
                records["conditional_is_small_proposal"], records["conditional_is"]
            ),
        },
        "compute_comparisons": {
            "standard_over_small_proposal_flop_factor": conditional_compute_reduction,
            "standard_over_small_proposal_flop_factor_definition": (
                "standard on-policy conditional-IS estimated dominant dense FLOPs "
                "divided by small-proposal off-policy conditional-IS FLOPs; each model's "
                "term is 2 * its parameter count * observed forward token slots; values "
                "above one are a reduction and values below one are a compute increase"
            ),
            "small_proposal_main_model_generation_slots_avoided": (
                int(standard_conditional["base_generation_forward_token_slots"])
                - int(accelerated_conditional["base_generation_forward_token_slots"])
            ),
            "small_proposal_forward_token_slots": int(
                accelerated_conditional["proposal_generation_forward_token_slots"]
            ),
            "standard_over_small_proposal_wall_time_factor": conditional_speedup,
            "standard_over_small_proposal_wall_time_factor_definition": (
                "standard on-policy conditional-IS wall time divided by small-proposal "
                "off-policy conditional-IS wall time; model, GSM8K rows, seeds, candidate "
                "count, rollout count, block size, output length, and reward are fixed; "
                "values above one are a speedup and values below one are a slowdown"
            ),
        },
        "compute_scope": (
            "primary compute units are observed model-input forward token slots and the "
            "dominant-matmul estimate 2 * parameters * token slots; quadratic attention, "
            "elementwise kernels, tokenization, sampling, and host work are excluded"
        ),
        "timing_scope": (
            "supplemental per-example CUDA-synchronized inference time; model loading, "
            "dataset loading, and checkpoint serialization are excluded"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
