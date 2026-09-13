"""Collect the GSM8K ablations into one machine-readable report."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


RUNNER_PATH = "experiments/arllm/gsm8k_reproduction.py"
RESULT_COMPATIBLE_RUNNERS = frozenset(
    {
        "257cd25b2cd4bd4e20f8ff96f81e799dfda1f0871b4ca0d320a772975f17fe2f",
        "823e365c3807e7633707cb9fe62c66cdd897b23e456e5f7cc226a123fd191a4c",
    }
)


def _implementation_provenance(
    variants: list[dict[str, str]],
) -> tuple[dict[str, str], list[str], str]:
    if not variants:
        return {}, [], "no ablation summaries were found"
    core_variants = {
        json.dumps(
            {path: digest for path, digest in variant.items() if path != RUNNER_PATH},
            sort_keys=True,
        )
        for variant in variants
    }
    if len(core_variants) != 1:
        raise ValueError("ablation summaries were produced by different algorithms")
    runner_hashes = sorted({variant[RUNNER_PATH] for variant in variants})
    if (
        len(runner_hashes) > 1
        and frozenset(runner_hashes) != RESULT_COMPATIBLE_RUNNERS
    ):
        raise ValueError("ablation summaries were produced by incompatible runners")
    core = json.loads(next(iter(core_variants)))
    note = (
        "All rows share identical algorithm/backend implementations and "
        "result-compatible runner semantics."
        if len(runner_hashes) > 1
        else "All ablation summaries use one runner hash."
    )
    return core, runner_hashes, note


def _is_method_summary(summary: dict[str, Any]) -> bool:
    """Exclude replay/aggregate summaries that share the profile directory."""

    return isinstance(summary.get("method"), str) and isinstance(
        summary.get("tag"), str
    )


def _groups(tag: str) -> tuple[str, ...]:
    references = {
        "conditional-reference": (
            "candidate_rollout_budget",
            "guidance_steps",
            "sampling_temperature",
            "quality_compute_curve",
            "reward",
        ),
        "best-of-n-reference": (
            "sampling_temperature",
            "quality_compute_curve",
            "reward",
        ),
        "beam-reference": ("quality_compute_curve", "sampling_temperature"),
        "conditional-small-proposal-reference": (
            "quality_compute_curve",
            "importance_ratio_clipping",
            "reward",
            "sampling_temperature",
        ),
        "conditional-small-proposal-unclipped": (
            "importance_ratio_clipping",
        ),
    }
    if tag in references:
        return references[tag]
    if tag.startswith("alpha-") or tag.startswith("steps-"):
        return ("power_sampling",)
    if tag.startswith("candidates-"):
        return ("candidate_rollout_budget",)
    if tag.startswith("guidance-steps-"):
        return ("guidance_steps",)
    if "-reward-" in tag:
        return ("reward",)
    if "-temperature-" in tag:
        return ("sampling_temperature",)
    if tag.startswith("budget-"):
        return ("quality_compute_curve",)
    if tag.startswith("length-"):
        return ("generation_length",)
    return ()


def _per_example(summary: dict[str, Any], field: str) -> float:
    return float(summary[field]) / int(summary["examples"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_standard.toml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--output", type=Path, default=Path("results/gsm8k_ablations.json"))
    parser.add_argument(
        "--tag-prefix",
        help="include only one suite tag and remove its prefix from ablation labels",
    )
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    profile = str(config["run"]["name"])
    profile_root = args.results_root / profile
    grouped: dict[str, list[dict[str, Any]]] = {}
    implementation_variants: dict[str, dict[str, str]] = {}
    for summary_path in sorted(profile_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not _is_method_summary(summary):
            continue
        recorded_tag = str(summary["tag"])
        if args.tag_prefix is not None:
            prefix = f"{args.tag_prefix}-"
            if not recorded_tag.startswith(prefix):
                continue
            tag = recorded_tag[len(prefix) :]
        else:
            tag = recorded_tag
        groups = _groups(tag)
        if not groups:
            continue
        manifest_path = summary_path.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        implementation = manifest["effective"]["implementation_sha256"]
        implementation_variants[
            json.dumps(implementation, sort_keys=True)
        ] = implementation
        effective = manifest["effective"]["config"]
        row = {
            "method": summary["method"],
            "tag": tag,
            "recorded_tag": recorded_tag,
            "runner_sha256": implementation[RUNNER_PATH],
            "examples": int(summary["examples"]),
            "accuracy": float(summary["accuracy"]),
            "accuracy_wilson_95": summary["accuracy_wilson_95"],
            "forward_token_slots_per_example": _per_example(
                summary, "total_forward_token_slots"
            ),
            "shared_prefill_tokens_saved_per_example": _per_example(
                summary, "total_shared_prefill_tokens_saved"
            ),
            "estimated_dense_flops_per_example": _per_example(
                summary, "estimated_dense_forward_flops"
            ),
            "seconds_per_example": float(summary["mean_example_seconds"]),
            "mean_selected_output_tokens": float(
                summary["mean_selected_output_tokens"]
            ),
            "settings": {
                "max_new_tokens": effective["generation"]["max_new_tokens"],
                "sampling_temperature": effective.get("sampling", {}).get(
                    "temperature", 1.0
                ),
                "num_beams": effective["beam"]["num_beams"],
                "best_of_n_samples": effective["best_of_n"]["samples"],
                "mh": effective["mh"],
                "conditional_is": effective["conditional_is"],
            },
        }
        for group in groups:
            grouped.setdefault(group, []).append(row)

    algorithm_implementation, runner_hashes, runner_compatibility = (
        _implementation_provenance(list(implementation_variants.values()))
    )

    report = {
        "schema_version": 3,
        "profile": profile,
        "algorithm_implementation_sha256": algorithm_implementation,
        "runner_sha256": runner_hashes,
        "runner_compatibility": runner_compatibility,
        "groups": grouped,
        "alignment": {
            "main_metric": "single final response accuracy (pass@1)",
            "method_families": [
                "Base",
                "Beam Search",
                "Best-of-N self-consistency",
                "power-distribution suffix MH",
                "conditional importance sampling",
                "small-proposal off-policy conditional importance sampling",
                "GRPO",
            ],
            "ablations": [
                "M x K candidate/rollout budget",
                "guidance steps",
                "accuracy-compute trade-off",
                "model-score, self-consistency, and configured-verifier rewards",
                "sampling temperature",
                "small-proposal log-ratio clipping",
                "generation length",
                "power and MCMC steps",
            ],
        },
        "primary_compute": (
            "observed padded forward token slots and 2 * parameter count * token "
            "slots, with base and small proposal models counted separately"
        ),
        "wall_time_role": "hardware-dependent supplemental metric",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
