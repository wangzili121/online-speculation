"""Compare GRPO and training-free inference in token and FLOP units."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from typing import Any

from inference_scaling.shared.compute import (
    estimate_grpo_compute,
    estimate_grpo_compute_from_logs,
)


def _summary(root: Path, profile: str, method: str, tag: str) -> dict[str, Any]:
    path = root / profile / f"{method}-{tag}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _per_query(summary: dict[str, Any], field: str) -> float:
    return float(summary[field]) / int(summary["examples"])


def _comparison(
    method: dict[str, Any],
    rl: dict[str, Any],
    *,
    training_forward_token_slots: int,
    training_flops: int,
    training_seconds: float,
    accuracy_tolerance: float,
    answer_distribution: dict[str, Any] | None,
    answer_tv_tolerance: float,
    answer_js_tolerance: float,
) -> dict[str, Any]:
    method_tokens = _per_query(method, "total_forward_token_slots")
    rl_tokens = _per_query(rl, "total_forward_token_slots")
    extra_tokens = method_tokens - rl_tokens
    method_flops = _per_query(method, "estimated_dense_forward_flops")
    rl_flops = _per_query(rl, "estimated_dense_forward_flops")
    extra_flops = method_flops - rl_flops
    method_seconds = _per_query(method, "sum_example_seconds")
    rl_seconds = _per_query(rl, "sum_example_seconds")
    extra_seconds = method_seconds - rl_seconds
    accuracy_gap = float(method["accuracy"]) - float(rl["accuracy"])
    accuracy_matched = abs(accuracy_gap) <= accuracy_tolerance
    if answer_distribution is None:
        answer_distribution_matched = None
        joint_empirical_match = None
    else:
        answer_distribution_matched = (
            float(answer_distribution["mean_total_variation"])
            <= answer_tv_tolerance
            and float(answer_distribution["mean_jensen_shannon_bits"])
            <= answer_js_tolerance
        )
        joint_empirical_match = accuracy_matched and answer_distribution_matched
    token_break_even = (
        math.ceil(training_forward_token_slots / extra_tokens)
        if extra_tokens > 0
        else None
    )
    flop_break_even = (
        math.ceil(training_flops / extra_flops) if extra_flops > 0 else None
    )
    wall_break_even = (
        math.ceil(training_seconds / extra_seconds) if extra_seconds > 0 else None
    )
    return {
        "method": method["method"],
        "rl_reference": rl["method"],
        "method_accuracy": float(method["accuracy"]),
        "rl_accuracy": float(rl["accuracy"]),
        "method_minus_rl_accuracy": accuracy_gap,
        "accuracy_tolerance": accuracy_tolerance,
        "accuracy_matched": accuracy_matched,
        "answer_distribution_level": (
            "parsed final answer" if answer_distribution is not None else None
        ),
        "answer_distribution_diagnostic": answer_distribution,
        "answer_total_variation_tolerance": answer_tv_tolerance,
        "answer_jensen_shannon_bits_tolerance": answer_js_tolerance,
        "answer_distribution_matched": answer_distribution_matched,
        "joint_accuracy_and_answer_distribution_match": joint_empirical_match,
        "method_forward_token_slots_per_query": method_tokens,
        "rl_forward_token_slots_per_query": rl_tokens,
        "method_minus_rl_forward_token_slots_per_query": extra_tokens,
        "raw_token_break_even_queries": token_break_even,
        "accuracy_matched_token_break_even_queries": (
            token_break_even if accuracy_matched else None
        ),
        "joint_empirical_match_token_break_even_queries": (
            token_break_even if joint_empirical_match else None
        ),
        "method_estimated_dense_flops_per_query": method_flops,
        "rl_estimated_dense_flops_per_query": rl_flops,
        "method_minus_rl_estimated_dense_flops_per_query": extra_flops,
        "raw_flop_break_even_queries": flop_break_even,
        "accuracy_matched_flop_break_even_queries": (
            flop_break_even if accuracy_matched else None
        ),
        "joint_empirical_match_flop_break_even_queries": (
            flop_break_even if joint_empirical_match else None
        ),
        "flop_break_even_definition": (
            "ceil(GRPO training estimated dense FLOPs / (training-free method "
            "estimated dense FLOPs per query - GRPO inference estimated dense "
            "FLOPs per query))"
        ),
        "method_seconds_per_query": method_seconds,
        "rl_seconds_per_query": rl_seconds,
        "method_minus_rl_seconds_per_query": extra_seconds,
        "raw_wall_time_break_even_queries": wall_break_even,
        "accuracy_matched_wall_time_break_even_queries": (
            wall_break_even if accuracy_matched else None
        ),
        "joint_empirical_match_wall_time_break_even_queries": (
            wall_break_even if joint_empirical_match else None
        ),
        "wall_time_break_even_definition": (
            "ceil(GRPO training wall seconds / (method inference seconds per query - "
            "GRPO inference seconds per query)); model loading is excluded from both "
            "inference terms"
        ),
        "interpretation": (
            "Below the FLOP break-even, training-free inference uses fewer estimated "
            "dense-model FLOPs; above it, GRPO training plus repeated GRPO inference "
            "uses fewer."
            if flop_break_even is not None
            else "The training-free method does not use more estimated FLOPs per query, "
            "so GRPO training cannot be amortized in this comparison."
        ),
    }


def _training_compute(
    training: dict[str, Any],
    log_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert observed GRPO tokens into explicit forward-equivalent compute."""

    if log_history:
        section = training["effective"]["training"]
        return estimate_grpo_compute_from_logs(
            log_history=log_history,
            sequences_per_optimizer_step=(
                int(section["per_device_train_batch_size"])
                * int(section["gradient_accumulation_steps"])
            ),
            generated_completions=int(training["rollouts"]["generated_completions"]),
            total_parameters=int(training["total_parameters"]),
            trainable_parameters=int(training["trainable_parameters"]),
            optimizer_steps=int(training["global_step"]),
            gradient_checkpointing=True,
            reference_scoring=float(section["beta"]) != 0,
        ).as_dict()
    return estimate_grpo_compute(
        model_sequence_tokens=int(training["trainer_reported_model_tokens"]),
        generated_completions=int(training["rollouts"]["generated_completions"]),
        total_parameters=int(training["total_parameters"]),
        trainable_parameters=int(training["trainable_parameters"]),
        optimizer_steps=int(training["global_step"]),
        gradient_checkpointing=True,
        reference_scoring=float(training["effective"]["training"]["beta"]) != 0,
    ).as_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_standard.toml"))
    parser.add_argument("--tag", default="default")
    parser.add_argument("--results-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument(
        "--training-cost",
        type=Path,
        default=Path("models/Qwen2.5-1.5B-Instruct-GRPO-GSM8K/training_cost.json"),
    )
    parser.add_argument(
        "--trainer-state",
        type=Path,
        help="trainer_state.json; defaults to checkpoint-<global_step> beside training cost",
    )
    parser.add_argument("--accuracy-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--distribution-audit",
        type=Path,
        help="optional answer-distribution audit from gsm8k_distribution_audit.py",
    )
    parser.add_argument("--answer-tv-tolerance", type=float, default=0.25)
    parser.add_argument("--answer-js-tolerance", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("results/gsm8k_compute.json"))
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    profile = str(config["run"]["name"])
    training = json.loads(args.training_cost.read_text(encoding="utf-8"))
    training_seconds = float(
        training.get("cumulative_training_wall_seconds", training["training_wall_seconds"])
    )
    power_integral_wh = float(
        training.get(
            "cumulative_gpu_power_integral_wh",
            training["gpu_monitor"].get("gpu_power_integral_wh", 0.0),
        )
    )
    trainer_state_path = args.trainer_state or (
        args.training_cost.parent
        / f"checkpoint-{int(training['global_step'])}"
        / "trainer_state.json"
    )
    log_history = None
    if trainer_state_path.is_file():
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        log_history = trainer_state.get("log_history")
    training_compute = (
        training.get("primary_compute")
        or _training_compute(training, log_history=log_history)
    )
    methods = {
        name: _summary(args.results_root, profile, name, args.tag)
        for name in (
            "base",
            "mh",
            "conditional_is",
            "conditional_is_small_proposal",
            "verifier_mh",
            "verifier_conditional_is",
            "verifier_conditional_is_small_proposal",
            "rl_sample",
            "rl_greedy",
        )
    }
    distribution_audit = (
        json.loads(args.distribution_audit.read_text(encoding="utf-8"))
        if args.distribution_audit is not None
        else None
    )

    def distribution_for(method: str) -> dict[str, Any] | None:
        if distribution_audit is None:
            return None
        key = f"{method}_vs_rl_sample"
        try:
            return distribution_audit["comparisons"][key]
        except KeyError as error:
            raise ValueError(
                f"distribution audit does not contain comparison {key!r}"
            ) from error

    report = {
        "schema_version": 3,
        "profile": profile,
        "tag": args.tag,
        "public_dataset": "OpenAI GSM8K official test split",
        "method_manifest_fingerprints": {
            method: summary["manifest_fingerprint"]
            for method, summary in methods.items()
        },
        "training": {
            "primary_compute": training_compute,
            "trainer_state_used": str(trainer_state_path) if log_history else None,
            "wall_seconds": training_seconds,
            "gpu_power_integral_wh": power_integral_wh,
            "global_step": int(training["global_step"]),
            "generated_completion_tokens": int(
                training["rollouts"]["generated_completion_tokens"]
            ),
            "peak_cuda_allocated_bytes": int(training["peak_cuda_allocated_bytes"]),
            "base_model": training["effective"]["model"],
        },
        "comparisons": {
            "mh_vs_grpo_sample": _comparison(
                methods["verifier_mh"],
                methods["rl_sample"],
                training_forward_token_slots=int(
                    training_compute["total_forward_equivalent_token_slots"]
                ),
                training_flops=int(training_compute["estimated_total_flops"]),
                training_seconds=training_seconds,
                accuracy_tolerance=args.accuracy_tolerance,
                answer_distribution=distribution_for("verifier_mh"),
                answer_tv_tolerance=args.answer_tv_tolerance,
                answer_js_tolerance=args.answer_js_tolerance,
            ),
            "conditional_is_vs_grpo_sample": _comparison(
                methods["verifier_conditional_is"],
                methods["rl_sample"],
                training_forward_token_slots=int(
                    training_compute["total_forward_equivalent_token_slots"]
                ),
                training_flops=int(training_compute["estimated_total_flops"]),
                training_seconds=training_seconds,
                accuracy_tolerance=args.accuracy_tolerance,
                answer_distribution=distribution_for("verifier_conditional_is"),
                answer_tv_tolerance=args.answer_tv_tolerance,
                answer_js_tolerance=args.answer_js_tolerance,
            ),
            "off_policy_conditional_is_vs_grpo_sample": _comparison(
                methods["verifier_conditional_is_small_proposal"],
                methods["rl_sample"],
                training_forward_token_slots=int(
                    training_compute["total_forward_equivalent_token_slots"]
                ),
                training_flops=int(training_compute["estimated_total_flops"]),
                training_seconds=training_seconds,
                accuracy_tolerance=args.accuracy_tolerance,
                answer_distribution=distribution_for(
                    "verifier_conditional_is_small_proposal"
                ),
                answer_tv_tolerance=args.answer_tv_tolerance,
                answer_js_tolerance=args.answer_js_tolerance,
            ),
        },
        "distribution_audit": (
            {
                "path": str(args.distribution_audit),
                "distribution_level": distribution_audit["distribution_level"],
                "problem_indices": distribution_audit["problem_indices"],
                "draws_per_problem": distribution_audit["draws_per_problem"],
                "limitation": distribution_audit["limitation"],
            }
            if distribution_audit is not None
            else None
        ),
        "scope_note": (
            "The three training-free methods use the same exact-numeric reward and reward "
            "temperature as GRPO's KL-regularized objective. The small-proposal method "
            "uses the configured explicit log-ratio clipping and is therefore a biased, "
            "variance-stabilized finite-rollout approximation. Token slots and estimated "
            "dominant-matmul FLOPs are the primary compute metrics. Any break-even is "
            "reported as accuracy-matched only when the observed accuracy gap is within "
            "the stated tolerance. A joint empirical match additionally requires the "
            "optional parsed-answer distribution audit to pass both TV and JS thresholds; "
            "this finite audit does not prove equality of full token-sequence distributions. "
            "Wall time, memory, and measured power integral are hardware-dependent supplements."
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
