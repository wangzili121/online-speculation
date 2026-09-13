from __future__ import annotations

import hashlib
import json

import pytest

from experiments.arllm.summarize_gsm8k_passk import (
    combine_reports,
    summarize_is_raw_chunks,
)


def _method(correct: tuple[int, int]) -> dict:
    return {
        "examples": 2,
        "draws_per_example": 2,
        "generated_answers": 4,
        "single_draw_accuracy": sum(correct) / 4,
        "estimated_pass_at_k": {"1": sum(correct) / 4, "2": 0.5},
        "estimated_pass_at_k_problem_bootstrap_95": {
            "1": [0.0, 1.0],
            "2": [0.0, 1.0],
        },
        "mean_unique_parsed_answers_across_all_draws": 1.5,
        "mean_unique_full_outputs_across_all_draws": 2.0,
        "unparseable_fraction": 0.0,
        "total_forward_token_slots": 10,
        "estimated_dense_forward_flops": 100,
        "estimated_dense_forward_petaflops": 1e-13,
        "seconds_excluding_model_load": 4.0,
        "seconds_per_generated_answer": 1.0,
        "continuous_batching": {},
        "per_problem": [
            {"problem_index": 3, "correct_draws": correct[0]},
            {"problem_index": 5, "correct_draws": correct[1]},
        ],
    }


def _report(method: str, correct: tuple[int, int]) -> dict:
    return {
        "benchmark": "test",
        "draws_per_problem": 2,
        "problem_indices": [3, 5],
        "methods": {method: _method(correct)},
    }


def test_combined_passk_summary_pairs_methods_on_problem_rows() -> None:
    combined = combine_reports(
        [_report("base", (0, 2)), _report("guided", (1, 2))],
        [
            {"path": "base.json", "sha256": "a"},
            {"path": "guided.json", "sha256": "b"},
        ],
        bootstrap_seed=7,
        bootstrap_replicates=1_000,
    )
    comparison = combined["paired_comparisons"]["guided_minus_base"]
    assert comparison["pass_at_k"]["1"]["candidate_minus_reference"] == pytest.approx(
        0.25
    )
    assert comparison["pass_at_k"]["2"]["candidate_minus_reference"] == pytest.approx(
        0.5
    )
    assert comparison["cost"]["reference_over_candidate_inference_flops"] == 1.0
    assert combined["source_by_method"] == {
        "base": "base.json",
        "guided": "guided.json",
    }


def test_combined_passk_summary_rejects_mismatched_grids() -> None:
    mismatched = _report("guided", (1, 2))
    mismatched["problem_indices"] = [5, 3]
    with pytest.raises(ValueError, match="different problem rows"):
        combine_reports(
            [_report("base", (0, 2)), mismatched],
            [
                {"path": "base.json", "sha256": "a"},
                {"path": "guided.json", "sha256": "b"},
            ],
            bootstrap_seed=7,
            bootstrap_replicates=100,
        )


def test_is_raw_diagnostics_validate_sha_and_compare_outputs(tmp_path) -> None:
    diagnostics = {
        "guidance_steps": 2,
        "rollout_evaluations": 6,
        "mean_rollout_ess": 2.0,
        "mean_rollout_reward": 0.5,
        "mean_absolute_raw_log_importance_correction": 4.0,
        "mean_absolute_applied_log_importance_correction": 3.0,
        "clipped_rollout_corrections": 2,
    }
    chunks = [
        {
            "manifest_fingerprint": "grid",
            "method": method,
            "records": [
                {
                    "problem_index": 3,
                    "draw_index": 0,
                    "prediction": "1",
                    "correct": True,
                    "output_sha256": output_hash,
                    "diagnostics": diagnostics,
                }
            ],
        }
        for method, output_hash in (("clipped", "a"), ("unclipped", "b"))
    ]
    raw = tmp_path / "chunks.jsonl"
    raw.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8"
    )
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    report = {
        "raw_chunks_sha256": digest,
        "manifest_fingerprint": "grid",
        "methods": {
            "clipped": {"generated_answers": 1},
            "unclipped": {"generated_answers": 1},
        },
    }
    summary = summarize_is_raw_chunks(raw, [report])
    assert summary["methods"]["clipped"]["clipped_rollout_corrections"] == 2
    assert summary["methods"]["clipped"]["clipped_rollout_correction_fraction"] == pytest.approx(
        1 / 3
    )
    agreement = summary["pairwise_agreement"]["clipped_vs_unclipped"]
    assert agreement["exact_output_match_fraction"] == 0.0
    assert agreement["parsed_answer_match_fraction"] == 1.0

    report["raw_chunks_sha256"] = "wrong"
    with pytest.raises(ValueError, match="exactly one input report"):
        summarize_is_raw_chunks(raw, [report])
