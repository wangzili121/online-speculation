from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.dllm.gsm8k_analysis import (
    load_draw_grid,
    summarize_distribution,
    summarize_passk,
)


def _compute(flops: float) -> dict[str, float]:
    return {
        "estimated_active_flops": flops,
        "sample_model_token_slots": flops / 2,
        "score_model_token_slots": flops / 4,
    }


def _record(method: str, draw: int, problem: int, prediction: str) -> dict:
    return {
        "method": method,
        "draw_index": draw,
        "problem_index": problem,
        "prediction": prediction,
        "correct": prediction == "1",
        "elapsed_seconds": 0.5,
        "main_compute": _compute(10.0),
        "proposal_compute": _compute(2.0),
    }


def _grid() -> dict[str, list[dict]]:
    return {
        "candidate": [
            _record("candidate", 0, 3, "1"),
            _record("candidate", 1, 3, "0"),
            _record("candidate", 0, 7, "1"),
            _record("candidate", 1, 7, "1"),
        ],
        "reference": [
            _record("reference", 0, 3, "1"),
            _record("reference", 1, 3, "1"),
            _record("reference", 0, 7, "0"),
            _record("reference", 1, 7, "0"),
        ],
    }


def test_passk_aggregation_uses_complete_draw_grid_and_separate_compute() -> None:
    report = summarize_passk(
        _grid(),
        draws=2,
        ks=(1, 2),
        bootstrap_seed=5,
        bootstrap_replicates=100,
    )

    candidate = report["methods"]["candidate"]
    assert candidate["pass_at_k"] == {"1": 0.75, "2": 1.0}
    assert candidate["compute"]["main_estimated_active_flops"] == 40.0
    assert candidate["compute"]["proposal_estimated_active_flops"] == 8.0


def test_distribution_aggregation_reports_distance_to_named_reference() -> None:
    report = summarize_distribution(
        _grid(),
        draws=2,
        reference="reference",
        bootstrap_replicates=50,
    )

    comparison = report["comparisons"]["candidate_vs_reference"]
    assert comparison["mean_total_variation"] == pytest.approx(0.75)
    assert comparison["mean_jensen_shannon_bits"] > 0


def test_draw_loader_rejects_incomplete_problem_sets(tmp_path: Path) -> None:
    for draw, problem in ((0, 3), (1, 7)):
        path = tmp_path / "candidate" / f"draw-{draw}" / "records.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(_record("candidate", draw, problem, "1")) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="same problem set"):
        load_draw_grid(tmp_path, ("candidate",), 2)

