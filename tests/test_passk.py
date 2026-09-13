from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.arllm.gsm8k_passk import (
    _chunk_plan,
    _prepare_manifest,
    _summarize_method,
    _validate_chunks,
)


def _chunk(
    fingerprint: str,
    method: str,
    chunk_index: int,
    tasks: tuple[tuple[int, int], ...],
) -> dict:
    return {
        "manifest_fingerprint": fingerprint,
        "method": method,
        "chunk_index": chunk_index,
        "records": [
            {"draw_index": draw, "problem_index": problem}
            for draw, problem in tasks
        ],
    }


def test_passk_chunk_plan_preserves_draw_problem_grid() -> None:
    plan = _chunk_plan(("base", "mh"), 2, (11, 13, 17), 4)
    assert plan[("base", 0)] == ((0, 11), (1, 11))
    assert plan[("base", 1)] == ((0, 13), (1, 13))
    assert plan[("base", 2)] == ((0, 17), (1, 17))
    assert plan[("mh", 0)] == plan[("base", 0)]
    assert len(plan) == 6


def test_passk_manifest_allows_only_identical_resume_grid(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text('{"question":"q","answer":"#### 1"}\n', encoding="utf-8")
    raw = tmp_path / "audit.chunks.jsonl"
    arguments = {
        "config": {"run": {"name": "test"}},
        "data_path": data,
        "methods": ("base",),
        "draws": 2,
        "workers": 2,
        "problem_indices": (7,),
        "input_weight_sha256": {"base": "weights"},
        "implementation_sha256": {"script": "code"},
        "raw_path": raw,
    }
    _, fingerprint, manifest_path = _prepare_manifest(**arguments)
    assert manifest_path.is_file()
    assert _prepare_manifest(**arguments)[1] == fingerprint
    with pytest.raises(ValueError, match="different pass@k grid"):
        _prepare_manifest(**{**arguments, "draws": 3})


def test_passk_chunk_validation_rejects_duplicates_and_wrong_tasks() -> None:
    plan = _chunk_plan(("base",), 2, (3, 5), 2)
    first = _chunk("run", "base", 0, plan[("base", 0)])
    second = _chunk("run", "base", 1, plan[("base", 1)])
    assert len(_validate_chunks((first, second), "run", plan)) == 2
    with pytest.raises(ValueError, match="duplicate pass@k chunk"):
        _validate_chunks((first, first), "run", plan)
    wrong = json.loads(json.dumps(second))
    wrong["records"][0]["problem_index"] = 999
    with pytest.raises(ValueError, match="wrong task grid"):
        _validate_chunks((first, wrong), "run", plan)


def test_passk_summary_uses_all_draws_and_chunk_compute() -> None:
    records = [
        {
            "draw_index": draw,
            "problem_index": problem,
            "correct": (problem == 1 and draw == 0) or problem == 2,
            "prediction": str(draw) if problem == 1 else "2",
            "output_sha256": f"{problem}-{draw}",
        }
        for draw in range(2)
        for problem in (1, 2)
    ]
    chunk = {
        "backend_delta": {
            "generation_forward_token_slots": 10,
            "score_forward_token_slots": 5,
            "estimated_dense_forward_flops": 100,
        },
        "seconds_excluding_model_load": 4.0,
        "continuous_batching": {
            "sample_batches": 2,
            "score_batches": 1,
            "sample_requests": 4,
            "score_sequences": 3,
            "maximum_sample_batch": 2,
            "maximum_score_batch": 3,
        },
    }
    summary = _summarize_method(
        records,
        (chunk,),
        (1, 2),
        2,
        (1, 2),
        bootstrap_seed=9,
    )
    assert summary["single_draw_accuracy"] == 0.75
    assert summary["estimated_pass_at_k"] == {"1": 0.75, "2": 1.0}
    assert summary["total_forward_token_slots"] == 15
    assert summary["estimated_dense_forward_flops"] == 100
    assert summary["seconds_per_generated_answer"] == 1.0
