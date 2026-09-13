from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.shared.reasoning_results import comparison_coverage, summarize_reasoning
from experiments.arllm.reasoning_benchmark import summarize


def row(problem="one", method="base_enabled", draw=0, correct=True):
    return {"problem_id": problem, "draw": draw, "method": method, "reward": "none",
            "budget_forward_tokens": 128, "used_forward_tokens": 10, "parameter_count": 7,
            "cost": {"generation_forward_token_slots": 8, "score_forward_token_slots": 2,
                     "estimated_dense_forward_flops": 140, "generated_tokens": 6},
            "correct": correct, "parseable": True, "selected_tokens": 6, "thinking_status": "complete",
            "manifest_fingerprint": "same"}


def test_coverage_counts_missing_conditions_and_duplicate_records():
    records = [row(), row(method="base_disabled")]
    options = dict(problem_ids=["one", "two"], budgets=[128], draws=1, methods=["base"], rewards=[])
    result = comparison_coverage(records, **options)
    assert result["expected_records"] == 4 and result["missing_records"] == 2
    assert result["completed_problems"] == 1 and not result["complete"]
    records += [row("two"), row("two", method="base_disabled")]
    assert comparison_coverage(records, **options)["complete"]
    with pytest.raises(ValueError, match="duplicate"):
        comparison_coverage(records + [records[0]], **options)


def test_coverage_includes_budget_only_baselines():
    coverage = comparison_coverage([], problem_ids=["one"], budgets=[128, 512], draws=1,
        methods=["budget_base", "base", "vote", "is", "mh"],
        rewards=["self_consistency", "sequence_log_probability", "consilience"])
    assert coverage["expected_records"] == 22
    assert not coverage["complete"]


def test_summary_distinguishes_length_matched_and_budget_only_baselines():
    summary = summarize_reasoning([row(correct=False), row(method="budget_base_enabled"), row(method="vote")])
    vote = next(item for item in summary if item["method"] == "vote")
    assert vote["paired_vs_thinking_base"]["difference"] == 1.0
    assert vote["paired_vs_budget_thinking_base"]["difference"] == 0.0


def test_summary_validates_cost_and_keeps_paired_problem_differences():
    records = [row(), row("two", correct=False), row(method="vote"), row("two", method="vote")]
    summary = summarize_reasoning(records)
    base, vote = summary
    assert base["accuracy"] == 0.5
    assert base["mean_used_forward_tokens"] == 10
    assert base["mean_pfLOPs"] == 140 / 1e15
    assert vote["paired_vs_thinking_base"]["difference"] == 0.5
    broken = deepcopy(records)
    broken[0]["used_forward_tokens"] = 11
    with pytest.raises(ValueError, match="token accounting"):
        summarize_reasoning(broken)
    broken = deepcopy(records)
    broken[0]["cost"]["estimated_dense_forward_flops"] = 141
    with pytest.raises(ValueError, match="FLOPs"):
        summarize_reasoning(broken)
    broken = deepcopy(records)
    broken[0]["manifest_fingerprint"] = "another"
    with pytest.raises(ValueError, match="mix"):
        summarize_reasoning(broken)


def test_repeated_draws_use_problem_bootstrap_and_partial_pairs_are_omitted():
    records = [row(draw=0), row(draw=1, correct=False), row(method="vote")]
    base, vote = summarize_reasoning(records)
    assert base["problems"] == 1 and base["trials"] == 2
    assert base["interval_method"] == "problem_bootstrap"
    assert base["accuracy_95"] == [0.5, 0.5]
    assert "paired_vs_thinking_base" not in vote


def test_limited_summary_uses_manifest_order_and_keeps_original_artifacts(tmp_path):
    manifest = {"split": "test", "subset": {"test": ["two", "one", "three"]}, "budgets": [128]}
    records = [row("one", correct=False), row("two"), row("two", method="base_disabled")]
    manifest_path, records_path = tmp_path / "manifest.json", tmp_path / "comparisons.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    original_manifest, original_records = manifest_path.read_bytes(), records_path.read_bytes()
    args = SimpleNamespace(limit=1, draws=1, methods=["base"], rewards=[], require_complete=True)

    result = summarize(tmp_path, args)
    assert result["coverage"]["complete"]
    assert result["coverage"]["expected_problems"] == 1
    assert result["records"] == 2
    assert result["selection"] == {"problem_ids": ["two"], "source_records": 3, "excluded_records": 1}
    assert all(item["accuracy"] == 1.0 for item in result["rows"])

    args.limit = 2
    with pytest.raises(ValueError, match="incomplete comparison grid"):
        summarize(tmp_path, args)
    args.limit, args.require_complete = None, False
    result = summarize(tmp_path, args)
    assert result["coverage"]["expected_problems"] == 3
    assert result["coverage"]["missing_records"] == 3
    assert result["selection"]["excluded_records"] == 0
    assert manifest_path.read_bytes() == original_manifest
    assert records_path.read_bytes() == original_records


def test_limited_summary_rejects_problems_outside_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "split": "test", "subset": {"test": ["one", "two"]}, "budgets": [128],
    }), encoding="utf-8")
    (tmp_path / "comparisons.jsonl").write_text(json.dumps(row("unknown")) + "\n", encoding="utf-8")
    args = SimpleNamespace(limit=1, draws=1, methods=["base"], rewards=[], require_complete=True)
    with pytest.raises(ValueError, match="outside the manifest"):
        summarize(tmp_path, args)
