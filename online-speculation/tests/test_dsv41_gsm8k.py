import importlib.util
import json
from pathlib import Path


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gsm8k_summary_scores_last_numeric_answer(tmp_path):
    summary = load_script("summarize_dsv41_gsm8k.py")
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({
        "elapsed_seconds": 2.0,
        "output_tokens": 8,
        "output_tokens_per_second": 4.0,
        "speculation": {"draft_rounds": 2, "proposed_candidates": 14, "accepted_candidates": 7},
        "requests": [{
            "token_ids": [1, 2], "output_text": "reason\n#### 1,024",
            "reference_answer": "reference\n#### 1024",
        }],
    }))
    row = summary.load_run(run)
    assert row["gsm8k_exact_match"] == 1
    assert row["candidate_acceptance_rate"] == .5
    assert row["mean_accepted_per_round"] == 3.5


def test_gsm8k_summary_does_not_guess_missing_delimiter(tmp_path):
    summary = load_script("summarize_dsv41_gsm8k.py")
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({
        "requests": [{"token_ids": [1], "output_text": "answer 5", "reference_answer": "#### 5"}],
        "speculation": {},
    }))
    assert summary.load_run(run)["gsm8k_exact_match"] == 0
