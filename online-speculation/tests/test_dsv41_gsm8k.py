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


def test_staged_gsm8k_summary_compares_equal_workloads_and_keeps_each_dps_first_update(tmp_path):
    summary = load_script("summarize_dsv41_gsm8k.py")
    run = tmp_path / "result.json"
    ar = {"generated": {"train": [1, 2]}, "decode_tok_per_s": 4}
    updates = [
        dict(dp=dp, applied=True, update_seconds=seconds, reverse_seconds=.2, stage_seconds=.1,
             optimizer_seconds=.1, publication_seconds=.25)
        for dp, seconds in [(0, 400.), (2, 270.), (0, 4.), (2, 3.)]
    ]
    study = {
        **ar,
        "node": 1,
        "passed": False,
        "arms": {
            "ar": ar,
            "orthrus": {
                **ar, "decode_tok_per_s": 6, "steady_decode": {"tok_per_s": 7},
                "online": dict(updates=updates, version=4, rounds_after_publication=22,
                               update_seconds=677., decode_tok_per_s_excluding_updates=8.),
            },
            "eval_ar": {"generated": {"heldout": [1, 2]}, "decode_tok_per_s": 5},
            "eval_frozen": {"generated": {"heldout": [1, 3]}, "decode_tok_per_s": 10},
        },
    }
    run.write_text(json.dumps(study))
    result = summary.load_run(run)
    online = result["arms"]["orthrus"]
    assert result["node"] == 1 and not result["passed"]
    assert online["output_tokens"] == 2 and online["decode_speedup"] == 1.5
    assert online["steady_decode"] == {"tok_per_s": 7}
    assert online["online"]["first_update_seconds_by_dp"] == {"0": 400., "2": 270.}
    assert online["online"]["median_warm_update_seconds"] == 3.5
    assert online["online"]["publication_seconds"] == 1
    assert "decode_speedup" not in result["arms"]["eval_frozen"]


def test_staged_head_identity_detects_changed_weight_bytes_with_the_same_index(tmp_path):
    runner = load_script("run_dsv41_pypto_staged_target.py")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    shard = tmp_path / "one.safetensors"
    shard.write_bytes(b"weights A")
    files = {"a": shard.name, "b": shard.name}
    first = runner.head_identity(tmp_path, files)
    assert first == runner.head_identity(tmp_path, {"b": shard.name})
    shard.write_bytes(b"weights B")
    assert first != runner.head_identity(tmp_path, files)
