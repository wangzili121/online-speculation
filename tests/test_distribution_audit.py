from pathlib import Path

import pytest

from experiments.arllm.gsm8k_distribution_audit import (
    _prepare_manifest,
    _split_half_noise_floor,
    _validate_existing_records,
)


def test_distribution_audit_manifest_can_resume_only_the_same_grid(tmp_path: Path) -> None:
    raw_path = tmp_path / "audit.records.jsonl"
    arguments = {
        "config": {"run": {"seed": 1}},
        "methods": ("rl_sample",),
        "problem_indices": [3],
        "draws": 2,
        "input_weight_hashes": {"base": "base"},
        "implementation_hashes": {"script": "implementation"},
        "raw_path": raw_path,
    }
    _, fingerprint, manifest_path = _prepare_manifest(**arguments)
    assert manifest_path.is_file()
    assert _prepare_manifest(**arguments)[1] == fingerprint

    raw_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different distribution audit"):
        _prepare_manifest(**{**arguments, "draws": 3})


def test_distribution_audit_rejects_duplicate_sample_keys() -> None:
    record = {
        "manifest_fingerprint": "run",
        "method": "rl_sample",
        "draw": 0,
        "problem_index": 3,
    }
    expected = {("rl_sample", 0, 3)}
    with pytest.raises(ValueError, match="duplicate"):
        _validate_existing_records([record, record], "run", expected)


def test_split_half_noise_floor_is_zero_for_repeated_halves() -> None:
    report = _split_half_noise_floor(
        {3: ["a", "b", "a", "b"], 7: ["x", "x", "x", "x"]},
        [3, 7],
    )

    assert report == {
        "mean_total_variation": 0.0,
        "mean_jensen_shannon_bits": 0.0,
    }
