"""Validation tooling must not mistake a sampling difference for a regression."""

from pathlib import Path
import sys

import pytest


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_qwen38_vllm import compare_greedy
    return compare_greedy


def report(greedy=(2, 3), sampled=(5, 7)):
    return {"outputs": [dict(case="greedy", temperature=0., tokens=list(greedy)),
                        dict(case="sampling", temperature=1., tokens=list(sampled))]}


def test_greedy_comparison_ignores_sampling_rng(validation):
    validation(report(), report(sampled=(8, 9)))


@pytest.mark.parametrize("candidate", [report(greedy=(2, 4)), {"outputs": []}])
def test_greedy_comparison_requires_all_cases_and_exact_tokens(validation, candidate):
    with pytest.raises(AssertionError, match="greedy mismatch"):
        validation(report(), candidate)


def test_checkpoint_frequency_is_forwarded(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_qwen38_vllm import arguments, speculative_options
    monkeypatch.setattr(sys, "argv", ["run", "--model", "/models/base", "--mask-token-id", "1",
                                     "--online", "--checkpoint-every", "1"])
    options = speculative_options(arguments())["additional_config"]["qwen38_orthrus"]
    assert options["checkpoint_every"] == 1 and options["online"]
