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


def test_soak_cases_cover_long_decode_and_exact_context_tail(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_qwen38_vllm import validation_cases
    cases = {name: (prompt, temperature, length) for name, prompt, temperature, length
             in validation_cases(extended=True, soak=True, max_model_len=4096)}
    assert cases["long-decode"][2] == 128
    assert len(cases["context-tail"][0]["prompt_token_ids"]) + cases["context-tail"][2] == 4096
    assert cases["after-context-tail"] == cases["short-greedy"]
    with pytest.raises(ValueError, match="max_model_len"):
        validation_cases(soak=True, max_model_len=32)


@pytest.mark.parametrize("budget", [None, 12884901888])
def test_explicit_kv_budget_is_optional_and_forwarded(monkeypatch, budget):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_qwen38_vllm import arguments, native_options
    command = ["run", "--model", "/models/base", "--mask-token-id", "1"]
    if budget is not None:
        command += ["--kv-cache-memory-bytes", str(budget)]
    monkeypatch.setattr(sys, "argv", command)
    assert native_options(arguments()).get("kv_cache_memory_bytes") == budget


def test_explicit_kv_budget_must_be_positive(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_qwen38_vllm import arguments
    monkeypatch.setattr(sys, "argv", ["run", "--model", "/models/base", "--mask-token-id", "1",
                                     "--kv-cache-memory-bytes", "0"])
    with pytest.raises(SystemExit):
        arguments()
