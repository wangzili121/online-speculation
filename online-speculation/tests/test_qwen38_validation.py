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


def test_repeated_greedy_checks_every_repetition(validation):
    candidate = report()
    candidate["outputs"].append(dict(case="repeat", reference_case="greedy", temperature=0., tokens=[2, 3]))
    validation(report(), candidate)
    candidate["outputs"][-1]["tokens"] = [2, 4]
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


def test_comprehensive_cases_cover_lengths_and_physical_page_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_qwen38_vllm import validation_cases
    cases = {name: (prompt, length) for name, prompt, _, length
             in validation_cases(comprehensive=True, max_model_len=4096)}
    assert [cases[f"length-{length}"][1] for length in (1, 2, 31, 32, 33)] == [1, 2, 31, 32, 33]
    assert len(cases["page-edge"][0]["prompt_token_ids"]) == 1535
    assert {"code", "json", "unicode"} <= cases.keys()


def test_sampling_calibration_detects_biased_counts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_qwen38_vllm import check_calibration
    check_calibration({1: 101, 2: 99}, {1: 100., 2: 100.}, {1: 50., 2: 50.})
    with pytest.raises(AssertionError, match="calibration failed"):
        check_calibration({1: 200, 2: 0}, {1: 100., 2: 100.}, {1: 50., 2: 50.})


def test_calibration_normalizes_requested_raw_scores_without_underflow(monkeypatch):
    import math
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_qwen38_vllm import restricted_probabilities
    raw = {token: SimpleNamespace(logprob=math.log(probability) - 1000)
           for token, probability in ((1, .1), (2, .2), (3, .3))}
    assert restricted_probabilities(raw, [1, 2, 3]) == pytest.approx({1: 1/6, 2: 1/3, 3: 1/2})


def test_runtime_environment_preserves_project_and_adds_cann(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_qwen38_vllm import prepare_runtime_environment
    site = tmp_path / "python" / "site-packages"
    site.mkdir(parents=True)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/work/src:/work/scripts")
    monkeypatch.setattr(sys, "path", list(sys.path))
    prepare_runtime_environment()
    prepare_runtime_environment()
    import os
    assert os.environ["PYTHONPATH"].split(":") == ["/work/src", "/work/scripts", str(site)]
    assert sys.path.count(str(site)) == 1


def test_memory_check_compares_matching_boundaries_after_warmup(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from accept_qwen38_vllm import memory_growth
    report = dict(outputs=[dict(case=f"run{i}", reference_case="same") for i in range(3)],
                  memory=[dict(case=f"run{i}", workers=[dict(allocated=size)])
                          for i, size in enumerate((10, 200, 201))])
    assert memory_growth(report, tolerance=2)["max_live_growth_bytes"] == 1
    report["memory"][-1]["workers"][0]["allocated"] = 300
    with pytest.raises(AssertionError, match="allocation grew"):
        memory_growth(report, tolerance=2)


def test_weight_fingerprint_tracks_values_not_tensor_identity(monkeypatch):
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from qwen38_validation_worker import _fingerprint
    parameter = torch.tensor([1., 2.], dtype=torch.bfloat16)
    before = _fingerprint([("weight", parameter)])
    assert before == _fingerprint([("weight", parameter.clone())])
    parameter[0] = 3
    assert before != _fingerprint([("weight", parameter)])
