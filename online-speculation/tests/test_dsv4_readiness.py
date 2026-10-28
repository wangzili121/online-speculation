import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_records_failure_and_timeout(tmp_path):
    runner = load_script("validate_dsv4_readiness")
    failed = runner.run_check([sys.executable, "-c", "print('diagnostic'); exit(3)"],
                              tmp_path / "failed.log", 10, os.environ.copy())
    assert failed["returncode"] == 3
    assert not failed["timed_out"]
    assert "diagnostic" in (tmp_path / "failed.log").read_text()
    timed = runner.run_check([sys.executable, "-c", "import time; time.sleep(60)"],
                             tmp_path / "timed.log", 0.1, os.environ.copy())
    assert timed["returncode"] == 124
    assert timed["timed_out"]
    assert timed["seconds"] < 15


@pytest.mark.parametrize("stage", ["cpu", "npu"])
def test_readiness_stages_do_not_launch_services(stage):
    runner = load_script("validate_dsv4_readiness")
    checks = runner.commands(SimpleNamespace(stage=stage))
    assert checks
    assert all(command[0] == sys.executable for _, command in checks)
    assert not any("serve" in argument or "docker" in argument
                   for _, command in checks for argument in command)


def test_first_token_difference_handles_lengths_and_equal_tokens(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    replay = load_script("check_dsv4_sampling_replay")
    assert replay.first_difference({"token_ids": [1, 2]}, {"token_ids": [1, 2]}) is None
    assert replay.first_difference({"token_ids": [1, 2]}, {"token_ids": [1]}) == {
        "index": 1, "reference": [2], "actual": []}
    assert replay.first_difference({"token_ids": [1]}, {"token_ids": [2]})["index"] == 0
    assert replay.first_difference({}, {}) == {"token_ids_available": False}


def test_healthcheck_gives_cold_decode_a_separate_timeout(monkeypatch):
    health = load_script("healthcheck_dsv4")
    calls = []

    def request(url, payload, timeout):
        calls.append((url, timeout))
        if url.endswith("/v1/models"):
            return {"data": [{"id": "test"}]}
        return {"choices": [{}], "id": "smoke"}

    monkeypatch.setattr(health, "request", request)
    monkeypatch.setattr(sys, "argv", ["healthcheck", "--model", "test"])
    assert health.main() == 0
    assert [timeout for _, timeout in calls] == [30.0, 30.0, 300.0]


def test_service_stage_preserves_long_cases_and_eos(tmp_path):
    runner = load_script("validate_dsv4_readiness")
    checks = dict(runner.commands(SimpleNamespace(
        stage="service", base_url="http://127.0.0.1:1234", model="test",
        output=tmp_path, cases=tmp_path / "cases.json", include_long=True, eos_token_id=1,
    )))
    assert "--include-long" in checks["snapshot"]
    eos = checks["lifecycle"].index("--eos-token-id")
    assert checks["lifecycle"][eos + 1] == "1"


def test_greedy_probe_scores_the_same_causal_position(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    probe = load_script("check_dsv4_greedy_divergence")
    case = {"prompt": "test", "max_tokens": 3}
    for name, ids in (("baseline", [3, 4, 5]), ("candidate", [3, 9, 5])):
        (tmp_path / f"{name}.json").write_text(json.dumps({"greedy": [{
            "case": case, "token_ids": ids, "usage": {"prompt_tokens": 2},
        }]}))
    prompts = []

    def request(url, payload):
        if url.endswith("/tokenize"):
            return {"tokens": [1, 2]}
        tokens = payload["prompt"]
        prompts.append(tokens)
        if not payload.get("echo"):
            return {"choices": [{"token_ids": [9]}]}
        return {"usage": {"prompt_tokens": len(tokens)}, "choices": [{"logprobs": {
            "tokens": tokens, "token_logprobs": [-1.0] * len(tokens),
            "top_logprobs": [{"4": -1.0}] * len(tokens),
        }}]}

    output = tmp_path / "probe.json"
    monkeypatch.setattr(probe, "_json_request", request)
    monkeypatch.setattr(sys, "argv", ["probe", "--base-url", "http://test", "--model", "test",
        "--baseline", str(tmp_path / "baseline.json"),
        "--candidate", str(tmp_path / "candidate.json"), "--output", str(output)])
    probe.main()
    assert prompts[0] == [1, 2, 3]
    assert len(prompts[1]) == 4 and len(prompts[2]) == 35
    rows = json.loads(output.read_text())["cases"][0]["echoed_prefills"]
    assert [row["scored_token"] for row in rows] == [4, 4]


def test_layer_trace_compares_shapes_and_missing_stages():
    import torch

    trace = load_script("trace_dsv4_native_layers")
    left = {"same": torch.tensor([1.0]), "diff": torch.tensor([2.0]),
            "shape": torch.ones(2), "decode_only": torch.ones(1)}
    right = {"same": torch.tensor([1.0]), "diff": torch.tensor([2.25]),
             "shape": torch.ones(3), "prefill_only": torch.ones(1)}
    report = trace.compare_traces(left, right)
    assert report["stages"][0]["equal"]
    assert report["stages"][1]["max_abs"] == 0.25
    assert report["stages"][2]["shapes"] == [[2], [3]]
    assert report["only_decode"] == ["decode_only"]
    assert report["only_prefill"] == ["prefill_only"]
    empty = trace.compare_traces({"kv": torch.empty(0)}, {"kv": torch.empty(0)})
    assert empty["stages"][0]["equal"] and empty["stages"][0]["max_abs"] == 0
    with pytest.raises(ValueError, match="captured activations"):
        trace.compare_traces({}, {})


@pytest.mark.parametrize("invariant", ["0", "1"])
def test_invariant_launch_uses_consistent_native_prolog(invariant):
    config = {"other_setting": 123, "multistream_dsv4_dsa_overlap": True}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; configure_dsv4_runtime; printf "%s" "$ADDITIONAL_CONFIG"',
         "test", str(SCRIPTS / "dsv4_service_env.sh")],
        env={**os.environ, "VLLM_BATCH_INVARIANT": invariant, "ADDITIONAL_CONFIG": json.dumps(config)},
        check=True, text=True, capture_output=True)
    actual = json.loads(result.stdout)
    assert actual["other_setting"] == 123
    assert actual["multistream_dsv4_dsa_overlap"] == (invariant == "0")
