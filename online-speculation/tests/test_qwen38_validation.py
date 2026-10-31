"""Validation tooling must not mistake a sampling difference for a regression."""

from pathlib import Path
import sys

import pytest


def test_failed_ar_gate_preserves_final_sampling_evidence(monkeypatch, tmp_path):
    import json
    import os
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import validate_qwen38_vllm as validation

    stats = dict(orthrus=True, verified=1, proposals=1, history_reads=1, updates=0,
                 accepted=0, history_backend="shared_target_snapshot", ar_view_checks=[dict(passed=False)])

    class Engine:
        def __init__(self, **kwargs):
            pass

        def collective_rpc(self, name, **kwargs):
            return [stats.copy()] if name == "orthrus_stats" else None

        def generate(self, *args, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(range(8)), text="test")])]

    for name in tuple(os.environ):
        if name.startswith("ORTHRUS_VALIDATE_"):
            monkeypatch.delenv(name)
    destination = tmp_path / "failed.json"
    monkeypatch.setenv("ORTHRUS_VALIDATE_OUTPUT", str(destination))
    monkeypatch.setenv("ORTHRUS_VALIDATE_AR_VIEW", "1")
    monkeypatch.setenv("ORTHRUS_VALIDATE_SAMPLING_TRIALS", "16")
    monkeypatch.setenv("ORTHRUS_VALIDATE_CASES", "short-greedy")
    monkeypatch.setattr(sys, "argv", ["run", "--model", "/model", "--mask-token-id", "1"])
    monkeypatch.setattr(validation, "prepare_runtime_environment", lambda: None)
    monkeypatch.setattr(validation, "native_options", lambda args: {})
    monkeypatch.setattr(validation, "speculative_options", lambda args: {})
    monkeypatch.setattr(validation, "sampling_calibration", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace))
    with pytest.raises(AssertionError):
        validation.main()
    report = json.loads(destination.read_text())
    assert report["sampling_calibration"] == {"passed": True}
    assert report["completed"] is False


@pytest.mark.parametrize("count", [1, 15, 16, 31])
def test_ar_audit_replays_verifier_width_including_bonus_input(monkeypatch, count):
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from qwen38_validation_worker import ar_verifier_inputs

    pending = {"inputs": torch.tensor([[8] + [1] * 31]), "candidates": torch.arange(100, 131)}
    actual = ar_verifier_inputs(pending, count)
    assert actual.shape == (1, count + 1)
    assert actual.tolist() == [[8] + list(range(100, 100 + count))]
    assert pending["inputs"][0, -1] == 1
    with pytest.raises(ValueError, match="verification count"):
        ar_verifier_inputs(pending, 32)


def test_layerwise_metrics_detect_drift_and_preserve_source(monkeypatch):
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_qwen38_layerwise import differences, first_row

    source = torch.tensor([[[1., 2.], [3., 4.]]])
    row = first_row(source)
    row[0] = 9
    assert source[0, 0, 0] == 1
    assert differences(source, source) == dict(equal=True, max_abs=0., relative_l2=0.)
    metrics = differences(source + .25, source)
    assert not metrics["equal"] and metrics["max_abs"] == .25 and metrics["relative_l2"] > 0
    with pytest.raises(ValueError, match="boundary shapes differ"):
        differences(source, source.flatten())


@pytest.mark.parametrize("mode", ["gamma_bf16", "both_bf16"])
def test_norm_diagnostic_is_scoped_and_restorable(monkeypatch, mode):
    import torch
    from types import SimpleNamespace
    from blockspec.vllm_ascend.qwen38_weights import SharedRMSNorm
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_qwen38_layerwise import LayerwiseWorker

    model = torch.nn.ModuleList([SharedRMSNorm(torch.nn.Parameter(torch.full((8,), weight,
        dtype=torch.bfloat16), requires_grad=False), 1e-6) for weight in (.125, .25)])
    worker = LayerwiseWorker()
    worker.model_runner = SimpleNamespace(drafter=SimpleNamespace(model=model))
    source = torch.arange(1, 9, dtype=torch.bfloat16)[None]
    originals = [norm(source).clone() for norm in model]
    worker.orthrus_set_norm_diagnostic(mode)
    for norm in model:
        normalized = source.float() * torch.rsqrt(source.float().square().mean(-1, keepdim=True) + norm.eps)
        if mode == "both_bf16":
            normalized = normalized.bfloat16().float()
        expected = (normalized * (1 + norm.weight).float()).bfloat16()
        torch.testing.assert_close(norm(source), expected, rtol=0, atol=0)
    worker.orthrus_set_norm_diagnostic("original")
    for norm, expected in zip(model, originals, strict=True):
        torch.testing.assert_close(norm(source), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unknown diagnostic norm"):
        worker.orthrus_set_norm_diagnostic("invalid")


def test_candidate_diagnostic_preserves_proposal_width(monkeypatch):
    import torch
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_qwen38_layerwise import LayerwiseWorker

    draft = SimpleNamespace(block_size=32, model=SimpleNamespace(config=SimpleNamespace(vocab_size=200)))

    def propose():
        draft.pending = dict(candidates=torch.arange(100, 131), q=torch.ones(31, 200))
        return [draft.pending["candidates"].tolist()]

    draft.propose_batch = propose
    worker = LayerwiseWorker()
    worker.model_runner = SimpleNamespace(drafter=draft)
    worker._layerwise_fixed_candidates = torch.arange(100, 131)
    for width, shifted in ((0, False), (1, False), (15, False), (31, True), (31, False)):
        worker.orthrus_set_proposal_limit(width)
        worker.orthrus_diagnostic_candidates("shifted" if shifted else "head")
        actual = draft.propose_batch()[0]
        assert len(actual) == width
        assert actual == list(range(137 if shifted else 100, (137 if shifted else 100) + width))
        if shifted:
            assert draft.pending["q"] is None
    worker._layerwise_fixed_candidates.add_(2)
    worker.orthrus_diagnostic_candidates("fixed")
    assert draft.propose_batch()[0] == list(range(102, 133))


def test_layerwise_report_rejects_mislabeled_native_width(monkeypatch):
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_qwen38_layerwise import LayerwiseWorker

    worker = LayerwiseWorker()
    worker._layerwise = dict(decode=dict(query_rows=1, native={"0.input": torch.ones(1, 2)}),
                            k2=dict(query_rows=32, native={"0.input": torch.ones(32, 2)}))
    with pytest.raises(AssertionError, match="k2 scheduled 32 rows, expected 2"):
        worker.orthrus_layerwise_report()


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
    with pytest.raises(AssertionError, match="nonfinite"):
        check_calibration({1: 10}, {1: float("nan")}, {1: 1.})


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


@pytest.mark.parametrize("speculative", [False, True])
def test_sampling_audit_captures_reached_rows_before_logit_mutation(monkeypatch, speculative):
    import torch
    from types import SimpleNamespace as NS
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from qwen38_validation_worker import ValidationWorker
    logits = torch.tensor([[1., 2., 3.], [3., 1., 2.], [0., 0., 0.]])
    expected = logits[:, [0, 2]].softmax(-1)
    tokens = torch.tensor([[2, 0, -1]])

    def sample(logits, metadata):
        logits.fill_(-torch.inf)
        return NS(sampled_token_ids=tokens)

    worker = ValidationWorker()
    worker.model_runner = NS(_sample=sample)
    worker.orthrus_enable_sampling_calibration([0, 2])
    metadata = NS(target_logits_indices=torch.tensor([0, 1]), bonus_logits_indices=torch.tensor([2]))
    worker.model_runner._sample(logits, metadata if speculative else None)
    rows = worker.orthrus_sampling_rows()
    assert [row["token"] for row in rows] == [2, 0]
    torch.testing.assert_close(torch.tensor([row["probabilities"] for row in rows]), expected[:2])
    assert worker.orthrus_sampling_rows() == []


@pytest.mark.parametrize("mode", ["oracle", "reject"])
def test_proposal_fixture_starts_after_the_committed_anchor(monkeypatch, mode):
    import torch
    from types import SimpleNamespace as NS
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from qwen38_validation_worker import ValidationWorker
    draft = NS(runner=NS(input_batch=NS(num_tokens_no_spec=[11], num_prompt_tokens=[10])),
               model=NS(config=NS(vocab_size=256)),
               history_reader=NS(read=lambda **kwargs: "history"))

    def propose():
        draft.pending = dict(candidates=torch.zeros(31, dtype=torch.long), q=torch.ones(31, 256) / 256)
        return [[0] * 31]

    draft.propose_batch = propose
    worker = ValidationWorker()
    worker.model_runner = NS(drafter=draft)
    worker.orthrus_set_proposal_fixture(list(range(128)), mode)
    expected = list(range(1, 32)) if mode == "oracle" else list(range(2, 33))
    assert draft.propose_batch() == [expected]
    assert draft.pending["q"] is None
    assert draft.history_reader.read(draft_count=31, sampled_count=32) == "history"
    assert worker._fixture_reads == [32]


def test_width_diagnostic_only_limits_returned_candidates(monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from qwen38_validation_worker import ValidationWorker
    draft = NS(block_size=32, propose_batch=lambda: [[2, 3, 4]])
    worker = ValidationWorker()
    worker.model_runner = NS(drafter=draft)
    worker.orthrus_set_proposal_limit(1)
    assert draft.propose_batch() == [[2]]
    worker.orthrus_set_proposal_limit(0)
    assert draft.propose_batch() == [[]]
    worker.orthrus_set_proposal_limit(31)
    assert draft.propose_batch() == [[2, 3, 4]]
    with pytest.raises(ValueError, match="fit"):
        worker.orthrus_set_proposal_limit(32)


def test_projection_diagnostic_preserves_native_weights(monkeypatch):
    import torch
    from types import SimpleNamespace as NS
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from qwen38_validation_worker import ValidationWorker
    module = torch.nn.Linear(4, 8, bias=False).to(torch.bfloat16).requires_grad_(False)
    original = module.weight.clone()
    model = NS(model=NS(layers=[NS(linear_attn=NS(in_proj_qkvz=module))]))
    worker = ValidationWorker()
    worker.model_runner = NS(get_model=lambda: model, input_batch=NS(req_ids=["r1"]))
    worker._logit_boundary = ([12], [1])
    worker.orthrus_enable_projection_audit()
    module(torch.ones(2, 4, dtype=torch.bfloat16))
    result = worker.orthrus_projection_audit()
    assert [row["width"] for row in result["direct_linear"]] == [2, 8, 16, 18, 32]
    assert result["native_calls"][0]["input_equal"]
    torch.testing.assert_close(module.weight, original, rtol=0, atol=0)


def test_fixed_replay_failure_is_returned_for_evidence_collection(monkeypatch):
    import torch
    from types import SimpleNamespace as NS
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import qwen38_validation_worker as audit
    parameter = torch.ones(1)
    losses = iter((.5, .6))
    learner = NS(execution={"weight": parameter}, replay=True, updates=0, trainable_parameters=1,
                 backward=lambda: next(losses), optimizer=NS(zero_grad=lambda **kwargs: None), _check=lambda: None)

    def update():
        parameter.add_(1)
        learner.updates += 1
        return {"loss": .55}

    learner.update = update
    worker = audit.ValidationWorker()
    worker.model_runner = NS(drafter=NS(learner=learner, history_reader=None, tp=NS(rank=0)))
    monkeypatch.setattr(audit, "_cache_snapshot", lambda reader: [torch.ones(1)])
    result = worker.orthrus_validate_online_replay(1)
    assert result["passed"] is False and result["cache_unchanged"]
    assert result["steps"] == 1 and result["changed_parameters"] == ["weight"]
