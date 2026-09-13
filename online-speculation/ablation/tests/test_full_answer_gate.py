from types import SimpleNamespace
import copy

import pytest
import torch

from blockspec.parallel import DualViewConfig, DualViewDecoder
from blockspec.parallel.fitting import FitConfig
from blockspec.parallel.generation import Generation
from blockspec_ablation import cold_start as cold
from blockspec_ablation.full_answer_gate import FullAnswerGate


def decoder():
    torch.manual_seed(463)
    return DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                          num_hidden_layers=2, num_attention_heads=2,
                                          num_key_value_heads=1, head_dim=8)).eval().requires_grad_(False)


def example(step=1, seconds=10., predicted=8.):
    return {"step": step, "reference_tokens": 4, "reference_seconds": seconds,
            "predicted_seconds": predicted, "evaluation_seconds": .2}


def test_screen_aggregation_uses_request_totals_and_only_arms_confirmation():
    gate = FullAnswerGate(2, 1.02, .4)
    first = gate.observe(example(seconds=1., predicted=.5), .2)
    assert not first["complete"] and not first["passed"] and not gate.ready
    second = gate.observe(example(seconds=10., predicted=10.), .3)
    assert second["ratio"] == pytest.approx(11 / 10.5)
    assert second["screen_passed"] and gate.ready and not second["passed"] and not second["complete"]
    assert gate.estimate == pytest.approx(1.2 * (10.5 / 2 + .1))
    with pytest.raises(ValueError):
        gate.observe(example(), .2)
    gate.confirmed()
    assert not gate.pending and not gate.ready


def test_screen_failure_releases_the_fixed_version():
    gate = FullAnswerGate(1, 1.02)
    result = gate.observe(example(predicted=12.), .2)
    assert result["complete"] and not result["screen_passed"] and not result["passed"]
    assert gate.pending == []


@pytest.mark.parametrize("count,margin,cost", [(0, 1.02, .4), (True, 1.02, .4), (2, .9, .4),
                                               (2, float("nan"), .4), (2, 1.02, 0)])
def test_invalid_gate_configuration(count, margin, cost):
    with pytest.raises(ValueError):
        FullAnswerGate(count, margin, cost)


def service_fixture(monkeypatch, *, screen_ratio=1.25, candidate_seconds=8., screen_count=2, reuse_speculative=False):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    fit = FitConfig(steps=8, warmup_steps=0, sequence_length=12, anchors_per_sequence=2, learning_rate=.001)
    settings = cold.ServiceConfig(probe_every=1, full_answer_screen=True, screen_requests=screen_count,
                                    publish_margin=1.05, reuse_speculative_answers=reuse_speculative)
    service = cold.ColdStartService(decoder(), fit, settings, gate_prompts=[prompt])
    service.learner.steps = 1
    service.budget.delivered(1000.)
    service.tokens, service.requests = 400, 100
    service.step_estimate = 1e6
    calls = []

    def connect(current):
        def generate(prompt, tokens, *, speculative, seed, prefix_tokens=None):
            calls.append((speculative, seed, prefix_tokens, len(current.replay.records)))
            if speculative:
                assert current.learner.serving_step == current.learner.steps
                ticks[0] += candidate_seconds
                return Generation(tokens=[4, 5, 6, 7], seconds=candidate_seconds)
            capture = .01 if prefix_tokens else 0.
            ticks[0] += 10 + capture
            return Generation(tokens=[4, 5, 6, 7], seconds=10 + capture,
                               prefix_tokens=1 if prefix_tokens else 0,
                               prefix_seconds=2. if prefix_tokens else None, prefix_capture_seconds=capture)
        monkeypatch.setattr(current, "_generate", generate)

    def screen(*args, **kwargs):
        ticks[0] += .2
        return SimpleNamespace(predicted_seconds=10 / screen_ratio, measured_seconds=.2,
                                score_seconds=.1, calibration_seconds=.1,
                                work=SimpleNamespace(decode_forwards=torch.tensor(2.)))
    monkeypatch.setattr(cold, "estimate_trace", screen)
    connect(service)
    return service, prompt, ticks, calls, connect


@pytest.mark.parametrize("reuse_speculative", [False, True])
def test_actual_full_confirmation_is_required_and_all_costs_are_charged(monkeypatch, reuse_speculative):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch, reuse_speculative=reuse_speculative)
    original = {name: p.clone() for name, p in service.model.named_parameters()}
    for index in range(2):
        _, row = service.serve(prompt, 4, seed=10 + index)
        assert row["probe"]["stage"] == "screen" and not service.speculating
        assert service.learner.steps == 1 and service.learner.serving_step == 0
    assert service.full_gate.ready and service.budget.costs["screening"] == pytest.approx(.4)
    assert service.budget.costs["validation_capture"] == pytest.approx(.02)
    _, row = service.serve(prompt, 4, seed=12)
    assert row["mode"] == "ar" and row["probe"]["stage"] == "confirmation"
    assert row["probe"]["ratio"] == pytest.approx(1.25) and row["probe"]["passed"]
    assert service.speculating and service.learner.serving_step == 1 and service.next_probe == 2
    assert service.budget.spent == pytest.approx(8.42)
    assert calls == [(False, 10, 1, 0), (False, 11, 1, 1), (False, 12, None, 2), (True, 12, None, 2)]
    assert len(service.replay.records) == 3
    assert service.full_gate.ar_tps == pytest.approx(.4)
    assert all(torch.equal(p, original[name]) for name, p in service.model.named_parameters()
               if ".attention.draft." not in name)


def test_waiting_confirmation_restores_evidence_and_can_finish(monkeypatch):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch)
    for seed in (10, 11):
        service.serve(prompt, 4, seed=seed)
    state = service.state_dict()
    restored = cold.ColdStartService(decoder(), service.fit, service.config, gate_prompts=[prompt])
    restored.load_state_dict(state)
    connect(restored)
    assert restored.full_gate.state_dict() == service.full_gate.state_dict()
    state["full_answer_gate"]["pending"][0]["predicted_seconds"] = 77.
    assert restored.full_gate.pending[0]["predicted_seconds"] == pytest.approx(8.)
    _, row = restored.serve(prompt, 4, seed=12)
    assert row["probe"]["passed"] and restored.speculating


def test_publication_then_update_retains_the_validated_serving_version(monkeypatch):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch, screen_count=1)
    service.serve(prompt, 4, seed=10)
    service.step_estimate = .1
    before = {name: p.clone() for name, p in service.learner.execution.items()}
    def update(batches):
        ticks[0] += .1
        service.learner.steps += 1
        with torch.no_grad():
            for p in service.learner.master.values():
                p.add_(.1)
        return {"step": service.learner.steps}
    monkeypatch.setattr(service.learner, "step", update)
    _, row = service.serve(prompt, 4, seed=11)
    assert row["probe"]["passed"] and row["update"]["step"] == 2
    assert service.learner.steps == 2 and service.learner.serving_step == 1
    assert all(torch.equal(p, before[name]) for name, p in service.learner.execution.items())
    restored = cold.ColdStartService(decoder(), service.fit, service.config, gate_prompts=[prompt])
    restored.load_state_dict(service.state_dict())
    assert restored.speculating and restored.learner.steps == 2 and restored.learner.serving_step == 1
    assert not restored.full_gate.pending and not restored.full_gate.ready


def test_unfunded_confirmation_holds_parameters_and_evidence(monkeypatch):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch)
    for seed in (10, 11):
        service.serve(prompt, 4, seed=seed)
    service.budget.charge("test_reservation", service.budget.credit - .01)
    before = copy.deepcopy(service.full_gate.state_dict())
    _, row = service.serve(prompt, 4, seed=12)
    assert row["probe"] is None and row["update"] is None
    assert service.full_gate.state_dict() == before and service.learner.steps == 1
    assert calls[-1] == (False, 12, None, 2)


def test_current_request_income_cannot_select_its_own_screen(monkeypatch):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch)
    service.budget.charge("test_reservation", service.budget.credit - .35)
    _, row = service.serve(prompt, 4, seed=12)
    assert service.budget.credit > service.config.initial_screen_estimate
    assert row["probe"] is None and calls == [(False, 12, None, 0)]


def test_failed_screen_and_failed_real_trial_resume_learning_schedule(monkeypatch):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch, screen_ratio=.9, screen_count=1)
    _, row = service.serve(prompt, 4, seed=12)
    assert row["probe"]["complete"] and not service.speculating and service.next_probe == 2
    service2, prompt, ticks, calls, connect = service_fixture(monkeypatch, candidate_seconds=11., screen_count=1)
    service2.serve(prompt, 4, seed=10)
    _, row = service2.serve(prompt, 4, seed=11)
    assert row["probe"]["stage"] == "confirmation" and not row["probe"]["passed"]
    assert not service2.speculating and service2.next_probe == 2 and not service2.full_gate.ready


@pytest.mark.parametrize("mutation", ["wrong_step", "missing", "ready_without_count", "bad_time", "speculating",
                                      "too_many", "bad_margin"])
def test_restore_rejects_inconsistent_screening_state(monkeypatch, mutation):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch)
    service.serve(prompt, 4, seed=10)
    state = service.state_dict()
    gate = state["full_answer_gate"]
    if mutation == "wrong_step":
        gate["pending"][0]["step"] = 2
    elif mutation == "missing":
        state.pop("full_answer_gate")
    elif mutation == "ready_without_count":
        gate["ready"], gate["confirmation_estimate"] = True, 10.
    elif mutation == "bad_time":
        gate["screen_estimate"] = float("nan")
    elif mutation == "speculating":
        state["speculating"], state["serving_step"] = True, 1
    elif mutation == "too_many":
        gate["pending"] *= 3
    else:
        gate["pending"] *= 2
        gate["ready"], gate["confirmation_estimate"] = True, 10.
        gate["pending"][0]["predicted_seconds"] = 20.
    with pytest.raises(ValueError):
        cold.ColdStartService(decoder(), service.fit, service.config, gate_prompts=[prompt]).load_state_dict(state)


def test_full_policy_requires_an_independent_post_publication_gate():
    with pytest.raises(ValueError):
        cold.ColdStartService(decoder(), FitConfig(), cold.ServiceConfig(full_answer_screen=True))
    with pytest.raises(ValueError):
        cold.ServiceConfig(full_answer_screen=True, reuse_ar_prefix=True)
    with pytest.raises(ValueError):
        cold.ServiceConfig(full_answer_screen=True, live_probe_requests=2)


def test_legacy_service_state_restores_without_new_gate_fields():
    service = cold.ColdStartService(decoder(), FitConfig())
    state = service.state_dict()
    state.pop("full_answer_gate")
    for key in ("full_answer_screen", "screen_requests", "screen_margin", "initial_screen_estimate"):
        state["controller"].pop(key)
    restored = cold.ColdStartService(decoder(), FitConfig())
    restored.load_state_dict(state)
    assert restored.full_gate is None


def test_confirmation_overrun_creates_debt_and_retains_exact_serving_guard(monkeypatch):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch, screen_count=1, candidate_seconds=12.)
    service.serve(prompt, 4, seed=10)
    service.budget.charge("test_reservation", service.budget.credit - service.full_gate.estimate)
    _, row = service.serve(prompt, 4, seed=11)
    assert row["budget"]["debt_seconds"] > 0 and not row["probe"]["passed"]
    assert not service.speculating
    _, following = service.serve(prompt, 4, seed=12)
    assert following["update"] is None and following["probe"] is None


def test_recomputed_support_failure_returns_ar_and_charges_attempt(monkeypatch):
    service, prompt, ticks, calls, connect = service_fixture(monkeypatch)
    original = {name: p.clone() for name, p in service.learner.execution.items()}
    def failed_screen(*args, **kwargs):
        ticks[0] += .2
        raise cold.TraceSupportError("numeric support drift")
    monkeypatch.setattr(cold, "estimate_trace", failed_screen)
    result, row = service.serve(prompt, 4, seed=10)
    assert result.tokens == [4, 5, 6, 7] and row["mode"] == "ar"
    assert row["probe"]["failure_kind"] == "recomputed_target_support" and row["probe"]["complete"]
    assert not service.speculating and service.next_probe == 2
    assert service.budget.costs["screening"] == pytest.approx(.2)
    assert all(torch.equal(p, original[name]) for name, p in service.learner.execution.items())
    assert len(service.replay.records) == 1


def live_update_fixture(monkeypatch, *, baseline_seconds=8., candidate_seconds=6., ar_seconds=10.,
                        predicted_seconds=6., screen_count=2):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    fit = FitConfig(steps=8, warmup_steps=0, sequence_length=12, anchors_per_sequence=2)
    config = cold.ServiceConfig(probe_every=1, full_answer_screen=True, reuse_speculative_answers=True,
                                screen_requests=screen_count, publish_margin=1.05)
    service = cold.ColdStartService(decoder(), fit, config, gate_prompts=[prompt])
    service.learner.steps = 1
    service.learner.publish()
    service.learner.steps, service.next_probe, service.speculating = 2, 2, True
    with torch.no_grad():
        for p in service.learner.master.values():
            p.add_(.01)
    service.full_gate.confirmed(ar_tps=4 / ar_seconds)
    service.budget.delivered(10000.)
    service.tokens, service.requests, service.step_estimate = 4000, 100, 1e6
    calls = []

    def connect(current):
        def generate(prompt, tokens, *, speculative, seed, prefix_tokens=None):
            name = ("candidate" if current.learner.serving_step == current.learner.steps else "baseline"
                    ) if speculative else "ar"
            calls.append((name, seed, prefix_tokens, len(current.replay.records)))
            elapsed = {"baseline": baseline_seconds, "candidate": candidate_seconds, "ar": ar_seconds}[name]
            capture = .01 if prefix_tokens else 0.
            ticks[0] += elapsed + capture
            return Generation(tokens=[4, 5, 6, 7], seconds=elapsed + capture,
                               prefix_tokens=1 if prefix_tokens else 0,
                               prefix_seconds=2. if prefix_tokens else None, prefix_capture_seconds=capture)
        monkeypatch.setattr(current, "_generate", generate)

    def screen(*args, **kwargs):
        assert kwargs["prefill_seconds"] == 2. and kwargs["output_budget"] == 4
        ticks[0] += .2
        return SimpleNamespace(predicted_seconds=predicted_seconds, measured_seconds=.2,
                                score_seconds=.1, calibration_seconds=.1,
                                work=SimpleNamespace(decode_forwards=torch.tensor(2.)))
    monkeypatch.setattr(cold, "estimate_trace", screen)
    connect(service)
    return service, prompt, ticks, calls, connect


def test_post_publication_reuses_delivered_answer_and_reserves_two_extra_paths(monkeypatch):
    service, prompt, ticks, calls, connect = live_update_fixture(monkeypatch)
    for seed in (10, 11):
        _, row = service.serve(prompt, 4, seed=seed)
        assert row["mode"] == "speculative" and not row["probe"]["passed"]
        assert row["probe"]["reference_step"] == 1 and row["probe"]["step"] == 2
        assert row["probe"]["candidate_over_serving"] == pytest.approx(8 / 6)
        assert "candidate_over_ar" not in row["probe"]
    assert service.full_gate.ready and service.full_gate.estimate == pytest.approx(1.2 * (6 + 10))
    assert service.learner.serving_step == 1 and service.learner.steps == 2
    assert service.budget.costs["validation_capture"] == pytest.approx(.02)
    _, row = service.serve(prompt, 4, seed=12)
    probe = row["probe"]
    assert probe["passed"] and probe["ratio"] == pytest.approx(8 / 6)
    assert probe["candidate_over_ar"] == pytest.approx(10 / 6)
    assert probe["reference_kind"] == "delivered_speculative_full" and probe["reference_step"] == 1
    assert service.learner.serving_step == 2 and service.speculating and service.next_probe == 3
    assert service.budget.spent == pytest.approx(16.42) and service.probe_legs == 2
    assert calls == [("baseline", 10, 1, 0), ("baseline", 11, 1, 1),
                     ("baseline", 12, None, 2), ("ar", 12, None, 2), ("candidate", 12, None, 2)]
    assert len(service.replay.records) == 3


@pytest.mark.parametrize("count", [1, 2])
def test_live_update_pending_evidence_and_serving_identity_restore(monkeypatch, count):
    service, prompt, ticks, calls, connect = live_update_fixture(monkeypatch, screen_count=count)
    service.serve(prompt, 4, seed=10)
    state = service.state_dict()
    restored = cold.ColdStartService(decoder(), service.fit, service.config, gate_prompts=[prompt])
    restored.load_state_dict(state)
    connect(restored)
    assert restored.full_gate.state_dict() == service.full_gate.state_dict()
    assert restored.learner.serving_step == 1 and restored.learner.steps == 2
    if count == 2:
        restored.serve(prompt, 4, seed=11)
    _, row = restored.serve(prompt, 4, seed=12)
    assert row["probe"]["passed"] and restored.learner.serving_step == 2
    expected = ["candidate", "ar"] if count == 1 else ["ar", "candidate"]
    assert [item[0] for item in calls[-2:]] == expected


@pytest.mark.parametrize("baseline,candidate,ar,passed,fallback", [
    (8., 8.5, 10., False, False), (11., 12., 10., False, True),
    (10., 8., 7., False, True), (8., 6., 10., True, False),
])
def test_live_confirmation_checks_both_serving_and_ar(monkeypatch, baseline, candidate, ar, passed, fallback):
    service, prompt, ticks, calls, connect = live_update_fixture(
        monkeypatch, baseline_seconds=baseline, candidate_seconds=candidate, ar_seconds=ar, screen_count=1)
    old = {name: p.clone() for name, p in service.learner.execution.items()}
    service.serve(prompt, 4, seed=10)
    _, row = service.serve(prompt, 4, seed=11)
    assert row["probe"]["passed"] is passed and row["probe"]["fallback"] is fallback
    assert service.speculating is (passed or not fallback) and service.next_probe == 3
    assert service.full_gate.ar_tps == pytest.approx(4 / ar)
    assert service.budget.costs["validation"] == pytest.approx(candidate + ar)
    if not passed:
        assert service.learner.serving_step == 1
        assert all(torch.equal(p, old[name]) for name, p in service.learner.execution.items())


def test_post_publication_unfunded_trial_holds_versions_and_does_not_run_independent_gate(monkeypatch):
    service, prompt, ticks, calls, connect = live_update_fixture(monkeypatch, screen_count=1)
    service.serve(prompt, 4, seed=10)
    service.budget.charge("reservation", service.budget.credit - .01)
    before = copy.deepcopy(service.full_gate.state_dict())
    _, row = service.serve(prompt, 4, seed=11)
    assert row["probe"] is None and row["update"] is None
    assert len(calls) == 2 and calls[-1][:3] == ("baseline", 11, None)
    assert service.full_gate.state_dict() == before
    assert service.learner.steps == 2 and service.learner.serving_step == 1


def test_post_publication_cost_overrun_stops_new_maintenance(monkeypatch):
    service, prompt, ticks, calls, connect = live_update_fixture(
        monkeypatch, screen_count=1, candidate_seconds=15.)
    service.serve(prompt, 4, seed=10)
    service.budget.charge("reservation", service.budget.credit - service.full_gate.estimate)
    _, row = service.serve(prompt, 4, seed=11)
    assert row["budget"]["debt_seconds"] > 0 and not row["probe"]["passed"]
    assert service.speculating and service.learner.serving_step == 1
    _, row = service.serve(prompt, 4, seed=12)
    assert row["update"] is None and row["probe"] is None


@pytest.mark.parametrize("mutation", ["serving_version", "reference_version", "mode", "ar_rate", "missing_rate"])
def test_live_screen_restore_rejects_mismatched_reference(monkeypatch, mutation):
    service, prompt, ticks, calls, connect = live_update_fixture(monkeypatch)
    service.serve(prompt, 4, seed=10)
    state = service.state_dict()
    if mutation == "serving_version":
        state["serving_step"] = 2
    elif mutation == "reference_version":
        state["full_answer_gate"]["pending"][0]["reference_step"] = 0
    elif mutation == "mode":
        state["speculating"] = False
    elif mutation == "ar_rate":
        state["full_answer_gate"]["ar_tps"] = 0.
    else:
        state["full_answer_gate"].pop("ar_tps")
    with pytest.raises(ValueError):
        cold.ColdStartService(decoder(), service.fit, service.config, gate_prompts=[prompt]).load_state_dict(state)


def test_live_reuse_requires_explicit_full_answer_policy():
    with pytest.raises(ValueError):
        cold.ServiceConfig(reuse_speculative_answers=True)
    with pytest.raises(ValueError):
        cold.ServiceConfig(full_answer_screen=True, reuse_speculative_answers=1)


def test_screen_cannot_mix_serving_versions():
    gate = FullAnswerGate(reuse_speculative=True)
    gate.confirmed(ar_tps=1.)
    gate.observe({**example(step=3), "reference_step": 1}, .2)
    with pytest.raises(ValueError):
        gate.observe({**example(step=3), "reference_step": 2}, .2)
