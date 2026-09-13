from dataclasses import replace
import copy
from itertools import product
import json

import pytest
from safetensors.torch import save_file
import torch

from blockspec.parallel import DualViewConfig, DualViewDecoder
from blockspec.parallel.fitting import FitConfig
from blockspec.parallel.generation import Generation
from blockspec.parallel.training import distillation_loss
from blockspec_ablation import cold_start as cold


def model():
    torch.manual_seed(442)
    return DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                          num_hidden_layers=2, num_attention_heads=2,
                                          num_key_value_heads=1, head_dim=8)).eval().requires_grad_(False)


def fit():
    return FitConfig(steps=8, warmup_steps=0, sequence_length=12, anchors_per_sequence=2, learning_rate=.001)


def test_time_credit_charges_setup_and_repays_estimation_debt():
    budget = cold.TimeBudget(.01)
    budget.charge("setup", .2)
    budget.delivered(25.)
    assert budget.credit == pytest.approx(.05)
    assert not budget.admits(.051)
    assert budget.admits(.049)
    budget.charge("training", .08)
    assert budget.summary()["debt_seconds"] == pytest.approx(.03)
    assert not budget.admits(.001)
    budget.delivered(20.)
    assert budget.admits(.16)
    assert budget.summary()["extra_over_service"] < .01


@pytest.mark.parametrize("fraction", [0, -.01, float("nan"), float("inf"), .2])
def test_invalid_budget(fraction):
    with pytest.raises(ValueError):
        cold.TimeBudget(fraction)


def test_clean_replay_owns_records_and_samples_complete_blocks():
    replay = cold.CleanReplay(2, 12, 4, 731)
    tokens = torch.arange(15)[None]
    replay.append(tokens)
    tokens.zero_()
    replay.append(torch.tensor([[3, 5, 7]]))
    assert len(replay.records) == 1
    assert int(replay.records[0].max()) == 14
    for sequence, anchors in replay.batch(8, 3):
        assert sequence.shape == (1, 12)
        assert (sequence[:, 1:] - sequence[:, :-1] == 1).all()
        assert ((anchors >= 0) & (anchors <= 8)).all()
    replay.append(torch.full((1, 4), 8))
    replay.append(torch.full((1, 6), 9))
    assert len(replay.records) == 2 and int(replay.records[0].max()) == 8


def test_cold_copy_and_serving_isolation_with_candidate_rollback():
    decoder = model()
    with torch.no_grad():
        for layer in decoder.layers:
            layer.attention.draft.q.weight.add_(1.)
    learner = cold.FullBlockLearner(decoder, fit())
    for layer in decoder.layers:
        for name, p in layer.attention.draft.named_parameters():
            ar = dict(layer.attention.ar.named_parameters())[name]
            assert torch.equal(p, ar) and p.data_ptr() != ar.data_ptr()
    original = {name: p.clone() for name, p in decoder.named_parameters()}
    learner.step([(torch.tensor([[2, 3, 5, 7, 4, 6, 8, 4, 5, 8, 2, 3]]), torch.tensor([[1, 6]]))])
    assert all(torch.equal(p, original[name]) for name, p in decoder.named_parameters())
    assert any(not torch.equal(p, original[name]) for name, p in learner.master.items())
    with pytest.raises(RuntimeError, match="probe failure"), learner.candidate():
        assert learner.serving_step == 1
        assert all(torch.equal(p, learner.master[name]) for name, p in learner.execution.items())
        raise RuntimeError("probe failure")
    assert learner.serving_step == 0
    assert all(torch.equal(p, original[name]) for name, p in decoder.named_parameters())
    learner.publish()
    assert learner.serving_step == 1
    learner.check()
    assert all(torch.equal(p, original[name]) for name, p in decoder.named_parameters()
               if ".attention.draft." not in name)


def test_frozen_ar_changes_are_detected():
    learner = cold.FullBlockLearner(model(), fit())
    with torch.no_grad():
        learner.model.layers[0].attention.ar.q.weight.add_(.01)
    with pytest.raises(RuntimeError, match="remain fixed"):
        learner.check()


def test_identical_replay_has_identical_cold_and_offline_updates():
    first = cold.FullBlockLearner(model(), fit())
    second = cold.FullBlockLearner(model(), fit())
    replay = cold.CleanReplay(8, 12, 4, 732)
    transcript = []
    for i in range(4):
        replay.append((torch.arange(16)[None] + i) % 15 + 2)
        batches = replay.batch(2, 2)
        transcript.append(copy.deepcopy(batches))
        first.step(batches)
    for batches in transcript:
        second.step(batches)
    assert all(torch.equal(p, second.master[name]) for name, p in first.master.items())
    assert first.steps == second.steps == 4


def test_service_gate_reserves_budget_and_keeps_cold_drafts_out_of_serving(monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(probe_every=1, probe_tokens=2),
                                    gate_prompts=[prompt], retain_batches=True)
    used = []

    def deliver(prompt, tokens, *, speculative, seed):
        used.append(speculative)
        ticks[0] += 10.
        return Generation(tokens=[4, 5, 6, 7], seconds=10.)

    monkeypatch.setattr(service, "_generate", deliver)
    service.step_estimate = .01
    _, record = service.serve(prompt, 4, seed=1)
    assert service.learner.steps == 1 and service.next_probe == 1
    assert record["mode"] == "ar" and used == [False]
    assert len(service.transcript) == 1
    # A due trial accumulates credit rather than spending it on more updates.
    service.serve(prompt, 4, seed=2)
    assert service.learner.steps == 1 and used == [False, False]
    service.budget.delivered(100000.)
    service.tokens += 40000
    monkeypatch.setattr(service, "_probe", lambda: {"passed": True, "ratio": 2., "step": 1})
    service.serve(prompt, 4, seed=3)
    assert service.speculating and service.learner.serving_step == 1
    service.serve(prompt, 4, seed=4)
    assert used[-1] is True
    assert service.learner.steps == 2
    released_version = service.learner.serving_step
    monkeypatch.setattr(service, "_probe", lambda: {"passed": False, "fallback": True, "ratio": .9, "step": 2})
    service.serve(prompt, 4, seed=5)
    assert not service.speculating and service.learner.serving_step == released_version


def test_grouped_replay_preserves_window_sampling_and_valid_anchors():
    packed, separate = [cold.CleanReplay(8, 12, 4, 952) for _ in range(2)]
    for replay in (packed, separate):
        for length in (4, 7, 11, 16):
            replay.append(torch.arange(length)[None] % 15 + 2)
    groups = packed.batch(4, 3, batch_size=3)
    singles = separate.batch(12, 3)
    for index, (tokens, anchors) in enumerate(groups):
        assert tokens.shape[0] == anchors.shape[0] == 3
        for row in range(3):
            expected, expected_anchors = singles[index * 3 + row]
            length = expected.shape[1]
            assert torch.equal(tokens[row, :length], expected[0])
            assert (tokens[row, length:] == 0).all()
            assert torch.equal(anchors[row], expected_anchors[0])
            assert (anchors[row] + 4 <= length).all()
    assert torch.equal(packed.rng.get_state(), separate.rng.get_state())
    assert torch.equal(packed.anchors_rng.get_state(), separate.anchors_rng.get_state())


@pytest.mark.parametrize("windows,anchors", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_grouped_gradient_variance_identity_by_exhaustive_sampling(windows, anchors):
    # Three windows, two possible anchors and a two-coordinate gradient.
    gradients = torch.tensor([[[1., 2.], [3., -1.]], [[5., 0.], [9., 4.]],
                              [[-2., 3.], [0., 1.]]], dtype=torch.float64)
    mean = gradients.mean((0, 1))
    window_mean = gradients.mean(1)
    between = (window_mean - mean).square().sum(-1).mean()
    within = (gradients - window_mean[:, None]).square().sum(-1).mean()
    estimates = []
    for chosen_windows in product(range(3), repeat=windows):
        for chosen_anchors in product(range(2), repeat=windows * anchors):
            samples = [gradients[window, chosen_anchors[i * anchors + k]]
                       for i, window in enumerate(chosen_windows) for k in range(anchors)]
            estimates.append(torch.stack(samples).mean(0))
    estimates = torch.stack(estimates)
    torch.testing.assert_close(estimates.mean(0), mean)
    measured = (estimates - mean).square().sum(-1).mean()
    torch.testing.assert_close(measured, between / windows + within / (windows * anchors))


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("pad_token", [0, 16])
def test_right_padding_preserves_full_block_loss_and_gradients(backend, pad_token):
    decoder = model().train_draft_only().set_backend(backend)
    tokens = [torch.tensor([[2, 3, 4, 5, 6, 7]]), torch.tensor([[5, 4, 2, 7, 9, 6, 3, 8, 2]])]
    anchors = [torch.tensor([[0, 2]]), torch.tensor([[1, 5]])]
    packed = torch.nn.utils.rnn.pad_sequence([row[0] for row in tokens], batch_first=True, padding_value=pad_token)
    loss = distillation_loss(decoder, packed, torch.cat(anchors))
    loss.backward()
    gradients = {name: p.grad.clone() for name, p in decoder.named_parameters() if p.requires_grad}
    decoder.zero_grad(set_to_none=True)
    reference = sum(distillation_loss(decoder, x, a) for x, a in zip(tokens, anchors, strict=True)) / 2
    reference.backward()
    torch.testing.assert_close(loss, reference, rtol=2e-5, atol=2e-7)
    for name, parameter in decoder.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-4, atol=2e-7)


def test_grouped_windows_train_all_layers_and_resume_with_same_configuration():
    config = replace(fit(), batch_size=3, optimizer_impl="fused")
    first = cold.ColdStartService(model(), config)
    for length in (5, 8, 13):
        first.replay.append(torch.arange(length)[None] + 2)
    first.learner.step(first.replay.batch(1, 2, batch_size=3))
    state = first.state_dict()
    second = cold.ColdStartService(model(), config)
    second.load_state_dict(state)
    for service in (first, second):
        service.learner.step(service.replay.batch(1, 2, batch_size=3))
    assert all(torch.equal(p, second.learner.master[name]) for name, p in first.learner.master.items())
    assert all(not torch.equal(p, first.learner.execution[name]) for name, p in first.learner.master.items())
    with pytest.raises(ValueError, match="policies"):
        cold.ColdStartService(model(), replace(config, optimizer_impl="single")).load_state_dict(state)


@pytest.mark.parametrize("published", [False, True])
def test_resume_preserves_master_moments_replay_rng_and_serving_version(published):
    first = cold.ColdStartService(model(), fit())
    first.replay.append(torch.tensor([[2, 3, 5, 7, 4, 6, 8, 4, 5, 8, 2, 3, 9]]))
    first.replay.append(torch.tensor([[9, 3, 2, 8, 4, 5, 7, 4, 5, 2, 9, 3]]))
    for _ in range(2):
        first.learner.step(first.replay.batch(1, 2))
    if published:
        first.learner.publish()
        first.speculating = True
    first.budget.delivered(100.)
    first.tokens, first.requests = 200, 3
    state = first.state_dict()
    second = cold.ColdStartService(model(), fit())
    second.load_state_dict(state)
    assert second.speculating == published
    assert second.learner.serving_step == (2 if published else 0)
    assert second.budget.spent > first.budget.spent
    assert (second.tokens, second.requests) == (200, 3)
    for _ in range(2):
        a, b = first.replay.batch(1, 2), second.replay.batch(1, 2)
        assert all(torch.equal(x, y) for pair_a, pair_b in zip(a, b, strict=True)
                   for x, y in zip(pair_a, pair_b, strict=True))
        assert first.learner.step(a) == second.learner.step(b)
    for name, p in first.learner.master.items():
        assert torch.equal(p, second.learner.master[name])
        a, b = first.learner.optimizer.state[p], second.learner.optimizer.state[second.learner.master[name]]
        assert all(torch.equal(value, b[key]) for key, value in a.items())
    for name, p in first.learner.execution.items():
        assert torch.equal(p, second.learner.execution[name])


def test_resume_rejects_a_different_frozen_base_and_corrupt_weights():
    service = cold.ColdStartService(model(), fit())
    state = service.state_dict()
    different = model()
    with torch.no_grad():
        different.embedding.weight.add_(.01)
    with pytest.raises(ValueError, match="frozen base"):
        cold.ColdStartService(different, fit()).load_state_dict(state)
    next(iter(state["master"].values())).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite learning"):
        service.load_state_dict(state)


def test_deterministic_update_restores_serving_execution_even_on_exception():
    previous = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="interrupted"), cold.update_determinism(not previous):
        assert torch.are_deterministic_algorithms_enabled() != previous
        raise RuntimeError("interrupted")
    assert torch.are_deterministic_algorithms_enabled() == previous


def test_publication_trials_compare_with_ar_after_speculation_is_enabled(monkeypatch):
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(), gate_prompts=[prompt])
    service.learner.steps, service.learner.serving_step, service.speculating = 2, 1, True

    def deliver(prompt, tokens, *, speculative, seed):
        tps = (40. if service.learner.serving_step == 2 else 30.) if speculative else 60.
        return Generation(tokens=[4] * 8, seconds=8 / tps)

    monkeypatch.setattr(service, "_generate", deliver)
    result = service._probe()
    assert result["ratio"] > 1.1 and result["candidate_over_ar"] < 1.
    assert result["fallback"] and not result["passed"]


def test_measured_probe_cost_replaces_startup_estimate_and_scales_with_methods():
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(probe_tokens=8),
                                    gate_prompts=[torch.tensor([[2, 3]])])
    service.budget.delivered(10.)
    service.tokens = 100
    assert service._probe_cost() == pytest.approx(3.2)
    service.probe_estimate, service.probe_legs = .6, 2
    assert service._probe_cost() == pytest.approx(.6)
    service.speculating = True
    assert service._probe_cost() == pytest.approx(.9)


def test_optional_completed_archive_contains_only_delivered_records(monkeypatch):
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(replay_records=1), retain_records=True)
    monkeypatch.setattr(service, "_generate", lambda *args, **kwargs: Generation(tokens=[4, 5, 6], seconds=.001))
    for i in range(3):
        service.serve(torch.tensor([[2, i + 3]]), 3, seed=i)
    assert len(service.replay.records) == 1 and len(service.completed_records) == 3
    assert [x.tolist() for x in service.completed_records] == [[[2, i + 3, 4, 5, 6]] for i in range(3)]
    assert service.learner.steps == 0


def test_restore_requires_a_fresh_service():
    service = cold.ColdStartService(model(), fit())
    state = service.state_dict()
    service.replay.append(torch.tensor([[2, 3, 4, 5]]))
    with pytest.raises(ValueError, match="fresh service"):
        service.load_state_dict(state)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("optimizer_impl", ["single", "fused"])
def test_same_data_offline_control_matches_full_buffer_update_sequence(batch_size, optimizer_impl):
    from blockspec_ablation.cold_experiment import fit_complete_buffer
    records = [(torch.arange(14)[None] + i) % 15 + 2 for i in range(3)]
    config = replace(fit(), batch_size=batch_size, optimizer_impl=optimizer_impl)
    reference = cold.FullBlockLearner(model(), config)
    replay = cold.CleanReplay(3, 12, 4, fit().seed)
    for record in records:
        replay.append(record)
    for _ in range(3):
        reference.step(replay.batch(1, 2, batch_size=batch_size))
    learner, result = fit_complete_buffer(model(), config, records, 3)
    assert result["completed_records"] == 3 and result["supervised_rows"] == 18 * batch_size
    assert learner.steps == reference.steps == 3
    assert all(torch.equal(p, learner.master[name]) for name, p in reference.master.items())


def test_paired_stream_counts_restoration_and_maintenance_after_mode_switch():
    from blockspec_ablation.cold_experiment import paired_stream
    summary = {"delivered_tokens": 160, "budget": {"service_seconds": 3., "extra_seconds": .05}}
    rows = [{"tokens": 20, "seconds": 1.}, {"tokens": 40, "seconds": 2.}]
    result = paired_stream(summary, rows, prior_tokens=100, prior_generation=1., prior_extra=.02)
    assert result["ar_tps"] == 20.
    assert result["net_tps"] == pytest.approx(60 / 2.03)
    assert result["net_over_ar"] == pytest.approx(3 / 2.03)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cold_entry_accepts_plain_ar_weights_with_precise_rotary_frequencies(tmp_path, dtype):
    from blockspec.parallel.weights import public_key_map
    from blockspec_ablation.cold_experiment import load_cold_model
    original = model()
    config = original.config.to_dict() | {"model_type": "qwen3"}
    for key in ("block_size", "mask_token_id"):
        config.pop(key)
    (tmp_path / "config.json").write_text(json.dumps(config))
    source = dict(original.named_parameters())
    save_file({public: source[own].detach().clone().contiguous()
               for own, public in public_key_map(original.config, include_draft=False).items()},
              tmp_path / "model.safetensors")
    loaded = load_cold_model(tmp_path, ar_base=True, block_size=4, mask_token_id=1, device="cpu", dtype=dtype)
    assert loaded.frequencies.dtype == torch.float32
    for name, parameter in loaded.named_parameters():
        expected = source[name.replace(".attention.draft.", ".attention.ar.")].to(dtype)
        assert torch.equal(parameter, expected) and not parameter.requires_grad
    for layer in loaded.layers:
        assert layer.attention.draft.q.weight.data_ptr() != layer.attention.ar.q.weight.data_ptr()


def test_stream_interval_includes_incremental_maintenance_and_setup_once():
    from blockspec_ablation.cold_experiment import stream_comparison
    rows = [{"delivered_tokens": 100 + i * 10, "service_seconds": 1.,
             "budget": {"extra_seconds": .5 + i * .1}, "ar_reference": {"tokens": 10, "seconds": 2.2}}
            for i in (1, 2, 3)]
    result = stream_comparison(rows, seed=18, prior_tokens=100, prior_extra=.5)
    assert result["ratio"] == pytest.approx(2.)
    assert result["paired_request_ci95"] == pytest.approx([2., 2.])


def test_resume_checks_publication_prompts():
    gate = torch.tensor([[2, 3, 4]])
    service = cold.ColdStartService(model(), fit(), gate_prompts=[gate])
    state = service.state_dict()
    gate.fill_(9)
    assert state["gate_prompts"][0].tolist() == [[2, 3, 4]]
    with pytest.raises(ValueError, match="publication gate"):
        cold.ColdStartService(model(), fit(), gate_prompts=[gate]).load_state_dict(state)
    cold.ColdStartService(model(), fit(), gate_prompts=[state["gate_prompts"][0]]).load_state_dict(state)


def test_legacy_single_window_state_keeps_original_execution_and_reservation():
    service = cold.ColdStartService(model(), fit())
    state = service.state_dict()
    state["fit"].pop("optimizer_impl")
    state["controller"].pop("initial_probe_factor")
    cold.ColdStartService(model(), fit()).load_state_dict(state)


def test_explicit_initial_probe_reservation_keeps_measured_cost_policy():
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(probe_tokens=8, initial_probe_factor=1.5),
                                    gate_prompts=[torch.tensor([[2, 3]])])
    service.budget.delivered(10.)
    service.tokens = 100
    assert service._probe_cost() == pytest.approx(2.4)
    service.probe_estimate, service.probe_legs = .7, 2
    assert service._probe_cost() == pytest.approx(.7)
    service.speculating = True
    assert service._probe_cost() == pytest.approx(1.05)


@pytest.mark.parametrize("factor", [0, .9, float("nan"), float("inf")])
def test_initial_probe_reservation_has_finite_headroom(factor):
    with pytest.raises(ValueError):
        cold.ServiceConfig(initial_probe_factor=factor)


@pytest.mark.parametrize("gate_count", [1, 3])
def test_live_gate_reuses_ar_before_collection_and_charges_capture_once(monkeypatch, gate_count):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    config = cold.ServiceConfig(probe_every=1, probe_tokens=2, reuse_ar_prefix=True)
    service = cold.ColdStartService(model(), fit(), config, gate_prompts=[prompt] * gate_count)
    service.learner.step([(torch.tensor([[2, 3, 4, 5, 6, 7]]), torch.tensor([[0, 2]]))])
    service.budget.delivered(1000.)
    service.tokens = 4000
    service.step_estimate = 100000.
    calls = []

    def deliver(prompt, tokens, *, speculative, seed, prefix_tokens=None):
        calls.append((tokens, speculative, seed, prefix_tokens))
        assert len(service.replay.records) == 0
        if speculative:
            assert service.learner.serving_step == 1
            ticks[0] += 1.
            return Generation(tokens=[4, 5], seconds=1.)
        ticks[0] += 10.01
        return Generation(tokens=[4, 5, 6, 7], seconds=10.01,
                           prefix_tokens=2, prefix_seconds=5., prefix_capture_seconds=.01)

    monkeypatch.setattr(service, "_generate", deliver)
    _, row = service.serve(prompt, 4, seed=92)
    assert calls == [(4, False, 92, 2), (2, True, 92, None)]
    assert row["mode"] == "ar" and row["probe"]["reference_kind"] == "delivered_ar_prefix"
    assert row["probe"]["ratio"] == pytest.approx(5.)
    assert row["service_seconds"] == pytest.approx(10.)
    assert service.budget.costs["validation_capture"] == pytest.approx(.01)
    assert service.budget.costs["validation"] == pytest.approx(1.)
    assert service.budget.spent == pytest.approx(1.01)
    assert service.speculating and service.learner.serving_step == 1
    assert len(service.replay.records) == 1
    assert service._probe_cost() == pytest.approx(3.6 * gate_count)
    restored = cold.ColdStartService(model(), fit(), config, gate_prompts=[prompt] * gate_count)
    restored.load_state_dict(service.state_dict())
    assert restored.probe_legs == 1 and restored.speculating


def test_live_gate_estimate_after_a_multi_prompt_trial():
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(reuse_ar_prefix=True),
                                    gate_prompts=[prompt] * 3)
    service.probe_estimate, service.probe_legs = 18., 3
    assert service._probe_unit_cost() == pytest.approx(2.)
    assert service._probe_cost() == pytest.approx(12.)
    service.speculating = True
    assert service._probe_cost() == pytest.approx(18.)


def test_live_publication_then_learning_preserves_the_validated_version(monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(),
                                    cold.ServiceConfig(probe_every=1, probe_tokens=2, reuse_ar_prefix=True),
                                    gate_prompts=[prompt])
    service.learner.step([(torch.tensor([[2, 3, 4, 5, 6, 7]]), torch.tensor([[0, 2]]))])
    candidate = {name: p.detach().clone() for name, p in service.learner.master.items()}
    service.budget.delivered(1000.)
    calls = []

    def deliver(prompt, tokens, *, speculative, seed, prefix_tokens=None):
        calls.append(speculative)
        assert not service.replay.records
        ticks[0] += 1. if speculative else 10.
        return (Generation(tokens=[4, 5], seconds=1.) if speculative else
                Generation(tokens=[4, 5, 6, 7], seconds=10., prefix_tokens=2, prefix_seconds=5.))

    monkeypatch.setattr(service, "_generate", deliver)
    _, row = service.serve(prompt, 4, seed=93)
    assert calls == [False, True]
    assert row["probe"]["step"] == 1 and row["update"]["step"] == 2
    assert service.learner.steps == service.next_probe == 2
    assert service.learner.serving_step == 1
    assert all(torch.equal(p, candidate[name]) for name, p in service.learner.execution.items())
    assert any(not torch.equal(p, candidate[name]) for name, p in service.learner.master.items())


def test_unfunded_live_gate_reserves_credit_and_uses_no_extra_ar(monkeypatch):
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(probe_tokens=2, reuse_ar_prefix=True),
                                    gate_prompts=[prompt])
    service.learner.steps = service.next_probe = 1
    calls = []

    def deliver(prompt, tokens, *, speculative, seed, prefix_tokens=None):
        calls.append(speculative)
        return Generation(tokens=[4, 5, 6], seconds=.001, prefix_tokens=2, prefix_seconds=.1)

    monkeypatch.setattr(service, "_generate", deliver)
    _, row = service.serve(prompt, 4, seed=92)
    assert calls == [False] and row["probe"] is None and row["update"] is None
    assert service.learner.steps == 1 and service.learner.serving_step == 0


def test_live_gate_failure_restores_serving_and_resumes_learning_schedule(monkeypatch):
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(probe_tokens=2, reuse_ar_prefix=True),
                                    gate_prompts=[prompt])
    service.learner.steps = service.next_probe = 1
    old = {name: p.clone() for name, p in service.learner.execution.items()}
    monkeypatch.setattr(service, "_generate", lambda *args, **kwargs: Generation(tokens=[4, 5], seconds=2.))
    reference = Generation(tokens=[4, 5, 6], prefix_tokens=2, prefix_seconds=1.)
    probe = service._probe_live(prompt, reference, 2, 92)
    service._complete_probe(probe, 2., 1)
    assert not probe["passed"] and not service.speculating
    assert service.learner.serving_step == 0 and service.next_probe == 17
    assert all(torch.equal(p, old[name]) for name, p in service.learner.execution.items())


def test_live_gate_requires_a_later_reference_policy():
    with pytest.raises(ValueError, match="post-publication"):
        cold.ColdStartService(model(), fit(), cold.ServiceConfig(reuse_ar_prefix=True))
    with pytest.raises(ValueError, match="policy"):
        cold.ServiceConfig(reuse_ar_prefix=1)


@pytest.mark.parametrize("legs", [0, True, 1.5, 4])
def test_resume_validates_prefix_gate_path_counts(legs):
    prompt = torch.tensor([[2, 3, 5]])
    config = cold.ServiceConfig(reuse_ar_prefix=True)
    service = cold.ColdStartService(model(), fit(), config, gate_prompts=[prompt])
    state = service.state_dict()
    state["probe_legs"] = legs
    with pytest.raises(ValueError, match="counters"):
        cold.ColdStartService(model(), fit(), config, gate_prompts=[prompt]).load_state_dict(state)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_live_probe_count_is_a_positive_integer(count):
    with pytest.raises(ValueError, match="probe count"):
        cold.ServiceConfig(reuse_ar_prefix=True, live_probe_requests=count)


def test_multiple_live_probes_require_prefix_reuse():
    with pytest.raises(ValueError, match="AR-prefix"):
        cold.ServiceConfig(live_probe_requests=2)


def test_prefix_pool_uses_total_tokens_and_time_with_a_fixed_version():
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(),
                                    cold.ServiceConfig(reuse_ar_prefix=True, live_probe_requests=2, publish_margin=1.05),
                                    gate_prompts=[prompt])
    service.learner.steps = 1
    first = dict(step=1, reference_tokens=16, reference_seconds=10., candidate_tokens=16, candidate_seconds=8.)
    row = service._pool_live_probe(first)
    assert row["ratio"] == pytest.approx(1.25) and not row["complete"] and not row["passed"]
    with pytest.raises(RuntimeError, match="fixed learning version"):
        service._pool_live_probe(first | {"step": 2})
    row = service._pool_live_probe(dict(step=1, reference_tokens=4, reference_seconds=2.,
                                        candidate_tokens=2, candidate_seconds=2.))
    assert row["complete"] and row["passed"] and not service.pending_live_probes
    assert row["ratio"] == pytest.approx(1.08)
    assert row["paired_ratios"] == pytest.approx([1.25, .5])
    assert row["probe_requests"] == 2 and row["reference_tokens"] == 20 and row["candidate_tokens"] == 18


def pooled_service(monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    config = cold.ServiceConfig(probe_every=1, probe_tokens=2, reuse_ar_prefix=True,
                                live_probe_requests=3, publish_margin=1.0504)
    service = cold.ColdStartService(model(), fit(), config, gate_prompts=[prompt])
    service.learner.step([(torch.tensor([[2, 3, 4, 5, 6, 7]]), torch.tensor([[0, 2]]))])
    service.budget.delivered(1000.)
    service.tokens = 4000
    service.step_estimate = 100000.

    def connect(target):
        calls = []

        def deliver(prompt, tokens, *, speculative, seed, prefix_tokens=None):
            calls.append(speculative)
            expected_records = target.requests - int(speculative)
            assert len(target.replay.records) == expected_records
            if speculative:
                duration = (2.2, 1.7, 1.6)[target.requests - 1]
                ticks[0] += duration
                return Generation(tokens=[4, 5], seconds=duration)
            ticks[0] += 10.
            return Generation(tokens=[4, 5, 6, 7], seconds=10., prefix_tokens=2, prefix_seconds=2.)

        monkeypatch.setattr(target, "_generate", deliver)
        return calls

    return service, prompt, connect


def test_pooled_trials_hold_learning_until_the_fixed_count_and_resume_exactly(monkeypatch):
    service, prompt, connect = pooled_service(monkeypatch)
    calls = connect(service)
    _, first = service.serve(prompt, 4, seed=92)
    assert first["update"] is None and not first["probe"]["complete"]
    assert service.learner.steps == service.next_probe == 1 and service.learner.serving_step == 0
    state = service.state_dict()
    restored = cold.ColdStartService(model(), fit(), service.config, gate_prompts=[prompt])
    restored.load_state_dict(state)
    restored_calls = connect(restored)
    assert restored.pending_live_probes == service.pending_live_probes
    state["pending_live_probes"][0]["candidate_seconds"] = 77.
    assert restored.pending_live_probes[0]["candidate_seconds"] == pytest.approx(2.2)
    for number in (2, 3):
        for target in (service, restored):
            _, row = target.serve(prompt, 4, seed=91 + number)
            assert row["update"] is None and row["probe"]["probe_requests"] == number
            assert row["probe"]["complete"] == (number == 3)
            assert target.learner.steps == 1
            assert target.speculating == (number == 3)
    assert calls == [False, True] * 3 and restored_calls == [False, True] * 2
    assert service.last_probe["ratio"] == pytest.approx(6 / 5.5)
    assert service.next_probe == restored.next_probe == 2
    assert service.learner.serving_step == restored.learner.serving_step == 1
    assert service.pending_live_probes == restored.pending_live_probes == []
    assert service.budget.costs["validation"] == pytest.approx(5.5)
    assert restored.budget.costs["validation"] == pytest.approx(5.5)
    for name, parameter in service.learner.execution.items():
        assert torch.equal(parameter, restored.learner.execution[name])


def test_unfunded_pending_pool_keeps_learning_and_evidence_fixed(monkeypatch):
    service, prompt, connect = pooled_service(monkeypatch)
    calls = connect(service)
    service.serve(prompt, 4, seed=92)
    evidence = copy.deepcopy(service.pending_live_probes)
    service.budget.service_seconds = service.budget.spent / service.budget.fraction
    _, row = service.serve(prompt, 4, seed=93)
    assert calls == [False, True, False]
    assert row["probe"] is None and row["update"] is None
    assert service.pending_live_probes == evidence
    assert service.learner.steps == service.next_probe == 1 and not service.speculating


def test_pooled_admission_uses_only_pre_request_credit(monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(),
                                    cold.ServiceConfig(reuse_ar_prefix=True, live_probe_requests=3, probe_tokens=2),
                                    gate_prompts=[prompt])
    service.learner.steps = service.next_probe = 1
    service.tokens = 4000
    service.budget.service_seconds, service.budget.costs = 1000., {"setup": 10.}
    calls = []

    def deliver(prompt, tokens, *, speculative, seed, prefix_tokens=None):
        calls.append((speculative, prefix_tokens))
        ticks[0] += 1. if speculative else 200.
        return (Generation(tokens=[4, 5], seconds=1.) if speculative else
                Generation(tokens=[4, 5, 6, 7], seconds=200., prefix_tokens=2, prefix_seconds=.2))

    monkeypatch.setattr(service, "_generate", deliver)
    _, first = service.serve(prompt, 4, seed=92)
    assert calls == [(False, None)] and first["probe"] is None
    assert service.budget.credit == pytest.approx(2.)
    _, second = service.serve(prompt, 4, seed=93)
    assert calls == [(False, None), (False, 2), (True, None)]
    assert second["probe"]["probe_requests"] == 1 and not second["probe"]["complete"]


@pytest.mark.parametrize("field,value", [("step", 2), ("candidate_seconds", float("nan")),
                                        ("reference_seconds", 0), ("candidate_tokens", 1.5),
                                        ("reference_tokens", 3), ("extra", 1)])
def test_restore_checks_pending_prefix_evidence(monkeypatch, field, value):
    service, prompt, connect = pooled_service(monkeypatch)
    connect(service)
    service.serve(prompt, 4, seed=92)
    state = service.state_dict()
    state["pending_live_probes"][0][field] = value
    with pytest.raises(ValueError, match="pending prefix"):
        cold.ColdStartService(model(), fit(), service.config, gate_prompts=[prompt]).load_state_dict(state)


def test_legacy_state_with_an_empty_prefix_pool_restores():
    service = cold.ColdStartService(model(), fit())
    state = service.state_dict()
    del state["pending_live_probes"], state["controller"]["live_probe_requests"]
    restored = cold.ColdStartService(model(), fit())
    restored.load_state_dict(state)
    assert restored.pending_live_probes == []
