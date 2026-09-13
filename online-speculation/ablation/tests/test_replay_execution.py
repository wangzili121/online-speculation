import copy
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from blockspec_ablation.benchmark import BenchmarkConfig, benchmark_streams
from blockspec_ablation.checkpoint import adapter_state, base_fingerprint
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.online import Feedback, OnlineConfig, OnlineLearner
from blockspec_ablation.replay_execution import SuffixReplayExecutor


def example(device="cpu"):
    torch.manual_seed(1729)
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(
        device=device, dtype=torch.float32 if device == "cuda" else torch.float64)
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.1)
    return model


def item(learner, prefix, length, valid, *, covered=False):
    model = learner.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = torch.randint(0, model.config.vocab_size, (1, length), device=device)
    mask = torch.ones_like(inputs, dtype=torch.bool)
    mask[:, 0] = False
    with torch.no_grad():
        cache = model(torch.zeros(1, prefix, device=device, dtype=torch.long), return_cache=True)[1] if prefix else None
        _, _, boundary = model(inputs, cache=cache, adapter_mask=mask, return_cache=True,
                                capture_layer=learner.capture_layer)
    teacher = torch.randn(valid, model.config.vocab_size, device=device, dtype=dtype)
    return Feedback(inputs, cache, teacher, valid, boundary, covered)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA replay hardware required"))])
@pytest.mark.parametrize("loss", ["l1", "forward_kl", "reverse_kl_l1"])
@pytest.mark.parametrize("last_layers", [1, 3])
def test_prepared_gradients_and_adam_match_eager_across_shapes_and_repeated_signatures(device, loss, last_layers):
    model = example(device)
    config = OnlineConfig(stride=100, replay_blocks=3, train_last_layers=last_layers, loss=loss,
                          learning_rate=.0003, update_policy="coverage")
    learner = OnlineLearner(model, config)
    reference = OnlineLearner(copy.deepcopy(model), config)
    frozen = base_fingerprint(model)
    before = adapter_state(model)
    engine = SuffixReplayExecutor(model, start_layer=learner.capture_layer, loss=loss,
                                  capacity=8, max_query=4, use_cuda_graph=device == "cuda")
    engine.prepare([(length, valid) for length in range(2, 5) for valid in range(1, length)])
    learner.replay_executor = engine
    torch.testing.assert_close(adapter_state(model), before, atol=0, rtol=0)
    assert learner.optimizer.state_dict()["state"] == {}
    assert all(p.grad is None for p in learner.parameters)
    for update in range(4):
        learner.clear_replay()
        reference.clear_replay()
        # Reuse the (4, 3) graph twice, separated by a different signature.
        for prefix, length, valid in [(8, 4, 3), (0, 2 + update % 2, 1), (2, 4, 3)]:
            feedback = item(reference, prefix, length, valid)
            learner.observe(feedback, may_update=False)
            reference.observe(feedback, may_update=False)
        actual = learner.update()
        expected = reference.update()
        tolerance = 2e-6 if device == "cuda" else 1e-12
        assert actual["positions"] == expected["positions"] == 7
        assert actual["loss"] == pytest.approx(expected["loss"], abs=tolerance, rel=tolerance)
        torch.testing.assert_close([p.grad for p in learner.parameters], [p.grad for p in reference.parameters],
                                   atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(adapter_state(model), adapter_state(reference.model), atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(learner.optimizer.state_dict(), reference.optimizer.state_dict(),
                                   atol=tolerance, rtol=tolerance)
    assert base_fingerprint(model) == base_fingerprint(reference.model) == frozen
    # A fully covered window preserves even the existing clipped gradient buffers.
    state, grads = copy.deepcopy(learner.optimizer.state_dict()), [p.grad.clone() for p in learner.parameters]
    learner.clear_replay()
    learner.observe(item(learner, 2, 4, 3, covered=True), may_update=False)
    assert learner.update() is None and learner.coverage_skips == 1
    torch.testing.assert_close(learner.optimizer.state_dict(), state, atol=0, rtol=0)
    torch.testing.assert_close([p.grad for p in learner.parameters], grads, atol=0, rtol=0)


def test_replay_contracts_and_failed_validation_preserve_existing_parameters():
    model = example()
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1))
    with pytest.raises(ValueError, match="FP32 CUDA"):
        SuffixReplayExecutor(model, start_layer=2, loss="l1", capacity=8, max_query=4)
    engine = SuffixReplayExecutor(model, start_layer=2, loss="l1", capacity=8, max_query=4, use_cuda_graph=False)
    with pytest.raises(ValueError, match="signature"):
        engine.prepare([(4, 4)])
    engine.prepare([(4, 3)])
    with pytest.raises(ValueError, match="share model"):
        OnlineLearner(model, replace(learner.config, loss="forward_kl"), replay_executor=engine)
    learner = OnlineLearner(model, learner.config, replay_executor=engine)
    feedback = item(learner, 2, 4, 3).detached(cache_start=2)
    with pytest.raises(ValueError, match="unprepared"):
        engine.backward([replace(feedback, valid=1, teacher_logits=feedback.teacher_logits[:1])])
    with pytest.raises(ValueError, match="boundary"):
        engine.backward([replace(feedback, boundary=None)])
    with pytest.raises(ValueError, match="layout"):
        engine.backward([replace(feedback, boundary=replace(feedback.boundary, adapter_mask=None))])
    with pytest.raises(ValueError, match="device and dtype"):
        engine.backward([replace(feedback, teacher_logits=feedback.teacher_logits.float())])
    with pytest.raises(ValueError, match="capacity"):
        engine.backward([item(learner, 9, 4, 3).detached(cache_start=2)])
    model.adapter_parameters()[-1].data = model.adapter_parameters()[-1].data.clone()
    with pytest.raises(RuntimeError, match="storage"):
        engine.backward([feedback])


def test_online_graph_configuration_and_finite_loss_guards():
    with pytest.raises(ValueError, match="online execution"):
        BenchmarkConfig(online_execution="unknown")
    model = example()
    with pytest.raises(ValueError, match="explicit trainable suffix"):
        benchmark_streams(model, [torch.tensor([[0, 1]])], BenchmarkConfig(online_execution="cuda_graph"))
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1, stride=1))
    engine = SuffixReplayExecutor(model, start_layer=2, loss="l1", capacity=8, max_query=4, use_cuda_graph=False)
    engine.prepare([(4, 3)])
    learner.replay_executor = engine
    feedback = item(learner, 2, 4, 3)
    before = adapter_state(model)
    with pytest.raises(FloatingPointError, match="nonfinite online loss"):
        learner.observe(replace(feedback, teacher_logits=torch.full_like(feedback.teacher_logits, float("nan"))))
    torch.testing.assert_close(adapter_state(model), before, atol=0, rtol=0)
    assert learner.updates == learner.version == 0 and learner.optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA replay hardware required"))])
def test_preparing_additional_shapes_preserves_existing_gradients_and_optimizer(device):
    model = example(device)
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1, stride=1))
    learner.observe(item(learner, 2, 4, 3))
    initial = adapter_state(model)
    gradients = [p.grad.clone() for p in learner.parameters]
    state = copy.deepcopy(learner.optimizer.state_dict())
    engine = SuffixReplayExecutor(model, start_layer=2, loss="l1", capacity=8, max_query=4,
                                  use_cuda_graph=device == "cuda")
    engine.prepare([(4, 3)])
    engine.prepare([(2, 1), (4, 3)])
    assert len(engine.slots) == 2
    torch.testing.assert_close(adapter_state(model), initial, atol=0, rtol=0)
    torch.testing.assert_close([p.grad for p in learner.parameters], gradients, atol=0, rtol=0)
    torch.testing.assert_close(learner.optimizer.state_dict(), state, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA replay hardware required"))])
def test_replay_owner_and_workspaces_release_when_last_owner_is_dropped(device):
    model = example(device)
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1))
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        engine = SuffixReplayExecutor(model, start_layer=2, loss="l1", capacity=8, max_query=4,
                                      use_cuda_graph=device == "cuda")
        engine.prepare([(2, 1), (4, 3)])
        owner_ref = weakref.ref(engine)
        slot_refs = [weakref.ref(slot) for slot in engine.slots.values()]
        buffer_refs = [weakref.ref(slot.hidden) for slot in engine.slots.values()]
        if device == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
        del engine
        assert owner_ref() is None
        assert all(ref() is None for ref in slot_refs + buffer_refs)
        if device == "cuda":
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() < allocated
        assert learner.model is model and learner.optimizer.state_dict()["state"] == {}
    finally:
        if collection_was_enabled:
            gc.enable()
        gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA replay hardware required")
@pytest.mark.parametrize("sampler", ["linear", "tree"])
@pytest.mark.parametrize("execution", ["eager", "cuda_graph"])
def test_replay_graph_benchmark_charges_setup_and_preserves_generation(sampler, execution):
    model = example("cuda")
    result = benchmark_streams(model, [torch.tensor([[0, 1, 2]]), torch.tensor([[0, 2, 1]])],
                               BenchmarkConfig(tokens=18, block_size=3, warmup_tokens=8,
                                               sampler=sampler, top_k=3, prefix_budget=5,
                                               execution=execution, online_execution="cuda_graph"),
                               OnlineConfig(stride=2, replay_blocks=2, train_last_layers=1,
                                            loss="forward_kl", update_policy="coverage"))
    assert result["greedy_identical"] and result["base_unchanged"] and result["adapter_restored"]
    assert all(result["online_adapter_changed_per_stream"])
    assert result["online_execution"]["signatures"] == 3
    cost = result["online_execution"]["setup_seconds"]
    assert cost > 0 and result["arms"]["online"]["engine_setup_seconds"] == pytest.approx(
        result["execution"].get("setup_seconds_by_arm", {"online": 0})["online"] + cost)
    for repeat in result["trajectories"]:
        assert repeat["arms"]["online"][-1]["cumulative"] == result["repeats"][repeat["repeat"]]["online"]
