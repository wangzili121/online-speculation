"""Shared-KV row grouping: values, derivatives, cache isolation and online state."""

import copy

import pytest
import torch
from torch.nn import functional as F

from blockspec_ablation.attention import grouped_attention
from blockspec_ablation.benchmark import BenchmarkConfig, benchmark_streams
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.model import Decoder, GatedLinear, ModelConfig
from blockspec_ablation.online import OnlineConfig, OnlineLearner
from blockspec_ablation.replay_execution import SuffixReplayExecutor
from blockspec_ablation.tree import generate_tree


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA attention hardware required"))]


def reference(q, k, v, allowed):
    groups = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, k.repeat_interleave(groups, 1),
                                          v.repeat_interleave(groups, 1), attn_mask=allowed)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("mask_kind", ["causal", "tree", "per_head"])
def test_grouped_values_and_qkv_gradients(device, groups, mask_kind):
    torch.manual_seed(730)
    dtype = torch.float64 if device == "cpu" else torch.float32
    q = torch.randn(2, 2 * groups, 5, 8, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(2, 2, 11, 8, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(2, 2, 11, 8, dtype=dtype, device=device, requires_grad=True)
    mask = (torch.arange(11, device=device)[None] <= 6 + torch.arange(5, device=device)[:, None])[None, None]
    if mask_kind == "tree":
        mask = mask.expand(2, 1, -1, -1).clone()
        mask[..., 2, 7] = False
        mask[..., 4, 7:10] = False
    if mask_kind == "per_head":
        mask = torch.rand(2, 2 * groups, 5, 11, device=device) > .4
        mask[..., 0, :] = False  # empty attention rows have finite zero gradients
    expected = reference(q, k, v, mask)
    actual = grouped_attention(q, k, v, mask)
    upstream = torch.randn_like(expected)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), (q, k, v), retain_graph=True)
    reference_grads = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
    tol = 2e-12 if device == "cpu" else 3e-6
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(actual_grads, reference_grads, atol=tol, rtol=tol)
    assert all(torch.isfinite(g).all() for g in actual_grads)


def test_grouped_attention_passes_finite_difference_gradcheck():
    torch.manual_seed(801)
    values = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in
              [(1, 4, 2, 2), (1, 2, 3, 2), (1, 2, 3, 2)]]
    mask = torch.tensor([[[[True, False, True], [False, False, False]]]])
    assert torch.autograd.gradcheck(lambda *xs: grouped_attention(*xs, mask), values,
                                    eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1., 3.])
def test_boolean_lowrank_residual_values_and_parameter_gradients_match(device, dtype, scale):
    torch.manual_seed(592)
    layer = GatedLinear(5, 7, 2, 2 * scale).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(2, 4, 5, device=device, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[False, True, False, True], [True, True, False, True]], device=device)
    base = F.linear(x, layer.weight)
    delta = F.linear(F.linear(x, layer.lora_A), layer.lora_B)
    expected = base + mask[..., None].to(dtype) * (delta * layer.scale)
    actual = layer(x, mask)
    upstream = torch.randn_like(actual)
    parameters = (x, layer.weight, layer.lora_A, layer.lora_B)
    reference_grads = torch.autograd.grad((expected * upstream).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), parameters)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_grads, reference_grads, atol=0, rtol=0)


def test_fractional_mask_keeps_the_general_lowrank_formula():
    torch.manual_seed(173)
    layer = GatedLinear(3, 4, 2, 2)
    layer.lora_B.data.normal_()
    x = torch.randn(1, 3, 3)
    mask = torch.tensor([[0., .3, 1.]])
    expected = F.linear(x, layer.weight) + mask[..., None] * F.linear(F.linear(x, layer.lora_A), layer.lora_B)
    torch.testing.assert_close(layer(x, mask), expected, atol=0, rtol=0)


def example(device="cpu"):
    torch.manual_seed(135)
    model = Decoder(ModelConfig(vocab_size=13, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device)
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.1)
    return model.train_adapters_only()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
def test_grouped_inference_and_gradient_graphs_continue_the_same_adapter(device, generate):
    model = example(device).set_attention_backend("grouped")
    oracle = copy.deepcopy(model).set_attention_backend("sdpa")
    engine = FixedShapeExecutor(model, capacity=48, max_query=5, use_cuda_graph=device == "cuda")
    engine.prepare([(i, False, None) for i in range(1, 6)] +
                   [(i, True, c) for i in range(2, 5) for c in (None, 2)])
    config = OnlineConfig(stride=2, replay_blocks=2, train_last_layers=1, learning_rate=.001,
                          loss="forward_kl", optimizer="standard")
    learner, expected_learner = OnlineLearner(model, config), OnlineLearner(oracle, config)
    replay = SuffixReplayExecutor(model, start_layer=2, loss=config.loss, capacity=48, max_query=4,
                                  use_cuda_graph=device == "cuda")
    replay.prepare([(b, m) for b in range(2, 5) for m in range(1, b)])
    learner.replay_executor = replay
    options = dict(block_size=4)
    if generate is generate_tree:
        options.update(top_k=2, prefix_budget=5)
    for ids in ([1, 3, 5], [4, 2, 6]):
        prompt = torch.tensor([ids], device=device)
        actual = generate(model, prompt, 20, learner=learner, executor=engine,
                          generator=torch.Generator(device=device).manual_seed(27), **options)
        expected = generate(oracle, prompt, 20, learner=expected_learner,
                            generator=torch.Generator(device=device).manual_seed(27), **options)
        assert actual.tokens == expected.tokens == generate_ar(model, prompt, 20, executor=engine).tokens
        assert actual.accepted_per_round == expected.accepted_per_round
        assert actual.updates == expected.updates > 0
        torch.testing.assert_close(list(model.parameters()), list(oracle.parameters()), atol=3e-6, rtol=3e-4)
        torch.testing.assert_close(learner.optimizer.state_dict(), expected_learner.optimizer.state_dict(),
                                   atol=3e-6, rtol=3e-4)
        assert not learner.replay


def test_attention_changes_invalidate_graphs_and_cached_features():
    model = example()
    engine = FixedShapeExecutor(model, capacity=8, max_query=4, use_cuda_graph=False)
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1))
    replay = SuffixReplayExecutor(model, start_layer=2, loss="l1", capacity=8, max_query=4,
                                  use_cuda_graph=False)
    model.set_attention_backend("grouped")
    with pytest.raises(RuntimeError, match="attention execution"):
        engine.validate(model)
    with pytest.raises(RuntimeError, match="attention execution"):
        replay.validate(model, 2, "l1")
    with pytest.raises(RuntimeError, match="attention execution"):
        learner._check_prefix()
    with pytest.raises(ValueError, match="attention backend"):
        model.set_attention_backend("unknown")
    with pytest.raises(ValueError, match="attention backend"):
        BenchmarkConfig(attention_backend="unknown")


@pytest.mark.parametrize("dtype,length", [(torch.float32, 33), (torch.bfloat16, 4)])
@torch.no_grad()
def test_long_or_low_precision_queries_keep_sdpa_execution(monkeypatch, dtype, length):
    model = example().to(dtype=dtype)
    ids = (torch.arange(length) % model.config.vocab_size)[None]
    expected = model(ids)
    model.set_attention_backend("grouped")
    def fail(*args):
        raise AssertionError("short FP32/FP64 path selected outside its scope")
    monkeypatch.setattr("blockspec_ablation.model.grouped_attention", fail)
    torch.testing.assert_close(model(ids), expected, atol=0, rtol=0)


def test_benchmark_selects_one_backend_and_restores_callers_execution_on_exception():
    model = example()
    model.model.layers[0].self_attn.backend = "grouped"
    before = model.attention_signature()
    def stop(_):
        assert model.attention_signature() == ("grouped",) * 3
        raise RuntimeError("test stop")
    with pytest.raises(RuntimeError, match="test stop"):
        benchmark_streams(model, [torch.tensor([[1, 2, 3]])],
                          BenchmarkConfig(tokens=4, warmup_tokens=4, attention_backend="grouped"), progress=stop)
    assert model.attention_signature() == before
