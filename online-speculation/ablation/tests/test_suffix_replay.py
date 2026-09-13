"""Cached features must equal the full gradient for the SAME trainable subset."""

import copy

import pytest
import torch

from blockspec_ablation.checkpoint import adapter_state, base_fingerprint
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.distillation import divergence
from blockspec_ablation.model import Decoder, ModelConfig, is_adapter
from blockspec_ablation.online import Feedback, OnlineConfig, OnlineLearner
from blockspec_ablation.tree import generate_tree


def example():
    torch.manual_seed(97)
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                num_hidden_layers=3, adapter_rank=2)).double().train_adapters_only()
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.1)
    inputs = torch.tensor([[3, 6, 7, 2]])
    mask = torch.tensor([[False, True, True, True]])
    with torch.no_grad():
        _, cache = model(torch.tensor([[0, 4, 1]]), return_cache=True)
    return model, inputs, mask, cache


@pytest.mark.parametrize("last_layers", [1, 2, 3])
def test_suffix_logits_and_every_trainable_gradient_equal_full_recomputation(last_layers):
    model, inputs, mask, cache = example()
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=last_layers))
    with torch.no_grad():
        logits, _, boundary = model(inputs, cache=cache, adapter_mask=mask, return_cache=True,
                                     capture_layer=learner.capture_layer)
    teacher = torch.randn(3, model.config.vocab_size, dtype=torch.float64)
    full = model(inputs, cache=cache, adapter_mask=mask)
    full_loss = divergence(full[0, 1:], teacher, "forward_kl").mean()
    full_grads = torch.autograd.grad(full_loss, learner.parameters)
    suffix = model.forward_suffix(boundary.detached(), cache=cache[learner.capture_layer:])
    suffix_loss = divergence(suffix[0, 1:], teacher, "forward_kl").mean()
    suffix_grads = torch.autograd.grad(suffix_loss, learner.parameters)
    torch.testing.assert_close(logits, suffix, atol=0, rtol=0)
    for original, cached in zip(full_grads, suffix_grads):
        torch.testing.assert_close(original, cached, atol=0, rtol=0)


def test_reused_boundary_matches_full_projection_across_multiple_suffix_updates():
    model, inputs, mask, cache = example()
    oracle = copy.deepcopy(model)
    learner = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=1, train_last_layers=1,
                                               learning_rate=.001, loss="forward_kl"))
    for name, p in oracle.named_parameters():
        p.requires_grad_(is_adapter(name) and name.startswith("model.layers.2."))
    parameters = [p for p in oracle.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0)
    teacher = torch.randn(3, model.config.vocab_size, dtype=torch.float64)
    with torch.no_grad():
        _, _, boundary = model(inputs, cache=cache, adapter_mask=mask, return_cache=True, capture_layer=2)
    learner.observe(Feedback(inputs, cache, teacher, 3, boundary), may_update=False)
    # Poison the caller's copy: replay must own its small feature/target buffers.
    with torch.no_grad():
        boundary.hidden.add_(100)
    frozen = base_fingerprint(model)
    initial = adapter_state(model)
    for _ in range(3):
        learner.update()
        optimizer.zero_grad(set_to_none=True)
        logits = oracle(inputs, cache=cache, adapter_mask=mask)
        loss = divergence(logits[0, 1:], teacher, "forward_kl").mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        for n, p in model.named_parameters():
            # Selecting supervised rows changes the GEMM shape; compare FP64
            # updates at rounding precision and keep frozen weights bitwise exact.
            tolerance = 1e-14 if p.requires_grad else 0
            torch.testing.assert_close(p, dict(oracle.named_parameters())[n], atol=tolerance, rtol=tolerance)
    assert len(learner.replay[0].cache) == 1
    assert learner.replay[0].boundary.hidden.grad_fn is None
    assert base_fingerprint(model) == frozen
    current = adapter_state(model)
    assert any(not torch.equal(v, current[n]) for n, v in initial.items() if n.startswith("model.layers.2."))
    assert all(torch.equal(v, current[n]) for n, v in initial.items() if not n.startswith("model.layers.2."))


@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
def test_suffix_online_keeps_greedy_output_and_updates_existing_adapter(generate):
    model, _, _, _ = example()
    prompt = torch.tensor([[0, 1, 2]])
    expected = generate_ar(model, prompt, 24)
    initial = adapter_state(model)
    learner = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=1, train_last_layers=1))
    output = generate(model, prompt, 24, block_size=4, learner=learner)
    assert expected.tokens == output.tokens and output.updates > 0
    assert not learner.replay
    current = adapter_state(model)
    assert all(torch.equal(v, current[n]) for n, v in initial.items() if not n.startswith("model.layers.2."))
    assert any(not torch.equal(v, current[n]) for n, v in initial.items() if n.startswith("model.layers.2."))


def test_suffix_refuses_missing_boundary_or_mutated_frozen_prefix():
    model, inputs, mask, cache = example()
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1, stride=1))
    teacher = torch.randn(1, model.config.vocab_size, dtype=torch.float64)
    with pytest.raises(ValueError, match="matching draft boundary"):
        learner.observe(Feedback(inputs, cache, teacher, 1))
    with torch.no_grad():
        _, _, boundary = model(inputs, cache=cache, adapter_mask=mask, return_cache=True, capture_layer=2)
    learner.observe(Feedback(inputs, cache, teacher, 1, boundary), may_update=False)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.lora_B.add_(.1)
    with pytest.raises(RuntimeError, match="frozen draft prefix changed"):
        learner.update()


def test_suffix_validates_shapes_and_configuration():
    model, inputs, mask, cache = example()
    with pytest.raises(ValueError, match="suffix exceeds"):
        OnlineLearner(model, OnlineConfig(train_last_layers=4))
    with pytest.raises(ValueError):
        OnlineConfig(train_last_layers=0)
    with pytest.raises(ValueError):
        model(inputs, capture_layer=1)
    with pytest.raises(ValueError):
        model(inputs, return_cache=True, capture_layer=.5)
    _, _, boundary = model(inputs, cache=cache, adapter_mask=mask, return_cache=True, capture_layer=2)
    with pytest.raises(ValueError, match="suffix layer count"):
        model.forward_suffix(boundary, cache=cache)
    with pytest.raises(ValueError, match="replay dimensions"):
        model.forward_suffix(boundary)
