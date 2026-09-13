import math

import pytest
import torch

from blockspec_ablation.distillation import divergence, paired_batch, paired_loss
from blockspec_ablation.model import Decoder, GatedLinear, ModelConfig, cache_length, trim_cache


def tiny_model(backend="sdpa"):
    torch.manual_seed(29)
    config = ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         num_hidden_layers=2, adapter_rank=2, adapter_alpha=2)
    return Decoder(config).double().set_attention_backend(backend)


def randomize_adapter(model):
    with torch.no_grad():
        for parameter in model.adapter_parameters():
            parameter.normal_(std=.1)


def test_sdpa_matches_explicit_attention_equations(monkeypatch):
    model = tiny_model()
    x = torch.tensor([[0, 1, 3, 2, 4]])
    expected = model(x)
    def attention(q, k, v, attn_mask=None, dropout_p=0.):
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(~attn_mask, -torch.inf).softmax(-1)
        return weights @ v
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", attention)
    torch.testing.assert_close(model(x), expected, atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize("backend", ["sdpa", "grouped"])
def test_kv_cache_matches_full_causal_forward(split, backend):
    model = tiny_model(backend)
    x = torch.tensor([[0, 1, 3, 2, 4]])
    full = model(x)
    _, cache = model(x[:, :split], return_cache=True)
    tail, full_cache = model(x[:, split:], cache=cache, return_cache=True)
    torch.testing.assert_close(tail, full[:, split:], atol=2e-15, rtol=2e-14)
    assert cache_length(full_cache) == x.shape[1]
    assert cache_length(trim_cache(full_cache, split)) == split
    assert trim_cache(full_cache, 0) is None
    with pytest.raises(ValueError):
        trim_cache(full_cache, 100)


@pytest.mark.parametrize("backend", ["sdpa", "grouped"])
def test_paired_teacher_has_no_noise_or_adapter_leakage(backend):
    model = tiny_model(backend)
    randomize_adapter(model)
    clean = torch.tensor([[0, 1, 3, 2, 4, 5]])
    batch = paired_batch(clean, 3, noisy=torch.full_like(clean, 6))
    logits = model(batch.tokens, positions=batch.positions, allowed=batch.allowed,
                   adapter_mask=batch.adapter_mask)
    torch.testing.assert_close(logits[:, :6], model(clean), atol=2e-15, rtol=2e-14)
    # Noisy BOS has adapter OFF and the same position/memory as clean BOS.
    torch.testing.assert_close(logits[:, 6], logits[:, 0], atol=2e-15, rtol=2e-14)
    assert not batch.eligible[0, 0]


def test_block_attention_exact_truth_table():
    clean = torch.tensor([[0, 1, 2, 3, 4, 5]])
    b = paired_batch(clean, 3, noisy=clean)
    mask = b.allowed[0, 0]
    for j in range(6):
        for k in range(12):
            assert bool(mask[j, k]) == (k <= j)
            expected = k < (j // 3) * 3 if k < 6 else (j // 3 == (k - 6) // 3 and k - 6 <= j)
            assert bool(mask[6 + j, k]) == expected


@pytest.mark.parametrize("backend", ["sdpa", "grouped"])
def test_paired_student_matches_individual_cached_blocks(backend):
    model = tiny_model(backend)
    randomize_adapter(model)
    clean = torch.tensor([[0, 1, 2, 3, 4, 5]])
    noise = torch.tensor([[0, 6, 5, 4, 3, 2]])
    b = paired_batch(clean, 3, noisy=noise)
    paired = model(b.tokens, positions=b.positions, allowed=b.allowed, adapter_mask=b.adapter_mask)
    for start in (0, 3):
        cache = None
        if start:
            _, cache = model(clean[:, :start], return_cache=True)
        mask = torch.ones_like(noise[:, start:start + 3], dtype=torch.bool)
        if start == 0:
            mask[:, 0] = False
        independent = model(noise[:, start:start + 3], cache=cache, adapter_mask=mask)
        torch.testing.assert_close(paired[:, 6 + start:9 + start], independent, atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("backend", ["sdpa", "grouped"])
def test_clean_seed_kv_survives_arbitrary_adapter_updates(backend):
    model = tiny_model(backend)
    prefix = torch.tensor([[0, 1, 2]])
    _, cache = model(prefix, return_cache=True)
    inputs = torch.tensor([[3, 6, 5, 4]])
    mask = torch.tensor([[False, True, True, True]])
    base_logits, base_cache = model(inputs[:, :1], cache=cache, return_cache=True)
    for _ in range(2):
        randomize_adapter(model)
        logits, draft_cache = model(inputs, cache=cache, adapter_mask=mask, return_cache=True)
        torch.testing.assert_close(logits[:, :1], base_logits, atol=2e-15, rtol=2e-14)
        for actual, expected in zip(trim_cache(draft_cache, 4), base_cache):
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("alpha", [2.0, 32.0])
def test_gated_lowrank_gradient_against_finite_differences(alpha):
    layer = GatedLinear(3, 4, 2, alpha).double()
    a = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    x = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False, True]])
    def call(a, b, x):
        return torch.func.functional_call(layer, {"lora_A": a, "lora_B": b}, (x, mask))
    assert torch.autograd.gradcheck(call, (a, b, x))


@pytest.mark.parametrize("kind", ["l1", "tv", "forward_kl", "reverse_kl"])
def test_distillation_gradient_and_teacher_stop_gradient(kind):
    q = torch.tensor([[.1, 1., -.7]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[.5, -.3, .7]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: divergence(value, teacher, kind), (q,))
    divergence(q, teacher, kind).sum().backward()
    assert teacher.grad is None
    if kind == "forward_kl":
        torch.testing.assert_close(q.grad, q.softmax(-1) - teacher.softmax(-1))


def test_paired_backprop_updates_only_adapter_and_has_correct_shift():
    model = tiny_model().train_adapters_only()
    clean = torch.tensor([[0, 1, 2, 3]])
    noise = torch.tensor([[0, 4, 5, 6]])
    b = paired_batch(clean, 2, noisy=noise)
    logits = model(b.tokens, positions=b.positions, allowed=b.allowed, adapter_mask=b.adapter_mask)
    expected = divergence(logits[:, 4:], logits[:, :4], "forward_kl")[:, 1:].mean()
    actual = paired_loss(model, clean, 2, noisy=noise, kind="forward_kl")
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter_parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
