"""Frozen HF bridge: clean-law parity, routing, functional KV and head-only learning."""

import copy

import pytest
import torch

from blockspec_ablation.checkpoint import adapter_state, base_fingerprint, config_from_hf
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.hf_execution import FrozenHFDecoder
from blockspec_ablation.model import trim_cache
from blockspec_ablation.relay import RelayConfig, RelayHead, RelayLearner, generate_relay
from blockspec.sampling import SamplingConfig


def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(388)
    config = transformers.Qwen3Config(vocab_size=17, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=2, num_attention_heads=2,
                                      num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    reference = transformers.Qwen3ForCausalLM(config).eval()
    model = FrozenHFDecoder(copy.deepcopy(reference), config_from_hf(config.to_dict(), rank=2, alpha=8))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("lora_A", "lora_B")):
                parameter.normal_(std=.08)
    return reference, model


@torch.no_grad()
def test_hf_clean_outputs_match_the_author_backbone_and_adapter_gate_is_scoped():
    reference, model = tiny()
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = reference(tokens).logits
    torch.testing.assert_close(model(tokens), expected, atol=0, rtol=0)
    mask = torch.tensor([[False, True, True, True]])
    logits, _, boundary = model(tokens, adapter_mask=mask, return_cache=True, capture_layer=2)
    torch.testing.assert_close(logits[:, 0], expected[:, 0], atol=0, rtol=0)
    assert (logits[:, 1:] - expected[:, 1:]).abs().max() > .001
    assert boundary.hidden.shape == (1, 4, 16)
    torch.testing.assert_close(model.lm_head(model.model.norm(boundary.hidden)), logits, atol=0, rtol=0)
    torch.testing.assert_close(model(tokens), expected, atol=0, rtol=0)
    assert model.routing.mask is None and not model.routing.active
    assert all(not p.requires_grad for p in model.parameters())


@torch.no_grad()
def test_hf_rejected_cache_views_keep_their_clean_values():
    reference, model = tiny()
    prefix = torch.tensor([[1, 2, 3]])
    _, cache = model(prefix, return_cache=True)
    saved = tuple((k.clone(), v.clone()) for k, v in cache)
    draft = torch.tensor([[4, 5, 6]])
    mask = torch.tensor([[False, True, True]])
    _, temporary = model(draft, cache=cache, adapter_mask=mask, return_cache=True)
    clean = trim_cache(temporary, 4)
    logits, updated = model(torch.tensor([[7, 8]]), cache=clean, return_cache=True)
    expected = reference(torch.tensor([[1, 2, 3, 4, 7, 8]])).logits[:, -2:]
    torch.testing.assert_close(logits, expected, atol=3e-7, rtol=3e-6)
    torch.testing.assert_close(cache, saved, atol=0, rtol=0)
    assert clean[0][0].shape[2] == 4 and updated[0][0].shape[2] == 6


def test_hf_online_only_updates_the_new_head_and_preserves_greedy_generation():
    _, model = tiny()
    base, adapter = base_fingerprint(model), adapter_state(model)
    head = RelayHead(RelayConfig(17, 16, 2))
    learner = RelayLearner(head, interval=1, sampling=SamplingConfig(1., 5, .95))
    prompt = torch.tensor([[1, 2, 3]])
    expected = generate_ar(model, prompt, 16).tokens
    assert generate_speculative(model, prompt, 16, block_size=4).tokens == expected
    result = generate_relay(model, head, prompt, 16, block_size=4, learner=learner)
    assert result.tokens == expected and result.updates > 0
    assert head.projection.weight.count_nonzero() > 0
    assert base_fingerprint(model) == base
    torch.testing.assert_close(adapter_state(model), adapter, atol=0, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_hf_routing_context_cleans_up_after_failed_forward():
    _, model = tiny()
    with pytest.raises((IndexError, RuntimeError)):
        model(torch.tensor([[900]]), adapter_mask=torch.ones(1, 1, dtype=torch.bool),
              return_cache=True, capture_layer=2)
    assert model.routing.mask is None and not model.routing.active
    assert len(model.model.norm._forward_pre_hooks) == 0
    assert torch.isfinite(model(torch.tensor([[1, 2]]))).all()
    with pytest.raises(ValueError, match="SDPA"):
        model.set_attention_backend("grouped")
