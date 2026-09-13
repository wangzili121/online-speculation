"""Position reuse preserves logits, KV, gradients and frozen-suffix replay."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from blockspec_ablation.distillation import paired_batch
from blockspec_ablation.model import Decoder, ModelConfig


@pytest.mark.parametrize("dtype,autocast", [(torch.float32, False), (torch.float64, False),
                                           (torch.bfloat16, False), (torch.float32, True)])
def test_shared_rotary_matches_per_layer_reference(dtype, autocast):
    torch.manual_seed(26)
    config = ModelConfig(hidden_size=24, intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_factor=16, rope_theta=1e6, adapter_rank=2)
    model = Decoder(config).to(dtype=dtype).train_adapters_only()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("lora_B"):
                p.normal_(std=.05)
    clean = torch.randint(config.vocab_size, (2, 9))
    batch = paired_batch(clean, 4, noisy=torch.randint(config.vocab_size, clean.shape))
    options = dict(positions=batch.positions + 65536, allowed=batch.allowed,
                   adapter_mask=batch.adapter_mask, return_cache=True, capture_layer=1)
    outputs, gradients = [], []
    for shared in (False, True):
        model.zero_grad(set_to_none=True)
        sharing = nullcontext() if shared else patch.object(model, "_rotary", return_value=None)
        with sharing, torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            logits, cache, boundary = model(batch.tokens, **options)
            suffix = model.forward_suffix(boundary)
            torch.testing.assert_close(logits, suffix, rtol=0, atol=0)
            logits.square().mean().backward()
        outputs.append((logits.detach(), cache))
        gradients.append([p.grad.clone() for p in model.adapter_parameters()])
    for actual, expected in zip(outputs[0][1], outputs[1][1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


def test_shared_rotary_cache_replay_after_weight_update():
    torch.manual_seed(9)
    model = Decoder(ModelConfig(num_hidden_layers=3)).train_adapters_only()
    with torch.no_grad():
        _, past = model(torch.tensor([[1, 5, 3]]), return_cache=True)
    ids = torch.tensor([[4, 8, 2, 7]])
    mask = torch.tensor([[False, True, True, True]])
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=.01)
    for _ in range(3):
        with patch.object(model, "_rotary", return_value=None), torch.no_grad():
            expected, expected_cache = model(ids, cache=past, adapter_mask=mask, return_cache=True)
        actual, actual_cache = model(ids, cache=past, adapter_mask=mask, return_cache=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
        optimizer.zero_grad()
        actual.square().mean().backward()
        optimizer.step()
