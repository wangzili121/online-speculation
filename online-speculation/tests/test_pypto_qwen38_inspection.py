"""Validation helpers must preserve the actual head and committed target history."""

import pytest
import torch

from blockspec.pypto_qwen38.hf_reference import ReferenceCache


def test_reference_gdn_history_orientation_and_conv_boundary():
    conv = torch.arange(24).reshape(4, 6).float()
    recurrent = torch.arange(18).reshape(2, 3, 3).float()
    original = (conv.clone(), recurrent.clone())
    cache = ReferenceCache(1, 4, "linear_attention", (conv, recurrent))
    assert cache.has_previous_state(1) and not cache.has_previous_state(0)
    assert torch.equal(cache.layers[1].recurrent_states[0][0], recurrent.transpose(-1, -2))
    new = torch.full((1, 6, 2), 77.)
    full = cache.update_conv_state(new, 1)
    assert torch.equal(full[..., :4], conv.T.unsqueeze(0))
    assert torch.equal(full[..., -2:], new)
    assert torch.equal(cache.layers[1].conv_states[0], full[..., -4:])
    cache.layers[1].recurrent_states[0].zero_()
    assert torch.equal(conv, original[0]) and torch.equal(recurrent, original[1])


def test_reference_full_cache_appends_without_mutating_history():
    key, value = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    original = (key.clone(), value.clone())
    cache = ReferenceCache(3, 4, "full_attention", (key, value))
    new_key, new_value = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    result_key, result_value = cache.update(new_key, new_value, 3)
    assert torch.equal(result_key, torch.cat((key, new_key), dim=2))
    assert torch.equal(result_value, torch.cat((value, new_value), dim=2))
    assert torch.equal(key, original[0]) and torch.equal(value, original[1])


@pytest.mark.parametrize("layer,kind", [(0, "linear_attention"), (3, "full_attention")])
def test_hf_layer_reference_loads_only_the_selected_layer(layer, kind):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    from blockspec.pypto_qwen38.hf_reference import layer_reference

    torch.manual_seed(7)
    config = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=["linear_attention"] * 3 + ["full_attention"])
    config._attn_implementation = "sdpa"
    original = Qwen3_5DecoderLayer(config, layer).to(torch.bfloat16)
    state = original.state_dict()
    prefix = "self_attn." if kind == "full_attention" else "linear_attn."
    shared = {name: value for name, value in state.items() if not name.startswith(prefix)}
    mixer = {name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)}
    hidden = torch.randn(4, 32).bfloat16()
    history = ((torch.randn(1, 2, 3, 16).bfloat16(), torch.randn(1, 2, 3, 16).bfloat16())
               if kind == "full_attention" else (torch.randn(4, 64).bfloat16(), torch.randn(4, 8, 8)))
    rotary = (torch.ones(4, 4), torch.zeros(4, 4))
    result = layer_reference(config, layer, hidden, shared, mixer, history, rotary)
    assert result.shape == hidden.shape and result.dtype == hidden.dtype and result.isfinite().all()
