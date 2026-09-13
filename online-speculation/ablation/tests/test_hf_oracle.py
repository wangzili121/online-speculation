"""Transformers is an optional TEST oracle, never the implementation backend."""

import pytest
import torch

from blockspec_ablation.checkpoint import config_from_hf
from blockspec_ablation.model import Decoder, is_adapter, rotary_frequencies


@pytest.mark.parametrize("head_dim", [8, 12])
def test_independent_decoder_matches_hf_qwen3(head_dim):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=31, hidden_size=32, intermediate_size=48,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, head_dim=head_dim,
                                     max_position_embeddings=256, attention_dropout=0)
    config._attn_implementation = "eager"
    torch.manual_seed(15)
    oracle = transformers.Qwen3ForCausalLM(config).eval()
    ours = Decoder(config_from_hf(config.to_dict(), rank=2))
    missing, unexpected = ours.load_state_dict(oracle.state_dict(), strict=False)
    assert not unexpected and all(is_adapter(k) for k in missing)
    ids = torch.tensor([[0, 3, 7, 11, 4, 2, 17], [1, 6, 3, 9, 13, 4, 2]])
    with torch.no_grad():
        expected = oracle(ids, use_cache=False).logits
        actual = ours(ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_yarn_frequencies_against_hf_definition():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    raw = {"model_type": "qwen3", "hidden_size": 64, "intermediate_size": 128,
           "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
           "vocab_size": 9, "head_dim": 16, "rope_theta": 1000000.,
           "max_position_embeddings": 131072,
           "rope_scaling": {"rope_type": "yarn", "factor": 16., "attention_factor": 1.2772588722,
                            "beta_fast": 128., "beta_slow": 4.,
                            "original_max_position_embeddings": 8192}}
    config = config_from_hf(raw)
    oracle, scale = _compute_yarn_parameters(Qwen3Config(**raw), torch.device("cpu"))
    torch.testing.assert_close(rotary_frequencies(config), oracle, atol=1e-9, rtol=1e-6)
    assert scale == config.rope_attention_factor
    model = Decoder(config).bfloat16()
    assert model.model.layers[0].self_attn.frequencies.dtype == torch.float32


@pytest.mark.parametrize("feature", [{"model_type": "unknown"}, {"num_experts": 4},
                                    {"rope_head_dim": 4}, {"attention_gate_func": "silu"},
                                    {"sliding_window": 128}])
def test_unsupported_hf_features_fail_closed(feature):
    raw = {"model_type": "k2_horizon", "hidden_size": 16, "intermediate_size": 24,
           "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
           "vocab_size": 9, "head_dim": 8}
    raw.update(feature)
    with pytest.raises(ValueError):
        config_from_hf(raw)
