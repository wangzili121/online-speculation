from types import SimpleNamespace

import pytest
import torch

from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import ScoreRequest
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.generation import generation_config_for_prompt


@pytest.mark.parametrize("family", ["gpt2", "qwen2"])
def test_chunked_scoring_matches_complete_causal_context(family):
    from transformers import AutoModelForCausalLM, GPT2Config, Qwen2Config

    config = (
        GPT2Config(vocab_size=31, n_layer=1, n_head=2, n_embd=16, n_positions=128)
        if family == "gpt2" else
        Qwen2Config(vocab_size=31, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, hidden_size=16, intermediate_size=32,
                    max_position_embeddings=128, attn_implementation="eager")
    )
    with torch.random.fork_rng():
        torch.manual_seed(21)
        model = AutoModelForCausalLM.from_config(config).eval()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
    complete = TransformersBackend(model, tokenizer, device="cpu", score_chunk_size=128)
    chunked = TransformersBackend(model, tokenizer, device="cpu", score_chunk_size=4)
    requests = [ScoreRequest((1, 4, 3, 5, 8, 9), ((8, 4, 3, 9, 1, 6, 2), (5, 6)), SamplingConfig(temperature=0.7)),
                ScoreRequest((), ((8,),), SamplingConfig())]
    expected = complete.score_statistics_batch(requests, confidence_top_k=5)
    actual = chunked.score_statistics_batch(requests, confidence_top_k=5)
    for left, right in zip(expected, actual, strict=True):
        for name in ("token_logprobs", "mean_logprob", "mean_negative_entropy", "mean_self_certainty", "token_topk_confidences"):
            assert getattr(right, name) == pytest.approx(getattr(left, name), abs=2e-6)
    assert chunked.score_batch(requests) == pytest.approx([item.token_logprobs for item in actual])
    # Long inputs are processed once, including prefix; the last target has no forward.
    assert chunked.snapshot().score_forward_token_slots == 2 * (12 + 7 + 2)
    requests = [GenerationRequest(tuple(range(1, length + 1)), 4, SamplingConfig(), 8, str(length)) for length in (11, 5)]
    left, right = complete.sample_batch(requests), chunked.sample_batch(requests)
    for expected, actual in zip(left, right, strict=True):
        assert expected.token_ids == actual.token_ids
        assert expected.token_logprobs == pytest.approx(actual.token_logprobs, abs=2e-6)
    del model, complete, chunked


def test_context_budget_preserves_requested_config_and_caps_blocks():
    backend = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024)),
                              tokenizer=SimpleNamespace(model_max_length=10**30))
    source = {"mh": {"block_size": 256}, "conditional_is": {"block_size": 64}}
    config, metadata = generation_config_for_prompt(source, 1000, [backend])
    assert config["generation"]["max_new_tokens"] == 24
    assert metadata["requested_max_new_tokens"] == 32768
    assert config["mh"]["block_size"] == 24
    assert "generation" not in source
    with pytest.raises(ValueError, match="fills"):
        generation_config_for_prompt(source, 1024, [backend])


def test_proposal_context_is_also_respected():
    config, _ = generation_config_for_prompt(
        {"generation": {"max_new_tokens": 100}}, 8,
        [SimpleNamespace(max_model_len=200), SimpleNamespace(max_model_len=24)],
    )
    assert config["generation"]["max_new_tokens"] == 16
