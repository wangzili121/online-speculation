"""Tensor correction equivalence, captured sampling maps and online publication."""


import pytest
import torch

from blockspec_ablation.calibration import OverlapMix
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec.sampling import SamplingConfig, probabilities
from blockspec_ablation.sampling_execution import SamplingExecutor


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph hardware required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("top_p", [1., .95, .5])
@torch.no_grad()
def test_prepared_probability_maps_match_reference_and_own_snapshots(device, top_p):
    torch.manual_seed(616)
    sampling = SamplingConfig(1., 5, top_p)
    engine = SamplingExecutor(17, 4, sampling, device=device, use_cuda_graph=device == "cuda")
    logits = torch.randn(4, 17, device=device).round()
    baseline = probabilities(logits, sampling)
    mix = OverlapMix(4, 5, device=device, interval=1)
    gen = lambda: torch.Generator(device=device).manual_seed(17)
    tokens, q, _ = engine.draft(logits, gen())
    ids, original, feedback = engine.draft(logits, gen(), mix)
    torch.testing.assert_close(q, baseline, rtol=0, atol=0)
    torch.testing.assert_close(original, q, rtol=0, atol=0)
    assert torch.equal(tokens, ids)
    saved = q.clone()
    for _ in range(5):
        mix.observe(feedback, torch.randn(3, 17, device=device).softmax(-1))
    mixed_ids, changed, _ = engine.draft(logits, gen(), mix)
    expected, _ = mix.propose(baseline)
    torch.testing.assert_close(changed, expected, rtol=0, atol=0)
    torch.testing.assert_close(q, saved, rtol=0, atol=0)
    torch.testing.assert_close(feedback.baseline, feedback.mixed, rtol=0, atol=0)
    teacher = torch.randn(4, 17, device=device)
    verified, target = engine.verify(mixed_ids[1:], changed[1:], teacher, gen())
    torch.testing.assert_close(target, probabilities(teacher, sampling), rtol=0, atol=0)
    assert 1 <= len(verified.tokens) <= 4
    assert verified.tokens[:verified.accepted] == mixed_ids[1:1 + verified.accepted].tolist()
    target_copy = target.clone()
    engine.verify(mixed_ids[1:], changed[1:], teacher + 1., gen())
    torch.testing.assert_close(target, target_copy, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [0, 1, 2, 9])
@torch.no_grad()
def test_captured_decoder_identity_and_frozen_online_stream(device, budget):
    torch.manual_seed(617)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device).eval().requires_grad_(False)
    prompt = torch.tensor([[1, 2, 3]], device=device)
    sampling = SamplingConfig(1., 5, .95)
    sampler = SamplingExecutor(7, 4, sampling, device=device, use_cuda_graph=device == "cuda")
    options = dict(block_size=4, sampling=sampling, sampler_executor=sampler)
    gen = lambda: torch.Generator(device=device).manual_seed(71)
    expected = generate_speculative(model, prompt, budget, generator=gen(), **options)
    identity = OverlapMix(4, 5, adaptive=False, device=device)
    actual = generate_speculative(model, prompt, budget, generator=gen(), calibrator=identity, **options)
    assert expected.tokens == actual.tokens
    assert expected.decode_forwards == actual.decode_forwards
    assert len(generate_ar(model, prompt, budget, sampling=sampling, sampler_executor=sampler).tokens) == budget
    mix = OverlapMix(4, 5, device=device, interval=1)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    result = generate_speculative(model, prompt, budget, generator=gen(), calibrator=mix, **options)
    assert len(result.tokens) == budget
    torch.testing.assert_close(state, model.state_dict(), rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
