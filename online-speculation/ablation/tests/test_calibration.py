"""Sparse proposal identities, convex updates and frozen-decoder integration."""

import pytest
import torch

from blockspec_ablation.calibration import OverlapMix, project_simplex
from blockspec_ablation.decoding import generate_speculative
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec.sampling import SamplingConfig, probabilities, residual


def test_simplex_projection_kkt_and_idempotence():
    torch.manual_seed(391)
    values = torch.randn(20, 5, dtype=torch.float64) * 3
    result = project_simplex(values)
    assert (result >= 0).all()
    torch.testing.assert_close(result.sum(-1), torch.ones(20, dtype=torch.float64), atol=2e-15, rtol=0)
    torch.testing.assert_close(project_simplex(result), result, atol=2e-15, rtol=0)
    for v, w in zip(values, result):
        active = w > 1e-12
        threshold = (v - w)[active].mean()
        torch.testing.assert_close((v - w)[active], threshold.expand(active.sum()), atol=2e-15, rtol=0)
        assert (v[~active] <= threshold + 1e-14).all()


@pytest.mark.parametrize("top_p", [1., .95, .5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_identity_preserves_root_support_ties_and_exact_probabilities(top_p, dtype):
    torch.manual_seed(392)
    logits = torch.randn(4, 17, dtype=dtype).round()
    baseline = probabilities(logits, SamplingConfig(1., 5, top_p))
    mix = OverlapMix(4, 5, dtype=dtype, adaptive=False)
    q, feedback = mix.propose(baseline)
    torch.testing.assert_close(q, baseline, atol=0, rtol=0)
    mix.weights.fill_(1 / len(mix.temperatures))
    changed, _ = mix.propose(baseline)
    torch.testing.assert_close(changed[0], baseline[0], atol=0, rtol=0)
    assert torch.equal(changed > 0, baseline > 0)
    torch.testing.assert_close(changed.sum(-1), torch.ones(4, dtype=dtype))
    torch.testing.assert_close(q, baseline, atol=0, rtol=0)
    torch.testing.assert_close(feedback.mixed, feedback.baseline, atol=0, rtol=0)


def test_sparse_tv_and_feasible_subgradient_match_full_vocabulary():
    torch.manual_seed(393)
    q0 = probabilities(torch.randn(4, 17, dtype=torch.float64), SamplingConfig(1., 7))
    teacher = torch.randn(3, 17, dtype=torch.float64).softmax(-1)
    mix = OverlapMix(4, 7, dtype=torch.float64)
    mix.weights.fill_(.2)
    q, feedback = mix.propose(q0)
    p = teacher.gather(-1, feedback.indices)
    tv = 1 - torch.minimum(p, feedback.mixed).sum(-1)
    torch.testing.assert_close(tv, .5 * (teacher - q[1:]).abs().sum(-1), atol=1e-14, rtol=0)
    gradient = -(feedback.experts * (feedback.mixed < p)[:, None, :]).sum(-1)
    direction = torch.zeros_like(mix.weights)
    direction[:, 0], direction[:, 1] = 1., -1.
    losses = []
    for sign in [1, -1]:
        work = ((mix.weights + sign * 1e-6 * direction)[:, :, None] * feedback.experts).sum(1)
        losses.append((1 - torch.minimum(work, p).sum(-1)).sum())
    torch.testing.assert_close((losses[0] - losses[1]) / 2e-6, (gradient * direction).sum(), atol=2e-9, rtol=0)


def test_updates_learn_sharpening_and_leave_unobserved_depths_fixed():
    baseline = torch.tensor([[.2, .3, .5], [.55, .30, .15], [.4, .3, .3]], dtype=torch.float64)
    teacher = torch.tensor([[.9, .05, .05]], dtype=torch.float64)
    mix = OverlapMix(3, 3, interval=1, learning_rate=1., dtype=torch.float64, diagnostics=True)
    initial, _ = mix.propose(baseline)
    untouched = mix.weights[1].clone()
    for _ in range(128):
        q, feedback = mix.propose(baseline)
        saved = q.clone()
        mix.observe(feedback, teacher)
        torch.testing.assert_close(q, saved, rtol=0, atol=0)
    current, _ = mix.propose(baseline)
    tv0 = .5 * (initial[1] - teacher[0]).abs().sum()
    tv1 = .5 * (current[1] - teacher[0]).abs().sum()
    assert tv1 < tv0 - .1
    torch.testing.assert_close(mix.weights[1], untouched, rtol=0, atol=0)
    assert mix.metrics()["depth_observations"] == [128., 0.]


def test_adaptive_sparse_output_law_by_enumeration():
    q0 = torch.tensor([[.2, .3, .5], [.6, .4, 0.]], dtype=torch.float64)
    p = torch.tensor([.15, .5, .35], dtype=torch.float64)
    mix = OverlapMix(2, 2, dtype=torch.float64, interval=1)
    for _ in range(20):
        q, feedback = mix.propose(q0)
        replacement, _ = residual(p, q[1])
        law = torch.zeros_like(p)
        for token in range(3):
            prob = q[1, token]
            accept = min(1., float(p[token] / prob)) if prob else 0.
            law[token] += prob * accept
            law += prob * (1 - accept) * replacement
        torch.testing.assert_close(law, p, atol=1e-14, rtol=0)
        mix.observe(feedback, p[None])


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 19])
def test_identity_decoder_reuses_rng_and_online_keeps_model_frozen(budget):
    torch.manual_seed(394)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).eval().requires_grad_(False)
    prompt = torch.tensor([[1, 2, 3]])
    options = dict(block_size=4, sampling=SamplingConfig(1., 5, .95))
    expected = generate_speculative(model, prompt, budget, generator=torch.Generator().manual_seed(39), **options)
    mix = OverlapMix(4, 5, adaptive=False)
    actual = generate_speculative(model, prompt, budget, generator=torch.Generator().manual_seed(39),
                                  calibrator=mix, **options)
    assert expected.tokens == actual.tokens
    assert expected.accepted == actual.accepted
    assert expected.decode_forwards == actual.decode_forwards
    state = {k: v.clone() for k, v in model.state_dict().items()}
    adaptive = OverlapMix(4, 5, interval=1)
    result = generate_speculative(model, prompt, budget, calibrator=adaptive, **options)
    assert len(result.tokens) == budget
    torch.testing.assert_close(model.state_dict(), state, atol=0, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"block_size": 1}, {"temperatures": (.5, .75)},
                                   {"temperatures": (1., -1.)}, {"interval": 0}, {"learning_rate": float("nan")}])
def test_invalid_mixer_config(kwargs):
    with pytest.raises(ValueError):
        OverlapMix(**{"block_size": 4, "top_k": 5, **kwargs})


def test_state_restore_preserves_parameters_pending_feedback_and_cadence():
    torch.manual_seed(397)
    original = OverlapMix(4, 5, interval=3, dtype=torch.float64, diagnostics=True)
    q0 = torch.randn(4, 5, dtype=torch.float64).softmax(-1)
    p = torch.randn(3, 5, dtype=torch.float64).softmax(-1)
    for _ in range(5):
        _, feedback = original.propose(q0)
        original.observe(feedback, p)
    state = original.state_dict()
    continued = OverlapMix(4, 5, interval=3, dtype=torch.float64)
    storage = continued.weights.data_ptr()
    continued.load_state_dict(state)
    frozen = OverlapMix(4, 5, interval=3, dtype=torch.float64, adaptive=False)
    frozen.load_state_dict(state)
    assert continued.weights.data_ptr() == storage
    assert continued.feedback_blocks == 5 and continued.updates == 1
    torch.testing.assert_close(frozen.propose(q0)[0], original.propose(q0)[0], atol=0, rtol=0)
    for mix in (original, continued):
        _, feedback = mix.propose(q0)
        mix.observe(feedback, p)
    torch.testing.assert_close(original.weights, continued.weights, atol=0, rtol=0)
    assert original.updates == continued.updates == 2
    torch.testing.assert_close(frozen.weights, state["tensors"]["weights"], atol=0, rtol=0)
    assert not torch.equal(original.weights, frozen.weights)
    state["tensors"]["weights"].zero_()
    assert frozen.weights.sum() > 0


@pytest.mark.parametrize("corruption", ["shape", "nan", "sum", "contract"])
def test_corrupt_state_is_rejected_before_parameter_publication(corruption):
    mix = OverlapMix(4, 5)
    state = mix.state_dict()
    before = mix.weights.clone()
    if corruption == "shape":
        state["tensors"]["weights"] = torch.ones(2, 2)
    elif corruption == "nan":
        state["tensors"]["counts"][0] = float("nan")
    elif corruption == "sum":
        state["tensors"]["weights"] += 1
    else:
        state["config"]["top_k"] = 4
    with pytest.raises(ValueError):
        mix.load_state_dict(state)
    torch.testing.assert_close(mix.weights, before, atol=0, rtol=0)
