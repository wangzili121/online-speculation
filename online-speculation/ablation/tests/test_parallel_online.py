import pytest
import torch

from blockspec_ablation.calibration import OverlapMix
from blockspec_ablation.parallel import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate
from blockspec_ablation.parallel.feedback import OnlineFeedback
from blockspec_ablation.parallel.sampling import ProposalSampler
from blockspec.sampling import SamplingConfig, residual
from blockspec_ablation.sampling_execution import SamplingExecutor


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def tiny(device):
    torch.manual_seed(410)
    return DualViewDecoder(DualViewConfig(vocab_size=13, hidden_size=16, intermediate_size=24,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                         head_dim=8)).to(device).eval().requires_grad_(False)


def test_first_masked_proposal_is_learnable_and_residual_law_is_exact():
    base = torch.tensor([[.6, .4, 0.]], dtype=torch.float64)
    target = torch.tensor([[.15, .5, .35]], dtype=torch.float64)
    mix = OverlapMix(2, 2, protected_rows=0, interval=1, dtype=torch.float64)
    initial = mix.weights.clone()
    for _ in range(20):
        q, feedback = mix.propose(base)
        saved = q.clone()
        correction, _ = residual(target[0], q[0])
        actual = torch.zeros_like(q[0])
        for token in range(3):
            mass = q[0, token]
            acceptance = min(1., float(target[0, token] / mass)) if mass else 0.
            actual[token] += mass * acceptance
            actual += mass * (1 - acceptance) * correction
        torch.testing.assert_close(actual, target[0], rtol=0, atol=1e-14)
        mix.observe(feedback, target)
        assert torch.equal(saved, q)
    assert not torch.equal(initial, mix.weights)
    restored = OverlapMix(2, 2, protected_rows=0, interval=1, dtype=torch.float64)
    restored.load_state_dict(mix.state_dict())
    assert torch.equal(restored.weights, mix.weights)
    with pytest.raises(ValueError, match="state must match"):
        OverlapMix(2, 2, interval=1, dtype=torch.float64).load_state_dict(mix.state_dict())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block", [2, 4])
@torch.no_grad()
def test_masked_sampling_graph_matches_tensor_execution(device, block):
    sampling = SamplingConfig(1., 5, .95)
    plain = SamplingExecutor(13, block, sampling, device=device, temperatures=(),
                             use_cuda_graph=False, protected_rows=0)
    prepared = SamplingExecutor(13, block, sampling, device=device,
                                use_cuda_graph=device == "cuda", protected_rows=0)
    model = tiny(device)
    prompt = torch.tensor([[3, 5, 8]], device=device)
    branch = MaskedAttentionBranch(model)
    for budget in (0, 1, 2, 9):
        gen = lambda: torch.Generator(device=device).manual_seed(411)
        expected = generate(branch, prompt, budget, block_size=block, sampling=sampling,
                            sampler=ProposalSampler(sampling, executor=plain), generator=gen())
        mix = OverlapMix(block, 5, device=device, protected_rows=0, adaptive=False)
        actual = generate(branch, prompt, budget, block_size=block, sampling=sampling,
                          sampler=ProposalSampler(sampling, executor=prepared, calibrator=mix),
                          feedback=OnlineFeedback(calibrator=mix), generator=gen())
        assert actual.tokens == expected.tokens
        assert actual.accepted_per_round == expected.accepted_per_round
        assert actual.decode_forwards == expected.decode_forwards
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    mix = OverlapMix(block, 5, device=device, protected_rows=0, interval=1)
    for _ in range(2):
        result = generate(branch, prompt, 12, block_size=block, sampling=sampling,
                          sampler=ProposalSampler(sampling, executor=prepared, calibrator=mix),
                          feedback=OnlineFeedback(calibrator=mix), generator=gen())
        assert result.updates > 0
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in saved.items())


def test_masked_branch_rejects_a_causal_calibrator_layout():
    sampling = SamplingConfig(1., 5)
    with pytest.raises(ValueError, match="protected-row"):
        generate(MaskedAttentionBranch(tiny("cpu")), torch.tensor([[3, 4, 5]]), 9,
                 sampler=ProposalSampler(sampling, calibrator=OverlapMix(4, 5)))


def test_legacy_temperature_state_remains_readable():
    mix = OverlapMix(4, 5)
    state = mix.state_dict()
    state["config"].pop("protected_rows")
    mix.load_state_dict(state)
    assert mix.protected_rows == 1
