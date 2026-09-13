"""Tensor correction equivalence, captured sampling maps and online publication."""

import itertools

import pytest
import torch

from blockspec.sampling import SamplingConfig, residual
from blockspec.sampling_execution import SamplingExecutor, exponential_choice, linear_correction


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph hardware required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_acceptance_branches_match_scalar_residual(device, dtype):
    q = torch.tensor([[.6, .3, .1], [.5, .2, .3]], device=device, dtype=dtype)
    p = torch.tensor([[.2, .5, .3], [.25, .35, .4], [.1, .15, .75]], device=device, dtype=dtype)
    exponential = torch.tensor([.8, .2, 1.1], device=device, dtype=dtype)
    for candidates in itertools.product(range(3), repeat=2):
        ids = torch.tensor(candidates, device=device)
        for values in itertools.product([0., .25, .6, .99], repeat=2):
            uniforms = torch.tensor(values, device=device, dtype=dtype)
            count, tail, valid = linear_correction(ids, q, p, uniforms, exponential)
            expected = 0
            while expected < 2 and values[expected] < min(1., float(p[expected, ids[expected]] / q[expected, ids[expected]])):
                expected += 1
            law = residual(p[expected], q[expected])[0] if expected < 2 else p[-1]
            assert valid and int(count) == expected
            assert int(tail) == int(exponential_choice(law, exponential))


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_graph_boundary_rejects_nonfinite_logits_and_invalid_proposals(device):
    engine = SamplingExecutor(7, 4, SamplingConfig(1., 5), device=device, use_cuda_graph=device == "cuda")
    logits = torch.zeros(4, 7, device=device)
    tokens, q, _ = engine.draft(logits)
    bad = logits.clone()
    bad[1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        engine.draft(bad)
    with pytest.raises(ValueError, match="finite"):
        engine.verify(tokens[1:], q[1:], bad)
    with pytest.raises(ValueError, match="valid"):
        engine.verify(tokens[1:], q[1:] * 0, logits)
    with pytest.raises(ValueError, match="valid"):
        engine.verify(tokens[1:] - 20, q[1:], logits)
