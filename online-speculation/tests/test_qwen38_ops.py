"""Numerical boundaries and gradients of the native execution adapters."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from blockspec.vllm_ascend.qwen38_ops import GatedRMSNorm, SharedRotary, causal_conv, causal_conv_update, l2norm, swiglu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_activations_preserve_fp32_intermediates_and_gradients(dtype):
    torch.manual_seed(51)
    x = torch.randn(3, 16, dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    gate, up = reference.float().chunk(2, -1)
    expected = (F.silu(gate) * up).to(dtype)
    actual = swiglu(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
    normalized = l2norm(x.detach())
    full = x.detach().float()
    torch.testing.assert_close(normalized, (full * torch.rsqrt(full.square().sum(-1, keepdim=True) + 1e-6)).to(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_norm_has_weight_input_and_gate_gradients(dtype):
    torch.manual_seed(52)
    x, gate = [torch.randn(4, 8, dtype=dtype, requires_grad=True) for _ in range(2)]
    weight = nn.Parameter(torch.randn(8, dtype=dtype))
    norm = GatedRMSNorm(weight, 1e-6)
    result = norm(x, gate)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                * weight.float() * F.silu(gate.float())).to(dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.float().square().mean().backward()
    for tensor in (x, gate, weight):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.count_nonzero()


def test_frozen_inference_norm_weight_still_allows_input_gradient():
    with torch.inference_mode():
        weight = nn.Parameter(torch.ones(8), requires_grad=False)
    norm = GatedRMSNorm(weight, 1e-6)
    x, gate = [torch.randn(4, 8, requires_grad=True) for _ in range(2)]
    norm(x, gate).square().mean().backward()
    assert x.grad is not None and gate.grad is not None and weight.grad is None


@pytest.mark.parametrize("length", [1, 4, 17, 32, 36, 65])
def test_causal_conv_forward_backward_and_future_visibility(length):
    torch.manual_seed(53)
    x = torch.randn(1, 8, length, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, 4, dtype=torch.bfloat16, requires_grad=True)
    actual = causal_conv(x, weight)
    expected = F.silu(F.conv1d(x.float(), weight[:, None].float(), padding=3, groups=8)[..., :length]).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert x.grad is not None and weight.grad is not None
    if length > 1:
        changed = x.detach().clone()
        changed[..., 1:] += 10
        assert torch.equal(causal_conv(changed, weight.detach())[..., :1], actual.detach()[..., :1])


def test_single_step_conv_updates_only_owned_state_and_is_differentiable():
    torch.manual_seed(54)
    original = torch.randn(1, 8, 4, dtype=torch.bfloat16)
    state = original.clone()
    x = torch.randn(1, 8, 1, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, 4, dtype=torch.bfloat16, requires_grad=True)
    updated = torch.cat((original[..., 1:], x), dim=-1)
    actual = causal_conv_update(x, state, weight)
    expected = causal_conv(updated, weight)[..., -1:]
    torch.testing.assert_close(actual, expected)
    assert torch.equal(state, updated) and not torch.equal(state, original)
    actual.float().square().mean().backward()
    assert x.grad is not None and weight.grad is not None


def test_shared_rotary_shares_only_coefficients_and_preserves_head_layout():
    import copy

    class Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cos_sin_cache", torch.arange(64.).reshape(16, 4), persistent=False)
            self.mrope_section = [2, 0, 0]
            self.mrope_interleaved = False
            self.rotary_dim = 4
            self.last_positions = None

        def forward(self, positions, query, key):
            self.last_positions = positions.clone()
            return query, key

    original = Rotary()
    shared = SharedRotary(original)
    assert shared.cache is original.cos_sin_cache
    assert copy.deepcopy(shared) is shared
    positions = shared.positions(5, 3, "cpu")
    assert original.last_positions is None
    assert shared.sections == (2, 0, 0) and shared.dim == 4 and not shared.interleaved
    cos, sin = shared.coefficients(positions, torch.bfloat16)
    assert cos.shape == sin.shape == (1, 1, 3, 4)
    assert torch.equal(cos[0, 0, :, :2], original.cos_sin_cache[5:8, :2].bfloat16())
