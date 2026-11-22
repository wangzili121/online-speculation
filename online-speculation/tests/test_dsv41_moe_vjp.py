import torch

from blockspec.dsv41_moe_vjp import (
    clipped_swiglu,
    clipped_swiglu_vjp,
    frozen_expert_input_vjp,
    moe_combine,
    moe_combine_vjp,
    normalized_route_vjp,
)


def test_fixed_route_moe_combine_vjp_matches_autograd():
    torch.manual_seed(23)
    shared = torch.randn(3, 11, requires_grad=True)
    experts = torch.randn(3, 6, 11, requires_grad=True)
    scores = torch.rand(3, 6, requires_grad=True)
    weights = 2.5 * scores / scores.sum(-1, keepdim=True)
    gradient = torch.randn(3, 11)
    output = moe_combine(shared, experts, weights)
    expected = torch.autograd.grad(output, (shared, experts, scores), gradient)
    _, actual = moe_combine_vjp(shared, experts, weights, gradient)
    score_gradient = normalized_route_vjp(scores, actual.weights, scale=2.5)
    for reference, result in zip(expected, (actual.shared, actual.experts, score_gradient), strict=True):
        torch.testing.assert_close(result, reference)


def test_clipped_swiglu_vjp_matches_autograd_and_clamp_masks():
    torch.manual_seed(29)
    gate_up = (torch.randn(5, 22) * 7).requires_grad_()
    gradient = torch.randn(5, 11)
    output = clipped_swiglu(gate_up)
    expected, = torch.autograd.grad(output, gate_up, gradient)
    actual_output, actual = clipped_swiglu_vjp(gate_up.detach(), gradient)
    torch.testing.assert_close(actual_output, output.detach())
    torch.testing.assert_close(actual, expected)


def test_frozen_expert_input_vjp_matches_autograd_without_weight_gradients():
    torch.manual_seed(31)
    value = torch.randn(4, 16, requires_grad=True)
    gate_up_weight = torch.randn(16, 24)
    down_weight = torch.randn(12, 16)
    gradient = torch.randn_like(value)
    reference_output = clipped_swiglu(value @ gate_up_weight) @ down_weight
    expected, = torch.autograd.grad(reference_output, value, gradient)
    output, actual = frozen_expert_input_vjp(
        value.detach(), gate_up_weight, down_weight, gradient,
    )
    torch.testing.assert_close(output, reference_output.detach())
    torch.testing.assert_close(actual, expected)
