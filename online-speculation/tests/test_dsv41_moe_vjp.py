import torch

from blockspec.dsv41_moe_vjp import moe_combine, moe_combine_vjp, normalized_route_vjp


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
