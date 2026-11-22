import pytest
import torch

from blockspec.dsv41_moe_vjp import (
    clipped_swiglu,
    clipped_swiglu_vjp,
    frozen_expert_input_vjp,
    frozen_moe_input_vjp,
    moe_combine,
    moe_combine_vjp,
    normalized_route_vjp,
    router_logits_vjp,
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


def test_router_logits_vjp_scatter_matches_selected_score_autograd():
    torch.manual_seed(24)
    logits = torch.randn(4, 384, requires_grad=True)
    route_ids = torch.stack([torch.randperm(384)[:6] for _ in range(4)]).int()
    upstream = torch.randn(4, 6)
    scores = logits.sigmoid().gather(1, route_ids.long())
    weights = 2.5 * scores / scores.sum(-1, keepdim=True)
    expected, = torch.autograd.grad(weights, logits, upstream)
    actual = router_logits_vjp(logits.detach(), route_ids, upstream, scale=2.5)
    torch.testing.assert_close(actual, expected)
    mask = torch.ones_like(actual, dtype=torch.bool).scatter(1, route_ids.long(), False)
    assert torch.count_nonzero(actual[mask]) == 0


def test_router_logits_vjp_rejects_duplicate_topk_ids():
    ids = torch.tensor([[1, 2, 3, 4, 5, 5]], dtype=torch.int32)
    with pytest.raises(ValueError, match="unique"):
        router_logits_vjp(torch.ones(1, 384), ids, torch.ones(1, 6))


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


def test_complete_frozen_moe_input_vjp_matches_fixed_route_autograd():
    torch.manual_seed(32)
    rows, hidden, intermediate, experts, topk = 3, 8, 6, 12, 4
    value = torch.randn(rows, hidden, requires_grad=True)
    router = torch.randn(hidden, experts)
    gate_up = torch.randn(experts, hidden, 2 * intermediate)
    down = torch.randn(experts, intermediate, hidden)
    shared_gate_up = torch.randn(hidden, 2 * intermediate)
    shared_down = torch.randn(intermediate, hidden)
    route_ids = torch.stack([torch.randperm(experts)[:topk] for _ in range(rows)]).int()
    logits = value @ router
    scores = logits.sigmoid().gather(1, route_ids.long())
    weights = 2.5 * scores / scores.sum(-1, keepdim=True)
    routed = torch.stack([
        torch.stack([
            clipped_swiglu(value[token:token + 1] @ gate_up[expert]) @ down[expert]
            for expert in ids.tolist()
        ]).squeeze(1)
        for token, ids in enumerate(route_ids)
    ])
    shared = clipped_swiglu(value @ shared_gate_up) @ shared_down
    output = shared + torch.einsum("tr,trd->td", weights, routed)
    upstream = torch.randn_like(output)
    expected, = torch.autograd.grad(output, value, upstream)
    actual_output, actual = frozen_moe_input_vjp(
        value.detach(), route_ids, router, gate_up, down,
        shared_gate_up, shared_down, upstream, route_scale=2.5,
    )
    torch.testing.assert_close(actual_output, output.detach())
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
