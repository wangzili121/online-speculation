import torch

from blockspec.dsv41_tail_vjp import (
    FrozenTailContract,
    hc_post,
    hc_post_vjp,
    hc_pre_forward,
    hc_pre_vjp,
)


def test_hc_post_closed_form_vjp_matches_autograd():
    torch.manual_seed(41)
    value = torch.randn(3, 16, requires_grad=True)
    residual = torch.randn(3, 4, 16, requires_grad=True)
    post = torch.randn(3, 4, requires_grad=True)
    combination = torch.randn(3, 4, 4, requires_grad=True)
    gradient = torch.randn(3, 4, 16)
    expected = torch.autograd.grad(
        hc_post(value, residual, post, combination),
        (value, residual, post, combination),
        gradient,
    )
    _, actual = hc_post_vjp(value, residual, post, combination, gradient)
    for reference, result in zip(expected, actual.__dict__.values(), strict=True):
        torch.testing.assert_close(result, reference)


def test_frozen_tail_contract_ends_at_trainable_attention():
    contract = FrozenTailContract()
    assert contract.stages[0] == "lm_head_input_vjp"
    assert contract.stages[-1] == "attention_hc_post_vjp"
    assert "selected_moe_input_vjp" in contract.stages


def test_hc_pre_vjp_matches_autograd_through_sinkhorn_and_collapse():
    generator = torch.Generator().manual_seed(884)
    rows, copies, hidden = 3, 4, 8
    channels = (copies + 2) * copies
    residual = torch.randn(rows, copies, hidden, generator=generator, requires_grad=True)
    incoming = torch.randn(rows, copies, generator=generator, requires_grad=True)
    function = torch.randn(channels, copies * hidden, generator=generator)
    scale = torch.tensor([.7, 1.1, .9])
    base = torch.randn(channels, generator=generator)
    collapsed, state = hc_pre_forward(
        residual, incoming, function, scale, base,
        norm_eps=1e-6, hc_eps=1e-6, iterations=4,
    )
    gradients = (
        torch.randn_like(collapsed),
        torch.randn_like(state.pre),
        torch.randn_like(state.post),
        torch.randn_like(state.sinkhorn[-1]),
    )
    expected = torch.autograd.grad(
        (collapsed, state.pre, state.post, state.sinkhorn[-1]),
        (residual, incoming), gradients,
    )
    actual = hc_pre_vjp(
        residual.detach(), incoming.detach(), function, scale, base, state,
        *gradients, norm_eps=1e-6, hc_eps=1e-6, iterations=4,
    )
    torch.testing.assert_close(actual.residual, expected[0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.incoming, expected[1], atol=2e-5, rtol=2e-5)
