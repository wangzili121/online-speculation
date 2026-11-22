import torch

from blockspec.dsv41_tail_vjp import FrozenTailContract, hc_post, hc_post_vjp


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
