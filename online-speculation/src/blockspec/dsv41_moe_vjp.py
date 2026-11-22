"""Reference VJPs for the fixed-route DSV4.1 MoE suffix."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoECombineVJP:
    shared: torch.Tensor
    experts: torch.Tensor
    weights: torch.Tensor


def moe_combine(shared, expert_outputs, weights):
    if expert_outputs.ndim != 3 or weights.shape != expert_outputs.shape[:2]:
        raise ValueError("selected expert outputs and route weights disagree")
    if shared.shape != (expert_outputs.shape[0], expert_outputs.shape[2]):
        raise ValueError("shared expert output has the wrong shape")
    return shared.float() + torch.einsum("tr,trd->td", weights.float(), expert_outputs.float())


def moe_combine_vjp(shared, expert_outputs, weights, output_gradient):
    output = moe_combine(shared, expert_outputs, weights)
    if output_gradient.shape != output.shape:
        raise ValueError("MoE combine gradient has the wrong shape")
    gradient = output_gradient.float()
    return output, MoECombineVJP(
        shared=gradient,
        experts=weights.float().unsqueeze(-1) * gradient.unsqueeze(1),
        weights=torch.einsum("td,trd->tr", gradient, expert_outputs.float()),
    )


def normalized_route_vjp(scores, weight_gradient, *, scale=1.0):
    """Differentiate ``scale * scores / scores.sum`` for fixed Top-K IDs."""
    if scores.ndim != 2 or weight_gradient.shape != scores.shape or not scale > 0:
        raise ValueError("fixed-route scores, gradients and positive scale required")
    scores, weight_gradient = scores.float(), weight_gradient.float()
    total = scores.sum(-1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("selected route scores must have positive sums")
    probability = scores / total
    return scale / total * (
        weight_gradient - (weight_gradient * probability).sum(-1, keepdim=True)
    )
