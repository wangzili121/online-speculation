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


def router_logits_vjp(logits, route_ids, weight_gradient, *, scale=1.0):
    """Differentiate sigmoid routing and scatter fixed-TopK logit gradients.

    Top-K membership is held fixed, exactly as it was in the serving forward.
    Unselected logits therefore receive zero gradient.  Selected scores are
    ``sigmoid(logit)`` and their normalized weights are
    ``scale * score / selected_scores.sum()``.
    """
    if (
        logits.ndim != 2 or route_ids.ndim != 2
        or weight_gradient.shape != route_ids.shape
        or logits.shape[0] != route_ids.shape[0]
        or route_ids.dtype not in (torch.int32, torch.int64)
        or bool(((route_ids < 0) | (route_ids >= logits.shape[1])).any())
    ):
        raise ValueError("fixed-route scores, IDs and gradients are incompatible")
    if any(torch.unique(row).numel() != row.numel() for row in route_ids):
        raise ValueError("Top-K expert IDs must be unique within each token")
    scores = logits.float().sigmoid().gather(1, route_ids.long())
    selected = normalized_route_vjp(scores, weight_gradient, scale=scale)
    selected = selected * scores * (1. - scores)
    return torch.zeros(
        logits.shape, dtype=selected.dtype, device=selected.device,
    ).scatter(1, route_ids.long(), selected)


def clipped_swiglu(gate_up, *, limit=10.0):
    """DSV4.1 routed-expert activation before dynamic INT8 requantization."""
    if gate_up.ndim != 2 or gate_up.shape[1] % 2 or not limit > 0:
        raise ValueError("gate/up rows and a positive clamp limit required")
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


def clipped_swiglu_vjp(gate_up, output_gradient, *, limit=10.0):
    """Exact fixed-value VJP of the routed expert's clipped SwiGLU.

    The dynamic INT8 activation boundary uses a straight-through derivative,
    matching the frozen-W8A8 input-VJP contract used by online adaptation.
    """
    output = clipped_swiglu(gate_up, limit=limit)
    if output_gradient.shape != output.shape:
        raise ValueError("SwiGLU output gradient has the wrong shape")
    gate_raw, up_raw = gate_up.float().chunk(2, dim=-1)
    gate = gate_raw.clamp(max=limit)
    up = up_raw.clamp(-limit, limit)
    sigmoid = gate.sigmoid()
    silu = gate * sigmoid
    gradient = output_gradient.float()
    gate_gradient = gradient * up * sigmoid * (1 + gate * (1 - sigmoid))
    up_gradient = gradient * silu
    gate_gradient = gate_gradient * (gate_raw <= limit)
    up_gradient = up_gradient * ((up_raw >= -limit) & (up_raw <= limit))
    return output, torch.cat((gate_gradient, up_gradient), dim=-1)


def frozen_expert_input_vjp(value, gate_up_weight, down_weight, output_gradient, *, limit=10.0):
    """Input-only VJP for one frozen routed expert.

    The two logical weights are dequantized views of the resident W8A8
    operands.  Only ``d(value)`` is returned; target weights stay frozen and
    therefore never acquire gradients or optimizer state.
    """
    if (
        value.ndim != 2 or gate_up_weight.ndim != 2 or down_weight.ndim != 2
        or gate_up_weight.shape[0] != value.shape[1]
        or gate_up_weight.shape[1] % 2
        or down_weight.shape != (gate_up_weight.shape[1] // 2, value.shape[1])
        or output_gradient.shape != value.shape
    ):
        raise ValueError("invalid frozen expert input-VJP tensors")
    gate_up = value.float() @ gate_up_weight.float()
    activation = clipped_swiglu(gate_up, limit=limit)
    output = activation @ down_weight.float()
    activation_gradient = output_gradient.float() @ down_weight.float().T
    _, gate_up_gradient = clipped_swiglu_vjp(gate_up, activation_gradient, limit=limit)
    input_gradient = gate_up_gradient @ gate_up_weight.float().T
    return output, input_gradient


def frozen_moe_input_vjp(
    value,
    route_ids,
    router_weight,
    expert_gate_up,
    expert_down,
    shared_gate_up,
    shared_down,
    output_gradient,
    *,
    route_scale=1.0,
    limit=10.0,
):
    """Reference input VJP for the complete fixed-route frozen MoE.

    This is the executable golden for the production EP16 sequence: routed
    experts, shared expert and sigmoid router-weight path all contribute to
    ``d(value)`` while every target-model weight remains frozen.
    """
    if (
        value.ndim != 2 or route_ids.ndim != 2 or route_ids.shape[0] != value.shape[0]
        or router_weight.ndim != 2 or router_weight.shape[0] != value.shape[1]
        or expert_gate_up.ndim != 3 or expert_down.ndim != 3
        or expert_gate_up.shape[0] != router_weight.shape[1]
        or expert_down.shape[0] != router_weight.shape[1]
        or output_gradient.shape != value.shape
    ):
        raise ValueError("invalid fixed-route frozen MoE tensors")
    logits = value.float() @ router_weight.float()
    scores = logits.sigmoid().gather(1, route_ids.long())
    weights = route_scale * scores / scores.sum(-1, keepdim=True)
    routed_outputs, routed_input_gradients = [], []
    for token, ids in enumerate(route_ids.tolist()):
        outputs, gradients = [], []
        for slot, expert in enumerate(ids):
            output, gradient = frozen_expert_input_vjp(
                value[token:token + 1], expert_gate_up[expert], expert_down[expert],
                output_gradient[token:token + 1] * weights[token, slot], limit=limit,
            )
            outputs.append(output)
            gradients.append(gradient)
        routed_outputs.append(torch.cat(outputs))
        routed_input_gradients.append(torch.stack(gradients).sum(0))
    routed_outputs = torch.stack(routed_outputs)
    routed_input_gradient = torch.cat(routed_input_gradients)
    shared_output, shared_input_gradient = frozen_expert_input_vjp(
        value, shared_gate_up, shared_down, output_gradient, limit=limit,
    )
    weight_gradient = torch.einsum("td,trd->tr", output_gradient.float(), routed_outputs.float())
    logits_gradient = router_logits_vjp(logits, route_ids, weight_gradient, scale=route_scale)
    router_input_gradient = logits_gradient @ router_weight.float().T
    output = shared_output.float() + torch.einsum(
        "tr,trd->td", weights.float(), routed_outputs.float(),
    )
    return output, routed_input_gradient + shared_input_gradient + router_input_gradient
