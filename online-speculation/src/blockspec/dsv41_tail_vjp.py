"""Exact reference VJPs for the frozen DSV4.1 mHC/MoE suffix.

Only diffusion Attention owns trainable parameters.  Nevertheless, a last-N
update must propagate the loss through the frozen LM head, final norm, MoE,
and both mHC sites.  These routines are the Torch oracle for the corresponding
PyPTO leaves; frozen tensors receive no optimizer state.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HCPreState:
    """Forward values needed to differentiate one DSV4.1 mHC prelude."""

    normalized: torch.Tensor
    inverse_rms: torch.Tensor
    projection: torch.Tensor
    pre: torch.Tensor
    post: torch.Tensor
    sinkhorn: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class HCPreVJP:
    residual: torch.Tensor
    incoming: torch.Tensor
    projection: torch.Tensor


def _row_normalize(value, eps):
    return value / (value.sum(-1, keepdim=True) + eps)


def _column_normalize(value, eps):
    return value / (value.sum(-2, keepdim=True) + eps)


def hc_pre_forward(residual, incoming, function, scale, base, *, norm_eps, hc_eps, iterations):
    """Torch oracle matching the native mHC pre/Sinkhorn sequence."""
    if (
        residual.ndim != 3 or incoming.shape != residual.shape[:2]
        or function.shape[1] != residual.shape[1] * residual.shape[2]
        or scale.shape != (3,) or base.shape != (function.shape[0],)
        or iterations < 1
    ):
        raise ValueError("invalid DSV4.1 hc_pre tensors")
    rows, copies, hidden = residual.shape
    channels = (copies + 2) * copies
    if function.shape[0] != channels:
        raise ValueError("mHC function has the wrong channel count")
    flat = residual.float().reshape(rows, copies * hidden)
    inverse = (flat.square().mean(-1, keepdim=True) + norm_eps).rsqrt()
    normalized = flat * inverse
    projection = normalized @ function.float().T
    pre_logits, post_logits, combination_logits = projection.split(
        (copies, copies, copies * copies), dim=-1,
    )
    pre = torch.sigmoid(pre_logits * scale[0] + base[:copies].float()) + hc_eps
    post = 2 * torch.sigmoid(
        post_logits * scale[1] + base[copies:2 * copies].float(),
    )
    probability = torch.softmax(
        combination_logits.reshape(rows, copies, copies) * scale[2]
        + base[2 * copies:].float().reshape(copies, copies),
        dim=-1,
    ) + hc_eps
    states = [probability]
    probability = _column_normalize(probability, hc_eps)
    states.append(probability)
    for _ in range(iterations - 1):
        probability = _row_normalize(probability, hc_eps)
        states.append(probability)
        probability = _column_normalize(probability, hc_eps)
        states.append(probability)
    collapsed = torch.einsum("tc,tcd->td", incoming.float(), residual.float())
    return collapsed, HCPreState(normalized, inverse, projection, pre, post, tuple(states))


def _normalize_vjp(value, gradient, *, dim, eps):
    denominator = value.sum(dim, keepdim=True) + eps
    output = value / denominator
    return (gradient - (gradient * output).sum(dim, keepdim=True)) / denominator


def hc_pre_vjp(
    residual,
    incoming,
    function,
    scale,
    base,
    state,
    collapsed_gradient,
    pre_gradient,
    post_gradient,
    combination_gradient,
    *,
    norm_eps,
    hc_eps,
    iterations,
):
    """Exact input VJP; mHC target weights stay frozen and receive no gradient."""
    collapsed, reference = hc_pre_forward(
        residual, incoming, function, scale, base,
        norm_eps=norm_eps, hc_eps=hc_eps, iterations=iterations,
    )
    if (
        collapsed_gradient.shape != collapsed.shape
        or pre_gradient.shape != incoming.shape
        or post_gradient.shape != incoming.shape
        or combination_gradient.shape != state.sinkhorn[-1].shape
    ):
        raise ValueError("invalid DSV4.1 hc_pre output gradients")
    # Require the captured forward state to describe the exact replayed call.
    for actual, expected in (
        (state.normalized, reference.normalized),
        (state.inverse_rms, reference.inverse_rms),
        (state.projection, reference.projection),
        (state.pre, reference.pre),
        (state.post, reference.post),
    ):
        torch.testing.assert_close(actual, expected)

    rows, copies, hidden = residual.shape
    combination = combination_gradient.float()
    # Reverse every column/row normalization in the opposite order.
    cursor = len(state.sinkhorn) - 1
    for _ in range(iterations - 1):
        before_column = state.sinkhorn[cursor - 1]
        combination = _normalize_vjp(before_column, combination, dim=-2, eps=hc_eps)
        cursor -= 1
        before_row = state.sinkhorn[cursor - 1]
        combination = _normalize_vjp(before_row, combination, dim=-1, eps=hc_eps)
        cursor -= 1
    before_column = state.sinkhorn[0]
    combination = _normalize_vjp(before_column, combination, dim=-2, eps=hc_eps)
    # The first state is softmax(logits)+eps; the additive epsilon has unit VJP.
    probability = torch.softmax(
        state.projection[:, 2 * copies:].reshape(rows, copies, copies) * scale[2]
        + base[2 * copies:].float().reshape(copies, copies), dim=-1,
    )
    combination_logits_gradient = probability * (
        combination - (combination * probability).sum(-1, keepdim=True)
    ) * scale[2]

    pre_sigmoid = state.pre - hc_eps
    post_sigmoid = state.post * .5
    projection_gradient = torch.cat((
        pre_gradient.float() * pre_sigmoid * (1 - pre_sigmoid) * scale[0],
        post_gradient.float() * 2 * post_sigmoid * (1 - post_sigmoid) * scale[1],
        combination_logits_gradient.flatten(1),
    ), dim=-1)
    normalized_gradient = projection_gradient @ function.float()
    flat = residual.float().reshape(rows, copies * hidden)
    inverse = state.inverse_rms
    residual_from_projection = inverse * (
        normalized_gradient
        - flat * (normalized_gradient * flat).mean(-1, keepdim=True) * inverse.square()
    )
    residual_gradient = residual_from_projection.reshape_as(residual)
    residual_gradient = residual_gradient + incoming.float().unsqueeze(-1) * collapsed_gradient.float().unsqueeze(1)
    incoming_gradient = torch.einsum("tcd,td->tc", residual.float(), collapsed_gradient.float())
    return HCPreVJP(residual_gradient, incoming_gradient, projection_gradient)


@dataclass(frozen=True)
class HCPostVJP:
    value: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    combination: torch.Tensor


def hc_post(value, residual, post, combination):
    """DSV4.1 hyper-connection post mix in FP32 accumulation."""
    if (
        value.ndim != 2
        or residual.ndim != 3
        or post.shape != residual.shape[:2]
        or combination.shape != (*residual.shape[:2], residual.shape[1])
        or value.shape != (residual.shape[0], residual.shape[2])
    ):
        raise ValueError("invalid DSV4.1 hc_post tensors")
    mixed = torch.einsum("tij,tid->tjd", combination.float(), residual.float())
    return mixed + post.float().unsqueeze(-1) * value.float().unsqueeze(1)


def hc_post_vjp(value, residual, post, combination, output_gradient):
    """Closed-form VJP of ``hc_post``; used to validate device kernels."""
    output = hc_post(value, residual, post, combination)
    if output_gradient.shape != output.shape:
        raise ValueError("hc_post gradient shape differs from its output")
    dy = output_gradient.float()
    return output, HCPostVJP(
        value=torch.einsum("tjd,tj->td", dy, post.float()),
        residual=torch.einsum("tjd,tij->tid", dy, combination.float()),
        post=torch.einsum("tjd,td->tj", dy, value.float()),
        combination=torch.einsum("tid,tjd->tij", residual.float(), dy),
    )


@dataclass(frozen=True)
class FrozenTailContract:
    """The reverse order required between final logits and draft Attention."""

    stages: tuple[str, ...] = (
        "lm_head_input_vjp",
        "final_rmsnorm_input_vjp",
        "final_hc_collapse_vjp",
        "ffn_hc_post_vjp",
        "selected_moe_input_vjp",
        "ffn_rmsnorm_input_vjp",
        "ffn_hc_pre_vjp",
        "attention_hc_post_vjp",
    )

    def __post_init__(self):
        if len(self.stages) != len(set(self.stages)) or self.stages[-1] != "attention_hc_post_vjp":
            raise ValueError("invalid frozen DSV4.1 suffix reverse contract")
