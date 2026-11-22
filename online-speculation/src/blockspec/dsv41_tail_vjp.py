"""Exact reference VJPs for the frozen DSV4.1 mHC/MoE suffix.

Only diffusion Attention owns trainable parameters.  Nevertheless, a last-N
update must propagate the loss through the frozen LM head, final norm, MoE,
and both mHC sites.  These routines are the Torch oracle for the corresponding
PyPTO leaves; frozen tensors receive no optimizer state.
"""

from dataclasses import dataclass

import torch


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
