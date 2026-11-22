"""Explicit Torch golden for the Orthrus-DSV4.1 Attention VJP."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionVJP:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    sink: torch.Tensor

    def trainable_block(self, history_length):
        """Return only gradients belonging to temporary K32 rows."""
        if type(history_length) is not int or not 0 <= history_length <= self.key.shape[0]:
            raise ValueError("history length is outside the concatenated KV")
        return AttentionVJP(self.query, self.key[history_length:], self.value[history_length:], self.sink)


def attention_forward(query, key, value, sink, allowed, *, scale=None):
    """Reference forward with a zero-valued per-head sink slot.

    ``key/value`` concatenate frozen target history and the temporary draft
    block. ``allowed`` is [draft rows, KV rows]; the Orthrus block columns are
    all visible while disallowed history/padding columns are masked.
    """
    if (query.ndim != 3 or key.ndim != 2 or value.shape != key.shape or sink.ndim != 1
            or query.shape[1] != sink.numel() or query.shape[2] != key.shape[1]
            or allowed.shape != (query.shape[0], key.shape[0]) or allowed.dtype != torch.bool):
        raise ValueError("invalid Orthrus Attention tensors")
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.einsum("qhd,sd->qhs", query.float(), key.float()) * scale
    scores = scores.masked_fill(~allowed[:, None], -torch.inf)
    scores = torch.cat((scores, sink.float()[None, :, None].expand(query.shape[0], -1, -1)), dim=-1)
    probability = scores.softmax(-1)
    output = torch.einsum("qhs,sd->qhd", probability[..., :-1], value.float())
    return output, probability


def attention_vjp(query, key, value, sink, allowed, output_gradient, *, scale=None):
    """Exact VJP used as the correctness oracle for a future PyPTO kernel."""
    output, probability = attention_forward(query, key, value, sink, allowed, scale=scale)
    if output_gradient.shape != output.shape:
        raise ValueError("Attention output gradient shape differs from the forward output")
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    gradient = output_gradient.float()
    token_probability = probability[..., :-1]
    d_value = torch.einsum("qhs,qhd->sd", token_probability, gradient)
    d_probability = torch.einsum("qhd,sd->qhs", gradient, value.float())
    d_probability = torch.cat((d_probability, torch.zeros_like(probability[..., -1:])), dim=-1)
    d_score = probability * (d_probability - (probability * d_probability).sum(-1, keepdim=True))
    d_token_score = d_score[..., :-1].masked_fill(~allowed[:, None], 0)
    d_query = torch.einsum("qhs,sd->qhd", d_token_score, key.float()) * scale
    d_key = torch.einsum("qhs,qhd->sd", d_token_score, query.float()) * scale
    d_sink = d_score[..., -1].sum(0)
    return output, AttentionVJP(d_query, d_key, d_value, d_sink)
