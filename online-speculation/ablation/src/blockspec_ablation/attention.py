"""Grouped short-query attention with shared K/V and an explicit masked softmax."""

import math

import torch


ATTENTION_BACKENDS = ("sdpa", "grouped")
GROUPED_QUERY_LIMIT = 32


def grouped_attention(q, k, v, allowed):
    """Flatten each KV group's query heads into rows, then restore head order.

    q is [batch, query_heads, queries, dim]; k/v have shared KV heads. The
    boolean mask broadcasts to [batch, query_heads, queries, keys]. Empty
    attention rows return zeros and have zero gradients, as in SDPA.
    """
    batch, heads, length, dim = q.shape
    kv_heads, keys = k.shape[1:3]
    rows = (heads // kv_heads) * length
    scores = (q.reshape(batch, kv_heads, rows, dim) @ k.transpose(-1, -2)) / math.sqrt(dim)
    scores = scores.reshape(batch, heads, length, keys).masked_fill(~allowed, float("-inf"))
    # A constant zero row makes softmax well-defined for an empty visible set.
    # Masking its probabilities afterwards gives the zero output/gradient.
    scores = torch.where(allowed.any(-1, keepdim=True), scores, 0.)
    probabilities = scores.softmax(-1).masked_fill(~allowed, 0.)
    attended = probabilities.reshape(batch, kv_heads, rows, keys) @ v
    return attended.reshape(batch, heads, length, v.shape[-1])
