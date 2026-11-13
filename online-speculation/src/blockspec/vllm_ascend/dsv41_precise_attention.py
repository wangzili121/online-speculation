"""FP32 draft attention over bounded SWA and selected global KV rows."""

from __future__ import annotations

import torch

from .dsv41_attention import PagedRows
from .dsv4_hybrid import tail_logical_indices


def precise_draft_attention(
    query: torch.Tensor,
    temporary: torch.Tensor,
    swa: PagedRows,
    anchors: torch.Tensor,
    sink: torch.Tensor,
    *,
    window: int,
    global_kv: PagedRows | None = None,
    global_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute output and LSE in FP32. Ascend callers must disable HF32 matmul.

    Local KV is shared across draft positions. Only the Top-K global rows
    differ by query. Both sources use one normalization including the sink.
    The caller supplies logical indices already bounded by the committed prefix.
    """
    if query.device.type == "npu" and torch.npu.matmul.allow_hf32:
        raise RuntimeError("precise draft attention requires HF32 matmul disabled")
    if (global_kv is None) != (global_indices is None):
        raise ValueError("global history requires both cache pages and indices")

    batch, block, heads, dim = query.shape
    local_ids = tail_logical_indices(anchors, window)
    local_kv = torch.cat((swa.gather(local_ids), temporary), dim=1).float()
    scale = dim ** -0.5
    local_q = query.float().reshape(batch, block * heads, dim)
    local_scores = torch.bmm(local_q, local_kv.transpose(1, 2)) * scale
    local_scores = local_scores.view(batch, block, heads, -1)
    local_scores[..., :window].masked_fill_(local_ids[:, None, None, :] < 0, -torch.inf)

    logits = [local_scores]
    sparse_kv = None
    if global_kv is not None and global_indices.shape[-1]:
        sparse_kv = global_kv.gather(global_indices).flatten(0, 1).float()
        sparse_q = local_q.view(batch * block, heads, dim)
        sparse_scores = torch.bmm(sparse_q, sparse_kv.transpose(1, 2)) * scale
        sparse_scores = sparse_scores.view(batch, block, heads, -1)
        logits.append(sparse_scores.masked_fill(global_indices[:, :, None, :] < 0, -torch.inf))
    logits.append(sink.float().view(1, 1, heads, 1).expand(batch, block, -1, -1))
    lse = torch.cat(logits, dim=-1).logsumexp(-1, keepdim=True)

    local_weights = (local_scores - lse).exp().view(batch, block * heads, -1)
    output = torch.bmm(local_weights, local_kv).view(batch, block, heads, dim)
    if sparse_kv is not None:
        sparse_weights = (logits[1] - lse).exp().flatten(0, 1)
        output = output + torch.bmm(sparse_weights, sparse_kv).view_as(output)
    return output, lse
