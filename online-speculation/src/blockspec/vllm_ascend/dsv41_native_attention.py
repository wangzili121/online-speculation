"""Native Ascend attention over read-only V4.1 history and an Orthrus block."""

from __future__ import annotations

import re

import torch

from .dsv41_attention import PagedRows
from .dsv4_hybrid import tail_logical_indices


def _paged_values(pages: PagedRows) -> torch.Tensor:
    values = pages.values
    if values.ndim == 3:
        values = values.unsqueeze(2)
    if values.ndim != 4 or values.shape[2] != 1:
        raise ValueError("native V4.1 cache must be [pages,page,1,head_dim]")
    return values.contiguous()


def _normalize_lse(
    lse: torch.Tensor, batch: int, block: int, heads: int
) -> torch.Tensor:
    if lse.shape == (1, batch * block, heads):
        return lse.squeeze(0).view(batch, block, heads, 1)
    if lse.shape == (heads, batch * block, 1):
        return lse.permute(1, 0, 2).view(batch, block, heads, 1)
    if lse.shape == (batch, heads, block, 1):
        return lse.transpose(1, 2)
    if lse.shape == (batch, block, heads, 1):
        return lse
    raise RuntimeError(f"unexpected SparseFlashMLA LSE shape: {tuple(lse.shape)}")


def _runtime_topk_length_supported() -> bool:
    """Account for the A2 SparseFlashMLA ABI change in torch-npu post7."""
    import torch_npu

    match = re.search(r"\.post(\d+)", torch_npu.__version__)
    return match is None or int(match.group(1)) >= 7


def _native_sparse_plane(
    query: torch.Tensor,
    pages: PagedRows,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sink: torch.Tensor,
    *,
    maximum_length: int | None = None,
    runtime_cmp_ratio: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one explicit sparse history plane without changing its cache."""
    batch, block, heads, dim = query.shape
    if indices.ndim == 2 and indices.shape[0] == batch:
        sparse = indices[:, None, :].expand(-1, block, -1)
    elif indices.ndim == 3 and indices.shape[:2] == (batch, block):
        sparse = indices
    else:
        raise ValueError("history indices must be [B,N] or [B,K,N]")
    if lengths.shape != (batch,) or sink.shape != (heads,):
        raise ValueError("history lengths or attention sink have incompatible shape")
    if pages.table.shape[0] != batch:
        raise ValueError("page table and query batch differ")
    sparse = sparse.reshape(batch * block, 1, -1).contiguous().int()
    visible = (sparse >= 0).sum(-1, dtype=torch.int32)
    maximum_length = (
        int(lengths.max().item()) if maximum_length is None else maximum_length
    )
    if maximum_length < 0:
        raise ValueError("maximum history length must be non-negative")
    if not _runtime_topk_length_supported():
        raise RuntimeError(
            "native Orthrus sparse history requires the post7 SparseFlashMLA ABI"
        )
    if runtime_cmp_ratio not in (0, 1):
        raise ValueError("SparseFlashMLA runtime cmp_ratio must be zero or one")
    cu = torch.arange(batch + 1, dtype=torch.int32, device=query.device) * block
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        heads,
        1,
        dim,
        cu_seqlens_q=cu,
        seqused_ori_kv=lengths.int(),
        ori_topk_length=visible,
        batch_size=batch,
        max_seqlen_q=block,
        max_seqlen_ori_kv=maximum_length,
        ori_topk=sparse.shape[-1],
        cmp_topk=0,
        cmp_ratio=0,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=0,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    output, lse = torch.ops._C_ascend.npu_sparse_flash_mla(
        query.flatten(0, 1),
        ori_kv=_paged_values(pages),
        ori_sparse_indices=sparse,
        ori_block_table=pages.table.int(),
        cu_seqlens_q=cu,
        seqused_ori_kv=lengths.int(),
        sinks=sink,
        metadata=metadata,
        softmax_scale=dim**-0.5,
        cmp_ratio=runtime_cmp_ratio,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=0,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=True,
        ori_topk_length=visible,
    )
    output = output.view_as(query)
    lse = _normalize_lse(lse, batch, block, heads)
    has_rows = (visible > 0).view(batch, block, 1, 1)
    output = torch.where(has_rows, output, torch.zeros_like(output))
    lse = torch.where(
        has_rows, lse, sink.view(1, 1, heads, 1).expand_as(lse)
    )
    return output.masked_fill(~torch.isfinite(lse), 0), lse


def _temporary_block(
    query: torch.Tensor, temporary: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fully visible K-by-K draft block with native fused attention."""
    import torch_npu

    batch, block, heads, dim = query.shape
    result = torch_npu.npu_fused_infer_attention_score(
        query,
        temporary.unsqueeze(2),
        temporary.unsqueeze(2),
        num_heads=heads,
        num_key_value_heads=1,
        input_layout="BSND",
        sparse_mode=0,
        scale=dim**-0.5,
        actual_seq_lengths=[block] * batch,
        actual_seq_lengths_kv=[block] * batch,
        softmax_lse_flag=True,
    )
    output, lse = result[:2]
    if lse.shape == (batch, heads, block, 1):
        lse = lse.transpose(1, 2)
    if lse.shape != (batch, block, heads, 1):
        raise RuntimeError(f"unexpected FIA LSE shape: {tuple(lse.shape)}")
    return output, lse


def _merge_fp32(
    outputs: tuple[torch.Tensor, ...], lses: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack([lse.float() for lse in lses])
    merged_lse = stacked.logsumexp(0)
    weights = (stacked - merged_lse.unsqueeze(0)).exp()
    output = sum(
        value.float() * weight
        for value, weight in zip(outputs, weights, strict=True)
    )
    return output, merged_lse


def native_draft_attention(
    query: torch.Tensor,
    temporary: torch.Tensor,
    swa: PagedRows,
    anchors: torch.Tensor,
    sink: torch.Tensor,
    *,
    window: int,
    global_kv: PagedRows | None = None,
    global_indices: torch.Tensor | None = None,
    global_lengths: torch.Tensor | None = None,
    max_swa_history: int | None = None,
    max_global_history: int | None = None,
    runtime_cmp_ratio: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse each history plane, then normalize it with the temporary block.

    Every draft query reads the same anchor-bounded SWA prefix. Compressed
    Top-K rows may differ by query. Target pages are inputs only; temporary KV
    never enters the target cache. The returned output and LSE stay FP32 so a
    caller can choose the cast point before its output projection.
    """
    batch, block, heads, dim = query.shape
    if temporary.shape != (batch, block, dim) or anchors.shape != (batch,):
        raise ValueError("query, temporary KV and anchors have incompatible shape")
    if not 1 <= block <= 32 or window <= 0:
        raise ValueError("draft block or SWA window is invalid")
    if (global_kv is None) != (global_indices is None):
        raise ValueError("global cache and indices must be supplied together")
    if global_kv is not None and global_lengths is None:
        raise ValueError("global cache requires committed compressed lengths")

    local_indices = tail_logical_indices(anchors, window)
    local = _native_sparse_plane(
        query,
        swa,
        local_indices,
        anchors,
        sink,
        maximum_length=max_swa_history,
        runtime_cmp_ratio=runtime_cmp_ratio,
    )
    outputs = [local[0]]
    lses = [local[1]]
    if global_kv is not None and global_indices.shape[-1]:
        disabled_sink = torch.full_like(sink, -torch.inf)
        sparse = _native_sparse_plane(
            query,
            global_kv,
            global_indices,
            global_lengths,
            disabled_sink,
            maximum_length=max_global_history,
            runtime_cmp_ratio=runtime_cmp_ratio,
        )
        outputs.append(sparse[0])
        lses.append(sparse[1])
    local_block = _temporary_block(query, temporary)
    outputs.append(local_block[0])
    lses.append(local_block[1])
    return _merge_fp32(tuple(outputs), tuple(lses))
