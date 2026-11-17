"""Read-only paged QLI fixture and comparison for the draft layer probe."""

import torch

from blockspec.pypto_dsv41_indexer import INDEX_DIM, INDEX_HEADS, index_controls
from blockspec.pypto_dsv41_sparse import TILE, TOPK, compressed_physical_rows
from blockspec.pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW
from blockspec.vllm_ascend.dsv41_attention import PagedRows


def fixture(config, anchor, ratio, *, max_context):
    gen = torch.Generator().manual_seed(751)
    visible = anchor // ratio
    page_rows = WINDOW // ratio
    num_pages, page_bytes = 19, 131072
    code_stride, scale_stride = page_bytes // INDEX_DIM, page_bytes // 2
    code_offset = 512 if ratio == 2 else 0
    scale_offset = 36864 if ratio == 2 else 8192
    codes = torch.full((num_pages * code_stride, INDEX_DIM), 127, dtype=torch.int8)
    scales = torch.full((num_pages * scale_stride, 1), float("nan"), dtype=torch.float16)
    kv = torch.full((num_pages * WINDOW, HEAD_DIM), float("nan"), dtype=torch.bfloat16)
    pages = torch.randperm(num_pages - 1, generator=gen) + 1
    logical = torch.arange(visible)
    physical = pages[logical // page_rows]
    row_offset = logical % page_rows
    codes[physical * code_stride + code_offset + row_offset] = torch.randint(
        -127, 128, (visible, INDEX_DIM), generator=gen, dtype=torch.int8)
    scales[physical * scale_stride + scale_offset + row_offset, 0] = (
        torch.rand(visible, generator=gen) * 0.005 + 0.001).half()
    kv[physical * WINDOW + row_offset] = torch.randn(visible, HEAD_DIM, generator=gen).bfloat16()

    def plane(value, stride, offset):
        return dict(handle=value, shape=(num_pages, page_rows, 1, value.shape[-1]), stride_rows=stride,
                    offset_rows=offset, pages=tuple(pages.tolist()))

    history = dict(role={"ratio": ratio}, committed_tokens=anchor, planes=dict(
        index=(plane(codes, code_stride, code_offset), plane(scales, scale_stride, scale_offset)),
        long=(plane(kv, WINDOW, 0),)))
    geometry, controls = index_controls(
        history, candidate_blocks=config["candidate_topk_blocks"],
        candidate_block_size=config["candidate_block_size"], max_context=max_context)
    bucket = geometry.bucket
    buffers = dict(controls, index_codes=codes, index_scales=scales,
                   zero_codes=torch.zeros((page_rows, INDEX_DIM), dtype=torch.int8))
    tensors = dict(
        idx_projected=((BLOCK, INDEX_HEADS * INDEX_DIM), torch.bfloat16),
        idx_raw_weights=((BLOCK, INDEX_HEADS), torch.bfloat16),
        idx_query=((BLOCK, INDEX_HEADS, INDEX_DIM), torch.bfloat16),
        idx_scaled_weights=((BLOCK, INDEX_HEADS), torch.float16),
        index_keys=((bucket, INDEX_DIM), torch.int8),
        key_scales=((bucket,), torch.float16),
        query_codes=((BLOCK, INDEX_HEADS, INDEX_DIM), torch.int8),
        query_scales=((BLOCK, INDEX_HEADS), torch.float16),
        index_scores=((BLOCK, bucket), torch.float32),
        logical_rows=((BLOCK, TOPK), torch.int32),
        physical_rows=((BLOCK, TOPK), torch.int32),
        lengths=((BLOCK,), torch.int32),
        bias=((BLOCK, TOPK), torch.float32),
    )
    buffers.update({name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in tensors.items()})
    zeros = torch.zeros((TILE, HEAD_DIM), dtype=torch.bfloat16)
    gold_kv = PagedRows(kv.view(num_pages, WINDOW, HEAD_DIM)[:, :page_rows], pages.int()[None])
    return geometry, buffers, tensors, kv, zeros, gold_kv


def check(native, actual, buffers, geometry, gold_kv, *, anchor):
    indexer = native("indexer")
    table, visible = buffers["page_table"], buffers["visible"]
    page_rows = WINDOW // geometry.ratio
    logical = torch.arange(visible)
    pages, offsets = table[logical // page_rows], logical % page_rows
    code_address = pages * geometry.code_stride + geometry.code_offset + offsets
    scale_address = pages * geometry.scale_stride + geometry.scale_offset + offsets
    keys = torch.zeros_like(actual["index_keys"])
    key_scales = torch.zeros_like(actual["key_scales"])
    keys[:visible] = buffers["index_codes"][code_address]
    key_scales[:visible] = buffers["index_scales"][scale_address, 0]
    assert torch.equal(keys, actual["index_keys"]) and torch.equal(key_scales, actual["key_scales"])

    query_codes, query_scales = indexer.decode_quantize_indexer_query_reference(actual["idx_query"])
    torch.testing.assert_close(actual["query_scales"], query_scales, rtol=0, atol=0)
    assert (actual["query_codes"].short() - query_codes.short()).abs().max() <= 1
    expected = dict(query=actual["query_codes"], query_scale=actual["query_scales"], key=keys,
                    key_scale=key_scales, weights=actual["idx_scaled_weights"], positions=buffers["positions"],
                    scores=torch.empty_like(actual["index_scores"]), indices=torch.empty_like(actual["logical_rows"]),
                    active=BLOCK)
    indexer.golden_indexer_select(expected, compress_ratio=geometry.ratio)
    torch.testing.assert_close(actual["index_scores"], expected["scores"], rtol=6e-5, atol=2e-7)
    chosen = actual["logical_rows"]
    assert torch.equal(chosen.sort(-1).values, expected["indices"].sort(-1).values)
    rows, lengths = compressed_physical_rows(
        chosen, table, anchor=anchor, ratio=geometry.ratio, physical_pages=gold_kv.values.shape[0],
        stride_rows=geometry.kv_stride, offset_rows=geometry.kv_offset)
    torch.testing.assert_close(actual["physical_rows"], rows, rtol=0, atol=0)
    torch.testing.assert_close(actual["lengths"], lengths, rtol=0, atol=0)
    assert torch.equal(actual["bias"] == 0, rows != -1)
    return dict(visible=visible, bucket=geometry.bucket, selected_per_query=min(visible, TOPK),
                sets_exact=BLOCK, max_score_error=float((actual["index_scores"] - expected["scores"]).abs().max()))
