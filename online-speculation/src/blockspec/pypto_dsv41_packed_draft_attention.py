"""Native packed-cache draft attention for layers owning an Indexer."""

from .pypto_dsv41_packed_indexer import make_packed_draft_indexer
from .pypto_dsv41_packed_swa import make_packed_compressed_chain
from .pypto_dsv41_sparse import TOPK


def make_packed_indexed_attention(window_pages, compressed_pages, bucket, ratio, block_size):
    """Keep selection on device for this layer and subsequent Reuse layers."""
    import pypto.language as pl

    index_entry = make_packed_draft_indexer(compressed_pages, bucket, ratio, block_size)
    attention_entry = make_packed_compressed_chain(window_pages, compressed_pages, block_size)
    index = pl.jit.inline(auto_scope=False)(index_entry._func)
    attention = pl.jit.inline(auto_scope=False)(attention_entry._func)

    @pl.jit
    def attend(
        x: pl.Tensor, low_rank: pl.Tensor, idx_wq: pl.Tensor, idx_weight: pl.Tensor,
        cosine: pl.Tensor, sine: pl.Tensor, index_keys: pl.Tensor, index_scales: pl.Tensor,
        index_table: pl.Tensor, request_ids: pl.Tensor, visible: pl.Tensor,
        window: pl.Tensor, window_scales: pl.Tensor, window_lookup: pl.Tensor,
        window_rows: pl.Tensor, window_starts: pl.Tensor, spans: pl.Tensor, window_bias: pl.Tensor,
        compressed: pl.Tensor, compressed_scales: pl.Tensor, compressed_lookup: pl.Tensor,
        query: pl.Tensor, temporary: pl.Tensor, sink: pl.Tensor, zeros: pl.Tensor,
        rows: pl.Out[pl.Tensor[[block_size, TOPK], pl.INT32]],
        lengths: pl.Out[pl.Tensor[[block_size], pl.INT32]],
        bias: pl.Out[pl.Tensor[[block_size, TOPK], pl.FP32]],
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        rows, lengths, bias = index(x, low_rank, idx_wq, idx_weight, cosine, sine, index_keys, index_scales,
                                    index_table, request_ids, visible, rows, lengths, bias)
        output, lse = attention(
            window, window_scales, window_lookup, window_rows, window_starts, spans, window_bias,
            compressed, compressed_scales, compressed_lookup, rows, lengths, bias,
            query, temporary, sink, zeros, output, lse,
        )
        return rows, lengths, bias, output, lse

    return attend


def make_packed_scored_draft_layer(window_pages, compressed_pages, bucket, ratio, block_size):
    """Emit the local TP contribution and retained selection for a scored layer."""
    from dataclasses import asdict

    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.hc import hc_head_norm

    from .pypto_dsv41_draft import make_packed_draft_output, make_packed_draft_qkv

    capacity = max(16, block_size)
    hidden, dim, latent = FLASH.hidden_size, FLASH.head_dim, FLASH.q_lora_rank
    heads = FLASH.num_attention_heads // TP_SIZE
    width = heads * dim
    mix = FLASH.hc_mult
    groups, output_rank = FLASH.o_groups // TP_SIZE, FLASH.o_lora_rank
    qkv_entry = make_packed_draft_qkv(asdict(FLASH), tp=TP_SIZE, capacity=capacity)
    attention_entry = make_packed_indexed_attention(window_pages, compressed_pages, bucket, ratio, block_size)
    wo_entry = make_packed_draft_output(block_size=block_size)
    project = pl.jit.inline(auto_scope=False)(qkv_entry._func)
    attend = pl.jit.inline(auto_scope=False)(attention_entry._func)
    project_output = pl.jit.inline(auto_scope=False)(wo_entry._func)

    @pl.jit
    def layer(
        x_hc: pl.Tensor[[capacity, mix, hidden], pl.FP32],
        pre_mix: pl.Tensor[[capacity, mix], pl.FP32],
        norm: pl.Tensor[[hidden], pl.BF16],
        wq_a_kv: pl.Tensor, q_gamma: pl.Tensor, wq_b: pl.Tensor, kv_gamma: pl.Tensor,
        idx_wq: pl.Tensor, idx_weight: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor,
        index_keys: pl.Tensor, index_scales: pl.Tensor, index_table: pl.Tensor,
        request_ids: pl.Tensor, visible: pl.Tensor,
        window: pl.Tensor, window_scales: pl.Tensor, window_lookup: pl.Tensor,
        window_rows: pl.Tensor, window_starts: pl.Tensor, spans: pl.Tensor, window_bias: pl.Tensor,
        compressed: pl.Tensor, compressed_scales: pl.Tensor, compressed_lookup: pl.Tensor,
        sink: pl.Tensor, zeros: pl.Tensor,
        wo_a: pl.Tensor[[groups, width // groups, output_rank], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[groups * output_rank, hidden], pl.BF16, pl.NZ],
        rows: pl.Out[pl.Tensor[[block_size, TOPK], pl.INT32]],
        lengths: pl.Out[pl.Tensor[[block_size], pl.INT32]],
        bias: pl.Out[pl.Tensor[[block_size, TOPK], pl.FP32]],
        contribution: pl.Out[pl.Tensor[[block_size, hidden], pl.BF16]],
    ):
        x = pl.create_tensor([capacity, hidden], dtype=pl.BF16)
        hc_head_norm(x_hc, pre_mix, norm, x, one_token=False)
        low_rank = pl.create_tensor([capacity, latent], dtype=pl.BF16)
        query = pl.create_tensor([capacity, width], dtype=pl.BF16)
        temporary = pl.create_tensor([capacity, dim], dtype=pl.BF16)
        low_rank, query, temporary = project(
            x, wq_a_kv, q_gamma, wq_b, kv_gamma, cosine, sine, low_rank, query, temporary, block_size,
        )
        context = pl.create_tensor([block_size * heads, dim], dtype=pl.FP32)
        lse = pl.create_tensor([block_size * heads, 1], dtype=pl.FP32)
        query_rows = pl.slice(query, [block_size, width], [0, 0])
        rows, lengths, bias, context, lse = attend(
            x, low_rank, idx_wq, idx_weight, cosine, sine, index_keys, index_scales,
            index_table, request_ids, visible, window, window_scales, window_lookup, window_rows,
            window_starts, spans, window_bias, compressed, compressed_scales, compressed_lookup,
            query_rows, temporary, sink, zeros, rows, lengths, bias, context, lse,
        )
        cos_rows = pl.slice(cosine, [block_size, 32], [0, 0])
        sin_rows = pl.slice(sine, [block_size, 32], [0, 0])
        contribution = project_output(context, wo_a, wo_b, cos_rows, sin_rows, contribution)
        return rows, lengths, bias, contribution

    return layer
