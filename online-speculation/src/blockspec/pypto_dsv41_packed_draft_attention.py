"""Read-only draft attention over the native packed cache."""

from .pypto_dsv41_packed_indexer import make_packed_draft_indexer
from .pypto_dsv41_packed_swa import make_packed_compressed_chain, make_packed_swa_chain
from .pypto_dsv41_sparse import TOPK


def make_packed_indexed_attention(window_pages, compressed_pages, bucket, ratio, block_size, *, table_bucket=None):
    """Keep selection on device for this layer and subsequent Reuse layers."""
    import pypto.language as pl

    index_entry = make_packed_draft_indexer(compressed_pages, bucket, ratio, block_size, table_bucket=table_bucket)
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


def make_packed_draft_projection(block_size):
    """Bind the native mHC normalization and BF16 draft projections."""
    from dataclasses import asdict

    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.hc import hc_head_norm

    from .pypto_dsv41_draft import make_packed_draft_qkv

    capacity = max(16, block_size)
    hidden, dim, latent = FLASH.hidden_size, FLASH.head_dim, FLASH.q_lora_rank
    heads = FLASH.num_attention_heads // TP_SIZE
    width = heads * dim
    mix = FLASH.hc_mult
    groups, output_rank = FLASH.o_groups // TP_SIZE, FLASH.o_lora_rank
    qkv_entry = make_packed_draft_qkv(asdict(FLASH), tp=TP_SIZE, capacity=capacity)
    project = pl.jit.inline(auto_scope=False)(qkv_entry._func)

    @pl.jit.inline(auto_scope=False)
    def prepare(
        x_hc: pl.Tensor, pre_mix: pl.Tensor, norm: pl.Tensor,
        wq_a_kv: pl.Tensor, q_gamma: pl.Tensor, wq_b: pl.Tensor, kv_gamma: pl.Tensor,
        cosine: pl.Tensor, sine: pl.Tensor,
    ):
        x = pl.create_tensor([capacity, hidden], dtype=pl.BF16)
        hc_head_norm(x_hc, pre_mix, norm, x, one_token=False)
        low_rank = pl.create_tensor([capacity, latent], dtype=pl.BF16)
        query = pl.create_tensor([capacity, width], dtype=pl.BF16)
        temporary = pl.create_tensor([capacity, dim], dtype=pl.BF16)
        low_rank, query, temporary = project(
            x, wq_a_kv, q_gamma, wq_b, kv_gamma, cosine, sine, low_rank, query, temporary, block_size,
        )
        return x, low_rank, pl.slice(query, [block_size, width], [0, 0]), temporary

    return prepare, (capacity, mix, hidden, heads, dim, width, groups, output_rank)


def _draft_projection(block_size, *, head_major=False, context_only=False):
    import pypto.language as pl

    from .pypto_dsv41_draft import make_packed_draft_output

    prepare, shapes = make_packed_draft_projection(block_size)
    _, _, hidden, heads, dim, _, _, _ = shapes
    if context_only:
        @pl.jit.inline(auto_scope=False)
        def use_output(output: pl.Tensor[[block_size * heads, dim], pl.FP32]):
            return output

        @pl.jit.inline(auto_scope=False)
        def keep_output(context: pl.Tensor, wo_a: pl.Tensor, wo_b: pl.Tensor,
                        cosine: pl.Tensor, sine: pl.Tensor, output: pl.Tensor[[block_size * heads, dim], pl.FP32]):
            return output

        return prepare, use_output, keep_output, shapes, (block_size * heads, dim), pl.FP32

    @pl.jit.inline(auto_scope=False)
    def scratch(output: pl.Tensor[[block_size, hidden], pl.BF16]):
        return pl.create_tensor([block_size * heads, dim], dtype=pl.FP32)

    wo_entry = make_packed_draft_output(block_size=block_size, head_major=head_major)
    project_output = pl.jit.inline(auto_scope=False)(wo_entry._func)

    @pl.jit.inline(auto_scope=False)
    def finish(context: pl.Tensor, wo_a: pl.Tensor, wo_b: pl.Tensor,
               cosine: pl.Tensor, sine: pl.Tensor, contribution: pl.Tensor):
        cos_rows = pl.slice(cosine, [block_size, 32], [0, 0])
        sin_rows = pl.slice(sine, [block_size, 32], [0, 0])
        return project_output(context, wo_a, wo_b, cos_rows, sin_rows, contribution)

    return prepare, scratch, finish, shapes, (block_size, hidden), pl.BF16


def make_packed_scored_draft_layer(window_pages, compressed_pages, bucket, ratio, block_size, *,
                                  table_bucket=None, context_only=False):
    """Emit the local TP contribution and retained selection for a scored layer."""
    import pypto.language as pl

    prepare, attention_buffer, finish, shape, output_shape, output_dtype = _draft_projection(block_size, context_only=context_only)
    capacity, mix, hidden, heads, dim, width, groups, output_rank = shape
    attention = make_packed_indexed_attention(
        window_pages, compressed_pages, bucket, ratio, block_size, table_bucket=table_bucket,
    )
    attend = pl.jit.inline(auto_scope=False)(attention._func)

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
        contribution: pl.Out[pl.Tensor[output_shape, output_dtype]],
    ):
        x, low_rank, query_rows, temporary = prepare(
            x_hc, pre_mix, norm, wq_a_kv, q_gamma, wq_b, kv_gamma, cosine, sine,
        )
        context = attention_buffer(contribution)
        lse = pl.create_tensor([block_size * heads, 1], dtype=pl.FP32)
        rows, lengths, bias, context, lse = attend(
            x, low_rank, idx_wq, idx_weight, cosine, sine, index_keys, index_scales,
            index_table, request_ids, visible, window, window_scales, window_lookup, window_rows,
            window_starts, spans, window_bias, compressed, compressed_scales, compressed_lookup,
            query_rows, temporary, sink, zeros, rows, lengths, bias, context, lse,
        )
        return rows, lengths, bias, finish(context, wo_a, wo_b, cosine, sine, contribution)

    return layer


def make_packed_swa_draft_layer(window_pages, block_size, *, context_only=False):
    """Project a draft block against only the local history and temporary KV."""
    import pypto.language as pl

    prepare, attention_buffer, finish, shape, output_shape, output_dtype = _draft_projection(block_size, head_major=True, context_only=context_only)
    capacity, mix, hidden, heads, dim, width, groups, output_rank = shape
    attention = make_packed_swa_chain(window_pages, heads, block_size)
    attend = pl.jit.inline(auto_scope=False)(attention._func)

    @pl.jit
    def layer(
        x_hc: pl.Tensor[[capacity, mix, hidden], pl.FP32],
        pre_mix: pl.Tensor[[capacity, mix], pl.FP32],
        norm: pl.Tensor[[hidden], pl.BF16],
        wq_a_kv: pl.Tensor, q_gamma: pl.Tensor, wq_b: pl.Tensor, kv_gamma: pl.Tensor,
        cosine: pl.Tensor, sine: pl.Tensor,
        window: pl.Tensor, window_scales: pl.Tensor, window_lookup: pl.Tensor,
        window_rows: pl.Tensor, window_starts: pl.Tensor, spans: pl.Tensor, window_bias: pl.Tensor,
        sink: pl.Tensor, zeros: pl.Tensor,
        wo_a: pl.Tensor[[groups, width // groups, output_rank], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[groups * output_rank, hidden], pl.BF16, pl.NZ],
        contribution: pl.Out[pl.Tensor[output_shape, output_dtype]],
    ):
        normalized, low_rank, query, temporary = prepare(
            x_hc, pre_mix, norm, wq_a_kv, q_gamma, wq_b, kv_gamma, cosine, sine,
        )
        context = attention_buffer(contribution)
        lse = pl.create_tensor([block_size * heads, 1], dtype=pl.FP32)
        context, lse = attend(window, window_scales, window_lookup, window_rows, window_starts,
                              spans, window_bias, query, temporary, sink, zeros, context, lse)
        return finish(context, wo_a, wo_b, cosine, sine, contribution)

    return layer


def make_packed_reuse_draft_layer(window_pages, compressed_pages, block_size, *, context_only=False):
    """Read retained Top-K rows with this layer's own queries and draft KV."""
    import pypto.language as pl

    prepare, attention_buffer, finish, shape, output_shape, output_dtype = _draft_projection(block_size, context_only=context_only)
    capacity, mix, hidden, heads, dim, width, groups, output_rank = shape
    attention = make_packed_compressed_chain(window_pages, compressed_pages, block_size)
    attend = pl.jit.inline(auto_scope=False)(attention._func)

    @pl.jit
    def layer(
        x_hc: pl.Tensor[[capacity, mix, hidden], pl.FP32],
        pre_mix: pl.Tensor[[capacity, mix], pl.FP32],
        norm: pl.Tensor[[hidden], pl.BF16],
        wq_a_kv: pl.Tensor, q_gamma: pl.Tensor, wq_b: pl.Tensor, kv_gamma: pl.Tensor,
        cosine: pl.Tensor, sine: pl.Tensor,
        window: pl.Tensor, window_scales: pl.Tensor, window_lookup: pl.Tensor,
        window_rows: pl.Tensor, window_starts: pl.Tensor, spans: pl.Tensor, window_bias: pl.Tensor,
        compressed: pl.Tensor, compressed_scales: pl.Tensor, compressed_lookup: pl.Tensor,
        rows: pl.Tensor[[block_size, TOPK], pl.INT32],
        lengths: pl.Tensor[[block_size], pl.INT32],
        bias: pl.Tensor[[block_size, TOPK], pl.FP32],
        sink: pl.Tensor, zeros: pl.Tensor,
        wo_a: pl.Tensor[[groups, width // groups, output_rank], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[groups * output_rank, hidden], pl.BF16, pl.NZ],
        contribution: pl.Out[pl.Tensor[output_shape, output_dtype]],
    ):
        normalized, low_rank, query, temporary = prepare(
            x_hc, pre_mix, norm, wq_a_kv, q_gamma, wq_b, kv_gamma, cosine, sine,
        )
        context = attention_buffer(contribution)
        lse = pl.create_tensor([block_size * heads, 1], dtype=pl.FP32)
        context, lse = attend(
            window, window_scales, window_lookup, window_rows, window_starts, spans, window_bias,
            compressed, compressed_scales, compressed_lookup, rows, lengths, bias,
            query, temporary, sink, zeros, context, lse,
        )
        return finish(context, wo_a, wo_b, cosine, sine, contribution)

    return layer


def make_packed_draft_tp_post(block_size):
    """Use the native TP4 and mHC post path after a draft Attention projection."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.hc import hc_decode_mixes
    from models.deepseek_v4_1_flash_a3.o_proj import decode_tp_output_post

    hidden, mix = FLASH.hidden_size, FLASH.hc_mult

    @pl.jit.inline(auto_scope=False)
    def draft_tp_post(
        x_hc: pl.Tensor, contribution: pl.Tensor,
        hc_fn: pl.Tensor, hc_scale: pl.Tensor, hc_base: pl.Tensor,
        output_window: pld.DistributedTensor,
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], attention_epoch: pl.Scalar[pl.INT32],
        attention_hidden: pl.Tensor, attention_pre_mix: pl.Tensor, ffn_input: pl.Tensor,
    ):
        # K8 QKV has a 16-row carrier. Publish and mix only the draft rows.
        residual = pl.slice(x_hc, [block_size, mix, hidden], [0, 0, 0])
        post = pl.create_tensor([block_size, mix], dtype=pl.FP32)
        residual_mix = pl.create_tensor([block_size, mix, mix], dtype=pl.FP32)
        hc_decode_mixes(
            residual, hc_fn, hc_scale, hc_base, attention_pre_mix, post, residual_mix, block_size,
        )
        decode_tp_output_post(
            contribution, output_window, output_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE,
            block_size, attention_epoch, residual, post, residual_mix, attention_pre_mix,
            attention_hidden, ffn_input,
        )
        return attention_hidden, attention_pre_mix, ffn_input

    return draft_tp_post
