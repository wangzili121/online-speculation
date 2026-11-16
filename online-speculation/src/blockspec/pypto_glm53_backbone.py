"""Bind the rank-local Orthrus attention to GLM's resident residual stream."""

from .pypto_glm53_draft_attention import make_draft_attention
from .pypto_glm53_head import head_layout
from .pypto_glm53_mla import HEADS, LATENT, QUERY_ROWS, TILE, VALUE


def make_backbone_mla(capacity, table_width):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.all_reduce import tp_all_reduce
    from models.glm5_3_flash.config import D, EP_SIZE, INDEX_DIM, INDEX_H, Q_LORA
    from models.glm5_3_flash.mhc import mhc_mixes, mhc_post, mhc_pre
    from models.glm5_3_flash.rmsnorm import rmsnorm

    attend = make_draft_attention(inline=True)
    fields, packed_size = head_layout()
    width = HEADS * VALUE
    ((q_a_at, q_a_n, _), (q_norm_at, q_norm_n, _), (q_b_at, q_b_n, _), (kv_a_at, kv_a_n, _),
     (kv_norm_at, kv_norm_n, _), (key_at, key_n, _), (value_at, value_n, _), (out_at, out_n, _),
     (index_q_at, index_q_n, _), (index_gate_at, index_gate_n, _)) = fields.values()

    @pl.jit.inline(auto_scope=False)
    def draft_mla(
        stream: pl.Tensor, heads: pl.Tensor, ar_key: pl.Tensor, ar_value: pl.Tensor,
        residual_map: pl.Tensor, residual_scale: pl.Tensor, residual_base: pl.Tensor, norm: pl.Tensor,
        history: pl.Tensor, tables: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor, counts: pl.Tensor, tail_start: pl.Tensor, tail_count: pl.Tensor,
        prefix_length: pl.Tensor, request_ids: pl.Tensor,
        pre: pl.Tensor, post: pl.Tensor, residual: pl.Tensor, collapsed: pl.Tensor, normed: pl.Tensor,
        partial: pl.Tensor, reduced: pl.Tensor, result: pl.Tensor,
        tp_partial: pld.DistributedTensor[[capacity, D], pl.FP32],
        tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], mla_layer: pl.Scalar[pl.INT32],
    ):
        with pl.scope():
            mhc_mixes(stream, residual_map, residual_scale, residual_base, pre, post, residual)
            mhc_pre(stream, pre, collapsed)
            rmsnorm(collapsed, norm, normed)
            slab = pl.slice(heads, [1, packed_size], [pl.cast(mla_layer, pl.INDEX), 0])
            request_slot = pl.cast(pl.read(request_ids, [0]), pl.INDEX)
            page_row = pl.slice(tables, [1, table_width], [request_slot, 0])
            table = pl.reshape(page_row, [table_width])
            bias = pl.create_tensor([1, TILE], dtype=pl.FP32)
            for task in pl.spmd(1, name_hint="orthrus_glm53_block_mask"):
                lane = pl.arange(0, [1, TILE], dtype=pl.INT32)
                invalid = pl.minimum(pl.maximum(pl.sub(lane, QUERY_ROWS - 1), 0), 1)
                bias[task:task + 1, :] = pl.mul(pl.cast(invalid, pl.FP32), -3.4e38)
            x: pl.Tensor[[16, D], pl.BF16] = pl.slice(normed, [16, D], [0, 0])
            wqa_flat = pl.slice(slab, [1, q_a_n], [0, q_a_at])
            wqa: pl.Tensor[[Q_LORA, D], pl.BF16] = pl.reshape(wqa_flat, [Q_LORA, D])
            nqa_flat = pl.slice(slab, [1, q_norm_n], [0, q_norm_at])
            nqa: pl.Tensor[[Q_LORA], pl.BF16] = pl.reshape(nqa_flat, [Q_LORA])
            wqb_flat = pl.slice(slab, [1, q_b_n], [0, q_b_at])
            wqb: pl.Tensor[[width, Q_LORA], pl.BF16] = pl.reshape(wqb_flat, [width, Q_LORA])
            wkv_flat = pl.slice(slab, [1, kv_a_n], [0, kv_a_at])
            wkv: pl.Tensor[[LATENT, D], pl.BF16] = pl.reshape(wkv_flat, [LATENT, D])
            nkv_flat = pl.slice(slab, [1, kv_norm_n], [0, kv_norm_at])
            nkv: pl.Tensor[[LATENT], pl.BF16] = pl.reshape(nkv_flat, [LATENT])
            target_key: pl.Tensor[[width, LATENT], pl.BF16] = pl.reshape(ar_key, [width, LATENT])
            target_value: pl.Tensor[[width, LATENT], pl.BF16] = pl.reshape(ar_value, [width, LATENT])
            wk_flat = pl.slice(slab, [1, key_n], [0, key_at])
            wk: pl.Tensor[[width, LATENT], pl.BF16] = pl.reshape(wk_flat, [width, LATENT])
            wv_flat = pl.slice(slab, [1, value_n], [0, value_at])
            wv: pl.Tensor[[width, LATENT], pl.BF16] = pl.reshape(wv_flat, [width, LATENT])
            wo_flat = pl.slice(slab, [1, out_n], [0, out_at])
            wo: pl.Tensor[[D, width], pl.BF16] = pl.reshape(wo_flat, [D, width])
            wi_flat = pl.slice(slab, [1, index_q_n], [0, index_q_at])
            wi: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16] = pl.reshape(wi_flat, [Q_LORA, INDEX_H * INDEX_DIM])
            wg_flat = pl.slice(slab, [1, index_gate_n], [0, index_gate_at])
            wg: pl.Tensor[[D, INDEX_H], pl.BF16] = pl.reshape(wg_flat, [D, INDEX_H])
            active_partial: pl.Tensor[[16, D], pl.FP32] = pl.slice(partial, [16, D], [0, 0])
            query_starts = pl.slice(starts, [QUERY_ROWS], [0])
            query_counts = pl.slice(counts, [QUERY_ROWS], [0])
            query_tail_start = pl.slice(tail_start, [QUERY_ROWS], [0])
            query_tail_count = pl.slice(tail_count, [QUERY_ROWS], [0])
            query_prefix_length = pl.slice(prefix_length, [QUERY_ROWS], [0])
            attend(
                x, wqa, nqa, wqb, wkv, nkv, target_key, wk, target_value, wv, wo, wi, wg,
                history, table, pool_cache, pool_blocks, query_starts, query_counts,
                query_tail_start, query_tail_count, query_prefix_length, bias, active_partial,
            )
            tp_all_reduce(partial, tp_partial, tp_arrived, reduced,
                          pl.cast(QUERY_ROWS, pl.INT32), rank, 0, epoch)
            for row in pl.spmd(capacity - QUERY_ROWS, name_hint="orthrus_glm53_tp_pad"):
                offset = row + QUERY_ROWS
                reduced[offset:offset + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
            mhc_post(reduced, stream, post, residual, result)
        return result

    return draft_mla
