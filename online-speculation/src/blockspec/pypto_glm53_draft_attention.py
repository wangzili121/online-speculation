"""One TP8 GLM draft MLA sublayer with resident BF16 projections."""

from .pypto_glm53_mla import (
    HEADS, HEAD_TILE, LATENT, QUERY_ROWS, TILE, VALUE,
    make_indexed_history_metadata, make_indexed_mla_kernel, make_mla_kernel, make_selected_latent_reader,
)


def make_draft_attention(*, inline=False, training=False):
    """Return the rank-local attention before the backbone's existing TP reduction."""
    import pypto.language as pl
    from models.glm5_3_flash.config import D, INDEX_DIM, INDEX_H, Q_LORA
    from models.glm5_3_flash.indexer import WEIGHTS_SCALE
    from models.glm5_3_flash.rmsnorm import rmsnorm

    indexed_attention, _ = make_indexed_mla_kernel(inline=True)
    rows = 16
    q_width = HEADS * VALUE

    def projection(k_width, n_width, n_tile=64, *, transposed=True, keep_fp32=False):
        weight_first, weight_second = (n_width, k_width) if transposed else (k_width, n_width)
        out_type = pl.FP32 if keep_fp32 else pl.BF16

        @pl.jit.inline
        def project(x: pl.Tensor[[rows, k_width], pl.BF16], weight: pl.Tensor[[weight_first, weight_second], pl.BF16],
                    output: pl.Tensor[[rows, n_width], out_type]):
            for fragment in pl.spmd(n_width // n_tile, name_hint="orthrus_glm53_projection"):
                col = fragment * n_tile
                acc = pl.create_tensor([rows, n_tile], dtype=pl.FP32)
                for start in pl.pipeline(0, k_width, TILE, stage=2):
                    activation = pl.slice(x, [rows, TILE], [0, start])
                    if transposed:
                        nk = pl.slice(weight, [n_tile, TILE], [col, start])
                        acc = pl.matmul_acc(acc, activation, nk, b_trans=True, init_cond=start == 0)
                    else:
                        kn = pl.slice(weight, [TILE, n_tile], [start, col])
                        acc = pl.matmul_acc(acc, activation, kn, init_cond=start == 0)
                if keep_fp32:
                    output[:, col:col + n_tile] = acc
                else:
                    output[:, col:col + n_tile] = pl.cast(acc, pl.BF16, mode="rint")
            return output

        return project

    project_q_a = projection(D, Q_LORA)
    project_q_b = projection(Q_LORA, q_width)
    project_kv_a = projection(D, LATENT)
    project_index = projection(Q_LORA, INDEX_H * INDEX_DIM, transposed=False)
    project_gate = projection(D, INDEX_H, 32, transposed=False)
    project_out = projection(q_width, D, keep_fp32=True)

    @pl.jit.inline
    def absorb(q: pl.Tensor[[rows, q_width], pl.BF16], weight: pl.Tensor[[q_width, LATENT], pl.BF16],
               expanded: pl.Tensor[[QUERY_ROWS * HEAD_TILE, LATENT], pl.BF16]):
        for task in pl.spmd(HEADS * (LATENT // TILE), name_hint="orthrus_glm53_absorb"):
            head = task // (LATENT // TILE)
            col = task % (LATENT // TILE) * TILE
            q_part = pl.slice(q, [rows, VALUE], [0, head * VALUE])
            w_part = weight[head * VALUE:(head + 1) * VALUE, col:col + TILE]
            latent = pl.cast(pl.matmul(q_part, w_part, out_dtype=pl.FP32), pl.BF16, mode="rint")
            for token in pl.unroll(QUERY_ROWS):
                expanded_row = token * HEAD_TILE + head
                expanded[expanded_row:expanded_row + 1, col:col + TILE] = latent[token:token + 1, :]
        return expanded

    score_rows = QUERY_ROWS * HEAD_TILE

    @pl.jit.inline
    def prepare(
        x: pl.Tensor[[rows, D], pl.BF16], q_a_weight: pl.Tensor[[Q_LORA, D], pl.BF16],
        q_norm: pl.Tensor[[Q_LORA], pl.BF16], q_b_weight: pl.Tensor[[q_width, Q_LORA], pl.BF16],
        kv_a_weight: pl.Tensor[[LATENT, D], pl.BF16], kv_norm: pl.Tensor[[LATENT], pl.BF16],
        ar_key: pl.Tensor[[q_width, LATENT], pl.BF16], draft_key: pl.Tensor[[q_width, LATENT], pl.BF16],
        indexer_q_weight: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
        indexer_gate: pl.Tensor[[D, INDEX_H], pl.BF16],
        q_a: pl.Tensor[[rows, Q_LORA], pl.BF16], q_resid: pl.Tensor[[rows, Q_LORA], pl.BF16],
        query: pl.Tensor[[rows, q_width], pl.BF16], kv_a: pl.Tensor[[rows, LATENT], pl.BF16],
        q_history: pl.Tensor[[score_rows, LATENT], pl.BF16], q_draft: pl.Tensor[[score_rows, LATENT], pl.BF16],
        idx: pl.Tensor[[QUERY_ROWS, INDEX_H, INDEX_DIM], pl.BF16],
        weights: pl.Tensor[[QUERY_ROWS, INDEX_H], pl.FP32],
        padded_block: pl.Tensor[[TILE, LATENT], pl.BF16],
    ):
        draft_latent = pl.create_tensor([rows, LATENT], dtype=pl.BF16)
        project_q_a(x, q_a_weight, q_a)
        rmsnorm(q_a, q_norm, q_resid)
        project_q_b(q_resid, q_b_weight, query)
        project_kv_a(x, kv_a_weight, kv_a)
        rmsnorm(kv_a, kv_norm, draft_latent)

        for part in pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_q_pad"):
            row = part * HEAD_TILE + HEADS
            padding = pl.full([HEAD_TILE - HEADS, LATENT], dtype=pl.BF16, value=0.0)
            q_history[row:row + HEAD_TILE - HEADS, :] = padding
            q_draft[row:row + HEAD_TILE - HEADS, :] = padding
        absorb(query, ar_key, q_history)
        absorb(query, draft_key, q_draft)
        flat_index = pl.create_tensor([rows, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        gate = pl.create_tensor([rows, INDEX_H], dtype=pl.BF16)
        project_index(q_resid, indexer_q_weight, flat_index)
        project_gate(x, indexer_gate, gate)
        # The upstream scorer applies its scale to a BF16-rounded gate dot.
        for part in pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_gate_round"):
            weights[part:part + 1, :] = pl.mul(pl.cast(gate[part:part + 1, :], pl.FP32), WEIGHTS_SCALE)

        flat_idx = pl.reshape(idx, [QUERY_ROWS, INDEX_H * INDEX_DIM])
        for part in pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_index_pack"):
            flat_idx[part:part + 1, :] = flat_index[part:part + 1, :]
        for part in pl.spmd(TILE // rows, name_hint="orthrus_glm53_block_pad"):
            start = part * rows
            if part == 0:
                padded_block[start:start + rows, :] = draft_latent[0:rows, :]
            else:
                padded_block[start:start + rows, :] = pl.full([rows, LATENT], dtype=pl.BF16, value=0.0)
        return q_a, q_resid, query, kv_a, q_history, q_draft, idx, weights, padded_block

    @pl.jit.inline
    def round_context(
        context: pl.Tensor[[QUERY_ROWS * HEADS, VALUE], pl.FP32],
        projected_in: pl.Tensor[[rows, q_width], pl.BF16],
    ):
        flat_context = pl.reshape(context, [QUERY_ROWS, q_width])
        for part in pl.spmd(rows, name_hint="orthrus_glm53_context_pad"):
            if part < QUERY_ROWS:
                projected_in[part:part + 1, :] = pl.cast(flat_context[part:part + 1, :], pl.BF16, mode="rint")
            else:
                projected_in[part:part + 1, :] = pl.full([1, q_width], dtype=pl.BF16, value=0.0)
        return projected_in

    jit = pl.jit.inline if inline else pl.jit

    @jit
    def draft_attention(
        x: pl.Tensor[[rows, D], pl.BF16],
        q_a_weight: pl.Tensor[[Q_LORA, D], pl.BF16], q_norm: pl.Tensor[[Q_LORA], pl.BF16],
        q_b_weight: pl.Tensor[[q_width, Q_LORA], pl.BF16], kv_a_weight: pl.Tensor[[LATENT, D], pl.BF16],
        kv_norm: pl.Tensor[[LATENT], pl.BF16], ar_key: pl.Tensor[[q_width, LATENT], pl.BF16],
        draft_key: pl.Tensor[[q_width, LATENT], pl.BF16], ar_value: pl.Tensor[[q_width, LATENT], pl.BF16],
        draft_value: pl.Tensor[[q_width, LATENT], pl.BF16], out_weight: pl.Tensor[[D, q_width], pl.BF16],
        indexer_q_weight: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
        indexer_gate: pl.Tensor[[D, INDEX_H], pl.BF16],
        history: pl.Tensor, table: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor, pool_counts: pl.Tensor, tail_start: pl.Tensor, tail_count: pl.Tensor,
        prefix: pl.Tensor, block_bias: pl.Tensor[[1, TILE], pl.FP32], output: pl.Out[pl.Tensor[[rows, D], pl.FP32]],
    ):
        q_a = pl.create_tensor([rows, Q_LORA], dtype=pl.BF16)
        q_resid = pl.create_tensor([rows, Q_LORA], dtype=pl.BF16)
        query = pl.create_tensor([rows, q_width], dtype=pl.BF16)
        kv_a = pl.create_tensor([rows, LATENT], dtype=pl.BF16)
        q_history = pl.create_tensor([score_rows, LATENT], dtype=pl.BF16)
        q_draft = pl.create_tensor([score_rows, LATENT], dtype=pl.BF16)
        idx = pl.create_tensor([QUERY_ROWS, INDEX_H, INDEX_DIM], dtype=pl.BF16)
        weights = pl.create_tensor([QUERY_ROWS, INDEX_H], dtype=pl.FP32)
        padded_block = pl.create_tensor([TILE, LATENT], dtype=pl.BF16)
        _, _q_resid, _query, _kv_a, q_history, q_draft, idx, weights, padded_block = prepare(
            x, q_a_weight, q_norm, q_b_weight, kv_a_weight, kv_norm, ar_key, draft_key, indexer_q_weight, indexer_gate,
            q_a, q_resid, query, kv_a, q_history, q_draft, idx, weights, padded_block,
        )
        context = pl.create_tensor([QUERY_ROWS * HEADS, VALUE], dtype=pl.FP32)
        indexed_attention(q_history, q_draft, history, table, idx, weights, pool_cache, pool_blocks,
                          starts, pool_counts, tail_start, tail_count, prefix, padded_block, block_bias,
                          ar_value, draft_value, context)
        outgoing = pl.create_tensor([rows, q_width], dtype=pl.BF16)
        projected_in = round_context(context, outgoing)
        project_out(projected_in, out_weight, output)
        return output

    if not training:
        return draft_attention

    metadata, index_width, capacity = make_indexed_history_metadata()
    read_history = make_selected_latent_reader(capacity)
    attend_selected = make_mla_kernel(capacity, selected_input=True)

    @jit
    def draft_training_replay(
        x: pl.Tensor[[rows, D], pl.BF16],
        q_a_weight: pl.Tensor[[Q_LORA, D], pl.BF16], q_norm: pl.Tensor[[Q_LORA], pl.BF16],
        q_b_weight: pl.Tensor[[q_width, Q_LORA], pl.BF16], kv_a_weight: pl.Tensor[[LATENT, D], pl.BF16],
        kv_norm: pl.Tensor[[LATENT], pl.BF16], ar_key: pl.Tensor[[q_width, LATENT], pl.BF16],
        draft_key: pl.Tensor[[q_width, LATENT], pl.BF16], ar_value: pl.Tensor[[q_width, LATENT], pl.BF16],
        draft_value: pl.Tensor[[q_width, LATENT], pl.BF16],
        indexer_q_weight: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
        indexer_gate: pl.Tensor[[D, INDEX_H], pl.BF16],
        history: pl.Tensor, table: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor[[QUERY_ROWS], pl.INT32], pool_counts: pl.Tensor[[QUERY_ROWS], pl.INT32],
        tail_start: pl.Tensor[[QUERY_ROWS], pl.INT32], tail_count: pl.Tensor[[QUERY_ROWS], pl.INT32],
        prefix: pl.Tensor[[QUERY_ROWS], pl.INT32], block_bias: pl.Tensor[[1, TILE], pl.FP32],
        q_raw: pl.Out[pl.Tensor[[rows, Q_LORA], pl.BF16]], q_normed: pl.Out[pl.Tensor[[rows, Q_LORA], pl.BF16]],
        queries: pl.Out[pl.Tensor[[rows, q_width], pl.BF16]], kv_raw: pl.Out[pl.Tensor[[rows, LATENT], pl.BF16]],
        absorbed_history: pl.Out[pl.Tensor[[score_rows, LATENT], pl.BF16]],
        absorbed_block: pl.Out[pl.Tensor[[score_rows, LATENT], pl.BF16]],
        block: pl.Out[pl.Tensor[[TILE, LATENT], pl.BF16]],
        selected: pl.Out[pl.Tensor[[QUERY_ROWS * capacity, LATENT], pl.BF16]],
        bias: pl.Out[pl.Tensor[[QUERY_ROWS, capacity], pl.FP32]],
        context: pl.Out[pl.Tensor[[rows, q_width], pl.BF16]],
    ):
        index = pl.create_tensor([QUERY_ROWS, INDEX_H, INDEX_DIM], dtype=pl.BF16)
        gates = pl.create_tensor([QUERY_ROWS, INDEX_H], dtype=pl.FP32)
        scratch_q = pl.create_tensor([rows, Q_LORA], dtype=pl.BF16)
        scratch_norm = pl.create_tensor([rows, Q_LORA], dtype=pl.BF16)
        scratch_kv = pl.create_tensor([rows, LATENT], dtype=pl.BF16)
        qa, qn, _q, kraw, q_h, q_d, index_q, index_gate, block_kv = prepare(
            x, q_a_weight, q_norm, q_b_weight, kv_a_weight, kv_norm, ar_key, draft_key, indexer_q_weight, indexer_gate,
            scratch_q, scratch_norm, queries, scratch_kv, absorbed_history, absorbed_block, index, gates, block,
        )
        for stripe in pl.spmd(Q_LORA // TILE, name_hint="glm_mla_tape_query"):
            col = stripe * TILE
            q_raw[:, col:col + TILE] = pl.slice(qa, [rows, TILE], [0, col])
            q_normed[:, col:col + TILE] = pl.slice(qn, [rows, TILE], [0, col])
        for stripe in pl.spmd(LATENT // TILE, name_hint="glm_mla_tape_latent"):
            col = stripe * TILE
            kv_raw[:, col:col + TILE] = pl.slice(kraw, [rows, TILE], [0, col])
        ids = pl.create_tensor([QUERY_ROWS, index_width], dtype=pl.INT32)
        lengths = pl.create_tensor([QUERY_ROWS], dtype=pl.INT32)
        chosen, additive_bias, counts = metadata(index_q, index_gate, pool_cache, pool_blocks, starts, pool_counts,
                                                 tail_start, tail_count, prefix, ids, bias, lengths)
        latents = read_history(history, table, chosen, counts, selected)
        local = pl.create_tensor([QUERY_ROWS * HEADS, VALUE], dtype=pl.FP32)
        values = attend_selected(q_h, q_d, latents, additive_bias, counts, block_kv,
                                 block_bias, ar_value, draft_value, local)
        round_context(values, context)

    return draft_training_replay


def make_draft_attention_program():
    import pypto.language as pl

    attend = make_draft_attention()

    @pl.jit.host
    def entry(
        x: pl.Tensor, q_a_weight: pl.Tensor, q_norm: pl.Tensor, q_b_weight: pl.Tensor,
        kv_a_weight: pl.Tensor, kv_norm: pl.Tensor, ar_key: pl.Tensor, draft_key: pl.Tensor,
        ar_value: pl.Tensor, draft_value: pl.Tensor, out_weight: pl.Tensor,
        indexer_q_weight: pl.Tensor, indexer_gate: pl.Tensor,
        history: pl.Tensor, table: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor, pool_counts: pl.Tensor, tail_start: pl.Tensor, tail_count: pl.Tensor,
        prefix: pl.Tensor, block_bias: pl.Tensor, output: pl.Out[pl.Tensor],
    ):
        attend(x, q_a_weight, q_norm, q_b_weight, kv_a_weight, kv_norm, ar_key, draft_key, ar_value, draft_value,
               out_weight, indexer_q_weight, indexer_gate, history, table, pool_cache, pool_blocks,
               starts, pool_counts, tail_start, tail_count, prefix, block_bias, output, device=0)

    return entry


def make_draft_attention_replay_program():
    """Expose the precise rank-local activation tape used by the MLA adjoint."""
    import pypto.language as pl

    replay = make_draft_attention(training=True)

    @pl.jit.host
    def entry(
        x: pl.Tensor, q_a_weight: pl.Tensor, q_norm: pl.Tensor, q_b_weight: pl.Tensor,
        kv_a_weight: pl.Tensor, kv_norm: pl.Tensor, ar_key: pl.Tensor, draft_key: pl.Tensor,
        ar_value: pl.Tensor, draft_value: pl.Tensor, indexer_q_weight: pl.Tensor, indexer_gate: pl.Tensor,
        history: pl.Tensor, table: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor, pool_counts: pl.Tensor, tail_start: pl.Tensor, tail_count: pl.Tensor,
        prefix: pl.Tensor, block_bias: pl.Tensor,
        q_raw: pl.Out[pl.Tensor], q_normed: pl.Out[pl.Tensor], queries: pl.Out[pl.Tensor], kv_raw: pl.Out[pl.Tensor],
        absorbed_history: pl.Out[pl.Tensor], absorbed_block: pl.Out[pl.Tensor], block: pl.Out[pl.Tensor],
        selected: pl.Out[pl.Tensor], bias: pl.Out[pl.Tensor], context: pl.Out[pl.Tensor],
    ):
        replay(x, q_a_weight, q_norm, q_b_weight, kv_a_weight, kv_norm, ar_key, draft_key, ar_value, draft_value,
               indexer_q_weight, indexer_gate, history, table, pool_cache, pool_blocks, starts, pool_counts,
               tail_start, tail_count, prefix, block_bias, q_raw, q_normed, queries, kv_raw, absorbed_history,
               absorbed_block, block, selected, bias, context, device=0)

    return entry


def make_resident_mla_replay(api):
    """Replay rank-local draft MLA directly from the Serving cache geometry."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, INDEX_DIM, INDEX_H, Q_LORA, TOPK_INDEX_WIDTH

    replay = make_draft_attention(training=True)
    ranks, rows, local = api.RANKS, 16, HEADS * VALUE
    table_width, pool_pages = api.TABLE_MAX, api.POOL_PAGE_ROWS
    cache_rows, cache_width, pool_rows = api.LATENT_ROWS, api.MLA_CACHE_WIDTH, api.POOL_TABLE_ROWS
    selected_capacity = (TOPK_INDEX_WIDTH + TILE - 1) // TILE * TILE
    selected_rows = QUERY_ROWS * selected_capacity
    score_rows = QUERY_ROWS * HEAD_TILE

    @pl.jit.host
    def entry(
        x: pl.Tensor[[ranks, rows, D], pl.BF16],
        q_a: pl.Tensor[[ranks, Q_LORA, D], pl.BF16], q_norm: pl.Tensor[[ranks, Q_LORA], pl.BF16],
        q_b: pl.Tensor[[ranks, local, Q_LORA], pl.BF16], kv_a: pl.Tensor[[ranks, LATENT, D], pl.BF16],
        kv_norm: pl.Tensor[[ranks, LATENT], pl.BF16], ar_key: pl.Tensor[[ranks, local, LATENT], pl.BF16],
        key: pl.Tensor[[ranks, local, LATENT], pl.BF16], ar_value: pl.Tensor[[ranks, local, LATENT], pl.BF16],
        value: pl.Tensor[[ranks, local, LATENT], pl.BF16],
        index_q: pl.Tensor[[ranks, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
        index_gate: pl.Tensor[[ranks, D, INDEX_H], pl.BF16],
        cache: pl.Tensor[[ranks, cache_rows, cache_width], pl.INT8],
        table: pl.Tensor[[ranks, table_width], pl.INT32],
        pool: pl.Tensor[[ranks, pool_rows, INDEX_DIM], pl.BF16], pool_blocks: pl.Tensor[[ranks, pool_pages], pl.INT32],
        start: pl.Tensor[[ranks, QUERY_ROWS], pl.INT32], count: pl.Tensor[[ranks, QUERY_ROWS], pl.INT32],
        tail: pl.Tensor[[ranks, QUERY_ROWS], pl.INT32], tail_count: pl.Tensor[[ranks, QUERY_ROWS], pl.INT32],
        prefix: pl.Tensor[[ranks, QUERY_ROWS], pl.INT32], block_bias: pl.Tensor[[ranks, 1, TILE], pl.FP32],
        raw_q: pl.Out[pl.Tensor[[ranks, rows, Q_LORA], pl.BF16]],
        normed_q: pl.Out[pl.Tensor[[ranks, rows, Q_LORA], pl.BF16]],
        query: pl.Out[pl.Tensor[[ranks, rows, local], pl.BF16]],
        raw_kv: pl.Out[pl.Tensor[[ranks, rows, LATENT], pl.BF16]],
        query_history: pl.Out[pl.Tensor[[ranks, score_rows, LATENT], pl.BF16]],
        query_block: pl.Out[pl.Tensor[[ranks, score_rows, LATENT], pl.BF16]],
        block: pl.Out[pl.Tensor[[ranks, TILE, LATENT], pl.BF16]],
        selected: pl.Out[pl.Tensor[[ranks, selected_rows, LATENT], pl.BF16]],
        history_bias: pl.Out[pl.Tensor[[ranks, QUERY_ROWS, selected_capacity], pl.FP32]],
        context: pl.Out[pl.Tensor[[ranks, rows, local], pl.BF16]],
    ):
        for rank in pl.range(pld.world_size()):
            replay(x[rank], q_a[rank], q_norm[rank], q_b[rank], kv_a[rank],
                         kv_norm[rank], ar_key[rank], key[rank], ar_value[rank], value[rank], index_q[rank],
                         index_gate[rank], cache[rank], table[rank], pool[rank], pool_blocks[rank], start[rank],
                         count[rank], tail[rank], tail_count[rank], prefix[rank], block_bias[rank],
                         raw_q[rank], normed_q[rank], query[rank], raw_kv[rank], query_history[rank],
                         query_block[rank], block[rank], selected[rank], history_bias[rank], context[rank], device=rank)

    return entry
