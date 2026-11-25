"""Resident VJP of SWA, the shared draft KV block, and per-token compressed history."""

from .pypto_dsv41_sparse import HEAD_TILE, PARTS, TILE, TOPK
from .pypto_dsv41_swa import HEAD_DIM, WINDOW


def make_packed_attention_vjp(block, *, world=1, tp=4):
    """Replay global probabilities and return rank-local FP32 Q, shared KV, and sink gradients."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    heads, dim = HEAD_TILE, HEAD_DIM
    rows = block * heads
    parts = 1 + PARTS
    selected_rows = block * TOPK
    scale = dim ** -0.5

    panel, panels = 128, dim // 128

    @pl.jit.inline(auto_scope=False)
    def history_panel(window: pl.Tensor[[WINDOW, dim], pl.BF16],
                      selected: pl.Tensor[[selected_rows, dim], pl.BF16],
                      token: pl.Scalar[pl.INT32], part: pl.Scalar[pl.INT32], column: pl.Scalar[pl.INT32]):
        values = pl.slice(window, [TILE, panel], [0, column])
        if part > 0:
            values = pl.slice(selected, [TILE, panel], [token * TOPK + (part - 1) * TILE, column])
        return values

    @pl.jit
    def reverse(
        query: pl.Tensor[[rows, dim], pl.BF16],
        temporary: pl.Tensor[[block, dim], pl.BF16],
        window: pl.Tensor[[WINDOW, dim], pl.BF16],
        selected: pl.Tensor[[selected_rows, dim], pl.BF16],
        window_bias: pl.Tensor[[1, WINDOW], pl.FP32],
        selected_bias: pl.Tensor[[block, TOPK], pl.FP32],
        lengths: pl.Tensor[[block], pl.INT32],
        context: pl.Tensor[[rows, dim], pl.FP32],
        lse: pl.Tensor[[rows, 1], pl.FP32],
        sink: pl.Tensor[[1, heads], pl.FP32],
        incoming: pl.Tensor[[rows, dim], pl.FP32],
        query_gradient: pl.Out[pl.Tensor[[rows, dim], pl.FP32]],
        temporary_gradient: pl.Out[pl.Tensor[[block, dim], pl.FP32]],
        sink_gradient: pl.Out[pl.Tensor[[1, heads], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        # The softmax contraction sum(P * dP) equals dot(dContext, Context).
        # Use the saved Attention output so the replay never normalizes per segment.
        grad_hi = pl.create_tensor([rows, dim], dtype=pl.BF16)
        grad_lo = pl.create_tensor([rows, dim], dtype=pl.BF16)
        dot = pl.create_tensor([rows, 1], dtype=pl.FP32)
        with pl.spmd(block, name_hint="orthrus_packed_attention_cotangent") as seed_ready:
            seed_token = pl.tile.get_block_idx()
            seed_start = seed_token * heads
            g = pl.full([heads, dim], dtype=pl.FP32, value=0.)
            if seed_token < active:
                g = pl.slice(incoming, [heads, dim], [seed_start, 0])
            high = pl.cast(g, pl.BF16, mode="rint")
            low = pl.cast(pl.sub(g, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
            grad_hi[seed_start:seed_start + heads, :] = high
            grad_lo[seed_start:seed_start + heads, :] = low
            out = pl.slice(context, [heads, dim], [seed_start, 0])
            dot[seed_start:seed_start + heads, :] = pl.row_sum(pl.mul(g, out))

        history_score = pl.create_tensor([rows * parts, TILE], dtype=pl.FP32)
        history_inner = pl.create_tensor([rows * parts, TILE], dtype=pl.FP32)
        with pl.spmd(active * parts, name_hint="orthrus_packed_attention_history_scores",
                     deps=[seed_ready]) as scores_ready:
            score_task = pl.tile.get_block_idx()
            score_token, score_part = score_task // parts, score_task % parts
            score_length = pl.read(lengths, [score_token])
            score_start, score_dest = score_token * heads, score_task * heads
            if score_part == 0 or (score_part - 1) * TILE < score_length:
                logits = pl.create_tensor([heads, TILE], dtype=pl.FP32)
                probability_gradient = pl.create_tensor([heads, TILE], dtype=pl.FP32)
                for panel_id in pl.range(panels):
                    column = panel_id * panel
                    key_tile = history_panel(window, selected, score_token, score_part, column)
                    query_tile = pl.slice(query, [heads, panel], [score_start, column])
                    incoming_hi = pl.slice(grad_hi, [heads, panel], [score_start, column])
                    incoming_lo = pl.slice(grad_lo, [heads, panel], [score_start, column])
                    logits = pl.matmul_acc(logits, query_tile, key_tile, b_trans=True, init_cond=(panel_id == 0))
                    probability_gradient = pl.matmul_acc(probability_gradient, incoming_hi, key_tile, b_trans=True,
                                                         init_cond=(panel_id == 0))
                    probability_gradient = pl.matmul_acc(probability_gradient, incoming_lo, key_tile, b_trans=True)
                history_score[score_dest:score_dest + heads, :] = logits
                history_inner[score_dest:score_dest + heads, :] = probability_gradient
            else:
                history_score[score_dest:score_dest + heads, :] = pl.full([heads, TILE], dtype=pl.FP32, value=0.)
                history_inner[score_dest:score_dest + heads, :] = pl.full([heads, TILE], dtype=pl.FP32, value=0.)

        history_ds_hi = pl.create_tensor([rows * parts, TILE], dtype=pl.BF16)
        history_ds_lo = pl.create_tensor([rows * parts, TILE], dtype=pl.BF16)
        history_mask = pl.create_tensor([block * parts, TILE], dtype=pl.FP32)
        with pl.spmd(active * parts, name_hint="orthrus_packed_attention_history_masks") as masks_ready:
            mask_task = pl.tile.get_block_idx()
            mask_token, mask_part = mask_task // parts, mask_task % parts
            if mask_part == 0:
                pl.store(pl.load(window_bias, [0, 0], [1, TILE]), [mask_task, 0], history_mask)
            else:
                pl.store(pl.load(selected_bias, [mask_token, (mask_part - 1) * TILE], [1, TILE]),
                         [mask_task, 0], history_mask)
        with pl.spmd(active * parts, name_hint="orthrus_packed_attention_history_softmax_vjp",
                     deps=[scores_ready, masks_ready]) as history_ready:
            softmax_task = pl.tile.get_block_idx()
            softmax_start, softmax_dest = softmax_task // parts * heads, softmax_task * heads
            bias_tile = pl.load(history_mask, [softmax_task, 0], [1, TILE])
            for head in pl.range(heads):
                source_row, dest_row = softmax_start + head, softmax_dest + head
                local_scores = pl.load(history_score, [dest_row, 0], [1, TILE])
                local_dprob = pl.load(history_inner, [dest_row, 0], [1, TILE])
                logp = pl.sub(pl.add(pl.mul(local_scores, scale), bias_tile), pl.read(lse, [source_row, 0]))
                difference = pl.sub(local_dprob, pl.read(dot, [source_row, 0]))
                history_dscore = pl.mul(pl.mul(pl.exp(logp), difference), scale)
                score_hi = pl.cast(history_dscore, pl.BF16, mode="rint")
                score_lo = pl.cast(pl.sub(history_dscore, pl.cast(score_hi, pl.FP32)), pl.BF16, mode="rint")
                pl.store(score_hi, [dest_row, 0], history_ds_hi)
                pl.store(score_lo, [dest_row, 0], history_ds_lo)

        draft_score = pl.create_tensor([rows, block], dtype=pl.FP32)
        draft_inner = pl.create_tensor([rows, block], dtype=pl.FP32)
        with pl.spmd(active, name_hint="orthrus_packed_attention_draft_scores",
                     deps=[seed_ready]) as draft_scores_ready:
            draft_score_start = pl.tile.get_block_idx() * heads
            draft_score_q = pl.slice(query, [heads, dim], [draft_score_start, 0])
            draft_score_g_hi = pl.slice(grad_hi, [heads, dim], [draft_score_start, 0])
            draft_score_g_lo = pl.slice(grad_lo, [heads, dim], [draft_score_start, 0])
            draft_score[draft_score_start:draft_score_start + heads, :] = pl.matmul(
                draft_score_q, temporary, b_trans=True, out_dtype=pl.FP32,
            )
            draft_score_v = pl.matmul(draft_score_g_hi, temporary, b_trans=True, out_dtype=pl.FP32)
            draft_inner[draft_score_start:draft_score_start + heads, :] = pl.matmul_acc(
                draft_score_v, draft_score_g_lo, temporary, b_trans=True,
            )

        draft_ds_hi = pl.create_tensor([rows, block], dtype=pl.BF16)
        draft_ds_lo = pl.create_tensor([rows, block], dtype=pl.BF16)
        draft_p_hi = pl.create_tensor([rows, block], dtype=pl.BF16)
        draft_p_lo = pl.create_tensor([rows, block], dtype=pl.BF16)
        with pl.spmd(block, name_hint="orthrus_packed_attention_draft_softmax_vjp",
                     deps=[draft_scores_ready]) as draft_ready:
            draft_softmax_token = pl.tile.get_block_idx()
            draft_softmax_start = draft_softmax_token * heads
            p = pl.full([heads, block], dtype=pl.FP32, value=0.)
            draft_dscore = pl.full([heads, block], dtype=pl.FP32, value=0.)
            if draft_softmax_token < active:
                draft_logits = pl.mul(pl.slice(draft_score, [heads, block], [draft_softmax_start, 0]), scale)
                p = pl.exp(pl.row_expand_sub(draft_logits, pl.slice(lse, [heads, 1], [draft_softmax_start, 0])))
                draft_dprob = pl.slice(draft_inner, [heads, block], [draft_softmax_start, 0])
                draft_dot = pl.slice(dot, [heads, 1], [draft_softmax_start, 0])
                draft_dscore = pl.mul(pl.mul(p, pl.row_expand_sub(draft_dprob, draft_dot)), scale)
            draft_softmax_dh = pl.cast(draft_dscore, pl.BF16, mode="rint")
            draft_softmax_dl = pl.cast(pl.sub(draft_dscore, pl.cast(draft_softmax_dh, pl.FP32)), pl.BF16, mode="rint")
            draft_softmax_ph = pl.cast(p, pl.BF16, mode="rint")
            draft_softmax_pr = pl.cast(pl.sub(p, pl.cast(draft_softmax_ph, pl.FP32)), pl.BF16, mode="rint")
            draft_ds_hi[draft_softmax_start:draft_softmax_start + heads, :] = draft_softmax_dh
            draft_ds_lo[draft_softmax_start:draft_softmax_start + heads, :] = draft_softmax_dl
            draft_p_hi[draft_softmax_start:draft_softmax_start + heads, :] = draft_softmax_ph
            draft_p_lo[draft_softmax_start:draft_softmax_start + heads, :] = draft_softmax_pr

        # Each Cube task writes full output rows for one history segment.
        # Column-partitioned writes to the same rows were unstable on 910B3.
        partial_query = pl.create_tensor([rows * parts, dim], dtype=pl.FP32)
        with pl.spmd(active * parts, name_hint="orthrus_packed_attention_history_query_vjp",
                     deps=[history_ready]) as partial_ready:
            row_task = pl.tile.get_block_idx()
            history_query_token, history_query_part = row_task // parts, row_task % parts
            history_query_length = pl.read(lengths, [history_query_token])
            if history_query_part == 0 or (history_query_part - 1) * TILE < history_query_length:
                history_query_dest = row_task * heads
                history_values = pl.create_l1([TILE, dim], dtype=pl.BF16)
                if history_query_part == 0:
                    history_values = pl.gather_row(history_values, window, [0, 0], [0, 0], [TILE, dim])
                else:
                    history_values = pl.gather_row(
                        history_values, selected, [0, 0],
                        [history_query_token * TOPK + (history_query_part - 1) * TILE, 0], [TILE, dim],
                    )
                h_hi = pl.slice(history_ds_hi, [heads, TILE], [history_query_dest, 0])
                h_lo = pl.slice(history_ds_lo, [heads, TILE], [history_query_dest, 0])
                partial = pl.matmul(h_hi, history_values, out_dtype=pl.FP32)
                partial_query[history_query_dest:history_query_dest + heads, :] = pl.matmul_acc(
                    partial, h_lo, history_values,
                )

        draft_query = pl.create_tensor([rows, dim], dtype=pl.FP32)
        with pl.spmd(active, name_hint="orthrus_packed_attention_draft_query_vjp",
                     deps=[draft_ready]) as draft_query_ready:
            draft_query_start = pl.tile.get_block_idx() * heads
            primary = pl.slice(draft_ds_hi, [heads, block], [draft_query_start, 0])
            residual = pl.slice(draft_ds_lo, [heads, block], [draft_query_start, 0])
            draft_dq = pl.matmul(primary, temporary, out_dtype=pl.FP32)
            draft_query[draft_query_start:draft_query_start + heads, :] = pl.matmul_acc(draft_dq, residual, temporary)

        with pl.spmd(block, name_hint="orthrus_packed_attention_query_sum", deps=[partial_ready, draft_query_ready]):
            merge_token = pl.tile.get_block_idx()
            merge_start = merge_token * heads
            dquery = pl.full([heads, dim], dtype=pl.FP32, value=0.)
            if merge_token < active:
                dquery = pl.slice(draft_query, [heads, dim], [merge_start, 0])
                merge_length = pl.read(lengths, [merge_token])
                for merge_part in pl.range(parts):
                    if merge_part == 0 or (merge_part - 1) * TILE < merge_length:
                        merge_dest = (merge_token * parts + merge_part) * heads
                        dquery = pl.add(dquery, pl.slice(partial_query, [heads, dim], [merge_dest, 0]))
            query_gradient[merge_start:merge_start + heads, :] = dquery

        # The draft shares K and V across all heads, including draft rows that
        # did not receive a direct teacher gradient. Sum both contributions.
        with pl.spmd(dim // 128, name_hint="orthrus_packed_attention_shared_kv_vjp", deps=[draft_ready]):
            col = pl.tile.get_block_idx() * 128
            result = pl.create_tensor([block, 128], dtype=pl.FP32)
            for kv_token in pl.range(active):
                kv_offset = kv_token * heads
                query_slice = pl.slice(query, [heads, 128], [kv_offset, col])
                seed_hi = pl.slice(grad_hi, [heads, 128], [kv_offset, col])
                seed_lo = pl.slice(grad_lo, [heads, 128], [kv_offset, col])
                dh = pl.slice(draft_ds_hi, [heads, block], [kv_offset, 0])
                dl = pl.slice(draft_ds_lo, [heads, block], [kv_offset, 0])
                ph = pl.slice(draft_p_hi, [heads, block], [kv_offset, 0])
                pr = pl.slice(draft_p_lo, [heads, block], [kv_offset, 0])
                result = pl.matmul_acc(result, dh, query_slice, a_trans=True, init_cond=(kv_token == 0))
                result = pl.matmul_acc(result, dl, query_slice, a_trans=True)
                result = pl.matmul_acc(result, ph, seed_hi, a_trans=True)
                result = pl.matmul_acc(result, ph, seed_lo, a_trans=True)
                result = pl.matmul_acc(result, pr, seed_hi, a_trans=True)
                result = pl.matmul_acc(result, pr, seed_lo, a_trans=True)
            temporary_gradient[:, col:col + 128] = result

        for group in pl.spmd(1, name_hint="orthrus_packed_attention_sink_vjp", deps=[seed_ready]):
            grad = pl.full([1, heads], dtype=pl.FP32, value=0.)
            sink_logits = pl.slice(sink, [1, heads], [0, 0])
            for sink_token in pl.range(active):
                norm = pl.reshape(pl.slice(lse, [heads, 1], [sink_token * heads, 0]), [1, heads])
                context_dot = pl.reshape(pl.slice(dot, [heads, 1], [sink_token * heads, 0]), [1, heads])
                grad = pl.sub(grad, pl.mul(pl.exp(pl.sub(sink_logits, norm)), context_dot))
            sink_gradient[group:group + 1, :] = grad
        return query_gradient, temporary_gradient, sink_gradient

    @pl.jit.host
    def dispatch(
        q: pl.Tensor[[world, rows, dim], pl.BF16],
        temporary: pl.Tensor[[world, block, dim], pl.BF16],
        window: pl.Tensor[[world, WINDOW, dim], pl.BF16],
        selected: pl.Tensor[[world, selected_rows, dim], pl.BF16],
        window_bias: pl.Tensor[[world, 1, WINDOW], pl.FP32],
        selected_bias: pl.Tensor[[world, block, TOPK], pl.FP32],
        lengths: pl.Tensor[[world, block], pl.INT32],
        context: pl.Tensor[[world, rows, dim], pl.FP32],
        lse: pl.Tensor[[world, rows, 1], pl.FP32],
        sink: pl.Tensor[[world, 1, heads], pl.FP32],
        incoming: pl.Tensor[[world, rows, dim], pl.FP32],
        dq: pl.Out[pl.Tensor[[world, rows, dim], pl.FP32]],
        dkv: pl.Out[pl.Tensor[[world, block, dim], pl.FP32]],
        dsink: pl.Out[pl.Tensor[[world, 1, heads], pl.FP32]],
        owner: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for idx in pl.range(pld.world_size()):
            if idx >= owner and idx < owner + tp:
                reverse(q[idx], temporary[idx], window[idx], selected[idx], window_bias[idx], selected_bias[idx],
                        lengths[idx], context[idx], lse[idx], sink[idx], incoming[idx], dq[idx], dkv[idx],
                        dsink[idx], active, device=idx)
    return dispatch
