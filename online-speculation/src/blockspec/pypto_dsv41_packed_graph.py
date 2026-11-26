"""Full resident draft graph using the 1122 A3 frozen model layers."""

from .dsv41_packed_attention_tape import (
    make_packed_attention_core_capture, make_packed_attention_post_capture,
    make_packed_qkv_capture, packed_attention_tape_shapes, packed_core_tape_shapes, packed_qkv_tape_shapes,
    packed_wo_tape_shapes,
)
from .dsv41_packed_ffn_tape import make_packed_ffn_capture, packed_ffn_tape_shapes
from .pypto_dsv41_draft import make_packed_draft_output_tape
from .pypto_dsv41_packed_draft_attention import make_packed_draft_projection
from .pypto_dsv41_packed_indexer import make_packed_draft_indexer
from .pypto_dsv41_packed_swa import (
    make_packed_gathered_attention, make_packed_selection_gather, make_packed_swa_chain,
)


def make_packed_draft_graph(pages, c1a_bucket, c2a_bucket, block, *, last_layers=0, staged_ranks=None):
    """Compile the 40-layer rank program for one fixed-width draft request per DP group."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3 import decode_fwd as native

    if block not in (16, 32):
        raise ValueError("The whole-model draft graph supports K16 and K32")
    capacity = block
    m = native.M
    ep, tp, layers = native.EP_SIZE, native.TP_SIZE, m.num_hidden_layers
    if not 0 <= last_layers <= layers:
        raise ValueError("Packed draft tape must cover a suffix of the model")
    hidden, mix, mix_width, mix_input = m.hidden_size, m.hc_mult, m.mix_hc, m.hc_dim
    dim, latent, heads = m.head_dim, m.q_lora_rank, native.LOCAL_H
    groups, wo_in, wo_rank = native.LOCAL_O_GROUPS, native.O_GROUP_IN, m.o_lora_rank
    index_heads, index_dim, index_topk = m.index_n_heads, m.index_head_dim, m.index_topk
    index_count, kv_count = native.INDEX_SOURCE_COUNT, native.KV_SOURCE_COUNT
    inter, experts, all_experts = m.moe_intermediate_size, native.N_LOCAL_EXPERTS, native.N_EXPERTS
    moe_rows, fanout = native.MOE_TOKENS, m.num_experts_per_tok
    signals, recv_rows, aux = native.SIGNAL_PAD, native.ROUTED_RECV_ROWS, native.AUX_WIDTH
    head_rows, local_vocab = native.MAX_LOGIT_ROWS, native.VOCAB_PER_TP
    group_logit_rows, sampled_pad, vocab = native.GROUP_LOGIT_ROWS, native.SAMPLED_IDS_PAD, m.vocab_size
    engram_k, engram_out = native.ENGRAM_K, native.KV_OUT
    tape_shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, mix)
    tape_residual = tape_shapes["ffn_tape_residual"][0]
    tape_mixes = tape_shapes["ffn_tape_mixes"][0]
    tape_vectors = tape_shapes["ffn_tape_vectors"][0]
    tape_routes = tape_shapes["ffn_tape_route_ids"][0]
    attention_shapes = packed_attention_tape_shapes(block, last_layers, hidden, mix)
    attention_residual = attention_shapes["attention_tape_residual"][0]
    attention_mixes = attention_shapes["attention_tape_mixes"][0]
    attention_delta = attention_shapes["attention_tape_delta"][0]
    wo_shapes = packed_wo_tape_shapes(block, last_layers, m, tp)
    wo_input_shape = wo_shapes["attention_tape_wo_input"][0]
    wo_latent_shape = wo_shapes["attention_tape_wo_latent"][0]
    core_shapes = packed_core_tape_shapes(block, last_layers, m, tp)
    core_normed = core_shapes["attention_tape_normed"][0]
    core_query = core_shapes["attention_tape_query"][0]
    core_temporary = core_shapes["attention_tape_temporary"][0]
    core_context = core_shapes["attention_tape_context"][0]
    core_lse = core_shapes["attention_tape_lse"][0]
    core_selection = core_shapes["attention_tape_selection"][0]
    core_lengths = core_shapes["attention_tape_lengths"][0]
    qkv_shapes = packed_qkv_tape_shapes(block, last_layers, m)
    qkv_low = qkv_shapes["attention_tape_q_normed"][0]
    qkv_raw = qkv_shapes["attention_tape_qkv_raw"][0]
    first_saved_layer = layers - last_layers
    save_ffn = make_packed_ffn_capture(block, last_layers, tp, hidden, mix, fanout)
    page = native.BLOCK_SIZE
    norm_span = native.ROPE_DIM // 2
    win_scale, kv_scale, key_scale = native.WINDOW_CACHE_GROUP, native.COMPRESSED_CACHE_GROUP, native.INDEX_CACHE_GROUP

    def inline(entry):
        return pl.jit.inline(auto_scope=False)(entry._func)

    prepare, _ = make_packed_draft_projection(block)
    swa = inline(make_packed_swa_chain(pages, heads, block))
    c1a = inline(make_packed_draft_indexer(pages, c1a_bucket, 1, block))
    # Native C1A/C2A share the DSL's table-width variable. Pad only the
    # metadata and keep C2A's score and Top-K arenas half-width.
    c2a = inline(make_packed_draft_indexer(pages, c2a_bucket, 2, block, table_bucket=c1a_bucket))
    gather_history = make_packed_selection_gather(pages, block)
    attend_history = make_packed_gathered_attention(pages, block)
    retain_history = staged_ranks is None
    history_groups = 8 if retain_history else 1
    post = make_packed_attention_post_capture(block, last_layers, m, tp)
    save_core = make_packed_attention_core_capture(block, last_layers, m, tp)
    save_qkv = make_packed_qkv_capture(block, last_layers, m)

    project_wo = make_packed_draft_output_tape(block, last_layers)

    # Bind the original Decode entries. The same collective protocol is used
    # for every rank, including ranks that only own remote experts this round.
    moe_source_mask, moe = native._moe_source_mask, native.prefill_moe_sublayer
    restore, clear = native.prefill_ffn_restore, native.clear_moe_signals
    retire, engram, output = native.retire_prefill_signals, native.engram_from_rows, native.final_output

    @pl.jit.inline(auto_scope=False)
    def draft_attention_layer(
        x_hc: pl.InOut[pl.Tensor[[capacity, mix, hidden], pl.FP32]],
        pre_mix: pl.InOut[pl.Tensor[[capacity, mix], pl.FP32]],
        engram_rows: pl.Tensor[[2, block, engram_k], pl.BF16],
        engram_wkv: pl.Tensor[[2, engram_k, engram_out], pl.BF16, pl.NZ],
        engram_weight: pl.Tensor[[2, mix, hidden], pl.FP32],
        engram_rotation: pl.Tensor[[hidden // 32, 32, 32], pl.FP32],
        hc_attn_fn: pl.Tensor[[layers * mix_width, mix_input], pl.FP32],
        hc_attn_scale: pl.Tensor[[layers * 3], pl.FP32],
        hc_attn_base: pl.Tensor[[layers * mix_width], pl.FP32],
        attn_norm_weight: pl.Tensor[[layers * hidden], pl.BF16],
        draft_q_a_kv: pl.Tensor[[layers, hidden, latent + dim], pl.BF16],
        draft_q_gamma: pl.Tensor[[layers, latent], pl.BF16],
        draft_q_b: pl.Tensor[[layers, latent, heads * dim], pl.BF16],
        draft_kv_gamma: pl.Tensor[[layers, dim], pl.BF16],
        draft_sink: pl.Tensor[[layers, heads], pl.FP32],
        draft_wo_a: pl.Tensor[[layers * groups, wo_in, wo_rank], pl.BF16, pl.NZ],
        draft_wo_b: pl.Tensor[[layers, groups * wo_rank, hidden], pl.BF16, pl.NZ],
        draft_index_q: pl.Tensor[[index_count, latent, index_heads * index_dim], pl.BF16],
        draft_index_weight: pl.Tensor[[index_count, hidden, index_heads], pl.BF16],
        draft_cos: pl.Tensor[[2, capacity, norm_span], pl.FP32],
        draft_sin: pl.Tensor[[2, capacity, norm_span], pl.FP32],
        window_cache_pool: pl.Tensor[[layers * pages, page, 1, dim], pl.UINT8],
        window_cache_scale_pool: pl.Tensor[[layers * pages, page, 1, dim // win_scale], pl.BF16],
        compressed_cache_pool: pl.Tensor[[kv_count * pages, page, 1, dim // 2], pl.UINT8],
        compressed_cache_scale_pool: pl.Tensor[[kv_count * pages, page, 1, dim // kv_scale], pl.BF16],
        index_cache_pool: pl.Tensor[[kv_count * pages, page, 1, index_dim // 2], pl.UINT8],
        index_cache_scale_pool: pl.Tensor[[kv_count * pages, page, 1, index_dim // key_scale], pl.UINT8],
        window_lookup: pl.Tensor[[1, 256], pl.INT32],
        compressed_lookup: pl.Tensor[[1, 256], pl.INT32],
        window_rows: pl.Tensor[[8, 16], pl.INT32],
        window_starts: pl.Tensor[[8], pl.INT32],
        window_spans: pl.Tensor[[2, 3], pl.INT32],
        window_bias: pl.Tensor[[1, page], pl.FP32],
        c1a_table: pl.Tensor[[1, c1a_bucket // page], pl.INT32],
        c2a_table: pl.Tensor[[1, c1a_bucket // page], pl.INT32],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        c1a_visible: pl.Tensor[[capacity], pl.INT32],
        c2a_visible: pl.Tensor[[capacity], pl.INT32],
        zero_kv: pl.Tensor[[page, dim], pl.BF16],
        attention_tape_residual: pl.Out[pl.Tensor[attention_residual, pl.FP32]],
        attention_tape_mixes: pl.Out[pl.Tensor[attention_mixes, pl.FP32]],
        attention_tape_delta: pl.Out[pl.Tensor[attention_delta, pl.BF16]],
        attention_tape_wo_input: pl.Out[pl.Tensor[wo_input_shape, pl.BF16]],
        attention_tape_wo_latent: pl.Out[pl.Tensor[wo_latent_shape, pl.BF16]],
        attention_tape_normed: pl.Out[pl.Tensor[core_normed, pl.BF16]],
        attention_tape_query: pl.Out[pl.Tensor[core_query, pl.BF16]],
        attention_tape_temporary: pl.Out[pl.Tensor[core_temporary, pl.BF16]],
        attention_tape_context: pl.Out[pl.Tensor[core_context, pl.FP32]],
        attention_tape_lse: pl.Out[pl.Tensor[core_lse, pl.FP32]],
        attention_tape_selection: pl.Out[pl.Tensor[core_selection, pl.INT32]],
        attention_tape_lengths: pl.Out[pl.Tensor[core_lengths, pl.INT32]],
        attention_tape_q_normed: pl.Out[pl.Tensor[qkv_low, pl.BF16]],
        attention_tape_qkv_raw: pl.Out[pl.Tensor[qkv_raw, pl.BF16]],
        ffn_capture_owner: pl.Scalar[pl.INT32],
        output_window: pld.DistributedTensor[[block, hidden], pl.BF16],
        output_arrived: pld.DistributedTensor[[tp, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        live_x: pl.Tensor[[block, mix, hidden], pl.FP32],
        live_mix: pl.Tensor[[block, mix], pl.FP32],
        selected: pl.Tensor[[block, index_topk], pl.INT32],
        lengths: pl.Tensor[[block], pl.INT32],
        mask: pl.Tensor[[block, index_topk], pl.FP32],
        history_banks: pl.InOut[pl.Tensor[[history_groups, block * index_topk, dim], pl.BF16]],
        count: pl.Scalar[pl.INT32],
        group: pl.Scalar[pl.INT32],
        layer: pl.Scalar[pl.INDEX],
        epoch: pl.Scalar[pl.INT32],
        h_attention: pl.Tensor[[block, mix, hidden], pl.FP32],
        pre_attention: pl.Tensor[[block, mix], pl.FP32],
        ffn_input: pl.Tensor[[block, hidden], pl.BF16],
    ):
        """Read the target history and produce the temporary bidirectional attention rows."""
        if count != 0:
            if layer == 1 or layer == 14:
                slot = layer // 14
                engram(engram_rows[slot], engram_wkv[slot], engram_weight[slot], engram_rotation, live_x, live_x)
            fn = pl.slice(hc_attn_fn, [mix_width, mix_input], [layer * mix_width, 0])
            scale = pl.slice(hc_attn_scale, [3], [layer * 3])
            base = pl.slice(hc_attn_base, [mix_width], [layer * mix_width])
            norm = pl.slice(attn_norm_weight, [hidden], [layer * hidden])
            window = pl.slice(window_cache_pool, [pages, page, 1, dim], [layer * pages, 0, 0, 0])
            window_scales = pl.slice(window_cache_scale_pool, [pages, page, 1, dim // win_scale], [layer * pages, 0, 0, 0])
            wo_a = pl.slice(draft_wo_a, [groups, wo_in, wo_rank], [layer * groups, 0, 0])
            wo_b = draft_wo_b[layer]
            q_a_kv, q_gamma = draft_q_a_kv[layer], draft_q_gamma[layer]
            q_b, kv_gamma, sink = draft_q_b[layer], draft_kv_gamma[layer], draft_sink[layer]
            delta = pl.create_tensor([block, hidden], dtype=pl.BF16)
            context = pl.create_tensor([block * heads, dim], dtype=pl.FP32)
            lse = pl.create_tensor([block * heads, 1], dtype=pl.FP32)
            raw = pl.create_tensor([block, latent + dim], dtype=pl.BF16)
            profile = 0
            if layer >= 2:
                profile = 1
            normed, low_rank, query, temporary = prepare(
                x_hc, pre_mix, norm, q_a_kv, q_gamma, q_b, kv_gamma, draft_cos[profile], draft_sin[profile], raw,
            )
            slot = pl.cast(layer, pl.INT32) - first_saved_layer
            capture_slot = -1
            if group == ffn_capture_owner:
                capture_slot = slot
            if layer < 2:
                swa(window, window_scales, window_lookup, window_rows, window_starts, window_spans,
                    window_bias, query, temporary, sink, zero_kv, context, lse)
                project_wo(context, wo_a, wo_b, draft_cos[0], draft_sin[0], delta,
                           attention_tape_wo_input, attention_tape_wo_latent, 1, capture_slot)
            else:
                history_group = (layer - 2) // 6
                if layer >= 20:
                    history_group = 3 + (layer - 20) // 4
                kv_source = pl.min(3, history_group)
                bank = history_banks[pl.min(history_group, history_groups - 1)]
                refresh = not retain_history
                if layer < 20 and (layer - 2) % 6 == 0:
                    key = pl.slice(index_cache_pool, [pages, page, 1, index_dim // 2], [kv_source * pages, 0, 0, 0])
                    key_scales = pl.slice(index_cache_scale_pool, [pages, page, 1, index_dim // key_scale], [kv_source * pages, 0, 0, 0])
                    c2a(
                        normed, low_rank,
                        draft_index_q[kv_source], draft_index_weight[kv_source], draft_cos[1], draft_sin[1],
                        key, key_scales, c2a_table, request_ids, c2a_visible, selected, lengths, mask,
                    )
                    refresh = True
                elif layer >= 20 and (layer - 20) % 4 == 0:
                    index_source = 3 + (layer - 20) // 4
                    key = pl.slice(index_cache_pool, [pages, page, 1, index_dim // 2], [kv_source * pages, 0, 0, 0])
                    key_scales = pl.slice(index_cache_scale_pool, [pages, page, 1, index_dim // key_scale], [kv_source * pages, 0, 0, 0])
                    c1a(
                        normed, low_rank,
                        draft_index_q[index_source], draft_index_weight[index_source], draft_cos[1], draft_sin[1],
                        key, key_scales, c1a_table, request_ids, c1a_visible, selected, lengths, mask,
                    )
                    refresh = True
                # A Reuse layer has the same physical compressed KV source and
                # the same Top-K. Its SWA window and temporary KV are layer-local.
                if refresh:
                    compressed = pl.slice(compressed_cache_pool, [pages, page, 1, dim // 2],
                                          [kv_source * pages, 0, 0, 0])
                    compressed_scales = pl.slice(compressed_cache_scale_pool, [pages, page, 1, dim // kv_scale],
                                                 [kv_source * pages, 0, 0, 0])
                    bank = gather_history(compressed, compressed_scales, compressed_lookup, selected, lengths, bank)
                context, lse = attend_history(
                    window, window_scales, window_lookup, window_rows, window_starts, window_spans, window_bias,
                    bank, selected, lengths, mask, query, temporary, sink, zero_kv, context, lse,
                )
                project_wo(context, wo_a, wo_b, draft_cos[1], draft_sin[1], delta,
                           attention_tape_wo_input, attention_tape_wo_latent, 0, capture_slot)
            if last_layers > 0:
                save_core(normed, query, temporary, context, lse, selected, lengths,
                          attention_tape_normed, attention_tape_query, attention_tape_temporary,
                          attention_tape_context, attention_tape_lse, attention_tape_selection,
                          attention_tape_lengths, layer < 2, capture_slot)
                save_qkv(low_rank, raw, attention_tape_q_normed, attention_tape_qkv_raw,
                         rank == ffn_capture_owner, capture_slot)
            h_attention, pre_attention, ffn_input = post(
                live_x, live_mix, delta, fn, scale, base, output_window, output_arrived, rank, epoch,
                h_attention, pre_attention, ffn_input, attention_tape_residual, attention_tape_mixes,
                attention_tape_delta, ffn_capture_owner, slot,
            )
        return h_attention, pre_attention, ffn_input

    if staged_ranks is not None:
        if staged_ranks < tp or staged_ranks % tp:
            raise ValueError("Staged draft requires complete local TP groups")

        @pl.jit(auto_scope=False)
        def attention_rank(
            x_hc: pl.Tensor,
            pre_mix: pl.Tensor,
            engram_rows: pl.Tensor,
            engram_wkv: pl.Tensor,
            engram_weight: pl.Tensor,
            engram_rotation: pl.Tensor,
            hc_attn_fn: pl.Tensor,
            hc_attn_scale: pl.Tensor,
            hc_attn_base: pl.Tensor,
            attn_norm_weight: pl.Tensor,
            draft_q_a_kv: pl.Tensor,
            draft_q_gamma: pl.Tensor,
            draft_q_b: pl.Tensor,
            draft_kv_gamma: pl.Tensor,
            draft_sink: pl.Tensor,
            draft_wo_a: pl.Tensor,
            draft_wo_b: pl.Tensor,
            draft_index_q: pl.Tensor,
            draft_index_weight: pl.Tensor,
            draft_cos: pl.Tensor,
            draft_sin: pl.Tensor,
            window_cache_pool: pl.Tensor,
            window_cache_scale_pool: pl.Tensor,
            compressed_cache_pool: pl.Tensor,
            compressed_cache_scale_pool: pl.Tensor,
            index_cache_pool: pl.Tensor,
            index_cache_scale_pool: pl.Tensor,
            window_lookup: pl.Tensor,
            compressed_lookup: pl.Tensor,
            window_rows: pl.Tensor,
            window_starts: pl.Tensor,
            window_spans: pl.Tensor,
            window_bias: pl.Tensor,
            c1a_table: pl.Tensor,
            c2a_table: pl.Tensor,
            request_ids: pl.Tensor,
            c1a_visible: pl.Tensor,
            c2a_visible: pl.Tensor,
            zero_kv: pl.Tensor,
            attention_tape_residual: pl.Tensor,
            attention_tape_mixes: pl.Tensor,
            attention_tape_delta: pl.Tensor,
            attention_tape_wo_input: pl.Tensor,
            attention_tape_wo_latent: pl.Tensor,
            attention_tape_normed: pl.Tensor,
            attention_tape_query: pl.Tensor,
            attention_tape_temporary: pl.Tensor,
            attention_tape_context: pl.Tensor,
            attention_tape_lse: pl.Tensor,
            attention_tape_selection: pl.Tensor,
            attention_tape_lengths: pl.Tensor,
            attention_tape_q_normed: pl.Tensor,
            attention_tape_qkv_raw: pl.Tensor,
            selected: pl.Tensor,
            lengths: pl.Tensor,
            mask: pl.Tensor,
            count_meta: pl.Tensor,
            h_attention: pl.Tensor,
            pre_attention: pl.Tensor,
            ffn_input: pl.Tensor,
            output_window: pld.DistributedTensor[[block, hidden], pl.BF16],
            output_arrived: pld.DistributedTensor[[tp, 1], pl.INT32],
            rank: pl.Scalar[pl.INT32],
            layer_index: pl.Scalar[pl.INT32],
            ffn_capture_owner: pl.Scalar[pl.INT32],
        ):
            count = pl.read(count_meta, [0])
            group = rank // tp * tp
            if count != 0:
                history_banks = pl.create_tensor([history_groups, block * index_topk, dim], dtype=pl.BF16)
                draft_attention_layer(
                    x_hc, pre_mix, engram_rows, engram_wkv, engram_weight, engram_rotation, hc_attn_fn,
                    hc_attn_scale, hc_attn_base, attn_norm_weight, draft_q_a_kv, draft_q_gamma, draft_q_b,
                    draft_kv_gamma, draft_sink, draft_wo_a, draft_wo_b, draft_index_q, draft_index_weight,
                    draft_cos, draft_sin, window_cache_pool, window_cache_scale_pool, compressed_cache_pool,
                    compressed_cache_scale_pool, index_cache_pool, index_cache_scale_pool, window_lookup,
                    compressed_lookup, window_rows, window_starts, window_spans, window_bias, c1a_table,
                    c2a_table, request_ids, c1a_visible, c2a_visible, zero_kv, attention_tape_residual,
                    attention_tape_mixes, attention_tape_delta, attention_tape_wo_input,
                    attention_tape_wo_latent, attention_tape_normed, attention_tape_query,
                    attention_tape_temporary, attention_tape_context, attention_tape_lse,
                    attention_tape_selection, attention_tape_lengths, attention_tape_q_normed,
                    attention_tape_qkv_raw, ffn_capture_owner, output_window, output_arrived, rank, x_hc,
                    pre_mix, selected, lengths, mask, history_banks, count, group, pl.cast(layer_index, pl.INDEX), 1,
                    h_attention, pre_attention, ffn_input,
                )
                retire(h_attention, output_arrived, group, rank % tp)

        @pl.jit.host
        def draft_attention_stage(
            x_hc: pl.InOut[pl.Tensor[[staged_ranks, capacity, mix, hidden], pl.FP32]],
            pre_mix: pl.InOut[pl.Tensor[[staged_ranks, capacity, mix], pl.FP32]],
            engram_rows: pl.Tensor[[staged_ranks, 2, block, engram_k], pl.BF16],
            engram_wkv: pl.Tensor[[staged_ranks, 2, engram_k, engram_out], pl.BF16, pl.NZ],
            engram_weight: pl.Tensor[[staged_ranks, 2, mix, hidden], pl.FP32],
            engram_rotation: pl.Tensor[[staged_ranks, hidden // 32, 32, 32], pl.FP32],
            hc_attn_fn: pl.Tensor[[staged_ranks, layers * mix_width, mix_input], pl.FP32],
            hc_attn_scale: pl.Tensor[[staged_ranks, layers * 3], pl.FP32],
            hc_attn_base: pl.Tensor[[staged_ranks, layers * mix_width], pl.FP32],
            attn_norm_weight: pl.Tensor[[staged_ranks, layers * hidden], pl.BF16],
            draft_q_a_kv: pl.Tensor[[staged_ranks, layers, hidden, latent + dim], pl.BF16],
            draft_q_gamma: pl.Tensor[[staged_ranks, layers, latent], pl.BF16],
            draft_q_b: pl.Tensor[[staged_ranks, layers, latent, heads * dim], pl.BF16],
            draft_kv_gamma: pl.Tensor[[staged_ranks, layers, dim], pl.BF16],
            draft_sink: pl.Tensor[[staged_ranks, layers, heads], pl.FP32],
            draft_wo_a: pl.Tensor[[staged_ranks, layers * groups, wo_in, wo_rank], pl.BF16, pl.NZ],
            draft_wo_b: pl.Tensor[[staged_ranks, layers, groups * wo_rank, hidden], pl.BF16, pl.NZ],
            draft_index_q: pl.Tensor[[staged_ranks, index_count, latent, index_heads * index_dim], pl.BF16],
            draft_index_weight: pl.Tensor[[staged_ranks, index_count, hidden, index_heads], pl.BF16],
            draft_cos: pl.Tensor[[staged_ranks, 2, capacity, norm_span], pl.FP32],
            draft_sin: pl.Tensor[[staged_ranks, 2, capacity, norm_span], pl.FP32],
            window_cache_pool: pl.Tensor[[staged_ranks, layers * pages, page, 1, dim], pl.UINT8],
            window_cache_scale_pool: pl.Tensor[[staged_ranks, layers * pages, page, 1, dim // win_scale], pl.BF16],
            compressed_cache_pool: pl.Tensor[[staged_ranks, kv_count * pages, page, 1, dim // 2], pl.UINT8],
            compressed_cache_scale_pool: pl.Tensor[[staged_ranks, kv_count * pages, page, 1, dim // kv_scale], pl.BF16],
            index_cache_pool: pl.Tensor[[staged_ranks, kv_count * pages, page, 1, index_dim // 2], pl.UINT8],
            index_cache_scale_pool: pl.Tensor[[staged_ranks, kv_count * pages, page, 1, index_dim // key_scale], pl.UINT8],
            window_lookup: pl.Tensor[[staged_ranks, 1, 256], pl.INT32],
            compressed_lookup: pl.Tensor[[staged_ranks, 1, 256], pl.INT32],
            window_rows: pl.Tensor[[staged_ranks, 8, 16], pl.INT32],
            window_starts: pl.Tensor[[staged_ranks, 8], pl.INT32],
            window_spans: pl.Tensor[[staged_ranks, 2, 3], pl.INT32],
            window_bias: pl.Tensor[[staged_ranks, 1, page], pl.FP32],
            c1a_table: pl.Tensor[[staged_ranks, 1, c1a_bucket // page], pl.INT32],
            c2a_table: pl.Tensor[[staged_ranks, 1, c1a_bucket // page], pl.INT32],
            request_ids: pl.Tensor[[staged_ranks, capacity], pl.INT32],
            c1a_visible: pl.Tensor[[staged_ranks, capacity], pl.INT32],
            c2a_visible: pl.Tensor[[staged_ranks, capacity], pl.INT32],
            zero_kv: pl.Tensor[[staged_ranks, page, dim], pl.BF16],
            attention_tape_residual: pl.InOut[pl.Tensor[[staged_ranks, *attention_residual], pl.FP32]],
            attention_tape_mixes: pl.InOut[pl.Tensor[[staged_ranks, *attention_mixes], pl.FP32]],
            attention_tape_delta: pl.InOut[pl.Tensor[[staged_ranks, *attention_delta], pl.BF16]],
            attention_tape_wo_input: pl.InOut[pl.Tensor[[staged_ranks, *wo_input_shape], pl.BF16]],
            attention_tape_wo_latent: pl.InOut[pl.Tensor[[staged_ranks, *wo_latent_shape], pl.BF16]],
            attention_tape_normed: pl.InOut[pl.Tensor[[staged_ranks, *core_normed], pl.BF16]],
            attention_tape_query: pl.InOut[pl.Tensor[[staged_ranks, *core_query], pl.BF16]],
            attention_tape_temporary: pl.InOut[pl.Tensor[[staged_ranks, *core_temporary], pl.BF16]],
            attention_tape_context: pl.InOut[pl.Tensor[[staged_ranks, *core_context], pl.FP32]],
            attention_tape_lse: pl.InOut[pl.Tensor[[staged_ranks, *core_lse], pl.FP32]],
            attention_tape_selection: pl.InOut[pl.Tensor[[staged_ranks, *core_selection], pl.INT32]],
            attention_tape_lengths: pl.InOut[pl.Tensor[[staged_ranks, *core_lengths], pl.INT32]],
            attention_tape_q_normed: pl.InOut[pl.Tensor[[staged_ranks, *qkv_low], pl.BF16]],
            attention_tape_qkv_raw: pl.InOut[pl.Tensor[[staged_ranks, *qkv_raw], pl.BF16]],
            selected: pl.InOut[pl.Tensor[[staged_ranks, block, index_topk], pl.INT32]],
            lengths: pl.InOut[pl.Tensor[[staged_ranks, block], pl.INT32]],
            mask: pl.InOut[pl.Tensor[[staged_ranks, block, index_topk], pl.FP32]],
            count_meta: pl.Tensor[[staged_ranks, 2], pl.INT32],
            h_attention: pl.Out[pl.Tensor[[staged_ranks, block, mix, hidden], pl.FP32]],
            pre_attention: pl.Out[pl.Tensor[[staged_ranks, block, mix], pl.FP32]],
            ffn_input: pl.Out[pl.Tensor[[staged_ranks, block, hidden], pl.BF16]],
            layer_index: pl.Scalar[pl.INT32],
            ffn_capture_owner: pl.Scalar[pl.INT32],
            execute_layers: pl.Scalar[pl.INT32],
        ):
            window_buffer = pld.alloc_window_buffer([block, hidden], dtype=pl.BF16)
            arrived_buffer = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
            for rank in pl.range(pld.world_size()):
                output_window = pld.window(window_buffer, [block, hidden], dtype=pl.BF16)
                output_arrived = pld.window(arrived_buffer, [tp, 1], dtype=pl.INT32)
                if execute_layers != 0:
                    attention_rank(
                        x_hc[rank], pre_mix[rank], engram_rows[rank], engram_wkv[rank], engram_weight[rank],
                        engram_rotation[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                        attn_norm_weight[rank], draft_q_a_kv[rank], draft_q_gamma[rank], draft_q_b[rank],
                        draft_kv_gamma[rank], draft_sink[rank], draft_wo_a[rank], draft_wo_b[rank],
                        draft_index_q[rank], draft_index_weight[rank], draft_cos[rank], draft_sin[rank],
                        window_cache_pool[rank], window_cache_scale_pool[rank], compressed_cache_pool[rank],
                        compressed_cache_scale_pool[rank], index_cache_pool[rank], index_cache_scale_pool[rank],
                        window_lookup[rank], compressed_lookup[rank], window_rows[rank], window_starts[rank],
                        window_spans[rank], window_bias[rank], c1a_table[rank], c2a_table[rank],
                        request_ids[rank], c1a_visible[rank], c2a_visible[rank], zero_kv[rank],
                        attention_tape_residual[rank], attention_tape_mixes[rank], attention_tape_delta[rank],
                        attention_tape_wo_input[rank], attention_tape_wo_latent[rank],
                        attention_tape_normed[rank], attention_tape_query[rank], attention_tape_temporary[rank],
                        attention_tape_context[rank], attention_tape_lse[rank], attention_tape_selection[rank],
                        attention_tape_lengths[rank], attention_tape_q_normed[rank],
                        attention_tape_qkv_raw[rank], selected[rank], lengths[rank], mask[rank],
                        count_meta[rank], h_attention[rank], pre_attention[rank], ffn_input[rank], output_window,
                        output_arrived, rank, layer_index, ffn_capture_owner,
                        device=rank,
                    )

        return draft_attention_stage

    @pl.jit(auto_scope=False)
    def draft_rank(
        x_hc: pl.InOut[pl.Tensor[[capacity, mix, hidden], pl.FP32]],
        pre_mix: pl.InOut[pl.Tensor[[capacity, mix], pl.FP32]],
        engram_rows: pl.Tensor[[2, block, engram_k], pl.BF16],
        engram_wkv: pl.Tensor[[2, engram_k, engram_out], pl.BF16, pl.NZ],
        engram_weight: pl.Tensor[[2, mix, hidden], pl.FP32],
        engram_rotation: pl.Tensor[[hidden // 32, 32, 32], pl.FP32],
        hc_attn_fn: pl.Tensor[[layers * mix_width, mix_input], pl.FP32],
        hc_attn_scale: pl.Tensor[[layers * 3], pl.FP32],
        hc_attn_base: pl.Tensor[[layers * mix_width], pl.FP32],
        attn_norm_weight: pl.Tensor[[layers * hidden], pl.BF16],
        draft_q_a_kv: pl.Tensor[[layers, hidden, latent + dim], pl.BF16],
        draft_q_gamma: pl.Tensor[[layers, latent], pl.BF16],
        draft_q_b: pl.Tensor[[layers, latent, heads * dim], pl.BF16],
        draft_kv_gamma: pl.Tensor[[layers, dim], pl.BF16],
        draft_sink: pl.Tensor[[layers, heads], pl.FP32],
        draft_wo_a: pl.Tensor[[layers * groups, wo_in, wo_rank], pl.BF16, pl.NZ],
        draft_wo_b: pl.Tensor[[layers, groups * wo_rank, hidden], pl.BF16, pl.NZ],
        draft_index_q: pl.Tensor[[index_count, latent, index_heads * index_dim], pl.BF16],
        draft_index_weight: pl.Tensor[[index_count, hidden, index_heads], pl.BF16],
        draft_cos: pl.Tensor[[2, capacity, norm_span], pl.FP32],
        draft_sin: pl.Tensor[[2, capacity, norm_span], pl.FP32],
        window_cache_pool: pl.Tensor[[layers * pages, page, 1, dim], pl.UINT8],
        window_cache_scale_pool: pl.Tensor[[layers * pages, page, 1, dim // win_scale], pl.BF16],
        compressed_cache_pool: pl.Tensor[[kv_count * pages, page, 1, dim // 2], pl.UINT8],
        compressed_cache_scale_pool: pl.Tensor[[kv_count * pages, page, 1, dim // kv_scale], pl.BF16],
        index_cache_pool: pl.Tensor[[kv_count * pages, page, 1, index_dim // 2], pl.UINT8],
        index_cache_scale_pool: pl.Tensor[[kv_count * pages, page, 1, index_dim // key_scale], pl.UINT8],
        window_lookup: pl.Tensor[[1, 256], pl.INT32],
        compressed_lookup: pl.Tensor[[1, 256], pl.INT32],
        window_rows: pl.Tensor[[8, 16], pl.INT32],
        window_starts: pl.Tensor[[8], pl.INT32],
        window_spans: pl.Tensor[[2, 3], pl.INT32],
        window_bias: pl.Tensor[[1, page], pl.FP32],
        c1a_table: pl.Tensor[[1, c1a_bucket // page], pl.INT32],
        c2a_table: pl.Tensor[[1, c1a_bucket // page], pl.INT32],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        c1a_visible: pl.Tensor[[capacity], pl.INT32],
        c2a_visible: pl.Tensor[[capacity], pl.INT32],
        zero_kv: pl.Tensor[[page, dim], pl.BF16],
        hc_ffn_fn: pl.Tensor[[layers * mix_width, mix_input], pl.FP32],
        hc_ffn_scale: pl.Tensor[[layers * 3], pl.FP32],
        hc_ffn_base: pl.Tensor[[layers * mix_width], pl.FP32],
        ffn_norm_weight: pl.Tensor[[layers * hidden], pl.BF16],
        gate_weight: pl.Tensor[[layers * all_experts, hidden], pl.FP32],
        correction_bias: pl.Tensor[[layers * all_experts], pl.FP32],
        routed_w1: pl.Tensor[[layers * experts, inter, hidden], pl.INT8],
        routed_w2: pl.Tensor[[layers * experts, hidden, inter], pl.INT8],
        routed_w3: pl.Tensor[[layers * experts, inter, hidden], pl.INT8],
        routed_w1_scale: pl.Tensor[[layers * experts, inter], pl.FP32],
        routed_w2_scale: pl.Tensor[[layers * experts, hidden], pl.FP32],
        routed_w3_scale: pl.Tensor[[layers * experts, inter], pl.FP32],
        shared_w13: pl.Tensor[[layers, hidden, 2 * inter], pl.INT8, pl.NZ],
        shared_w2: pl.Tensor[[layers, inter, hidden], pl.INT8, pl.NZ],
        shared_w1_scale: pl.Tensor[[layers, inter], pl.FP32],
        shared_w2_scale: pl.Tensor[[layers, hidden], pl.FP32],
        shared_w3_scale: pl.Tensor[[layers, inter], pl.FP32],
        final_norm_weight: pl.Tensor[[hidden], pl.BF16],
        lm_head_weight: pl.Tensor[[local_vocab, hidden], pl.BF16],
        moe_num_tokens: pl.Tensor[[ep], pl.INT32],
        logit_row_indices: pl.Tensor[[head_rows], pl.INT32],
        logits: pl.Out[pl.Tensor[[head_rows, vocab], pl.FP32]],
        sampled_ids: pl.Out[pl.Tensor[[head_rows, sampled_pad], pl.INT32]],
        ffn_tape_residual: pl.Out[pl.Tensor[tape_residual, pl.FP32]],
        ffn_tape_mixes: pl.Out[pl.Tensor[tape_mixes, pl.FP32]],
        ffn_tape_vectors: pl.Out[pl.Tensor[tape_vectors, pl.BF16]],
        ffn_tape_route_ids: pl.Out[pl.Tensor[tape_routes, pl.INT32]],
        ffn_tape_route_weights: pl.Out[pl.Tensor[tape_routes, pl.FP32]],
        attention_tape_residual: pl.Out[pl.Tensor[attention_residual, pl.FP32]],
        attention_tape_mixes: pl.Out[pl.Tensor[attention_mixes, pl.FP32]],
        attention_tape_delta: pl.Out[pl.Tensor[attention_delta, pl.BF16]],
        attention_tape_wo_input: pl.Out[pl.Tensor[wo_input_shape, pl.BF16]],
        attention_tape_wo_latent: pl.Out[pl.Tensor[wo_latent_shape, pl.BF16]],
        attention_tape_normed: pl.Out[pl.Tensor[core_normed, pl.BF16]],
        attention_tape_query: pl.Out[pl.Tensor[core_query, pl.BF16]],
        attention_tape_temporary: pl.Out[pl.Tensor[core_temporary, pl.BF16]],
        attention_tape_context: pl.Out[pl.Tensor[core_context, pl.FP32]],
        attention_tape_lse: pl.Out[pl.Tensor[core_lse, pl.FP32]],
        attention_tape_selection: pl.Out[pl.Tensor[core_selection, pl.INT32]],
        attention_tape_lengths: pl.Out[pl.Tensor[core_lengths, pl.INT32]],
        attention_tape_q_normed: pl.Out[pl.Tensor[qkv_low, pl.BF16]],
        attention_tape_qkv_raw: pl.Out[pl.Tensor[qkv_raw, pl.BF16]],
        ffn_capture_owner: pl.Scalar[pl.INT32],
        recv_meta: pld.DistributedTensor[[ep, experts], pl.INT32],
        recv_x: pld.DistributedTensor[[recv_rows, hidden], pl.INT8],
        recv_weights: pld.DistributedTensor[[recv_rows, aux], pl.FP32],
        arrived: pld.DistributedTensor[[ep, signals], pl.INT32],
        data_arrived: pld.DistributedTensor[[ep, experts, signals], pl.INT32],
        routed_output: pld.DistributedTensor[[moe_rows * fanout, hidden], pl.BF16],
        combine_arrived: pld.DistributedTensor[[ep, experts, signals], pl.INT32],
        output_window: pld.DistributedTensor[[block, hidden], pl.BF16],
        output_arrived: pld.DistributedTensor[[tp, 1], pl.INT32],
        ffn_window: pld.DistributedTensor[[block, hidden], pl.BF16],
        ffn_arrived: pld.DistributedTensor[[tp, 1], pl.INT32],
        hidden_window: pld.DistributedTensor[[group_logit_rows, hidden], pl.BF16],
        hidden_done: pld.DistributedTensor[[tp, 1], pl.INT32],
        logits_window: pld.DistributedTensor[[head_rows, vocab], pl.FP32],
        logits_done: pld.DistributedTensor[[tp, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        count = pl.read(moe_num_tokens, [rank])
        source_mask = moe_source_mask(moe_num_tokens)
        lane, group = rank % tp, rank // tp * tp
        live_x = pl.slice(x_hc, [block, mix, hidden], [0, 0, 0])
        live_mix = pl.slice(pre_mix, [block, mix], [0, 0])
        selected = pl.create_tensor([block, index_topk], dtype=pl.INT32)
        lengths = pl.create_tensor([block], dtype=pl.INT32)
        mask = pl.create_tensor([block, index_topk], dtype=pl.FP32)
        # A group's writer must not overwrite rows still read by another group's queued tasks.
        history_banks = pl.create_tensor([history_groups, block * index_topk, dim], dtype=pl.BF16)
        for layer in pl.range(layers):
            with pl.scope():
                h_attention = pl.create_tensor([block, mix, hidden], dtype=pl.FP32)
                pre_attention = pl.create_tensor([block, mix], dtype=pl.FP32)
                ffn_input = pl.create_tensor([block, hidden], dtype=pl.BF16)
                h_attention, pre_attention, ffn_input = draft_attention_layer(
                    x_hc, pre_mix, engram_rows, engram_wkv, engram_weight, engram_rotation, hc_attn_fn,
                    hc_attn_scale, hc_attn_base, attn_norm_weight, draft_q_a_kv, draft_q_gamma, draft_q_b,
                    draft_kv_gamma, draft_sink, draft_wo_a, draft_wo_b, draft_index_q, draft_index_weight,
                    draft_cos, draft_sin, window_cache_pool, window_cache_scale_pool, compressed_cache_pool,
                    compressed_cache_scale_pool, index_cache_pool, index_cache_scale_pool, window_lookup,
                    compressed_lookup, window_rows, window_starts, window_spans, window_bias, c1a_table,
                    c2a_table, request_ids, c1a_visible, c2a_visible, zero_kv, attention_tape_residual,
                    attention_tape_mixes, attention_tape_delta, attention_tape_wo_input,
                    attention_tape_wo_latent, attention_tape_normed, attention_tape_query,
                    attention_tape_temporary, attention_tape_context, attention_tape_lse,
                    attention_tape_selection, attention_tape_lengths, attention_tape_q_normed,
                    attention_tape_qkv_raw, ffn_capture_owner, output_window, output_arrived, rank, live_x,
                    live_mix, selected, lengths, mask, history_banks, count, group, layer, layer + 1, h_attention,
                    pre_attention, ffn_input,
                )
                if count == 0:
                    h_attention, pre_attention = live_x, live_mix

                expert_base = layer * experts
                ffn_owned = pl.create_tensor([block, hidden], dtype=pl.BF16)
                ffn_post = pl.create_tensor([block, mix], dtype=pl.FP32)
                ffn_residual = pl.create_tensor([block, mix, mix], dtype=pl.FP32)
                local_normed = pl.create_tensor([moe_rows, hidden], dtype=pl.BF16)
                local_indices = pl.create_tensor([moe_rows, fanout], dtype=pl.INT32)
                local_weights = pl.create_tensor([moe_rows, fanout], dtype=pl.FP32)
                route_payload = pl.create_tensor([moe_rows, hidden], dtype=pl.INT8)
                route_scales = pl.create_tensor([1, moe_rows], dtype=pl.FP32)
                deps = pl.array.create(1, pl.TASK_ID)
                moe(
                    h_attention, pre_attention,
                    pl.slice(hc_ffn_fn, [mix_width, mix_input], [layer * mix_width, 0]),
                    pl.slice(hc_ffn_scale, [3], [layer * 3]),
                    pl.slice(hc_ffn_base, [mix_width], [layer * mix_width]),
                    pl.slice(ffn_norm_weight, [hidden], [layer * hidden]),
                    pl.slice(gate_weight, [all_experts, hidden], [layer * all_experts, 0]),
                    pl.slice(correction_bias, [all_experts], [layer * all_experts]),
                    pl.slice(routed_w1, [experts, inter, hidden], [expert_base, 0, 0]),
                    pl.slice(routed_w1_scale, [experts, inter], [expert_base, 0]),
                    pl.slice(routed_w2, [experts, hidden, inter], [expert_base, 0, 0]),
                    pl.slice(routed_w2_scale, [experts, hidden], [expert_base, 0]),
                    pl.slice(routed_w3, [experts, inter, hidden], [expert_base, 0, 0]),
                    pl.slice(routed_w3_scale, [experts, inter], [expert_base, 0]),
                    shared_w13[layer], pl.slice(shared_w1_scale, [1, inter], [layer, 0]),
                    shared_w2[layer], pl.slice(shared_w2_scale, [1, hidden], [layer, 0]),
                    pl.slice(shared_w3_scale, [1, inter], [layer, 0]),
                    recv_meta, recv_x, recv_weights, arrived, data_arrived, routed_output, combine_arrived,
                    ffn_input, ffn_owned, live_mix, ffn_post, ffn_residual, local_normed, local_indices, local_weights,
                    count, rank, lane, layer + 1, deps, False, False, route_payload, route_scales, source_mask,
                )
                mixed = pl.create_tensor([block, hidden], dtype=pl.BF16)
                restore(h_attention, ffn_owned, ffn_post, ffn_residual, ffn_window, ffn_arrived,
                        mixed, live_x, group, lane, count, layer + 1, prefill=False)
                if last_layers > 0:
                    if group == ffn_capture_owner and layer >= first_saved_layer:
                        save_ffn(h_attention, pre_attention, ffn_post, ffn_residual, ffn_input, mixed,
                                 local_normed, local_indices, local_weights, ffn_tape_residual, ffn_tape_mixes,
                                 ffn_tape_vectors, ffn_tape_route_ids, ffn_tape_route_weights, layer - first_saved_layer)
        clear(live_x, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
        retire(live_x, output_arrived, group, lane)
        retire(live_x, ffn_arrived, group, lane)
        logits, sampled_ids = output(
            live_x, live_mix, final_norm_weight, lm_head_weight, logit_row_indices, logits, sampled_ids,
            hidden_window, hidden_done, logits_window, logits_done, group, lane, 1, 1,
        )
        return logits, sampled_ids


    @pl.jit.host
    def draft_forward(
        x_hc: pl.InOut[pl.Tensor[[ep, capacity, mix, hidden], pl.FP32]],
        pre_mix: pl.InOut[pl.Tensor[[ep, capacity, mix], pl.FP32]],
        engram_rows: pl.Tensor[[ep, 2, block, engram_k], pl.BF16],
        engram_wkv: pl.Tensor[[ep, 2, engram_k, engram_out], pl.BF16, pl.NZ],
        engram_weight: pl.Tensor[[ep, 2, mix, hidden], pl.FP32],
        engram_rotation: pl.Tensor[[ep, hidden // 32, 32, 32], pl.FP32],
        hc_attn_fn: pl.Tensor[[ep, layers * mix_width, mix_input], pl.FP32],
        hc_attn_scale: pl.Tensor[[ep, layers * 3], pl.FP32],
        hc_attn_base: pl.Tensor[[ep, layers * mix_width], pl.FP32],
        attn_norm_weight: pl.Tensor[[ep, layers * hidden], pl.BF16],
        draft_q_a_kv: pl.Tensor[[ep, layers, hidden, latent + dim], pl.BF16],
        draft_q_gamma: pl.Tensor[[ep, layers, latent], pl.BF16],
        draft_q_b: pl.Tensor[[ep, layers, latent, heads * dim], pl.BF16],
        draft_kv_gamma: pl.Tensor[[ep, layers, dim], pl.BF16],
        draft_sink: pl.Tensor[[ep, layers, heads], pl.FP32],
        draft_wo_a: pl.Tensor[[ep, layers * groups, wo_in, wo_rank], pl.BF16, pl.NZ],
        draft_wo_b: pl.Tensor[[ep, layers, groups * wo_rank, hidden], pl.BF16, pl.NZ],
        draft_index_q: pl.Tensor[[ep, index_count, latent, index_heads * index_dim], pl.BF16],
        draft_index_weight: pl.Tensor[[ep, index_count, hidden, index_heads], pl.BF16],
        draft_cos: pl.Tensor[[ep, 2, capacity, norm_span], pl.FP32],
        draft_sin: pl.Tensor[[ep, 2, capacity, norm_span], pl.FP32],
        window_cache_pool: pl.Tensor[[ep, layers * pages, page, 1, dim], pl.UINT8],
        window_cache_scale_pool: pl.Tensor[[ep, layers * pages, page, 1, dim // win_scale], pl.BF16],
        compressed_cache_pool: pl.Tensor[[ep, kv_count * pages, page, 1, dim // 2], pl.UINT8],
        compressed_cache_scale_pool: pl.Tensor[[ep, kv_count * pages, page, 1, dim // kv_scale], pl.BF16],
        index_cache_pool: pl.Tensor[[ep, kv_count * pages, page, 1, index_dim // 2], pl.UINT8],
        index_cache_scale_pool: pl.Tensor[[ep, kv_count * pages, page, 1, index_dim // key_scale], pl.UINT8],
        window_lookup: pl.Tensor[[ep, 1, 256], pl.INT32],
        compressed_lookup: pl.Tensor[[ep, 1, 256], pl.INT32],
        window_rows: pl.Tensor[[ep, 8, 16], pl.INT32],
        window_starts: pl.Tensor[[ep, 8], pl.INT32],
        window_spans: pl.Tensor[[ep, 2, 3], pl.INT32],
        window_bias: pl.Tensor[[ep, 1, page], pl.FP32],
        c1a_table: pl.Tensor[[ep, 1, c1a_bucket // page], pl.INT32],
        c2a_table: pl.Tensor[[ep, 1, c1a_bucket // page], pl.INT32],
        request_ids: pl.Tensor[[ep, capacity], pl.INT32],
        c1a_visible: pl.Tensor[[ep, capacity], pl.INT32],
        c2a_visible: pl.Tensor[[ep, capacity], pl.INT32],
        zero_kv: pl.Tensor[[ep, page, dim], pl.BF16],
        hc_ffn_fn: pl.Tensor[[ep, layers * mix_width, mix_input], pl.FP32],
        hc_ffn_scale: pl.Tensor[[ep, layers * 3], pl.FP32],
        hc_ffn_base: pl.Tensor[[ep, layers * mix_width], pl.FP32],
        ffn_norm_weight: pl.Tensor[[ep, layers * hidden], pl.BF16],
        gate_weight: pl.Tensor[[ep, layers * all_experts, hidden], pl.FP32],
        correction_bias: pl.Tensor[[ep, layers * all_experts], pl.FP32],
        routed_w1: pl.Tensor[[ep, layers * experts, inter, hidden], pl.INT8],
        routed_w2: pl.Tensor[[ep, layers * experts, hidden, inter], pl.INT8],
        routed_w3: pl.Tensor[[ep, layers * experts, inter, hidden], pl.INT8],
        routed_w1_scale: pl.Tensor[[ep, layers * experts, inter], pl.FP32],
        routed_w2_scale: pl.Tensor[[ep, layers * experts, hidden], pl.FP32],
        routed_w3_scale: pl.Tensor[[ep, layers * experts, inter], pl.FP32],
        shared_w13: pl.Tensor[[ep, layers, hidden, 2 * inter], pl.INT8, pl.NZ],
        shared_w2: pl.Tensor[[ep, layers, inter, hidden], pl.INT8, pl.NZ],
        shared_w1_scale: pl.Tensor[[ep, layers, inter], pl.FP32],
        shared_w2_scale: pl.Tensor[[ep, layers, hidden], pl.FP32],
        shared_w3_scale: pl.Tensor[[ep, layers, inter], pl.FP32],
        final_norm_weight: pl.Tensor[[ep, hidden], pl.BF16],
        lm_head_weight: pl.Tensor[[ep, local_vocab, hidden], pl.BF16],
        moe_num_tokens: pl.Tensor[[ep], pl.INT32],
        logit_row_indices: pl.Tensor[[ep, head_rows], pl.INT32],
        logits: pl.Out[pl.Tensor[[ep, head_rows, vocab], pl.FP32]],
        sampled_ids: pl.Out[pl.Tensor[[ep, head_rows, sampled_pad], pl.INT32]],
        ffn_tape_residual: pl.Out[pl.Tensor[[ep, *tape_residual], pl.FP32]],
        ffn_tape_mixes: pl.Out[pl.Tensor[[ep, *tape_mixes], pl.FP32]],
        ffn_tape_vectors: pl.Out[pl.Tensor[[ep, *tape_vectors], pl.BF16]],
        ffn_tape_route_ids: pl.Out[pl.Tensor[[ep, *tape_routes], pl.INT32]],
        ffn_tape_route_weights: pl.Out[pl.Tensor[[ep, *tape_routes], pl.FP32]],
        attention_tape_residual: pl.Out[pl.Tensor[[ep, *attention_residual], pl.FP32]],
        attention_tape_mixes: pl.Out[pl.Tensor[[ep, *attention_mixes], pl.FP32]],
        attention_tape_delta: pl.Out[pl.Tensor[[ep, *attention_delta], pl.BF16]],
        attention_tape_wo_input: pl.Out[pl.Tensor[[ep, *wo_input_shape], pl.BF16]],
        attention_tape_wo_latent: pl.Out[pl.Tensor[[ep, *wo_latent_shape], pl.BF16]],
        attention_tape_normed: pl.Out[pl.Tensor[[ep, *core_normed], pl.BF16]],
        attention_tape_query: pl.Out[pl.Tensor[[ep, *core_query], pl.BF16]],
        attention_tape_temporary: pl.Out[pl.Tensor[[ep, *core_temporary], pl.BF16]],
        attention_tape_context: pl.Out[pl.Tensor[[ep, *core_context], pl.FP32]],
        attention_tape_lse: pl.Out[pl.Tensor[[ep, *core_lse], pl.FP32]],
        attention_tape_selection: pl.Out[pl.Tensor[[ep, *core_selection], pl.INT32]],
        attention_tape_lengths: pl.Out[pl.Tensor[[ep, *core_lengths], pl.INT32]],
        attention_tape_q_normed: pl.Out[pl.Tensor[[ep, *qkv_low], pl.BF16]],
        attention_tape_qkv_raw: pl.Out[pl.Tensor[[ep, *qkv_raw], pl.BF16]],
        ffn_capture_owner: pl.Scalar[pl.INT32],
    ):
        meta_buf = pld.alloc_window_buffer([ep, experts], dtype=pl.INT32)
        recv_buf = pld.alloc_window_buffer([recv_rows, hidden], dtype=pl.INT8)
        weight_buf = pld.alloc_window_buffer([recv_rows, aux], dtype=pl.FP32)
        arrived_buf = pld.alloc_window_buffer([ep, signals], dtype=pl.INT32)
        data_arrived_buf = pld.alloc_window_buffer([ep, experts, signals], dtype=pl.INT32)
        routed_buf = pld.alloc_window_buffer([moe_rows * fanout, hidden], dtype=pl.BF16)
        combine_arrived_buf = pld.alloc_window_buffer([ep, experts, signals], dtype=pl.INT32)
        output_buf = pld.alloc_window_buffer([block, hidden], dtype=pl.BF16)
        output_arrived_buf = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        ffn_buf = pld.alloc_window_buffer([block, hidden], dtype=pl.BF16)
        ffn_arrived_buf = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        head_hidden_buf = pld.alloc_window_buffer([group_logit_rows, hidden], dtype=pl.BF16)
        head_hidden_done_buf = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        head_logits_buf = pld.alloc_window_buffer([head_rows, vocab], dtype=pl.FP32)
        head_logits_done_buf = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            recv_meta = pld.window(meta_buf, [ep, experts], dtype=pl.INT32)
            recv_x = pld.window(recv_buf, [recv_rows, hidden], dtype=pl.INT8)
            recv_weights = pld.window(weight_buf, [recv_rows, aux], dtype=pl.FP32)
            arrived = pld.window(arrived_buf, [ep, signals], dtype=pl.INT32)
            data_arrived = pld.window(data_arrived_buf, [ep, experts, signals], dtype=pl.INT32)
            routed_output = pld.window(routed_buf, [moe_rows * fanout, hidden], dtype=pl.BF16)
            combine_arrived = pld.window(combine_arrived_buf, [ep, experts, signals], dtype=pl.INT32)
            output_window = pld.window(output_buf, [block, hidden], dtype=pl.BF16)
            output_arrived = pld.window(output_arrived_buf, [tp, 1], dtype=pl.INT32)
            ffn_window = pld.window(ffn_buf, [block, hidden], dtype=pl.BF16)
            ffn_arrived = pld.window(ffn_arrived_buf, [tp, 1], dtype=pl.INT32)
            head_hidden = pld.window(head_hidden_buf, [group_logit_rows, hidden], dtype=pl.BF16)
            head_hidden_done = pld.window(head_hidden_done_buf, [tp, 1], dtype=pl.INT32)
            head_logits = pld.window(head_logits_buf, [head_rows, vocab], dtype=pl.FP32)
            head_logits_done = pld.window(head_logits_done_buf, [tp, 1], dtype=pl.INT32)
            draft_rank(
                x_hc[rank], pre_mix[rank], engram_rows[rank], engram_wkv[rank], engram_weight[rank], engram_rotation[rank],
                hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank],
                draft_q_a_kv[rank], draft_q_gamma[rank], draft_q_b[rank], draft_kv_gamma[rank], draft_sink[rank],
                draft_wo_a[rank], draft_wo_b[rank], draft_index_q[rank], draft_index_weight[rank],
                draft_cos[rank], draft_sin[rank], window_cache_pool[rank], window_cache_scale_pool[rank],
                compressed_cache_pool[rank], compressed_cache_scale_pool[rank], index_cache_pool[rank],
                index_cache_scale_pool[rank], window_lookup[rank], compressed_lookup[rank], window_rows[rank],
                window_starts[rank], window_spans[rank], window_bias[rank], c1a_table[rank], c2a_table[rank],
                request_ids[rank], c1a_visible[rank], c2a_visible[rank], zero_kv[rank],
                hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], ffn_norm_weight[rank], gate_weight[rank],
                correction_bias[rank], routed_w1[rank], routed_w2[rank], routed_w3[rank], routed_w1_scale[rank],
                routed_w2_scale[rank], routed_w3_scale[rank], shared_w13[rank], shared_w2[rank], shared_w1_scale[rank],
                shared_w2_scale[rank], shared_w3_scale[rank], final_norm_weight[rank], lm_head_weight[rank],
                moe_num_tokens, logit_row_indices[rank], logits[rank], sampled_ids[rank],
                ffn_tape_residual[rank], ffn_tape_mixes[rank], ffn_tape_vectors[rank],
                ffn_tape_route_ids[rank], ffn_tape_route_weights[rank], attention_tape_residual[rank],
                attention_tape_mixes[rank], attention_tape_delta[rank], attention_tape_wo_input[rank],
                attention_tape_wo_latent[rank], attention_tape_normed[rank], attention_tape_query[rank],
                attention_tape_temporary[rank], attention_tape_context[rank], attention_tape_lse[rank],
                attention_tape_selection[rank], attention_tape_lengths[rank], attention_tape_q_normed[rank],
                attention_tape_qkv_raw[rank], ffn_capture_owner,
                recv_meta, recv_x, recv_weights, arrived, data_arrived, routed_output, combine_arrived,
                output_window, output_arrived, ffn_window, ffn_arrived,
                head_hidden, head_hidden_done, head_logits, head_logits_done, rank, device=rank,
            )
        return logits, sampled_ids

    return draft_forward
