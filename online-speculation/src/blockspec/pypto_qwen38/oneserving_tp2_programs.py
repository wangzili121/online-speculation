"""Orthrus K32 program for the OneServing packed TP2 Qwen3.8 worker."""

from .attention import AttentionShape
from .attention_programs import make_attention
from .paged_attention_programs import make_paged_attention


# One device dispatch by default, with an option to split at full-attention boundaries.
GROUP_LAYERS = 64


def make_draft_program(*, pool_pages, group_layers=GROUP_LAYERS, attention_backend="paged",
                       batch_size=1, compact_tp=False, capture_layer=None, draft_convolution="sequential",
                       gdn_valid_rows=False, physical_block_rows=32):
    from .tp2_row_layout import row_geometry, make_pack, make_expand, make_feedback
    external_rows, rows_per_dispatch = row_geometry(batch_size, physical_block_rows)
    if group_layers not in (4, 8, 16, 32, 64):
        raise ValueError("draft layer groups must divide the 64-layer backbone and end on full attention")
    if batch_size not in (1, 2) or batch_size == 2 and not compact_tp:
        raise ValueError("a paired draft requires compact TP")
    if draft_convolution not in ("sequential", "parallel64", "parallel128"):
        raise ValueError("unsupported draft convolution backend")
    if type(gdn_valid_rows) is not bool:
        raise ValueError("gdn_valid_rows must be a boolean")
    capture_feedback = capture_layer is not None
    # JIT 会解析条件分支；关闭反馈时也须保持层号比较为数值类型。
    capture_index = capture_layer if capture_feedback else -1
    fused_residual = batch_size == 2
    window_rows = rows_per_dispatch if compact_tp else 16384

    import pypto.language as pl
    import pypto.language.distributed as pld
    from serving.model import (
        delta_gate, delta_normalize, embed, ffn_gate_up_nz_m32, ffn_gate_up_nz_m64,
        linear_bf16_m32, linear_bf16_m64, linear_partial_m32, linear_partial_m64,
        packed_greedy_fwd, prepare_heads,
        residual_add, rms_norm, tp_finish, tp_sum,
    )

    projection = linear_bf16_m64 if rows_per_dispatch == 64 else linear_bf16_m32
    row_projection = linear_partial_m64 if rows_per_dispatch == 64 else linear_partial_m32
    mlp_gate_up = ffn_gate_up_nz_m64 if rows_per_dispatch == 64 else ffn_gate_up_nz_m32
    pack_hidden = make_pack(batch_size, physical_block_rows, 5120)
    pack_rope = make_pack(batch_size, physical_block_rows, 64)
    expand_hidden = make_expand(batch_size, physical_block_rows, 5120)
    expand_logits = make_expand(batch_size, physical_block_rows, 124160)
    capture_input = make_feedback(physical_block_rows)

    # 专门化器需要从实参或 create_tensor 直接推导 LM 输出形状。
    if physical_block_rows == 32:
        @pl.jit.inline
        def project_logits(hidden: pl.Tensor[[rows_per_dispatch, 5120], pl.BF16], weight: pl.Tensor,
                           storage: pl.Out[pl.Tensor[[external_rows, 124160], pl.BF16]]):
            projection(hidden, weight, storage)
            return storage
    else:
        @pl.jit.inline
        def project_logits(hidden: pl.Tensor[[rows_per_dispatch, 5120], pl.BF16], weight: pl.Tensor,
                           storage: pl.Out[pl.Tensor[[external_rows, 124160], pl.BF16]]):
            compact = pl.create_tensor([rows_per_dispatch, 124160], dtype=pl.BF16)
            projection(hidden, weight, compact)
            return expand_logits(compact, storage)

    @pl.jit.inline(auto_scope=False)
    def reduce_draft(partial: pl.Tensor,
                     window: pld.DistributedTensor[[rows_per_dispatch, 5120], pl.FP32],
                     signal: pld.DistributedTensor[[2, 1], pl.INT32], out: pl.Tensor,
                     residual_input: pl.Tensor,
                     rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32]):
        # Both ranks must retire the preceding read before either overwrites its window.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="orthrus_tp_reuse",
                   allow_early_resolve=False) as reuse:
            for peer in pl.range(2):
                pld.system.wait(signal, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge)
        with pl.spmd(32, name_hint="orthrus_tp_publish", deps=[reuse]) as publish:
            for tile in pl.range(pl.tile.get_block_idx(), rows_per_dispatch // 4 * 10, 32):
                row = tile // 10 * 4
                col = tile % 10 * 512
                value = pl.load(partial, [row, col], [4, 512])
                pld.tile.remote_store(value, window, peer=rank, offsets=[row, col])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="orthrus_tp_ready", deps=[publish]) as ready:
            for peer in pl.range(2):
                pld.system.notify(signal, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="orthrus_tp_wait",
                   deps=[ready], allow_early_resolve=False) as wait:
            pld.system.wait(signal, offsets=[1 - rank, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
        with pl.spmd(32, name_hint="orthrus_tp_reduce", deps=[wait]) as reduced:
            for tile in pl.range(pl.tile.get_block_idx(), rows_per_dispatch // 4 * 10, 32):
                row = tile // 10 * 4
                col = tile % 10 * 512
                own = pl.load(partial, [row, col], [4, 512])
                other = pld.tile.remote_load(window, peer=1 - rank, offsets=[row, col], shape=[4, 512])
                rounded_sum = pl.cast(pl.add(own, other), pl.BF16, mode="rint")
                if fused_residual:
                    # Keep the separate reduction's BF16 rounding before adding the skip path.
                    skip = pl.cast(pl.load(residual_input, [row, col], [4, 512]), pl.FP32)
                    with_skip = pl.cast(pl.add(skip, pl.cast(rounded_sum, pl.FP32)), pl.BF16, mode="rint")
                    out = pl.store(with_skip, [row, col], out)
                else:
                    out = pl.store(rounded_sum, [row, col], out)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="orthrus_tp_release", deps=[reduced]):
            for peer in pl.range(2):
                pld.system.notify(signal, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        return out

    @pl.jit.inline(auto_scope=False)
    def reduce_and_add(partial: pl.Tensor,
                       window: pld.DistributedTensor[[window_rows, 5120], pl.FP32],
                       signal: pld.DistributedTensor[[2, 1], pl.INT32],
                       skip: pl.Tensor, reduced: pl.Tensor, out: pl.Tensor,
                       rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32]):
        if fused_residual:
            reduce_draft(partial, window, signal, out, skip, rank, epoch)
        elif compact_tp:
            reduce_draft(partial, window, signal, reduced, skip, rank, epoch)
            residual_add(skip, reduced, out)
        else:
            tp_sum(partial, window, signal, reduced, rank, epoch)
            residual_add(skip, reduced, out)
        return out

    @pl.jit.inline
    def parameters(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor[[1, 32], pl.FP32], dt: pl.Tensor,
                   decay: pl.Tensor, beta: pl.Tensor):
        # Match native decay and beta while preserving the learned FP32 A_log.
        with pl.spmd(rows_per_dispatch // 16, name_hint="orthrus_tp2_gdn_parameters"):
            row = pl.tile.get_block_idx() * 16
            av = pl.cast(pl.slice(a, [16, 32], [row, 0]), pl.FP32)
            bv = pl.cast(pl.slice(b, [16, 32], [row, 0]), pl.FP32)
            bias = pl.cast(pl.slice(dt, [1, 32], [0, 0]), pl.FP32)
            x = pl.col_expand_add(av, bias)
            softplus = pl.add(pl.maximum(x, 0.), pl.log(pl.add(pl.exp(pl.neg(pl.abs(x))), 1.)))
            decay[row:row + 16, 0:32] = pl.neg(pl.col_expand_mul(softplus, pl.exp(a_log)))
            y = pl.div(pl.full([16, 32], pl.FP32, value=1.), pl.add(pl.exp(pl.neg(bv)), 1.),
                       high_precision=True)
            beta[row:row + 16, 0:32] = pl.cast(pl.cast(y, pl.BF16, mode="rint"), pl.FP32)
        return decay, beta

    shape = AttentionShape(block_size=physical_block_rows, q_heads=12, kv_heads=2, page_size=128,
                           query_tile=physical_block_rows)
    if attention_backend == "paged":
        attention, _ = make_paged_attention(shape, physical_pages=pool_pages, pool_layers=17)
    else:
        attention, _ = make_attention(shape, backend=attention_backend, physical_pages=pool_pages, pool_layers=17)

    from .tp2_gdn_convolution import make_convolution
    convolution = make_convolution(batch_size, backend="sequential" if draft_convolution == "sequential" else "parallel",
                                   channel_tile=64 if draft_convolution == "parallel64" else 128,
                                   block_rows=physical_block_rows)

    from .tp2_gdn_rows import make_recurrence
    recurrent = make_recurrence(batch_size, valid_rows=gdn_valid_rows, block_rows=physical_block_rows)

    @pl.jit.inline(auto_scope=False)
    def gdn(x: pl.Tensor, qw: pl.Tensor, zw: pl.Tensor, ow: pl.Tensor, aw: pl.Tensor,
            bw: pl.Tensor, cw: pl.Tensor, al: pl.Tensor[[1, 32], pl.FP32], dt: pl.Tensor, norm: pl.Tensor,
            conv_state: pl.Tensor, recurrent_state: pl.Tensor, partial: pl.Tensor,
            slot: pl.Scalar[pl.INT32], slot_second: pl.Scalar[pl.INT32],
            layer: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32], width_second: pl.Scalar[pl.INT32]):
        with pl.scope():
            projected = pl.create_tensor([rows_per_dispatch, 5120], dtype=pl.BF16)
            z = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.BF16)
            a = pl.create_tensor([rows_per_dispatch, 128], dtype=pl.BF16)
            b = pl.create_tensor([rows_per_dispatch, 128], dtype=pl.BF16)
            conv = pl.create_tensor([rows_per_dispatch, 5120], dtype=pl.BF16)
            decay = pl.create_tensor([rows_per_dispatch, 32], dtype=pl.FP32)
            beta = pl.create_tensor([rows_per_dispatch, 32], dtype=pl.FP32)
            qn = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.FP32)
            kn = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.FP32)
            core = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.BF16)
            gated = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.BF16)
            projection(x, qw, projected)
            projection(x, zw, z)
            projection(x, aw, a)
            projection(x, bw, b)
            convolution(projected, cw, conv_state, conv, slot, slot_second, layer)
            parameters(a, b, al, dt, decay, beta)
            delta_normalize(conv, qn, kn)
            recurrent(qn, kn, conv, decay, beta, recurrent_state, core, slot, slot_second, layer,
                      width, width_second)
            delta_gate(core, z, norm, gated)
            row_projection(gated, ow, partial)
        return partial

    @pl.jit.inline(auto_scope=False)
    def full(x: pl.Tensor, qw: pl.Tensor, kw: pl.Tensor, vw: pl.Tensor, qgamma: pl.Tensor,
             kgamma: pl.Tensor, ow: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
             key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
             partial: pl.Tensor, layer: pl.Scalar[pl.INT32], past: pl.Scalar[pl.INT32],
             width: pl.Scalar[pl.INT32], past_second: pl.Scalar[pl.INT32],
             width_second: pl.Scalar[pl.INT32]):
        with pl.scope():
            qp = pl.create_tensor([rows_per_dispatch, 6144], dtype=pl.BF16)
            kp = pl.create_tensor([rows_per_dispatch, 512], dtype=pl.BF16)
            v = pl.create_tensor([rows_per_dispatch, 512], dtype=pl.BF16)
            q = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.BF16)
            k = pl.create_tensor([rows_per_dispatch, 512], dtype=pl.BF16)
            gate = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.BF16)
            attended = pl.create_tensor([rows_per_dispatch, 3072], dtype=pl.BF16)
            projection(x, qw, qp)
            projection(x, kw, kp)
            projection(x, vw, v)
            prepare_heads(qp, kp, qgamma, kgamma, cos, sin, q, k, gate)
            for seq in pl.range(batch_size):
                first = seq * physical_block_rows
                local_past = past + seq * (past_second - past)
                local_width = width + seq * (width_second - width)
                table_row = pl.slice(table, [1, 4], [seq, 0])
                local_table: pl.Tensor[[4], pl.INT32] = pl.reshape(table_row, [4])
                local_bias = pl.slice(bias, [1, 512], [seq, 0])
                local_q = pl.slice(q, [physical_block_rows, 3072], [first, 0])
                local_k = pl.slice(k, [physical_block_rows, 512], [first, 0])
                local_v = pl.slice(v, [physical_block_rows, 512], [first, 0])
                local_gate = pl.slice(gate, [physical_block_rows, 3072], [first, 0])
                local_output = pl.create_tensor([physical_block_rows, 3072], dtype=pl.BF16)
                attention(local_q, local_k, local_v, local_gate, key_pool, value_pool,
                          local_table, local_bias, local_output, local_past, local_width, layer)
                with pl.spmd(physical_block_rows // 16 * 24, name_hint="orthrus_store_attention_block"):
                    tile = pl.tile.get_block_idx()
                    row = tile // 24 * 16
                    col = tile % 24 * 128
                    attended[first + row:first + row + 16, col:col + 128] = pl.slice(
                        local_output, [16, 128], [row, col])
            row_projection(attended, ow, partial)
        return partial

    @pl.jit
    def embed_draft(ids: pl.Tensor[[external_rows], pl.INT32], rows: pl.Tensor[[external_rows], pl.INT32],
                    embedding: pl.Tensor, hidden: pl.Out[pl.Tensor[[external_rows, 5120], pl.BF16]]):
        # The pinned compiler misbinds the loop-carried hidden buffer if this is
        # a conditional write inside the same device entry as the layer loop.
        return embed(ids, embedding, rows, hidden)

    @pl.jit(auto_scope=False)
    def predict(
        final_gamma: pl.Tensor, lm_head: pl.Tensor,
        input_gamma: pl.Tensor, post_gamma: pl.Tensor,
        mlp_gate: pl.Tensor[[64, 8704, 5120], pl.BF16, pl.NZ],
        mlp_up: pl.Tensor[[64, 8704, 5120], pl.BF16, pl.NZ], mlp_down: pl.Tensor,
        delta_qkv: pl.Tensor, delta_z: pl.Tensor, delta_out: pl.Tensor,
        delta_a: pl.Tensor, delta_b: pl.Tensor, delta_conv: pl.Tensor,
        delta_a_log: pl.Tensor, delta_dt_bias: pl.Tensor, delta_norm: pl.Tensor,
        full_q: pl.Tensor, full_k: pl.Tensor, full_v: pl.Tensor, full_q_gamma: pl.Tensor,
        full_k_gamma: pl.Tensor, full_out: pl.Tensor,
        head_qkv: pl.Tensor, head_z: pl.Tensor, head_out: pl.Tensor,
        head_a: pl.Tensor, head_b: pl.Tensor, head_conv: pl.Tensor,
        head_a_log: pl.Tensor, head_dt_bias: pl.Tensor, head_norm: pl.Tensor,
        head_q: pl.Tensor, head_k: pl.Tensor, head_v: pl.Tensor, head_q_gamma: pl.Tensor,
        head_k_gamma: pl.Tensor, head_o: pl.Tensor,
        conv_state: pl.Tensor, recurrent_state: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor,
        cos_storage: pl.Tensor, sin_storage: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
        hidden_storage: pl.InOut[pl.Tensor[[external_rows, 5120], pl.BF16]],
        logits_storage: pl.Out[pl.Tensor[[external_rows, 124160], pl.BF16]],
        feedback: pl.InOut[pl.Tensor[[32, 5120], pl.BF16]],
        feedback_row: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
        slot_second: pl.Scalar[pl.INT32], past_second: pl.Scalar[pl.INT32],
        width_second: pl.Scalar[pl.INT32], begin: pl.Scalar[pl.INT32],
        comm_data: pld.DistributedTensor[[window_rows, 5120], pl.FP32],
        comm_signals: pld.DistributedTensor[[2, 1], pl.INT32], rank: pl.Scalar[pl.INT32],
    ):
        with pl.scope():
            hidden: pl.Tensor[[rows_per_dispatch, 5120], pl.BF16] = pack_hidden(hidden_storage)
            cos: pl.Tensor[[rows_per_dispatch, 64], pl.BF16] = pack_rope(cos_storage)
            sin: pl.Tensor[[rows_per_dispatch, 64], pl.BF16] = pack_rope(sin_storage)
            normalized = pl.create_tensor([rows_per_dispatch, 5120], dtype=pl.BF16)
            partial = pl.create_tensor([rows_per_dispatch, 5120], dtype=pl.FP32)
            reduced = pl.create_tensor([rows_per_dispatch, 5120], dtype=pl.BF16)
            residual = pl.create_tensor([rows_per_dispatch, 5120], dtype=pl.BF16)
            post_norm = pl.create_tensor([rows_per_dispatch, 5120], dtype=pl.BF16)
            middle = pl.create_tensor([rows_per_dispatch, 8704], dtype=pl.BF16)
            for offset in pl.range(group_layers):
                layer = begin + offset
                if capture_feedback:
                    if layer == capture_index:
                        if feedback_row >= 0:
                            if rank == 0:
                                feedback = capture_input(hidden, feedback, feedback_row)
                with pl.scope():
                    gamma = pl.slice(input_gamma, [1, 5120], [layer, 0])
                    rms_norm(hidden, gamma, normalized)
                    if layer % 4 != 3:
                        linear = layer - layer // 4
                        if layer < 60:
                            qw = pl.slice(delta_qkv, [5120, 5120], [linear * 5120, 0])
                            zw = pl.slice(delta_z, [3072, 5120], [linear * 3072, 0])
                            ow = pl.slice(delta_out, [5120, 3072], [linear * 5120, 0])
                            aw = pl.slice(delta_a, [128, 5120], [linear * 128, 0])
                            bw = pl.slice(delta_b, [128, 5120], [linear * 128, 0])
                            cw = pl.slice(delta_conv, [4, 5120], [linear * 4, 0])
                            al = pl.slice(delta_a_log, [1, 32], [linear, 0])
                            dt = pl.slice(delta_dt_bias, [1, 32], [linear, 0])
                            norm = pl.slice(delta_norm, [1, 128], [linear, 0])
                            gdn(normalized, qw, zw, ow, aw, bw, cw, al, dt, norm,
                                conv_state, recurrent_state, partial, slot, slot_second, linear, width, width_second)
                        else:
                            tail = layer - 60
                            tqw = pl.slice(head_qkv, [5120, 5120], [tail * 5120, 0])
                            tzw = pl.slice(head_z, [3072, 5120], [tail * 3072, 0])
                            tow = pl.slice(head_out, [5120, 3072], [tail * 5120, 0])
                            taw = pl.slice(head_a, [128, 5120], [tail * 128, 0])
                            tbw = pl.slice(head_b, [128, 5120], [tail * 128, 0])
                            tcw = pl.slice(head_conv, [4, 5120], [tail * 4, 0])
                            tal = pl.slice(head_a_log, [1, 32], [tail, 0])
                            tdt = pl.slice(head_dt_bias, [1, 32], [tail, 0])
                            tnorm = pl.slice(head_norm, [1, 128], [tail, 0])
                            gdn(normalized, tqw, tzw, tow, taw, tbw, tcw, tal, tdt, tnorm,
                                conv_state, recurrent_state, partial, slot, slot_second, linear, width, width_second)
                    else:
                        index = layer // 4
                        if layer < 63:
                            aq = pl.slice(full_q, [6144, 5120], [index * 6144, 0])
                            ak = pl.slice(full_k, [512, 5120], [index * 512, 0])
                            av = pl.slice(full_v, [512, 5120], [index * 512, 0])
                            qg = pl.slice(full_q_gamma, [1, 256], [index, 0])
                            kg = pl.slice(full_k_gamma, [1, 256], [index, 0])
                            ao = pl.slice(full_out, [5120, 3072], [index * 5120, 0])
                            full(normalized, aq, ak, av, qg, kg, ao, cos, sin,
                                 key_pool, value_pool, table, bias, partial, index, past, width, past_second, width_second)
                        else:
                            full(normalized, head_q, head_k, head_v, head_q_gamma, head_k_gamma, head_o,
                                 cos, sin, key_pool, value_pool, table, bias, partial, index, past, width, past_second, width_second)
                reduce_and_add(partial, comm_data, comm_signals, hidden, reduced, residual, rank, offset * 2 + 1)
                post = pl.slice(post_gamma, [1, 5120], [layer, 0])
                rms_norm(residual, post, post_norm)
                gate_w = pl.slice(mlp_gate, [1, 8704, 5120], [layer, 0, 0])
                up_w = pl.slice(mlp_up, [1, 8704, 5120], [layer, 0, 0])
                down_w = pl.slice(mlp_down, [5120, 8704], [layer * 5120, 0])
                with pl.scope():
                    mlp_gate_up(post_norm, gate_w, up_w, middle)
                    row_projection(middle, down_w, partial)
                reduce_and_add(partial, comm_data, comm_signals, residual, reduced, hidden, rank, offset * 2 + 2)
            if begin == 64 - group_layers:
                rms_norm(hidden, final_gamma, hidden)
                logits_storage = project_logits(hidden, lm_head, logits_storage)
            hidden_storage = expand_hidden(hidden, hidden_storage)
            tp_finish(comm_signals, rank, group_layers * 2)
        return hidden_storage, logits_storage

    @pl.jit.host
    def draft_host(
        ids: pl.Tensor, rows: pl.Tensor, embedding: pl.Tensor, final_gamma: pl.Tensor, lm_head: pl.Tensor,
        input_gamma: pl.Tensor, post_gamma: pl.Tensor,
        mlp_gate: pl.Tensor[[2, 64, 8704, 5120], pl.BF16],
        mlp_up: pl.Tensor[[2, 64, 8704, 5120], pl.BF16], mlp_down: pl.Tensor,
        delta_qkv: pl.Tensor, delta_z: pl.Tensor, delta_out: pl.Tensor,
        delta_a: pl.Tensor, delta_b: pl.Tensor, delta_conv: pl.Tensor, delta_a_log: pl.Tensor,
        delta_dt_bias: pl.Tensor, delta_norm: pl.Tensor, full_q: pl.Tensor, full_k: pl.Tensor,
        full_v: pl.Tensor, full_q_gamma: pl.Tensor, full_k_gamma: pl.Tensor, full_out: pl.Tensor,
        head_qkv: pl.Tensor, head_z: pl.Tensor, head_out: pl.Tensor, head_a: pl.Tensor,
        head_b: pl.Tensor, head_conv: pl.Tensor, head_a_log: pl.Tensor, head_dt_bias: pl.Tensor,
        head_norm: pl.Tensor, head_q: pl.Tensor, head_k: pl.Tensor, head_v: pl.Tensor,
        head_q_gamma: pl.Tensor, head_k_gamma: pl.Tensor, head_o: pl.Tensor,
        conv_state: pl.Tensor, recurrent_state: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor,
        cos: pl.Tensor, sin: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
        hidden: pl.InOut[pl.Tensor], logits: pl.Out[pl.Tensor],
        valid: pl.Tensor, decisions: pl.Out[pl.Tensor],
        feedback: pl.InOut[pl.Tensor], feedback_row: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
        slot_second: pl.Scalar[pl.INT32], past_second: pl.Scalar[pl.INT32],
        width_second: pl.Scalar[pl.INT32], begin: pl.Scalar[pl.INT32],
    ):
        data = pld.alloc_window_buffer([window_rows, 5120], dtype=pl.FP32)
        signals = pld.alloc_window_buffer([2, 1], dtype=pl.INT32)
        for rank in pl.range(2):
            window = pld.window(data, [window_rows, 5120], dtype=pl.FP32)
            signal = pld.window(signals, [2, 1], dtype=pl.INT32)
            if begin == 0:
                embed_draft(ids[rank], rows[rank], embedding[rank], hidden[rank], device=rank)
            predict(
                final_gamma[rank], lm_head[rank],
                input_gamma[rank], post_gamma[rank], mlp_gate[rank], mlp_up[rank], mlp_down[rank],
                delta_qkv[rank], delta_z[rank], delta_out[rank], delta_a[rank], delta_b[rank],
                delta_conv[rank], delta_a_log[rank], delta_dt_bias[rank], delta_norm[rank],
                full_q[rank], full_k[rank], full_v[rank], full_q_gamma[rank], full_k_gamma[rank], full_out[rank],
                head_qkv[rank], head_z[rank], head_out[rank], head_a[rank], head_b[rank], head_conv[rank],
                head_a_log[rank], head_dt_bias[rank], head_norm[rank], head_q[rank], head_k[rank],
                head_v[rank], head_q_gamma[rank], head_k_gamma[rank], head_o[rank],
                conv_state[rank], recurrent_state[rank], key_pool[rank], value_pool[rank],
                cos[rank], sin[rank], table[rank], bias[rank], hidden[rank], logits[rank],
                feedback[rank], feedback_row,
                slot, past, width, slot_second, past_second, width_second, begin,
                window, signal, rank, device=rank)
            if begin == 64 - group_layers:
                packed_greedy_fwd(logits[rank], valid[rank], decisions[rank], rank, device=rank)

    return draft_host
