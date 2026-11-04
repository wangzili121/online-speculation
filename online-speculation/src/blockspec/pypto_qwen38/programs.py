"""Draft orchestration built from the pinned model's existing PyPTO primitives.

The native primitives operate on 16 rows. A K32 layer therefore prepares both
halves before full attention, rather than finishing the first half's network
before the second half exists. No target cache is an output of these graphs.
"""

from .baseline import native_module


def make_programs(source):
    import pypto.language as pl

    model = native_module(source, "model")
    linear = model.linear_hidden_projection_inline
    norm = model.rmsnorm_rmsnorm_inline
    residual = model.residual_residual_add_inline
    swiglu = model.swiglu_swiglu_inline
    stem = model.layer0_stem_layer0_stem_inline
    convolution = model.causal_conv_causal_conv_silu_inline
    expand = model.delta_window16_expand_qkv_window16_inline
    recurrence = model.delta_window16_recurrence_window16_inline
    tail = model.delta_window16_attention_tail_window16_inline
    prepare = model.attention3_prepare_attention3_prepare_inline
    attention = model.attention512_core_attention512_core_inline

    @pl.jit.inline
    def gates(a: pl.Tensor, b: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
              log_decay: pl.Tensor, beta: pl.Tensor):
        # Same decay/beta formula as the native model. Trained A_log is already
        # FP32; this compiler rejects the native helper's redundant FP32 cast.
        with pl.spmd(1, name_hint="orthrus_decay_beta"):
            row = pl.tile.get_block_idx() * 16
            a32 = pl.cast(pl.slice(a, [16, 64], [row, 0]), pl.FP32)
            b32 = pl.cast(pl.slice(b, [16, 64], [row, 0]), pl.FP32)
            bias = pl.cast(pl.slice(dt_bias, [1, 64], [0, 0]), pl.FP32)
            parameter = pl.slice(a_log, [1, 64], [0, 0])
            softplus = pl.log(pl.add(pl.exp(pl.col_expand_add(a32, bias)), 1.0))
            log_decay[row:row + 16, 0:64] = pl.mul(pl.col_expand_mul(softplus, pl.exp(parameter)), -1.0)
            beta[row:row + 16, 0:64] = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(b32)), 1.0)), pl.BF16, mode="rint")
        return log_decay, beta

    @pl.jit.inline
    def shared_tail(hidden: pl.Tensor, attn: pl.Tensor, post_gamma: pl.Tensor,
                    gate_weight: pl.Tensor, up_weight: pl.Tensor, down_weight: pl.Tensor,
                    result: pl.Tensor, layer: pl.Scalar[pl.INT32]):
        gamma = pl.slice(post_gamma, [1, 5120], [layer, 0])
        wg = pl.slice(gate_weight, [17408, 5120], [layer * 17408, 0])
        wu = pl.slice(up_weight, [17408, 5120], [layer * 17408, 0])
        wd = pl.slice(down_weight, [5120, 17408], [layer * 5120, 0])
        skip = pl.create_tensor([16, 5120], dtype=pl.BF16)
        skip = residual(hidden, attn, skip)
        normalized = pl.create_tensor([16, 5120], dtype=pl.BF16)
        normalized = norm(skip, gamma, normalized)
        gate = pl.create_tensor([16, 17408], dtype=pl.BF16)
        up = pl.create_tensor([16, 17408], dtype=pl.BF16)
        middle = pl.create_tensor([16, 17408], dtype=pl.BF16)
        output = pl.create_tensor([16, 5120], dtype=pl.BF16)
        gate = linear(normalized, wg, gate)
        up = linear(normalized, wu, up)
        middle = swiglu(gate, up, middle)
        output = linear(middle, wd, output)
        result = residual(skip, output, result)
        return result

    @pl.jit
    def embedding(ids: pl.Tensor, weight: pl.Tensor, result: pl.Out[pl.Tensor]):
        with pl.spmd(32 * 40, name_hint="orthrus_embedding"):
            task = pl.tile.get_block_idx()
            row = task // 40
            column = task % 40 * 128
            token = pl.read(ids, [row])
            result[row:row + 1, column:column + 128] = pl.slice(weight, [1, 128], [token, column])
        return result

    @pl.jit
    def gdn(hidden: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor,
            mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
            qkv_weight: pl.Tensor, z_weight: pl.Tensor, a_weight: pl.Tensor, b_weight: pl.Tensor,
            conv_weight: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
            norm_weight: pl.Tensor, out_weight: pl.Tensor,
            conv_history: pl.Tensor, recurrent_history: pl.Tensor, result: pl.Out[pl.Tensor],
            layer: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32]):
        index = layer - layer // 4
        gamma = pl.slice(input_gamma, [1, 5120], [layer, 0])
        conv_source = pl.reshape(pl.slice(conv_history, [1, 1, 4, 10240],
                                          [slot, index, 0, 0]), [4, 10240])
        state_source = pl.reshape(pl.slice(recurrent_history, [1, 1, 48, 128, 128],
                                           [slot, index, 0, 0, 0]), [6144, 128])
        # Orthrus: start from the AR state, mutate only temporary copies.
        conv = pl.create_tensor([4, 10240], dtype=pl.BF16)
        state = pl.create_tensor([48, 128, 128], dtype=pl.FP32)
        state_rows = pl.reshape(state, [6144, 128])
        with pl.spmd(320, name_hint="orthrus_copy_conv"):
            task = pl.tile.get_block_idx()
            row = task // 80
            column = task % 80 * 128
            conv[row:row + 1, column:column + 128] = pl.slice(conv_source, [1, 128], [row, column])
        with pl.spmd(384, name_hint="orthrus_copy_recurrence"):
            row = pl.tile.get_block_idx() * 16
            state_rows[row:row + 16, 0:128] = pl.slice(state_source, [16, 128], [row, 0])
        for half in pl.range(2):
            row = half * 16
            current = pl.slice(hidden, [16, 5120], [row, 0])
            normalized = pl.create_tensor([16, 5120], dtype=pl.BF16)
            qkv = pl.create_tensor([16, 10240], dtype=pl.BF16)
            z = pl.create_tensor([16, 6144], dtype=pl.BF16)
            a = pl.create_tensor([16, 128], dtype=pl.BF16)
            b = pl.create_tensor([16, 128], dtype=pl.BF16)
            normalized, qkv, z, a, b = stem(current, gamma, qkv_weight, z_weight,
                                           a_weight, b_weight, normalized, qkv, z, a, b)
            convolved = pl.create_tensor([16, 10240], dtype=pl.BF16)
            convolved, conv = convolution(qkv, conv_weight, conv, convolved, 16)
            q = pl.create_tensor([16, 48, 128], dtype=pl.BF16)
            k = pl.create_tensor([16, 48, 128], dtype=pl.BF16)
            v = pl.create_tensor([16, 48, 128], dtype=pl.BF16)
            q, k, v = expand(convolved, q, k, v)
            decay_padded = pl.create_tensor([16, 64], dtype=pl.FP32)
            beta_padded = pl.create_tensor([16, 64], dtype=pl.BF16)
            decay_padded, beta_padded = gates(a, b, a_log, dt_bias, decay_padded, beta_padded)
            decay = pl.create_tensor([16, 48], dtype=pl.FP32)
            beta = pl.create_tensor([16, 48], dtype=pl.FP32)
            with pl.spmd(16, name_hint="orthrus_unpad_gates"):
                token = pl.tile.get_block_idx()
                decay[token:token + 1, 0:48] = pl.slice(decay_padded, [1, 48], [token, 0])
                beta[token:token + 1, 0:48] = pl.cast(pl.slice(beta_padded, [1, 48], [token, 0]), pl.FP32)
            core = pl.create_tensor([16, 48, 128], dtype=pl.BF16)
            core, state = recurrence(q, k, v, decay, beta, state, core, 16)
            gated = pl.create_tensor([768, 128], dtype=pl.BF16)
            attn = pl.create_tensor([16, 5120], dtype=pl.BF16)
            gated, attn = tail(core, z, norm_weight, out_weight, gated, attn)
            following = pl.create_tensor([16, 5120], dtype=pl.BF16)
            following = shared_tail(current, attn, post_gamma, mlp_gate, mlp_up, mlp_down, following, layer)
            result[row:row + 16, 0:5120] = following
        return result

    @pl.jit
    def full_prepare(hidden: pl.Tensor, input_gamma: pl.Tensor,
                     q_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor,
                     q_gamma: pl.Tensor, k_gamma: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
                     query: pl.Out[pl.Tensor], key: pl.Out[pl.Tensor], value: pl.Out[pl.Tensor],
                     gate: pl.Out[pl.Tensor], layer: pl.Scalar[pl.INT32]):
        gamma = pl.slice(input_gamma, [1, 5120], [layer, 0])
        for half in pl.range(2):
            row = half * 16
            current = pl.slice(hidden, [16, 5120], [row, 0])
            normalized = pl.create_tensor([16, 5120], dtype=pl.BF16)
            normalized = norm(current, gamma, normalized)
            qn = pl.create_tensor([16, 6144], dtype=pl.BF16)
            kn = pl.create_tensor([16, 1024], dtype=pl.BF16)
            qr = pl.create_tensor([16, 6144], dtype=pl.BF16)
            kr = pl.create_tensor([16, 1024], dtype=pl.BF16)
            val = pl.create_tensor([16, 1024], dtype=pl.BF16)
            gating = pl.create_tensor([16, 6144], dtype=pl.BF16)
            cos_half = pl.slice(cos, [16, 64], [row, 0])
            sin_half = pl.slice(sin, [16, 64], [row, 0])
            qn, kn, qr, kr, val, gating = prepare(
                normalized, q_weight, k_weight, v_weight, q_gamma, k_gamma,
                cos_half, sin_half, qn, kn, qr, kr, val, gating)
            query[row:row + 16, 0:6144] = qr
            key[row:row + 16, 0:1024] = kr
            value[row:row + 16, 0:1024] = val
            gate[row:row + 16, 0:6144] = gating
        return query, key, value, gate

    @pl.jit
    def full(hidden: pl.Tensor, post_gamma: pl.Tensor,
             mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
             out_weight: pl.Tensor, query: pl.Tensor, key: pl.Tensor, value: pl.Tensor,
             gate: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor,
             block_table: pl.Tensor, mask: pl.Tensor, result: pl.Out[pl.Tensor],
             layer: pl.Scalar[pl.INT32], offset: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        index = layer // 4
        keys = pl.create_tensor([4, 512, 256], dtype=pl.BF16)
        values = pl.create_tensor([4, 512, 256], dtype=pl.BF16)
        with pl.spmd(128, name_hint="orthrus_read_history"):
            task = pl.tile.get_block_idx()
            head = task // 32
            logical = task % 32
            physical = pl.read(block_table, [logical])
            row = logical * 16
            keys[head:head + 1, row:row + 16, 0:256] = pl.reshape(
                pl.slice(key_pool, [1, 1, 1, 16, 256], [index, head, physical, 0, 0]), [1, 16, 256])
            values[head:head + 1, row:row + 16, 0:256] = pl.reshape(
                pl.slice(value_pool, [1, 1, 1, 16, 256], [index, head, physical, 0, 0]), [1, 16, 256])
        # All active block K/V exist before either half's query runs. The pools
        # above remain read-only: only these temporary tensors receive draft KV.
        with pl.spmd(4, name_hint="orthrus_append_temporary_block"):
            head = pl.tile.get_block_idx()
            for token in pl.range(width):
                keys[head:head + 1, offset + token:offset + token + 1, 0:256] = pl.reshape(
                    pl.slice(key, [1, 256], [token, head * 256]), [1, 1, 256])
                values[head:head + 1, offset + token:offset + token + 1, 0:256] = pl.reshape(
                    pl.slice(value, [1, 256], [token, head * 256]), [1, 1, 256])
        for half in pl.range(2):
            row = half * 16
            scores = pl.create_tensor([384, 512], dtype=pl.FP32)
            probabilities = pl.create_tensor([384, 512], dtype=pl.BF16)
            weighted = pl.create_tensor([16, 6144], dtype=pl.BF16)
            gated = pl.create_tensor([16, 6144], dtype=pl.BF16)
            attn = pl.create_tensor([16, 5120], dtype=pl.BF16)
            query_half = pl.slice(query, [16, 6144], [row, 0])
            gate_half = pl.slice(gate, [16, 6144], [row, 0])
            scores, probabilities, weighted, gated, attn = attention(
                query_half, keys, values, gate_half, mask, out_weight,
                scores, probabilities, weighted, gated, attn)
            following = pl.create_tensor([16, 5120], dtype=pl.BF16)
            current = pl.slice(hidden, [16, 5120], [row, 0])
            following = shared_tail(current, attn,
                                    post_gamma, mlp_gate, mlp_up, mlp_down, following, layer)
            result[row:row + 16, 0:5120] = following
        return result

    @pl.jit
    def finish(hidden: pl.Tensor, gamma: pl.Tensor, weight: pl.Tensor, logits: pl.Out[pl.Tensor]):
        for half in pl.range(2):
            row = half * 16
            current = pl.slice(hidden, [16, 5120], [row, 0])
            normalized = pl.create_tensor([16, 5120], dtype=pl.BF16)
            normalized = norm(current, gamma, normalized)
            output = pl.create_tensor([16, 248320], dtype=pl.BF16)
            output = linear(normalized, weight, output)
            logits[row:row + 16, 0:248320] = output
        return logits

    # Register all entry points with the same DistributedWorker as the target;
    # separate workers cannot safely share resident DeviceTensor allocations.
    @pl.jit.host
    def embed_host(ids: pl.Tensor, weight: pl.Tensor, result: pl.Out[pl.Tensor]):
        embedding(ids, weight, result, device=0)

    @pl.jit.host
    def gdn_host(hidden: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor,
                 mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
                 qkv_weight: pl.Tensor, z_weight: pl.Tensor, a_weight: pl.Tensor, b_weight: pl.Tensor,
                 conv_weight: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
                 norm_weight: pl.Tensor, out_weight: pl.Tensor,
                 conv_history: pl.Tensor, recurrent_history: pl.Tensor, result: pl.Out[pl.Tensor],
                 layer: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32]):
        gdn(hidden, input_gamma, post_gamma, mlp_gate, mlp_up, mlp_down,
            qkv_weight, z_weight, a_weight, b_weight, conv_weight, a_log, dt_bias,
            norm_weight, out_weight, conv_history, recurrent_history, result, layer, slot, device=0)

    @pl.jit.host
    def prepare_host(hidden: pl.Tensor, input_gamma: pl.Tensor,
                     q_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor,
                     q_gamma: pl.Tensor, k_gamma: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
                     query: pl.Out[pl.Tensor], key: pl.Out[pl.Tensor], value: pl.Out[pl.Tensor],
                     gate: pl.Out[pl.Tensor], layer: pl.Scalar[pl.INT32]):
        full_prepare(hidden, input_gamma, q_weight, k_weight, v_weight, q_gamma, k_gamma,
                     cos, sin, query, key, value, gate, layer, device=0)

    @pl.jit.host
    def full_host(hidden: pl.Tensor, post_gamma: pl.Tensor,
                  mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
                  out_weight: pl.Tensor, query: pl.Tensor, key: pl.Tensor, value: pl.Tensor,
                  gate: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor,
                  block_table: pl.Tensor, mask: pl.Tensor, result: pl.Out[pl.Tensor],
                  layer: pl.Scalar[pl.INT32], offset: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        full(hidden, post_gamma, mlp_gate, mlp_up, mlp_down, out_weight, query, key, value,
             gate, key_pool, value_pool, block_table, mask, result, layer, offset, width, device=0)

    @pl.jit.host
    def finish_host(hidden: pl.Tensor, gamma: pl.Tensor, weight: pl.Tensor, logits: pl.Out[pl.Tensor]):
        finish(hidden, gamma, weight, logits, device=0)

    return dict(embedding=embed_host, gdn=gdn_host, full_prepare=prepare_host, full=full_host, finish=finish_host)
