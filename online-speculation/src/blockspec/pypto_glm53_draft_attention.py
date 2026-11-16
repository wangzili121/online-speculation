"""One TP8 GLM draft MLA sublayer with resident BF16 projections."""

from .pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, TILE, VALUE, make_indexed_mla_kernel


def make_draft_attention(*, inline=False):
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
                    activation = x[:, start:start + TILE]
                    if transposed:
                        nk = weight[col:col + n_tile, start:start + TILE]
                        acc = pl.matmul_acc(acc, activation, nk, b_trans=True, init_cond=start == 0)
                    else:
                        kn = weight[start:start + TILE, col:col + n_tile]
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
            q_part = q[:, head * VALUE:(head + 1) * VALUE]
            w_part = weight[head * VALUE:(head + 1) * VALUE, col:col + TILE]
            latent = pl.cast(pl.matmul(q_part, w_part, out_dtype=pl.FP32), pl.BF16, mode="rint")
            for token in pl.unroll(QUERY_ROWS):
                expanded_row = token * HEAD_TILE + head
                expanded[expanded_row:expanded_row + 1, col:col + TILE] = latent[token:token + 1, :]
        return expanded

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
        draft_latent = pl.create_tensor([rows, LATENT], dtype=pl.BF16)
        project_q_a(x, q_a_weight, q_a)
        rmsnorm(q_a, q_norm, q_resid)
        project_q_b(q_resid, q_b_weight, query)
        project_kv_a(x, kv_a_weight, kv_a)
        rmsnorm(kv_a, kv_norm, draft_latent)

        q_history = pl.create_tensor([QUERY_ROWS * HEAD_TILE, LATENT], dtype=pl.BF16)
        q_draft = pl.create_tensor([QUERY_ROWS * HEAD_TILE, LATENT], dtype=pl.BF16)
        for part in pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_q_pad"):
            row = part * HEAD_TILE + HEADS
            padding = pl.full([HEAD_TILE - HEADS, LATENT], dtype=pl.BF16, value=0.0)
            q_history[row:row + HEAD_TILE - HEADS, :] = padding
            q_draft[row:row + HEAD_TILE - HEADS, :] = padding
        absorb(query, ar_key, q_history)
        absorb(query, draft_key, q_draft)
        flat_index = pl.create_tensor([rows, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        gate = pl.create_tensor([rows, INDEX_H], dtype=pl.BF16)
        weights = pl.create_tensor([QUERY_ROWS, INDEX_H], dtype=pl.FP32)
        project_index(q_resid, indexer_q_weight, flat_index)
        project_gate(x, indexer_gate, gate)
        # The upstream scorer applies its scale to a BF16-rounded gate dot.
        for part in pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_gate_round"):
            weights[part:part + 1, :] = pl.mul(pl.cast(gate[part:part + 1, :], pl.FP32), WEIGHTS_SCALE)

        idx = pl.create_tensor([QUERY_ROWS, INDEX_H, INDEX_DIM], dtype=pl.BF16)
        flat_idx = pl.reshape(idx, [QUERY_ROWS, INDEX_H * INDEX_DIM])
        for part in pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_index_pack"):
            flat_idx[part:part + 1, :] = flat_index[part:part + 1, :]
        padded_block = pl.create_tensor([TILE, LATENT], dtype=pl.BF16)
        for part in pl.spmd(TILE // rows, name_hint="orthrus_glm53_block_pad"):
            start = part * rows
            if part == 0:
                padded_block[start:start + rows, :] = draft_latent[0:rows, :]
            else:
                padded_block[start:start + rows, :] = pl.full([rows, LATENT], dtype=pl.BF16, value=0.0)
        context = pl.create_tensor([QUERY_ROWS * HEADS, VALUE], dtype=pl.FP32)
        indexed_attention(q_history, q_draft, history, table, idx, weights, pool_cache, pool_blocks,
                          starts, pool_counts, tail_start, tail_count, prefix, padded_block, block_bias,
                          ar_value, draft_value, context)
        flat_context = pl.reshape(context, [QUERY_ROWS, q_width])
        projected_in = pl.create_tensor([rows, q_width], dtype=pl.BF16)
        for part in pl.spmd(rows, name_hint="orthrus_glm53_context_pad"):
            if part < QUERY_ROWS:
                projected_in[part:part + 1, :] = pl.cast(flat_context[part:part + 1, :], pl.BF16, mode="rint")
            else:
                projected_in[part:part + 1, :] = pl.full([1, q_width], dtype=pl.BF16, value=0.0)
        project_out(projected_in, out_weight, output)
        return output

    return draft_attention


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
