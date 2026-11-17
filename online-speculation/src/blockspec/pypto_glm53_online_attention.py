"""Explicit derivatives for the parameterized GLM draft MLA output."""

from .pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, TILE, VALUE


def make_output_backward(*, device_entry=False):
    """Differentiate the BF16 MLA output projection with FP32 accumulation."""
    import pypto.language as pl
    from models.glm5_3_flash.config import D

    rows, width = 16, HEADS * VALUE
    columns, reduction = 64, 128

    @pl.jit
    def backward(
        context: pl.Tensor[[rows, width], pl.BF16], weight: pl.Tensor[[D, width], pl.BF16],
        upstream: pl.Tensor[[rows, D], pl.BF16], weight_grad: pl.Out[pl.Tensor[[D, width], pl.FP32]],
        context_grad: pl.Out[pl.Tensor[[rows, width], pl.FP32]],
    ):
        for tile in pl.spmd((D // columns) * (width // columns), name_hint="orthrus_glm53_dwo"):
            output_column = tile // (width // columns) * columns
            input_column = tile % (width // columns) * columns
            dy = pl.slice(upstream, [rows, columns], [0, output_column])
            values = pl.slice(context, [rows, columns], [0, input_column])
            dw = pl.matmul(dy, values, a_trans=True, out_dtype=pl.FP32)
            weight_grad[output_column:output_column + columns, input_column:input_column + columns] = dw

        for tile in pl.spmd(width // columns, name_hint="orthrus_glm53_dcontext"):
            column = tile * columns
            dc = pl.create_tensor([rows, columns], dtype=pl.FP32)
            for start in pl.pipeline(0, D, reduction, stage=2):
                grad_tile = pl.slice(upstream, [rows, reduction], [0, start])
                weight_tile = pl.slice(weight, [reduction, columns], [start, column])
                dc = pl.matmul_acc(dc, grad_tile, weight_tile, init_cond=start == 0)
            context_grad[:, column:column + columns] = dc
        return weight_grad, context_grad

    @pl.jit.host
    def entry(context: pl.Tensor, weight: pl.Tensor, upstream: pl.Tensor,
              weight_grad: pl.Out[pl.Tensor], context_grad: pl.Out[pl.Tensor]):
        backward(context, weight, upstream, weight_grad, context_grad, device=0)

    return backward if device_entry else entry


def make_projection_backward(input_width, output_width, *, device_entry=False, refine=False):
    """Apply the input and parameter adjoints of a packed BF16 MLA projection."""
    import pypto.language as pl

    rows, columns, reduction = 16, 64, 128
    if input_width < columns or input_width % columns or output_width < reduction or output_width % reduction:
        raise ValueError("GLM MLA projections require aligned feature axes")

    @pl.jit
    def projection_backward(
        activation: pl.Tensor[[rows, input_width], pl.BF16],
        weight: pl.Tensor[[output_width, input_width], pl.BF16],
        upstream: pl.Tensor[[rows, output_width], pl.FP32],
        weight_grad: pl.Out[pl.Tensor[[output_width, input_width], pl.FP32]],
        input_grad: pl.Out[pl.Tensor[[rows, input_width], pl.FP32]],
    ):
        for tile in pl.spmd((output_width // columns) * (input_width // columns), name_hint="orthrus_glm53_projection_dw"):
            output_column = tile // (input_width // columns) * columns
            input_column = tile % (input_width // columns) * columns
            dy = pl.slice(upstream, [rows, columns], [0, output_column])
            values = pl.slice(activation, [rows, columns], [0, input_column])
            high = pl.cast(dy, pl.BF16, mode="rint")
            dw = pl.matmul(high, values, a_trans=True, out_dtype=pl.FP32)
            low = pl.cast(pl.sub(dy, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
            dw = pl.matmul_acc(dw, low, values, a_trans=True)
            if refine:
                remainder = pl.sub(pl.sub(dy, pl.cast(high, pl.FP32)), pl.cast(low, pl.FP32))
                dw = pl.matmul_acc(dw, pl.cast(remainder, pl.BF16, mode="rint"), values, a_trans=True)
            weight_grad[output_column:output_column + columns, input_column:input_column + columns] = dw

        for tile in pl.spmd(input_width // columns, name_hint="orthrus_glm53_projection_dx"):
            column = tile * columns
            dc = pl.create_tensor([rows, columns], dtype=pl.FP32)
            for start in pl.pipeline(0, output_width, reduction, stage=2):
                grad_tile = pl.slice(upstream, [rows, reduction], [0, start])
                weight_tile = pl.slice(weight, [reduction, columns], [start, column])
                high_input = pl.cast(grad_tile, pl.BF16, mode="rint")
                dc = pl.matmul_acc(dc, high_input, weight_tile, init_cond=start == 0)
                low_input = pl.cast(pl.sub(grad_tile, pl.cast(high_input, pl.FP32)), pl.BF16, mode="rint")
                dc = pl.matmul_acc(dc, low_input, weight_tile)
                if refine:
                    remainder_input = pl.sub(pl.sub(grad_tile, pl.cast(high_input, pl.FP32)), pl.cast(low_input, pl.FP32))
                    dc = pl.matmul_acc(dc, pl.cast(remainder_input, pl.BF16, mode="rint"), weight_tile)
            input_grad[:, column:column + columns] = dc
        return weight_grad, input_grad

    @pl.jit.host
    def entry(context: pl.Tensor, weight: pl.Tensor, upstream: pl.Tensor,
              weight_grad: pl.Out[pl.Tensor], context_grad: pl.Out[pl.Tensor]):
        projection_backward(context, weight, upstream, weight_grad, context_grad, device=0)

    return projection_backward if device_entry else entry


def make_joint_softmax_backward(rows, history_capacity, block_capacity=128, *, device_entry=False):
    """Apply the score adjoint across the shared history and the draft block."""
    import pypto.language as pl

    row_tile, tile = 8, 128
    if rows < row_tile or rows % row_tile or min(history_capacity, block_capacity) < tile:
        raise ValueError("joint MLA softmax requires eight query heads and 128-column tiles")
    if history_capacity % tile or block_capacity % tile:
        raise ValueError("history and draft probabilities must be tile-aligned")
    capacity = history_capacity + block_capacity

    @pl.jit
    def backward(
        probability: pl.Tensor[[rows, capacity], pl.FP32],
        upstream: pl.Tensor[[rows, capacity], pl.FP32],
        score_gradient: pl.Out[pl.Tensor[[rows, capacity], pl.FP32]],
    ):
        for fragment in pl.spmd(rows // row_tile, name_hint="orthrus_glm53_joint_softmax_backward"):
            begin = fragment * row_tile
            # One row dot across both regions, including the mass contributed
            # by the temporary draft keys. Padding has probability zero.
            dot = pl.full([row_tile, 8], dtype=pl.FP32, value=0.0)
            unit = pl.full([row_tile, 8], dtype=pl.FP32, value=1.0)
            for column in pl.range(0, capacity, tile):
                p = probability[begin:begin + row_tile, column:column + tile]
                g = upstream[begin:begin + row_tile, column:column + tile]
                dot = pl.add(dot, pl.row_expand_mul(unit, pl.row_sum(pl.mul(p, g))))
            coupling = pl.row_max(dot)
            for column in pl.range(0, capacity, tile):
                p = probability[begin:begin + row_tile, column:column + tile]
                g = upstream[begin:begin + row_tile, column:column + tile]
                score_gradient[begin:begin + row_tile, column:column + tile] = pl.mul(p, pl.row_expand_sub(g, coupling))
        return score_gradient

    @pl.jit.host
    def entry(probability: pl.Tensor, upstream: pl.Tensor, score_gradient: pl.Out[pl.Tensor]):
        backward(probability, upstream, score_gradient, device=0)

    return backward if device_entry else entry


def make_mla_score_backward(history_capacity, *, device_entry=False):
    """Backpropagate scaled joint scores into latent queries and the draft block."""
    import pypto.language as pl

    if history_capacity < TILE or history_capacity % TILE:
        raise ValueError("GLM MLA score backward needs tile-aligned history")
    rows = QUERY_ROWS * HEAD_TILE
    selected = QUERY_ROWS * history_capacity
    scores_width = history_capacity + TILE
    scale = VALUE ** -0.5

    @pl.jit
    def score_backward(
        grad_scores: pl.Tensor[[rows, scores_width], pl.FP32],
        history: pl.Tensor[[selected, LATENT], pl.BF16],
        block: pl.Tensor[[TILE, LATENT], pl.BF16],
        q_block: pl.Tensor[[rows, LATENT], pl.BF16],
        d_hist_query: pl.Out[pl.Tensor[[rows, LATENT], pl.FP32]],
        d_block_query: pl.Out[pl.Tensor[[rows, LATENT], pl.FP32]],
        d_block_key: pl.Out[pl.Tensor[[TILE, LATENT], pl.FP32]],
    ):
        for task in pl.spmd(QUERY_ROWS * (LATENT // TILE), name_hint="glm_mla_score_query_vjp"):
            row = task // (LATENT // TILE) * HEAD_TILE
            col = task % (LATENT // TILE) * TILE
            history_offset = row // HEAD_TILE * history_capacity
            accumulated = pl.create_tensor([HEAD_TILE, TILE], dtype=pl.FP32)
            for start in pl.pipeline(0, history_capacity, TILE, stage=2):
                k = history[history_offset + start:history_offset + start + TILE, col:col + TILE]
                g = pl.mul(grad_scores[row:row + HEAD_TILE, start:start + TILE], scale)
                hi = pl.cast(g, pl.BF16, mode="rint")
                lo = pl.cast(pl.sub(g, pl.cast(hi, pl.FP32)), pl.BF16, mode="rint")
                accumulated = pl.matmul_acc(accumulated, hi, k, init_cond=start == 0)
                accumulated = pl.matmul_acc(accumulated, lo, k)
            d_hist_query[row:row + HEAD_TILE, col:col + TILE] = accumulated
            block_key = block[:, col:col + TILE]
            block_grad = pl.mul(grad_scores[row:row + HEAD_TILE, history_capacity:scores_width], scale)
            high = pl.cast(block_grad, pl.BF16, mode="rint")
            low = pl.cast(pl.sub(block_grad, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
            dq = pl.matmul(high, block_key, out_dtype=pl.FP32)
            dq = pl.matmul_acc(dq, low, block_key)
            d_block_query[row:row + HEAD_TILE, col:col + TILE] = dq
        for task in pl.spmd(LATENT // TILE, name_hint="glm_mla_score_key_vjp"):
            col = task * TILE
            key_gradient = pl.create_tensor([TILE, TILE], dtype=pl.FP32)
            for row in pl.pipeline(0, rows, HEAD_TILE, stage=2):
                q = q_block[row:row + HEAD_TILE, col:col + TILE]
                dlogit = pl.mul(grad_scores[row:row + HEAD_TILE, history_capacity:scores_width], scale)
                major = pl.cast(dlogit, pl.BF16, mode="rint")
                minor = pl.cast(pl.sub(dlogit, pl.cast(major, pl.FP32)), pl.BF16, mode="rint")
                key_gradient = pl.matmul_acc(key_gradient, major, q, a_trans=True, init_cond=row == 0)
                key_gradient = pl.matmul_acc(key_gradient, minor, q, a_trans=True)
            d_block_key[:, col:col + TILE] = key_gradient
        return d_hist_query, d_block_query, d_block_key

    @pl.jit.host
    def entry(grad_scores: pl.Tensor, history: pl.Tensor, block: pl.Tensor, q_block: pl.Tensor,
              d_hist_query: pl.Out[pl.Tensor], d_block_query: pl.Out[pl.Tensor], d_block_key: pl.Out[pl.Tensor]):
        score_backward(grad_scores, history, block, q_block, d_hist_query, d_block_query, d_block_key, device=0)

    return score_backward if device_entry else entry


def make_mla_value_backward(*, device_entry=False):
    """Differentiate the two value bases that are added at the attention output."""
    import pypto.language as pl

    rows = QUERY_ROWS * HEAD_TILE
    weight_rows = HEADS * VALUE
    lane_width = 32
    latent_tiles = LATENT // TILE
    value_tiles = VALUE // lane_width

    @pl.jit
    def value_backward(
        ctx_grad: pl.Tensor[[rows, VALUE], pl.FP32],
        draft_mixture: pl.Tensor[[rows, LATENT], pl.FP32],
        history_weight: pl.Tensor[[weight_rows, LATENT], pl.BF16],
        draft_weight: pl.Tensor[[weight_rows, LATENT], pl.BF16],
        d_hist_mixture: pl.Out[pl.Tensor[[rows, LATENT], pl.FP32]],
        d_block_mixture: pl.Out[pl.Tensor[[rows, LATENT], pl.FP32]],
        d_draft_value: pl.Out[pl.Tensor[[weight_rows, LATENT], pl.FP32]],
    ):
        for task in pl.spmd(rows * latent_tiles, name_hint="glm_mla_latent_vjp"):
            row = task // latent_tiles
            head = row % HEAD_TILE
            col = task % latent_tiles * TILE
            hist = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
            block = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
            if head < HEADS:
                for start in pl.range(0, VALUE, lane_width):
                    index = head * VALUE + start
                    wh = pl.cast(history_weight[index:index + lane_width, col:col + TILE], pl.FP32)
                    wb = pl.cast(draft_weight[index:index + lane_width, col:col + TILE], pl.FP32)
                    context = pl.reshape(ctx_grad[row:row + 1, start:start + lane_width], [lane_width, 1])
                    hist = pl.add(hist, pl.col_sum(pl.row_expand_mul(wh, context)))
                    block = pl.add(block, pl.col_sum(pl.row_expand_mul(wb, context)))
            d_hist_mixture[row:row + 1, col:col + TILE] = hist
            d_block_mixture[row:row + 1, col:col + TILE] = block
        for task in pl.spmd(HEADS * value_tiles * latent_tiles, name_hint="glm_mla_wv_vjp"):
            head = task // (value_tiles * latent_tiles)
            start = task // latent_tiles % value_tiles * lane_width
            col = task % latent_tiles * TILE
            accumulated = pl.full([lane_width, TILE], dtype=pl.FP32, value=0.0)
            for token in pl.range(QUERY_ROWS):
                row = token * HEAD_TILE + head
                context = pl.reshape(ctx_grad[row:row + 1, start:start + lane_width], [lane_width, 1])
                mixture = draft_mixture[row:row + 1, col:col + TILE]
                broadcast = pl.col_expand(pl.full([lane_width, TILE], dtype=pl.FP32, value=0.0), mixture)
                accumulated = pl.add(accumulated, pl.row_expand_mul(broadcast, context))
            index = head * VALUE + start
            d_draft_value[index:index + lane_width, col:col + TILE] = accumulated
        return d_hist_mixture, d_block_mixture, d_draft_value

    @pl.jit.host
    def entry(ctx_grad: pl.Tensor, draft_mixture: pl.Tensor, history_weight: pl.Tensor, draft_weight: pl.Tensor,
              d_hist_mixture: pl.Out[pl.Tensor], d_block_mixture: pl.Out[pl.Tensor], d_draft_value: pl.Out[pl.Tensor]):
        value_backward(ctx_grad, draft_mixture, history_weight, draft_weight,
                       d_hist_mixture, d_block_mixture, d_draft_value, device=0)

    return value_backward if device_entry else entry


def make_mla_mixture_backward(history_capacity, *, device_entry=False):
    """Differentiate the history and block sums under the same attention probabilities."""
    import pypto.language as pl

    if history_capacity < TILE or history_capacity % TILE:
        raise ValueError("GLM MLA mixture backward needs tile-aligned history")
    rows = QUERY_ROWS * HEAD_TILE
    selected = QUERY_ROWS * history_capacity
    scores_width = history_capacity + TILE
    segments = history_capacity // TILE

    @pl.jit
    def mixture_backward(
        probability: pl.Tensor[[rows, scores_width], pl.FP32],
        history: pl.Tensor[[selected, LATENT], pl.BF16],
        block: pl.Tensor[[TILE, LATENT], pl.BF16],
        d_hist_mixture: pl.Tensor[[rows, LATENT], pl.FP32],
        d_block_mixture: pl.Tensor[[rows, LATENT], pl.FP32],
        d_probability: pl.Out[pl.Tensor[[rows, scores_width], pl.FP32]],
        d_block_value: pl.Out[pl.Tensor[[TILE, LATENT], pl.FP32]],
    ):
        for task in pl.spmd(QUERY_ROWS * segments, name_hint="glm_mla_history_probability_vjp"):
            token = task // segments
            offset = task % segments * TILE
            row = token * HEAD_TILE
            selected_row = token * history_capacity + offset
            hist_grad = pl.create_tensor([HEAD_TILE, TILE], dtype=pl.FP32)
            for col in pl.pipeline(0, LATENT, TILE, stage=2):
                hist_keys = pl.slice(history, [TILE, TILE], [selected_row, col])
                dh = pl.slice(d_hist_mixture, [HEAD_TILE, TILE], [row, col])
                hi_hist = pl.cast(dh, pl.BF16, mode="rint")
                lo_hist = pl.cast(pl.sub(dh, pl.cast(hi_hist, pl.FP32)), pl.BF16, mode="rint")
                hist_grad = pl.matmul_acc(hist_grad, hi_hist, hist_keys, b_trans=True, init_cond=col == 0)
                hist_grad = pl.matmul_acc(hist_grad, lo_hist, hist_keys, b_trans=True)
            d_probability[row:row + HEAD_TILE, offset:offset + TILE] = hist_grad
        for token in pl.spmd(QUERY_ROWS, name_hint="glm_mla_block_probability_vjp"):
            row = token * HEAD_TILE
            draft_grad = pl.create_tensor([HEAD_TILE, TILE], dtype=pl.FP32)
            for col in pl.pipeline(0, LATENT, TILE, stage=2):
                draft_keys = pl.slice(block, [TILE, TILE], [0, col])
                db = pl.slice(d_block_mixture, [HEAD_TILE, TILE], [row, col])
                hi_draft = pl.cast(db, pl.BF16, mode="rint")
                lo_draft = pl.cast(pl.sub(db, pl.cast(hi_draft, pl.FP32)), pl.BF16, mode="rint")
                draft_grad = pl.matmul_acc(draft_grad, hi_draft, draft_keys, b_trans=True, init_cond=col == 0)
                draft_grad = pl.matmul_acc(draft_grad, lo_draft, draft_keys, b_trans=True)
            d_probability[row:row + HEAD_TILE, history_capacity:scores_width] = draft_grad
        for task in pl.spmd(LATENT // TILE, name_hint="glm_mla_block_value_vjp"):
            col = task * TILE
            value_grad = pl.create_tensor([TILE, TILE], dtype=pl.FP32)
            for row in pl.pipeline(0, rows, HEAD_TILE, stage=2):
                p = probability[row:row + HEAD_TILE, history_capacity:scores_width]
                g = d_block_mixture[row:row + HEAD_TILE, col:col + TILE]
                p_high = pl.cast(p, pl.BF16, mode="rint")
                p_low = pl.cast(pl.sub(p, pl.cast(p_high, pl.FP32)), pl.BF16, mode="rint")
                g_high = pl.cast(g, pl.BF16, mode="rint")
                g_low = pl.cast(pl.sub(g, pl.cast(g_high, pl.FP32)), pl.BF16, mode="rint")
                value_grad = pl.matmul_acc(value_grad, p_high, g_high, a_trans=True, init_cond=row == 0)
                value_grad = pl.matmul_acc(value_grad, p_high, g_low, a_trans=True)
                value_grad = pl.matmul_acc(value_grad, p_low, g_high, a_trans=True)
                value_grad = pl.matmul_acc(value_grad, p_low, g_low, a_trans=True)
            d_block_value[:, col:col + TILE] = value_grad
        return d_probability, d_block_value

    @pl.jit.host
    def entry(probability: pl.Tensor, history: pl.Tensor, block: pl.Tensor,
              d_hist_mixture: pl.Tensor, d_block_mixture: pl.Tensor,
              d_probability: pl.Out[pl.Tensor], d_block_value: pl.Out[pl.Tensor]):
        mixture_backward(probability, history, block, d_hist_mixture, d_block_mixture,
                         d_probability, d_block_value, device=0)

    return mixture_backward if device_entry else entry


def make_mla_latent_backward(history_capacity):
    """Replay the joint probabilities and apply the latent attention adjoint."""
    import pypto.language as pl

    rows = QUERY_ROWS * HEAD_TILE
    width = history_capacity + TILE
    value = make_mla_value_backward(device_entry=True)
    mixture = make_mla_mixture_backward(history_capacity, device_entry=True)
    softmax = make_joint_softmax_backward(rows, history_capacity, TILE, device_entry=True)
    scores = make_mla_score_backward(history_capacity, device_entry=True)
    replay = make_mla_replay_tape(history_capacity, device_entry=True)

    @pl.jit
    def add_block_grads(
        score_grad: pl.Tensor[[TILE, LATENT], pl.FP32],
        value_grad: pl.Tensor[[TILE, LATENT], pl.FP32],
        output: pl.Out[pl.Tensor[[TILE, LATENT], pl.FP32]],
    ):
        for row in pl.spmd(TILE, name_hint="glm_mla_add_block_adjoints"):
            output[row:row + 1, :] = pl.add(score_grad[row:row + 1, :], value_grad[row:row + 1, :])
        return output

    @pl.jit.host
    def entry(
        ctx_grad: pl.Tensor, history_weight: pl.Tensor, draft_weight: pl.Tensor,
        q_history: pl.Tensor, q_block: pl.Tensor, history: pl.Tensor, block: pl.Tensor,
        history_bias: pl.Tensor, block_bias: pl.Tensor,
        d_hist_query: pl.Out[pl.Tensor], d_block_query: pl.Out[pl.Tensor],
        d_draft_value: pl.Out[pl.Tensor], d_block_latent: pl.Out[pl.Tensor],
    ):
        probability = pl.create_tensor([rows, width], dtype=pl.FP32)
        draft_mixture = pl.create_tensor([rows, LATENT], dtype=pl.FP32)
        hist_mixture_grad = pl.create_tensor([rows, LATENT], dtype=pl.FP32)
        block_mixture_grad = pl.create_tensor([rows, LATENT], dtype=pl.FP32)
        probability_grad = pl.create_tensor([rows, width], dtype=pl.FP32)
        scores_grad = pl.create_tensor([rows, width], dtype=pl.FP32)
        block_value_grad = pl.create_tensor([TILE, LATENT], dtype=pl.FP32)
        block_key_grad = pl.create_tensor([TILE, LATENT], dtype=pl.FP32)
        p, draft_mix = replay(q_history, q_block, history, block, history_bias, block_bias,
                              probability, draft_mixture, device=0)
        dm_history, dm_block, _w = value(ctx_grad, draft_mix, history_weight, draft_weight,
                                         hist_mixture_grad, block_mixture_grad, d_draft_value, device=0)
        dp, dv = mixture(p, history, block, dm_history, dm_block,
                         probability_grad, block_value_grad, device=0)
        ds = softmax(p, dp, scores_grad, device=0)
        _qh, _qb, dk = scores(ds, history, block, q_block, d_hist_query, d_block_query, block_key_grad, device=0)
        add_block_grads(dk, dv, d_block_latent, device=0)

    return entry


def make_mla_replay_tape(history_capacity, *, device_entry=False):
    """Recompute the shared probabilities and draft mixture for the last MLA layer."""
    import pypto.language as pl

    if history_capacity < TILE or history_capacity % TILE:
        raise ValueError("GLM MLA replay needs tile-aligned history")
    rows = QUERY_ROWS * HEAD_TILE
    selected = QUERY_ROWS * history_capacity
    capacity = history_capacity + TILE
    parts = history_capacity // TILE
    scale = VALUE ** -0.5

    @pl.jit
    def replay(
        q_history: pl.Tensor[[rows, LATENT], pl.BF16],
        q_block: pl.Tensor[[rows, LATENT], pl.BF16],
        history: pl.Tensor[[selected, LATENT], pl.BF16],
        block: pl.Tensor[[TILE, LATENT], pl.BF16],
        history_bias: pl.Tensor[[QUERY_ROWS, history_capacity], pl.FP32],
        block_bias: pl.Tensor[[1, TILE], pl.FP32],
        probability: pl.Out[pl.Tensor[[rows, capacity], pl.FP32]],
        mixed_block: pl.Out[pl.Tensor[[rows, LATENT], pl.FP32]],
    ):
        logits = pl.create_tensor([rows, capacity], dtype=pl.FP32)
        for job in pl.spmd(QUERY_ROWS * parts, name_hint="glm_replay_history_scores"):
            token = job // parts
            position = job % parts * TILE
            row = token * HEAD_TILE
            queries = pl.slice(q_history, [HEAD_TILE, LATENT], [row, 0])
            latent = pl.slice(history, [TILE, LATENT], [job * TILE, 0])
            h_scores = pl.matmul(queries, latent, b_trans=True, out_dtype=pl.FP32)
            h_bias = history_bias[token:token + 1, position:position + TILE]
            logits[row:row + HEAD_TILE, position:position + TILE] = pl.col_expand_add(pl.mul(h_scores, scale), h_bias)
        for token in pl.spmd(QUERY_ROWS, name_hint="glm_replay_block_scores"):
            row = token * HEAD_TILE
            queries = pl.slice(q_block, [HEAD_TILE, LATENT], [row, 0])
            b_scores = pl.matmul(queries, block, b_trans=True, out_dtype=pl.FP32)
            logits[row:row + HEAD_TILE, history_capacity:capacity] = pl.col_expand_add(
                pl.mul(b_scores, scale), block_bias,
            )
        for token in pl.spmd(QUERY_ROWS, name_hint="glm_replay_shared_softmax"):
            row = token * HEAD_TILE
            block_logits = logits[row:row + HEAD_TILE, history_capacity:capacity]
            maximum = pl.row_max(block_logits)
            for offset in pl.range(0, history_capacity, TILE):
                maximum = pl.maximum(maximum, pl.row_max(logits[row:row + HEAD_TILE, offset:offset + TILE]))
            denominator = pl.row_sum(pl.exp(pl.row_expand_sub(block_logits, maximum)))
            for offset in pl.range(0, history_capacity, TILE):
                shifted = pl.row_expand_sub(logits[row:row + HEAD_TILE, offset:offset + TILE], maximum)
                denominator = pl.add(denominator, pl.row_sum(pl.exp(shifted)))
            for offset in pl.range(0, capacity, TILE):
                shifted = pl.row_expand_sub(logits[row:row + HEAD_TILE, offset:offset + TILE], maximum)
                p = pl.row_expand_div(pl.exp(shifted), denominator)
                probability[row:row + HEADS, offset:offset + TILE] = pl.slice(p, [HEADS, TILE], [0, 0])
                probability[row + HEADS:row + HEAD_TILE, offset:offset + TILE] = pl.full(
                    [HEAD_TILE - HEADS, TILE], dtype=pl.FP32, value=0.0,
                )
        for token in pl.spmd(QUERY_ROWS, name_hint="glm_replay_draft_mixture"):
            row = token * HEAD_TILE
            p_block = pl.slice(probability, [HEAD_TILE, TILE], [row, history_capacity])
            high = pl.cast(p_block, pl.BF16, mode="rint")
            low = pl.cast(pl.sub(p_block, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
            mix = pl.matmul(high, block, out_dtype=pl.FP32)
            mixed_block[row:row + HEAD_TILE, :] = pl.matmul_acc(mix, low, block)
        return probability, mixed_block

    @pl.jit.host
    def entry(q_history: pl.Tensor, q_block: pl.Tensor, history: pl.Tensor, block: pl.Tensor, history_bias: pl.Tensor,
              block_bias: pl.Tensor, probability: pl.Out[pl.Tensor], mixed_block: pl.Out[pl.Tensor]):
        replay(q_history, q_block, history, block, history_bias, block_bias, probability, mixed_block, device=0)

    return replay if device_entry else entry


def make_mla_absorb_backward():
    """Differentiate the shared query and the draft-specific absorbed key basis."""
    import pypto.language as pl

    tokens, channels = 16, 32
    rows = QUERY_ROWS * HEAD_TILE
    width = HEADS * VALUE
    pieces = width // channels

    @pl.jit
    def absorb_backward(
        query: pl.Tensor[[tokens, width], pl.BF16],
        history_basis: pl.Tensor[[width, LATENT], pl.BF16],
        draft_basis: pl.Tensor[[width, LATENT], pl.BF16],
        d_history: pl.Tensor[[rows, LATENT], pl.FP32],
        d_draft: pl.Tensor[[rows, LATENT], pl.FP32],
        d_query: pl.Out[pl.Tensor[[tokens, width], pl.FP32]],
        d_basis: pl.Out[pl.Tensor[[width, LATENT], pl.FP32]],
    ):
        for task in pl.spmd(tokens * pieces, name_hint="glm_mla_absorb_query_vjp"):
            token = task // pieces
            feature = task % pieces * channels
            head = feature // VALUE
            output = pl.full([channels, TILE], dtype=pl.FP32, value=0.0)
            if token < QUERY_ROWS:
                row = token * HEAD_TILE + head
                for col in pl.range(0, LATENT, TILE):
                    wh = pl.cast(history_basis[feature:feature + channels, col:col + TILE], pl.FP32)
                    wd = pl.cast(draft_basis[feature:feature + channels, col:col + TILE], pl.FP32)
                    dh = d_history[row:row + 1, col:col + TILE]
                    dd = d_draft[row:row + 1, col:col + TILE]
                    output = pl.add(output, pl.add(pl.col_expand_mul(wh, dh), pl.col_expand_mul(wd, dd)))
            d_query[token:token + 1, feature:feature + channels] = pl.reshape(pl.row_sum(output), [1, channels])
        for task in pl.spmd(pieces * (LATENT // TILE), name_hint="glm_mla_absorb_key_vjp"):
            feature = task // (LATENT // TILE) * channels
            head = feature // VALUE
            col = task % (LATENT // TILE) * TILE
            grad = pl.full([channels, TILE], dtype=pl.FP32, value=0.0)
            for token in pl.range(QUERY_ROWS):
                row = token * HEAD_TILE + head
                latent = d_draft[row:row + 1, col:col + TILE]
                broadcast = pl.col_expand(pl.full([channels, TILE], dtype=pl.FP32, value=0.0), latent)
                q = pl.cast(query[token:token + 1, feature:feature + channels], pl.FP32)
                grad = pl.add(grad, pl.row_expand_mul(broadcast, pl.reshape(q, [channels, 1])))
            d_basis[feature:feature + channels, col:col + TILE] = grad
        return d_query, d_basis

    @pl.jit.host
    def entry(query: pl.Tensor, history_basis: pl.Tensor, draft_basis: pl.Tensor, d_history: pl.Tensor,
              d_draft: pl.Tensor, d_query: pl.Out[pl.Tensor], d_basis: pl.Out[pl.Tensor]):
        absorb_backward(query, history_basis, draft_basis, d_history, d_draft, d_query, d_basis, device=0)

    return entry


def make_mla_norm_backward(width, epsilon, *, device_entry=False):
    """Differentiate the draft-owned query or latent RMSNorm, including its scale."""
    import pypto.language as pl

    tokens, tile = 16, 128
    if width < tile or width % tile or epsilon <= 0:
        raise ValueError("invalid GLM trainable MLA norm geometry")

    @pl.jit
    def norm_backward(
        value: pl.Tensor[[tokens, width], pl.BF16], weight: pl.Tensor[[width], pl.BF16],
        upstream: pl.Tensor[[tokens, width], pl.FP32],
        d_value: pl.Out[pl.Tensor[[tokens, width], pl.FP32]], d_weight: pl.Out[pl.Tensor[[1, width], pl.FP32]],
    ):
        partial = pl.create_tensor([tokens, width], dtype=pl.FP32)
        for row in pl.spmd(tokens, name_hint="glm_mla_rmsnorm_input_vjp"):
            sq = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            dot = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, width, tile):
                x = pl.cast(value[row:row + 1, col:col + tile], pl.FP32)
                gamma = pl.cast(pl.reshape(weight[col:col + tile], [1, tile]), pl.FP32)
                dy = upstream[row:row + 1, col:col + tile]
                sq = pl.add(sq, pl.mul(x, x))
                dot = pl.add(dot, pl.mul(pl.mul(x, dy), gamma))
            zeros = pl.full([8, tile], dtype=pl.FP32, value=0.0)
            var = pl.mul(pl.row_sum(pl.col_expand(zeros, sq)), 1.0 / width)
            inv_tile = pl.recip(pl.sqrt(pl.add(var, epsilon)))
            inv = pl.read(inv_tile, [0, 0])
            dot_tile = pl.mul(pl.row_sum(pl.col_expand(zeros, dot)), 1.0 / width)
            correction = pl.read(pl.mul(dot_tile, pl.mul(inv_tile, inv_tile)), [0, 0])
            for col in pl.range(0, width, tile):
                x = pl.cast(value[row:row + 1, col:col + tile], pl.FP32)
                gamma = pl.cast(pl.reshape(weight[col:col + tile], [1, tile]), pl.FP32)
                dy = upstream[row:row + 1, col:col + tile]
                d_value[row:row + 1, col:col + tile] = pl.mul(pl.sub(pl.mul(dy, gamma), pl.mul(x, correction)), inv)
                partial[row:row + 1, col:col + tile] = pl.mul(pl.mul(dy, x), inv)
        for piece in pl.spmd(width // tile, name_hint="glm_mla_rmsnorm_scale_vjp"):
            col = piece * tile
            d_weight[:, col:col + tile] = pl.col_sum(partial[:, col:col + tile])
        return d_value, d_weight

    @pl.jit.host
    def entry(value: pl.Tensor, weight: pl.Tensor, upstream: pl.Tensor,
              d_value: pl.Out[pl.Tensor], d_weight: pl.Out[pl.Tensor]):
        norm_backward(value, weight, upstream, d_value, d_weight, device=0)

    return norm_backward if device_entry else entry


def make_mla_input_projection_backward(ranks):
    """Reduce the branch gradients before differentiating the replicated MLA weights."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, FLASH, Q_LORA
    from .pypto_glm53_online_suffix import make_fp32_tp_sum

    rows = 16
    output_width = HEADS * VALUE
    if not 1 <= ranks <= 8:
        raise ValueError("GLM MLA gradients use up to eight TP shards")
    q_b = make_projection_backward(Q_LORA, output_width, device_entry=True, refine=True)
    q_a = make_projection_backward(D, Q_LORA, device_entry=True, refine=True)
    kv_a = make_projection_backward(D, LATENT, device_entry=True, refine=True)
    q_norm = make_mla_norm_backward(Q_LORA, FLASH.rms_norm_eps, device_entry=True)
    kv_norm = make_mla_norm_backward(LATENT, FLASH.rms_norm_eps, device_entry=True)
    reduce_q = make_fp32_tp_sum(ranks, rows, Q_LORA)
    reduce_kv = make_fp32_tp_sum(ranks, rows, LATENT)

    @pl.jit
    def sum_inputs(
        q: pl.Tensor[[rows, D], pl.FP32], kv: pl.Tensor[[rows, D], pl.FP32],
        result: pl.Out[pl.Tensor[[rows, D], pl.FP32]],
    ):
        for task in pl.spmd(rows * (D // TILE), name_hint="glm_mla_sum_input_gradients"):
            row = task // (D // TILE)
            col = task % (D // TILE) * TILE
            result[row:row + 1, col:col + TILE] = pl.add(q[row:row + 1, col:col + TILE], kv[row:row + 1, col:col + TILE])
        return result

    @pl.jit.host
    def entry(
        x: pl.Tensor[[ranks, rows, D], pl.BF16],
        q_preactivation: pl.Tensor[[ranks, rows, Q_LORA], pl.BF16],
        q_normalized: pl.Tensor[[ranks, rows, Q_LORA], pl.BF16],
        kv_preactivation: pl.Tensor[[ranks, rows, LATENT], pl.BF16],
        d_query: pl.Tensor[[ranks, rows, output_width], pl.FP32],
        d_latent: pl.Tensor[[ranks, rows, LATENT], pl.FP32],
        q_a_weight: pl.Tensor[[ranks, Q_LORA, D], pl.BF16],
        q_norm_weight: pl.Tensor[[ranks, Q_LORA], pl.BF16],
        q_b_weight: pl.Tensor[[ranks, output_width, Q_LORA], pl.BF16],
        kv_a_weight: pl.Tensor[[ranks, LATENT, D], pl.BF16],
        kv_norm_weight: pl.Tensor[[ranks, LATENT], pl.BF16],
        d_q_a: pl.Out[pl.Tensor[[ranks, Q_LORA, D], pl.FP32]],
        d_q_norm: pl.Out[pl.Tensor[[ranks, 1, Q_LORA], pl.FP32]],
        d_q_b: pl.Out[pl.Tensor[[ranks, output_width, Q_LORA], pl.FP32]],
        d_kv_a: pl.Out[pl.Tensor[[ranks, LATENT, D], pl.FP32]],
        d_kv_norm: pl.Out[pl.Tensor[[ranks, 1, LATENT], pl.FP32]],
        dx: pl.Out[pl.Tensor[[ranks, rows, D], pl.FP32]],
        d_inner: pl.InOut[pl.Tensor[[ranks, rows, Q_LORA], pl.FP32]],
        d_full_q: pl.InOut[pl.Tensor[[ranks, rows, Q_LORA], pl.FP32]],
        d_full_kv: pl.InOut[pl.Tensor[[ranks, rows, LATENT], pl.FP32]],
        d_raw_query: pl.InOut[pl.Tensor[[ranks, rows, Q_LORA], pl.FP32]],
        d_raw_latent: pl.InOut[pl.Tensor[[ranks, rows, LATENT], pl.FP32]],
        d_input_q: pl.InOut[pl.Tensor[[ranks, rows, D], pl.FP32]],
        d_input_kv: pl.InOut[pl.Tensor[[ranks, rows, D], pl.FP32]],
        epoch: pl.Scalar[pl.INT32],
    ):
        q_window = pld.alloc_window_buffer([rows, Q_LORA], dtype=pl.FP32)
        q_ready_window = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        kv_window = pld.alloc_window_buffer([rows, LATENT], dtype=pl.FP32)
        kv_ready_window = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            q_buffer = pld.window(q_window, [rows, Q_LORA], dtype=pl.FP32)
            q_ready = pld.window(q_ready_window, [ranks, 1], dtype=pl.INT32)
            kv_buffer = pld.window(kv_window, [rows, LATENT], dtype=pl.FP32)
            kv_ready = pld.window(kv_ready_window, [ranks, 1], dtype=pl.INT32)
            inner = d_inner[rank]
            full_q = d_full_q[rank]
            full_kv = d_full_kv[rank]
            raw_query = d_raw_query[rank]
            raw_latent = d_raw_latent[rank]
            input_q = d_input_q[rank]
            input_kv = d_input_kv[rank]

            _sharded_weight, d_local_q = q_b(q_normalized[rank], q_b_weight[rank], d_query[rank], d_q_b[rank],
                                           inner, device=rank)
            d_q = reduce_q(d_local_q, full_q, q_buffer, q_ready, rank, epoch, device=rank)
            d_kv = reduce_kv(d_latent[rank], full_kv, kv_buffer, kv_ready, rank, epoch, device=rank)
            d_raw_q, _query_scale = q_norm(q_preactivation[rank], q_norm_weight[rank], d_q, raw_query,
                                          d_q_norm[rank], device=rank)
            _query_weight, from_query = q_a(x[rank], q_a_weight[rank], d_raw_q, d_q_a[rank], input_q,
                                           device=rank)
            d_raw_kv, _latent_scale = kv_norm(kv_preactivation[rank], kv_norm_weight[rank], d_kv, raw_latent,
                                             d_kv_norm[rank], device=rank)
            _latent_weight, from_latent = kv_a(x[rank], kv_a_weight[rank], d_raw_kv, d_kv_a[rank], input_kv,
                                              device=rank)
            sum_inputs(from_query, from_latent, dx[rank], device=rank)

    return entry
