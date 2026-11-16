"""Orthrus MLA for the GLM Flash packed C8 target-cache layout."""

QUERY_ROWS = 6
HEADS = 8
HEAD_TILE = 16
LATENT = 512
VALUE = 256
PAGE = 640
TILE = 128


def make_mla_kernel(selected_capacity: int, *, inline=False):
    """Build the device attention implementation for TP8 and a six-row block."""
    if selected_capacity <= 0 or selected_capacity % TILE:
        raise ValueError("selection capacity must contain whole tiles")
    import pypto.language as pl
    from models.glm5_3_flash.attention import mla_dequant_cache_block

    parts = selected_capacity // TILE
    scale = VALUE ** -0.5
    selected_rows = QUERY_ROWS * selected_capacity
    partial_rows = QUERY_ROWS * parts * HEAD_TILE

    @pl.jit.inline
    def attention_tile(q: pl.Tensor, kv: pl.Tensor, bias: pl.Tensor):
        scores = pl.col_expand_add(pl.mul(pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32), scale), bias)
        max_score = pl.row_max(scores)
        weights = pl.exp(pl.row_expand_sub(scores, max_score))
        denom = pl.row_sum(weights)
        high = pl.cast(weights, pl.BF16, mode="rint")
        low = pl.cast(pl.sub(weights, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
        value = pl.matmul(high, kv, out_dtype=pl.FP32)
        return max_score, denom, pl.matmul_acc(value, low, kv)

    jit = pl.jit.inline if inline else pl.jit

    @jit
    def run(
        q_history: pl.Tensor, q_draft: pl.Tensor, history: pl.Tensor, table: pl.Tensor,
        indices: pl.Tensor, history_bias: pl.Tensor, lengths: pl.Tensor,
        block: pl.Tensor, block_bias: pl.Tensor, ar_value: pl.Tensor, draft_value: pl.Tensor,
        output: pl.Out[pl.Tensor],
    ):
        # Unpack each selected row once for all local heads. Bad/padding rows
        # never address the target arena; the binders supply anchor-limited IDs.
        selected = pl.create_tensor([selected_rows, LATENT], dtype=pl.BF16)
        with pl.spmd(48, name_hint="orthrus_glm53_unpack") as unpack:
            worker = pl.tile.get_block_idx()
            for turn in pl.range((selected_rows - worker + 47) // 48):
                unpack_row = worker + turn * 48
                unpack_token = unpack_row // selected_capacity
                unpack_offset = unpack_row % selected_capacity
                if unpack_offset < pl.cast(pl.read(lengths, [unpack_token]), pl.INDEX):
                    logical = pl.cast(pl.read(indices, [unpack_token, unpack_offset]), pl.INDEX)
                    page = pl.cast(pl.read(table, [logical // PAGE]), pl.INDEX)
                    physical = page * PAGE + logical % PAGE
                    for segment in pl.unroll(LATENT // TILE):
                        column = pl.cast(segment * TILE, pl.INDEX)
                        selected[unpack_row:unpack_row + 1, column:column + TILE] = mla_dequant_cache_block(
                            history, physical, column
                        )
                else:
                    selected[unpack_row:unpack_row + 1, :] = pl.full([1, LATENT], dtype=pl.BF16, value=0.0)

        part_max = pl.create_tensor([partial_rows, 1], dtype=pl.FP32)
        part_denom = pl.create_tensor([partial_rows, 1], dtype=pl.FP32)
        part_value = pl.create_tensor([partial_rows, LATENT], dtype=pl.FP32)
        with pl.spmd(QUERY_ROWS * parts, name_hint="orthrus_glm53_history", deps=[unpack]) as scored:
            history_task = pl.tile.get_block_idx()
            history_token, history_offset = history_task // parts, history_task % parts * TILE
            start = history_task * HEAD_TILE
            if history_offset < pl.cast(pl.read(lengths, [history_token]), pl.INDEX):
                hq = pl.slice(q_history, [HEAD_TILE, LATENT], [history_token * HEAD_TILE, 0])
                kv = pl.slice(selected, [TILE, LATENT], [history_task * TILE, 0])
                bias = pl.slice(history_bias, [1, TILE], [history_token, history_offset])
                hm, hl, hu = attention_tile(hq, kv, bias)
                part_max[start:start + HEAD_TILE, :] = hm
                part_denom[start:start + HEAD_TILE, :] = hl
                part_value[start:start + HEAD_TILE, :] = hu

        block_max = pl.create_tensor([QUERY_ROWS * HEAD_TILE, 1], dtype=pl.FP32)
        block_denom = pl.create_tensor([QUERY_ROWS * HEAD_TILE, 1], dtype=pl.FP32)
        block_value = pl.create_tensor([QUERY_ROWS * HEAD_TILE, LATENT], dtype=pl.FP32)
        with pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_bidir") as bidir:
            block_row = pl.tile.get_block_idx() * HEAD_TILE
            bq = pl.slice(q_draft, [HEAD_TILE, LATENT], [block_row, 0])
            bm, bl, bu = attention_tile(bq, pl.slice(block, [TILE, LATENT], [0, 0]), block_bias)
            block_max[block_row:block_row + HEAD_TILE, :] = bm
            block_denom[block_row:block_row + HEAD_TILE, :] = bl
            block_value[block_row:block_row + HEAD_TILE, :] = bu

        mix_history = pl.create_tensor([QUERY_ROWS * HEAD_TILE, LATENT], dtype=pl.FP32)
        mix_draft = pl.create_tensor([QUERY_ROWS * HEAD_TILE, LATENT], dtype=pl.FP32)
        with pl.spmd(QUERY_ROWS, name_hint="orthrus_glm53_normalize", deps=[scored, bidir]) as normalized:
            merge_token = pl.tile.get_block_idx()
            dest = merge_token * HEAD_TILE
            count = pl.cast(pl.read(lengths, [merge_token]), pl.INDEX)
            m = pl.slice(block_max, [HEAD_TILE, 1], [dest, 0])
            denom = pl.slice(block_denom, [HEAD_TILE, 1], [dest, 0])
            block_scale = pl.add(pl.mul(denom, 0.0), 1.0)
            hist = pl.full([HEAD_TILE, LATENT], dtype=pl.FP32, value=0.0)
            for part in pl.range(parts):
                if part * TILE < count:
                    row = (merge_token * parts + part) * HEAD_TILE
                    segment_max = pl.slice(part_max, [HEAD_TILE, 1], [row, 0])
                    next_max = pl.maximum(m, segment_max)
                    previous_scale = pl.exp(pl.sub(m, next_max))
                    segment_scale = pl.exp(pl.sub(segment_max, next_max))
                    hist = pl.add(pl.row_expand_mul(hist, previous_scale), pl.row_expand_mul(
                        pl.slice(part_value, [HEAD_TILE, LATENT], [row, 0]), segment_scale
                    ))
                    denom = pl.add(pl.mul(denom, previous_scale), pl.mul(
                        pl.slice(part_denom, [HEAD_TILE, 1], [row, 0]), segment_scale
                    ))
                    block_scale = pl.mul(block_scale, previous_scale)
                    m = next_max
            mix_history[dest:dest + HEAD_TILE, :] = pl.row_expand_div(hist, denom)
            mix_draft[dest:dest + HEAD_TILE, :] = pl.row_expand_div(
                pl.row_expand_mul(pl.slice(block_value, [HEAD_TILE, LATENT], [dest, 0]), block_scale), denom
            )

        # The two latent spaces have separate value bases. Only the projected
        # results can be added. This vector implementation is the first reference.
        vt = 8
        with pl.spmd(48, name_hint="orthrus_glm53_value", deps=[normalized]):
            lane = pl.tile.get_block_idx()
            for turn in pl.range((QUERY_ROWS * HEADS * (VALUE // vt) - lane + 47) // 48):
                task = lane + turn * 48
                token = task // (HEADS * (VALUE // vt))
                head = task // (VALUE // vt) % HEADS
                channel = task % (VALUE // vt) * vt
                source_row = token * HEAD_TILE + head
                wrow = head * VALUE + channel
                h = pl.slice(mix_history, [1, LATENT], [source_row, 0])
                draft_latent = pl.slice(mix_draft, [1, LATENT], [source_row, 0])
                wh = pl.cast(pl.slice(ar_value, [vt, LATENT], [wrow, 0]), pl.FP32)
                wb = pl.cast(pl.slice(draft_value, [vt, LATENT], [wrow, 0]), pl.FP32)
                values = pl.row_sum(pl.add(pl.col_expand_mul(wh, h), pl.col_expand_mul(wb, draft_latent)))
                output[token * HEADS + head:token * HEADS + head + 1, channel:channel + vt] = pl.reshape(values, [1, vt])
        return output

    return run


def make_mla_program(selected_capacity: int):
    """Expose the device kernel to a standalone worker for isolated validation."""
    import pypto.language as pl

    run = make_mla_kernel(selected_capacity)

    @pl.jit.host
    def entry(
        q_history: pl.Tensor, q_draft: pl.Tensor, history: pl.Tensor, table: pl.Tensor,
        indices: pl.Tensor, history_bias: pl.Tensor, lengths: pl.Tensor,
        block: pl.Tensor, block_bias: pl.Tensor, ar_value: pl.Tensor, draft_value: pl.Tensor,
        output: pl.Out[pl.Tensor],
    ):
        run(q_history, q_draft, history, table, indices, history_bias, lengths,
            block, block_bias, ar_value, draft_value, output, device=0)

    return entry


def make_history_selector(*, inline=False):
    """Retrieve each of the six queries' history using the resident target pools."""
    import pypto.language as pl
    from models.glm5_3_flash.config import KPOOL_SELECT_K, TOPK_INDEX_WIDTH
    from models.glm5_3_flash.indexer import indexer_expand, indexer_select

    jit = pl.jit.inline if inline else pl.jit

    @jit
    def select(
        query: pl.Tensor, head_weights: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor, counts: pl.Tensor, tail_start: pl.Tensor, tail_count: pl.Tensor,
        prefix_length: pl.Tensor, indices: pl.Out[pl.Tensor],
    ):
        # starts and counts describe the committed prefix, independently of the
        # draft row's absolute position. The draft block never enters the index.
        pools = pl.create_tensor([QUERY_ROWS, KPOOL_SELECT_K], dtype=pl.INT32)
        valid = pl.create_tensor([QUERY_ROWS, KPOOL_SELECT_K], dtype=pl.INT32)
        indexer_select(query, pool_cache, pool_blocks, head_weights, starts, counts, pools, valid)
        indexer_expand(pools, valid, tail_start, tail_count, prefix_length, indices)
        return indices

    return select, TOPK_INDEX_WIDTH


def make_indexed_mla_kernel(*, inline=False):
    """Run history retrieval and attention in one device-resident dispatch."""
    import pypto.language as pl
    from models.glm5_3_flash.config import INDEX_KPOOL, KPOOL_SELECT_K

    select, width = make_history_selector(inline=True)
    capacity = (width + TILE - 1) // TILE * TILE
    attend = make_mla_kernel(capacity, inline=True)
    jit = pl.jit.inline if inline else pl.jit

    @jit
    def indexed_attention(
        q_history: pl.Tensor, q_draft: pl.Tensor, history: pl.Tensor, table: pl.Tensor,
        index_query: pl.Tensor, head_weights: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor, pool_counts: pl.Tensor, tail_start: pl.Tensor, tail_count: pl.Tensor,
        prefix: pl.Tensor, block: pl.Tensor, block_bias: pl.Tensor, ar_value: pl.Tensor, draft_value: pl.Tensor,
        output: pl.Out[pl.Tensor],
    ):
        ids = pl.create_tensor([QUERY_ROWS, width], dtype=pl.INT32)
        select(index_query, head_weights, pool_cache, pool_blocks, starts, pool_counts, tail_start, tail_count,
               prefix, ids)
        bias = pl.create_tensor([QUERY_ROWS, capacity], dtype=pl.FP32)
        lengths = pl.create_tensor([QUERY_ROWS], dtype=pl.INT32)
        # Produce an aligned mask while retaining the native, 2051-wide indexer
        # result. An index is read only when its lane is within the true count.
        for worker in pl.spmd(1, name_hint="orthrus_glm53_lengths"):
            for row in pl.range(worker, QUERY_ROWS):
                count = pl.min(pl.read(pool_counts, [row]), KPOOL_SELECT_K) * INDEX_KPOOL
                actual = count + pl.read(tail_count, [row])
                pl.write(lengths, [row], pl.cast(actual, pl.INT32))
        for item in pl.spmd(QUERY_ROWS * (capacity // TILE), name_hint="orthrus_glm53_bias"):
            token = item // (capacity // TILE)
            offset = item % (capacity // TILE) * TILE
            active = pl.read(lengths, [token])
            lane = pl.arange(pl.cast(offset, pl.INT32), [1, TILE], dtype=pl.INT32)
            invalid = pl.minimum(pl.maximum(pl.sub(pl.add(lane, 1), active), 0), 1)
            bias[token:token + 1, offset:offset + TILE] = pl.mul(pl.cast(invalid, pl.FP32), -3.4e38)
        attend(q_history, q_draft, history, table, ids, bias, lengths, block, block_bias,
               ar_value, draft_value, output)
        return output

    return indexed_attention, width


def make_indexed_mla_program():
    import pypto.language as pl

    indexed_attention, width = make_indexed_mla_kernel()

    @pl.jit.host
    def entry(
        q_history: pl.Tensor, q_draft: pl.Tensor, history: pl.Tensor, table: pl.Tensor,
        index_query: pl.Tensor, head_weights: pl.Tensor, pool_cache: pl.Tensor, pool_blocks: pl.Tensor,
        starts: pl.Tensor, pool_counts: pl.Tensor, tail_start: pl.Tensor, tail_count: pl.Tensor,
        prefix: pl.Tensor, block: pl.Tensor, block_bias: pl.Tensor, ar_value: pl.Tensor, draft_value: pl.Tensor,
        output: pl.Out[pl.Tensor],
    ):
        indexed_attention(q_history, q_draft, history, table, index_query, head_weights, pool_cache, pool_blocks,
                          starts, pool_counts, tail_start, tail_count, prefix, block, block_bias,
                          ar_value, draft_value, output, device=0)

    return entry, width
