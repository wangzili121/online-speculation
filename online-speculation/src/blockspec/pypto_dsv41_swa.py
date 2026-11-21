"""K32 Orthrus attention over read-only V4.1 SWA pages and the draft block."""

BLOCK = 32
WINDOW = 128
HEAD_DIM = 512


def swa_spans(page_table, anchor: int, physical_pages: int, *, stride_rows=WINDOW, offset_rows=0):
    """Lower an anchor's logical SWA window to two (destination, source, length) spans."""
    if anchor < 0 or physical_pages <= 0 or not 0 <= offset_rows <= stride_rows - WINDOW:
        raise ValueError("invalid SWA anchor or pool capacity")
    start = max(0, anchor - WINDOW)
    row = start
    spans = []
    while row < anchor:
        logical, offset = divmod(row, WINDOW)
        if logical >= len(page_table) or not 0 <= page_table[logical] < physical_pages:
            raise ValueError("committed SWA row has no physical page")
        length = min(WINDOW - offset, anchor - row)
        spans.append((row - start, page_table[logical] * stride_rows + offset_rows + offset, length))
        row += length
    return tuple(spans + [(0, 0, 0)] * (2 - len(spans)))


def make_swa_attention(heads: int, *, token_major_query: bool = False, fused: bool = False):
    import pypto.language as pl

    if fused:
        return _make_fused_swa_attention(heads, token_major_query=token_major_query)

    @pl.jit
    def attention(
        query: pl.Tensor,
        pages: pl.Tensor,
        spans: pl.Tensor,
        temporary: pl.Tensor,
        history_bias: pl.Tensor,
        sink: pl.Tensor,
        zeros: pl.Tensor,
        output: pl.Out[pl.Tensor],
        lse: pl.Out[pl.Tensor],
    ):
        # Run Cube and Vector tasks with explicit device-memory dependencies.
        # The former mixed-core SWA produced intermittent output errors.
        local_scores = pl.create_tensor([heads * BLOCK, WINDOW], dtype=pl.FP32)
        block_scores = pl.create_tensor([heads * BLOCK, BLOCK], dtype=pl.FP32)
        local_high = pl.create_tensor([heads * BLOCK, WINDOW], dtype=pl.BF16)
        local_low = pl.create_tensor([heads * BLOCK, WINDOW], dtype=pl.BF16)
        block_high = pl.create_tensor([heads * BLOCK, BLOCK], dtype=pl.BF16)
        block_low = pl.create_tensor([heads * BLOCK, BLOCK], dtype=pl.BF16)
        with pl.spmd(heads, name_hint="orthrus_v41_swa_qk"):
            head = pl.tile.get_block_idx()
            history = pl.create_l1([WINDOW, HEAD_DIM], pl.BF16)
            history = pl.gather_row(
                history, zeros, [0, 0], [0, 0], [WINDOW, HEAD_DIM]
            )
            for span in pl.range(2):
                dst = pl.cast(pl.read(spans, [span, 0]), pl.INDEX)
                src = pl.cast(pl.read(spans, [span, 1]), pl.INDEX)
                length = pl.cast(pl.read(spans, [span, 2]), pl.INDEX)
                if length > 0:
                    history = pl.gather_row(
                        history, pages, [dst, 0], [src, 0], [WINDOW, HEAD_DIM],
                        valid_shape=[length, HEAD_DIM],
                    )

            row = head * BLOCK
            if token_major_query:
                q = pl.slice(query, [BLOCK, HEAD_DIM], [0, head * HEAD_DIM])
            else:
                q = pl.slice(query, [BLOCK, HEAD_DIM], [row, 0])
            local_scores[row:row + BLOCK, :] = pl.matmul(q, history, b_trans=True, out_dtype=pl.FP32)
            block_scores[row:row + BLOCK, :] = pl.matmul(q, temporary, b_trans=True, out_dtype=pl.FP32)
        with pl.spmd(heads, name_hint="orthrus_v41_swa_softmax"):
            head = pl.tile.get_block_idx()
            row = head * BLOCK
            local = pl.mul(pl.slice(local_scores, [BLOCK, WINDOW], [row, 0]), HEAD_DIM ** -0.5)
            local = pl.col_expand_add(local, history_bias)
            block = pl.mul(pl.slice(block_scores, [BLOCK, BLOCK], [row, 0]), HEAD_DIM ** -0.5)
            sink_value = pl.read(sink, [head])
            # Normalize all visible rows together. The sink has zero value.
            ref = pl.maximum(pl.maximum(pl.row_max(local), pl.row_max(block)), sink_value)
            local_exp = pl.exp(pl.row_expand_sub(local, ref))
            block_exp = pl.exp(pl.row_expand_sub(block, ref))
            sink_exp = pl.exp(pl.add(pl.neg(ref), sink_value))
            total = pl.add(pl.add(pl.row_sum(local_exp), pl.row_sum(block_exp)), sink_exp)
            p_local = pl.row_expand_div(local_exp, total)
            p_block = pl.row_expand_div(block_exp, total)
            local_hi = pl.cast(p_local, pl.BF16, mode="rint")
            block_hi = pl.cast(p_block, pl.BF16, mode="rint")
            # One BF16 probability rounding failed the near-zero error gate.
            # Accumulate its residual with a second BF16 Cube multiplication.
            local_lo = pl.cast(pl.sub(p_local, pl.cast(local_hi, pl.FP32)), pl.BF16, mode="rint")
            block_lo = pl.cast(pl.sub(p_block, pl.cast(block_hi, pl.FP32)), pl.BF16, mode="rint")
            local_high[row:row + BLOCK, :] = local_hi
            local_low[row:row + BLOCK, :] = local_lo
            block_high[row:row + BLOCK, :] = block_hi
            block_low[row:row + BLOCK, :] = block_lo
            lse[row:row + BLOCK, 0:1] = pl.add(ref, pl.log(total))
        with pl.spmd(heads, name_hint="orthrus_v41_swa_pv"):
            head = pl.tile.get_block_idx()
            history = pl.create_l1([WINDOW, HEAD_DIM], pl.BF16)
            history = pl.gather_row(
                history, zeros, [0, 0], [0, 0], [WINDOW, HEAD_DIM]
            )
            for span in pl.range(2):
                dst = pl.cast(pl.read(spans, [span, 0]), pl.INDEX)
                src = pl.cast(pl.read(spans, [span, 1]), pl.INDEX)
                length = pl.cast(pl.read(spans, [span, 2]), pl.INDEX)
                if length > 0:
                    history = pl.gather_row(
                        history, pages, [dst, 0], [src, 0], [WINDOW, HEAD_DIM],
                        valid_shape=[length, HEAD_DIM],
                    )

            row = head * BLOCK
            local_hi = pl.slice(local_high, [BLOCK, WINDOW], [row, 0])
            local_lo = pl.slice(local_low, [BLOCK, WINDOW], [row, 0])
            block_hi = pl.slice(block_high, [BLOCK, BLOCK], [row, 0])
            block_lo = pl.slice(block_low, [BLOCK, BLOCK], [row, 0])
            value = pl.matmul(local_hi, history, out_dtype=pl.FP32)
            value = pl.matmul_acc(value, block_hi, temporary)
            value = pl.matmul_acc(value, local_lo, history)
            output[row:row + BLOCK, :] = pl.matmul_acc(value, block_lo, temporary)
        return output, lse

    return attention


def _make_fused_swa_attention(heads: int, *, token_major_query: bool = False):
    import pypto.language as pl

    @pl.jit
    def attention(
        query: pl.Tensor,
        pages: pl.Tensor,
        spans: pl.Tensor,
        temporary: pl.Tensor,
        history_bias: pl.Tensor,
        sink: pl.Tensor,
        zeros: pl.Tensor,
        output: pl.Out[pl.Tensor],
        lse: pl.Out[pl.Tensor],
    ):
        # The anchor's committed SWA window intersects at most two 128-row pages.
        # Only those rows are copied to L1. The cache plane has no output binding.
        with pl.spmd(heads, name_hint="orthrus_v41_swa"):
            head = pl.tile.get_block_idx()
            history = pl.create_l1([WINDOW, HEAD_DIM], pl.BF16)
            history = pl.gather_row(
                history, zeros, [0, 0], [0, 0], [WINDOW, HEAD_DIM]
            )
            for span in pl.range(2):
                dst = pl.cast(pl.read(spans, [span, 0]), pl.INDEX)
                src = pl.cast(pl.read(spans, [span, 1]), pl.INDEX)
                length = pl.cast(pl.read(spans, [span, 2]), pl.INDEX)
                if length > 0:
                    history = pl.gather_row(
                        history, pages, [dst, 0], [src, 0], [WINDOW, HEAD_DIM],
                        valid_shape=[length, HEAD_DIM],
                    )

            row = head * BLOCK
            if token_major_query:
                q = pl.slice(query, [BLOCK, HEAD_DIM], [0, head * HEAD_DIM])
            else:
                q = pl.slice(query, [BLOCK, HEAD_DIM], [row, 0])
            local = pl.mul(pl.matmul(q, history, b_trans=True, out_dtype=pl.FP32), HEAD_DIM ** -0.5)
            local = pl.col_expand_add(local, history_bias)
            block = pl.mul(pl.matmul(q, temporary, b_trans=True, out_dtype=pl.FP32), HEAD_DIM ** -0.5)
            sink_value = pl.read(sink, [head])
            # Normalize all visible rows together. The sink has zero value.
            ref = pl.maximum(pl.maximum(pl.row_max(local), pl.row_max(block)), sink_value)
            local_exp = pl.exp(pl.row_expand_sub(local, ref))
            block_exp = pl.exp(pl.row_expand_sub(block, ref))
            sink_exp = pl.exp(pl.add(pl.neg(ref), sink_value))
            total = pl.add(pl.add(pl.row_sum(local_exp), pl.row_sum(block_exp)), sink_exp)
            p_local = pl.row_expand_div(local_exp, total)
            p_block = pl.row_expand_div(block_exp, total)
            local_hi = pl.cast(p_local, pl.BF16, mode="rint")
            block_hi = pl.cast(p_block, pl.BF16, mode="rint")
            # One BF16 probability rounding failed the near-zero error gate.
            # Accumulate its residual with a second BF16 Cube multiplication.
            local_lo = pl.cast(pl.sub(p_local, pl.cast(local_hi, pl.FP32)), pl.BF16, mode="rint")
            block_lo = pl.cast(pl.sub(p_block, pl.cast(block_hi, pl.FP32)), pl.BF16, mode="rint")
            value = pl.matmul(local_hi, history, out_dtype=pl.FP32)
            value = pl.matmul_acc(value, block_hi, temporary)
            value = pl.matmul_acc(value, local_lo, history)
            output[row:row + BLOCK, 0:HEAD_DIM] = pl.matmul_acc(value, block_lo, temporary)
            lse[row:row + BLOCK, 0:1] = pl.add(ref, pl.log(total))
        return output, lse

    return attention


def make_swa_program(heads: int, *, token_major_query: bool = False, fused: bool = False):
    import pypto.language as pl

    attention = make_swa_attention(heads, token_major_query=token_major_query, fused=fused)

    @pl.jit.host
    def entry(
        query: pl.Tensor, pages: pl.Tensor, spans: pl.Tensor, temporary: pl.Tensor,
        history_bias: pl.Tensor, sink: pl.Tensor, zeros: pl.Tensor,
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        attention(query, pages, spans, temporary, history_bias, sink, zeros, output, lse, device=0)

    return entry
