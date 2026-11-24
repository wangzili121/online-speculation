"""PyPTO leaves for the Orthrus-DSV4.1 C1A Attention backward.

The forward C1A path joins two *frozen* history segments (SWA and a different
set of selected compressed rows for each query) with the temporary K32 block.
Backward reuses or reconstructs the exact per-query joined K/V and FP32
probability, returns dQ and only the temporary block's dK/dV, and reduces the
sink gradient per local head. Frozen history never receives a gradient.

Large matrix products are intentionally delegated to ``make_linear`` in
``pypto_dsv41_online_programs``.  This file owns the part that a GEMM cannot
express: masked softmax VJP, sink reduction, and extraction of K32 rows.
"""

from dataclasses import dataclass

from .pypto_dsv41_resident_abi import with_resident_entry


@dataclass(frozen=True)
class C1AAttentionVJPGeometry:
    block: int = 32
    local_heads: int = 16
    dimension: int = 512
    swa: int = 128
    compressed: int = 512
    padded_columns: int = 704

    def __post_init__(self):
        total = self.swa + self.compressed + self.block + 1
        if self.block != 32 or self.local_heads < 1 or self.dimension != 512:
            raise ValueError("expected the DSV4.1 TP4 C1A K32 geometry")
        if self.padded_columns < total or self.padded_columns % 64:
            raise ValueError("softmax capacity must cover history, block and sink and align to 64")

    @property
    def rows(self):
        return self.block * self.local_heads

    @property
    def token_columns(self):
        return self.swa + self.compressed + self.block

    @property
    def sink_column(self):
        return self.token_columns

    @property
    def block_begin(self):
        return self.swa + self.compressed


def make_softmax_vjp(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Differentiate saved FP32 probabilities without rebuilding the mask.

    Masked and padding columns already have zero probability, so the usual
    ``p * (dy - dot(p, dy))`` formula also produces zero gradients there.
    """
    import pypto.language as pl

    rows, columns = geometry.rows, geometry.padded_columns
    block, sink_column = geometry.block, geometry.sink_column
    scale = geometry.dimension ** -0.5
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def softmax_vjp(probability: pl.Tensor, probability_gradient: pl.Tensor,
                    score_gradient: pl.Out[pl.Tensor], sink_rows: pl.Out[pl.Tensor]):
        with pl.spmd(rows, name_hint="orthrus_dsv41_c1a_softmax_backward"):
            row = pl.tile.get_block_idx()
            p = pl.reshape(pl.slice(probability, [1, columns], [row, 0]), [8, columns // 8])
            dy = pl.reshape(
                pl.cast(pl.slice(probability_gradient, [1, columns], [row, 0]), pl.FP32),
                [8, columns // 8],
            )
            partial = pl.row_expand_mul(
                pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(pl.mul(p, dy)),
            )
            dot = pl.row_sum(pl.transpose(partial, 0, 1))
            result = pl.mul(p, pl.row_expand_sub(dy, dot))
            # QK scores were scaled by 1/sqrt(d). Apply that factor before the
            # subsequent dQ/dK GEMMs. The sink is not a QK score, so retain its
            # unscaled derivative separately in ``sink_rows`` below.
            score_gradient[row:row + 1, 0:columns] = pl.cast(
                pl.mul(pl.reshape(result, [1, columns]), scale),
                pl.BF16, mode="rint",
            )
            pl.write(
                # One 64-byte FP32 row per SPMD block prevents scalar stores
                # from different cores from clobbering a shared cache line.
                sink_rows, [row, 0],
                pl.read(result, [sink_column // (columns // 8),
                                 sink_column % (columns // 8)]),
            )
        return score_gradient, sink_rows

    @pl.jit.host
    def entry(probability: pl.Tensor, probability_gradient: pl.Tensor,
              score_gradient: pl.Out[pl.Tensor], sink_rows: pl.Out[pl.Tensor]):
        softmax_vjp(probability, probability_gradient, score_gradient, sink_rows, device=0)

    return softmax_vjp if inline else with_resident_entry(entry, softmax_vjp)


def make_joined_history(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Materialize each query's exact SWA + selected C1A + K32 key/value rows.

    C1A selections differ by query, so the output is query-major
    ``[block * padded_columns, dimension]`` rather than one shared KV matrix.
    Rows after the 672 real KV entries are zero; this gives Cube a 64-aligned
    K dimension while the sink remains a score-only column.
    Cache inputs are read-only; only the temporary K32 rows belong to draft.
    """
    import pypto.language as pl

    if geometry.swa != 128 or geometry.compressed != 512:
        raise ValueError("resident C1A replay expects SWA128 and selected C512")
    kernel = pl.jit.inline if inline else pl.jit
    block, dimension = geometry.block, geometry.dimension
    swa, compressed = geometry.swa, geometry.compressed
    columns, stride = geometry.token_columns, geometry.padded_columns

    @kernel
    def joined_history(
        swa_pages: pl.Tensor,
        spans: pl.Tensor,
        compressed_pages: pl.Tensor,
        selected_rows: pl.Tensor,
        selected_lengths: pl.Tensor,
        temporary: pl.Tensor,
        zeros: pl.Tensor,
        joined: pl.Out[pl.Tensor],
    ):
        with pl.spmd(block, name_hint="orthrus_dsv41_c1a_replay_history"):
            token = pl.tile.get_block_idx()
            base = token * stride
            # 32x512 tiles stay below the A2/A3 vector-ring budget. A full
            # 128x512 SWA tile plus the compressed tile exceeds it.
            for part in pl.range(swa // 32):
                begin = part * 32
                local = pl.tile.full([32, dimension], dtype=pl.BF16, value=0.)
                for span in pl.range(2):
                    dst = pl.cast(pl.read(spans, [span, 0]), pl.INDEX)
                    src = pl.cast(pl.read(spans, [span, 1]), pl.INDEX)
                    length = pl.cast(pl.read(spans, [span, 2]), pl.INDEX)
                    start = pl.max(begin, dst)
                    end = pl.min(begin + 32, dst + length)
                    if end > start:
                        local = pl.tile.gather_row(
                            local, swa_pages, [start - begin, 0],
                            [src + start - dst, 0], [32, dimension],
                            valid_shape=[end - start, dimension],
                        )
                pl.store(
                    local,
                    [base + begin, 0], joined,
                )

            count = pl.cast(pl.read(selected_lengths, [token]), pl.INDEX)
            for part in pl.range(compressed // 32):
                offset = part * 32
                selected = pl.tile.full([32, dimension], dtype=pl.BF16, value=0.)
                if offset < count:
                    for row in pl.range(pl.min(32, count - offset)):
                        physical = pl.cast(
                            pl.read(selected_rows, [token, offset + row]), pl.INDEX,
                        )
                        selected = pl.tile.gather_row(
                            selected, compressed_pages, [row, 0], [physical, 0],
                            [1, dimension],
                        )
                pl.store(
                    selected,
                    [base + swa + offset, 0], joined,
                )
            pl.store(
                pl.load(temporary, [0, 0], [block, dimension]),
                [base + swa + compressed, 0], joined,
            )
            if stride > columns:
                pl.store(
                    pl.tile.full(
                        [stride - columns, dimension], dtype=pl.BF16, value=0.,
                    ),
                    [base + columns, 0], joined,
                )
        return joined

    @pl.jit.host
    def entry(
        swa_pages: pl.Tensor,
        spans: pl.Tensor,
        compressed_pages: pl.Tensor,
        selected_rows: pl.Tensor,
        selected_lengths: pl.Tensor,
        temporary: pl.Tensor,
        zeros: pl.Tensor,
        joined: pl.Out[pl.Tensor],
    ):
        joined_history(
            swa_pages, spans, compressed_pages, selected_rows, selected_lengths,
            temporary, zeros, joined, device=0,
        )

    return joined_history if inline else with_resident_entry(entry, joined_history)


def make_probability_replay(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Rebuild forward probabilities from exact keys and the saved final LSE."""
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    block, heads, dimension = geometry.block, geometry.local_heads, geometry.dimension
    swa, compressed, columns = geometry.swa, geometry.compressed, geometry.padded_columns
    real_columns = geometry.token_columns
    padded, sink_column = geometry.padded_columns, geometry.sink_column

    @kernel
    def probability_replay(
        query: pl.Tensor,
        joined_key: pl.Tensor,
        history_bias: pl.Tensor,
        compressed_bias: pl.Tensor,
        sink: pl.Tensor,
        final_lse: pl.Tensor,
        probability: pl.Out[pl.Tensor],
    ):
        with pl.spmd(block, name_hint="orthrus_dsv41_c1a_replay_probability"):
            token = pl.tile.get_block_idx()
            row = token * heads
            q = pl.slice(query, [heads, dimension], [token * heads, 0])
            lse = pl.slice(final_lse, [heads, 1], [row, 0])
            history_key = pl.slice(joined_key, [swa, dimension], [token * columns, 0])
            history_score = pl.col_expand_add(
                pl.mul(
                    pl.matmul(q, history_key, b_trans=True, out_dtype=pl.FP32),
                    dimension ** -0.5,
                ),
                history_bias,
            )
            probability[row:row + heads, 0:swa] = pl.exp(
                pl.row_expand_sub(history_score, lse),
            )
            # The 704-row key matrix does not fit beside the score buffers in
            # A2/A3 Mat memory. Replay C1A in 128-row segments; saved final LSE
            # still makes each segment exactly equivalent to one full softmax.
            segment = 128
            for part in pl.range(compressed // segment):
                offset = part * segment
                selected_key = pl.slice(
                    joined_key, [128, dimension],
                    [token * columns + swa + offset, 0],
                )
                selected_score = pl.col_expand_add(
                    pl.mul(
                        pl.matmul(q, selected_key, b_trans=True, out_dtype=pl.FP32),
                        dimension ** -0.5,
                    ),
                    pl.slice(compressed_bias, [1, 128], [token, offset]),
                )
                probability[
                    row:row + heads, swa + offset:swa + offset + segment
                ] = pl.exp(pl.row_expand_sub(selected_score, lse))
            block_key = pl.slice(
                joined_key, [block, dimension],
                [token * columns + swa + compressed, 0],
            )
            block_score = pl.mul(
                pl.matmul(q, block_key, b_trans=True, out_dtype=pl.FP32),
                dimension ** -0.5,
            )
            probability[
                row:row + heads, swa + compressed:real_columns
            ] = pl.exp(pl.row_expand_sub(block_score, lse))
            # Sink plus physical padding is exactly 32 columns. Build and
            # store it as one aligned tile; mixing scalar writes with the MTE
            # stores above is not coherent on the current PyPTO runtime.
            sink_input = pl.tile.full([heads, 8], dtype=pl.FP32, value=0.)
            for head in pl.range(heads):
                pl.tile.write(
                    sink_input, [head, 0],
                    pl.read(sink, [head]) - pl.read(final_lse, [row + head, 0]),
                )
            sink_probability = pl.exp(sink_input)
            tail = pl.tile.full(
                [heads, padded - sink_column], dtype=pl.FP32, value=0.,
            )
            for head in pl.range(heads):
                pl.tile.write(
                    tail, [head, 0], pl.tile.read(sink_probability, [head, 0]),
                )
            pl.store(tail, [row, sink_column], probability)
        return probability

    @pl.jit.host
    def entry(
        query: pl.Tensor,
        joined_key: pl.Tensor,
        history_bias: pl.Tensor,
        compressed_bias: pl.Tensor,
        sink: pl.Tensor,
        final_lse: pl.Tensor,
        probability: pl.Out[pl.Tensor],
    ):
        probability_replay(
            query, joined_key, history_bias, compressed_bias, sink,
            final_lse, probability, device=0,
        )

    return probability_replay if inline else with_resident_entry(entry, probability_replay)


def make_sink_reduce(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Reduce the dedicated sink column over the 32 draft query rows."""
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    block, heads, column = geometry.block, geometry.local_heads, geometry.sink_column

    @kernel
    def sink_reduce(sink_rows: pl.Tensor, sink_gradient: pl.Out[pl.Tensor]):
        with pl.spmd(heads, name_hint="orthrus_dsv41_c1a_sink_backward"):
            head = pl.tile.get_block_idx()
            total = pl.full([1, 8], dtype=pl.FP32, value=0.)
            for row in pl.range(block):
                # Softmax rows are token-major: token * heads + head.
                value = pl.read(sink_rows, [row * heads + head, 0])
                total = pl.add(total, value)
            # Each head owns a complete 64-byte row; consumers use column 0.
            pl.write(sink_gradient, [head, 0], pl.read(total, [0, 0]))
        return sink_gradient

    @pl.jit.host
    def entry(sink_rows: pl.Tensor, sink_gradient: pl.Out[pl.Tensor]):
        sink_reduce(sink_rows, sink_gradient, device=0)

    return sink_reduce if inline else with_resident_entry(entry, sink_reduce)


def make_block_extract(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Extract BF16 K32 score-gradient columns for the shared-block dK GEMM."""
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    begin, block, rows = geometry.block_begin, geometry.block, geometry.rows

    @kernel
    def block_extract(joined: pl.Tensor, block_gradient: pl.Out[pl.Tensor]):
        with pl.spmd(rows, name_hint="orthrus_dsv41_c1a_block_columns"):
            row = pl.tile.get_block_idx()
            block_gradient[row:row + 1, 0:block] = pl.slice(
                joined, [1, block], [row, begin],
            )
        return block_gradient

    @pl.jit.host
    def entry(joined: pl.Tensor, block_gradient: pl.Out[pl.Tensor]):
        block_extract(joined, block_gradient, device=0)

    return block_extract if inline else with_resident_entry(entry, block_extract)


def make_probability_block_extract(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Round the FP32 K32 probability columns to BF16 for the dV GEMM."""
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    begin, block, rows = geometry.block_begin, geometry.block, geometry.rows

    @kernel
    def probability_block_extract(
        probability: pl.Tensor, block_probability: pl.Out[pl.Tensor],
    ):
        with pl.spmd(rows, name_hint="orthrus_dsv41_c1a_block_probability"):
            row = pl.tile.get_block_idx()
            block_probability[row:row + 1, 0:block] = pl.cast(
                pl.slice(probability, [1, block], [row, begin]),
                pl.BF16, mode="rint",
            )
        return block_probability

    @pl.jit.host
    def entry(probability: pl.Tensor, block_probability: pl.Out[pl.Tensor]):
        probability_block_extract(probability, block_probability, device=0)

    return probability_block_extract if inline else with_resident_entry(entry, probability_block_extract)


def c1a_vjp_factories(geometry=C1AAttentionVJPGeometry()):
    """Return leaves for per-query history and shared temporary K32 rows.

    Joined K/V are laid out as ``[query, selected-row, dimension]``. This is
    required because C1A indexer selections differ between MASK positions;
    treating them as one shared history matrix is incorrect.
    """
    from .pypto_dsv41_online_programs import make_linear

    # Matrix stages are configured here so production-shape compilation tests
    # cover the exact TP4 dimensions used by the model-specific assembly.
    return {
        "joined_history": lambda: make_joined_history(geometry),
        "probability_replay": lambda: make_probability_replay(geometry),
        "probability_vjp": lambda: make_softmax_vjp(geometry),
        "sink_reduce": lambda: make_sink_reduce(geometry),
        "score_block_extract": lambda: make_block_extract(geometry),
        "probability_block_extract": lambda: make_probability_block_extract(geometry),
        "d_probability": lambda: make_linear(
            (geometry.local_heads, geometry.dimension),
            (geometry.padded_columns, geometry.dimension),
            transpose_b=True, batches=geometry.block,
        ),
        "d_query": lambda: make_linear(
            (geometry.local_heads, geometry.padded_columns),
            (geometry.padded_columns, geometry.dimension),
            transpose_b=False, batches=geometry.block,
        ),
        "d_key_block": lambda: make_linear(
            (geometry.rows, geometry.block),
            (geometry.rows, geometry.dimension),
            transpose_a=True, transpose_b=False, gradient_weight=True,
        ),
        "d_value_block": lambda: make_linear(
            (geometry.rows, geometry.block),
            (geometry.rows, geometry.dimension),
            transpose_a=True, transpose_b=False, gradient_weight=True,
        ),
    }
