"""PyPTO leaves for the Orthrus-DSV4.1 C1A Attention backward.

The forward C1A path joins two *frozen* history segments (SWA and selected
compressed rows) with the temporary K32 block.  Backward reuses the saved
FP32 probability, returns dQ and only the temporary block's dK/dV, and reduces
the sink gradient per local head.  Frozen history never receives a gradient.

Large matrix products are intentionally delegated to ``make_linear`` in
``pypto_dsv41_online_programs``.  This file owns the part that a GEMM cannot
express: masked softmax VJP, sink reduction, and extraction of K32 rows.
"""

from dataclasses import dataclass


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
            score_gradient[row:row + 1, 0:columns] = pl.cast(
                pl.reshape(result, [1, columns]), pl.BF16, mode="rint",
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

    return softmax_vjp if inline else entry


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
                value = pl.read(sink_rows, [head * block + row, 0])
                total = pl.add(total, value)
            # Each head owns a complete 64-byte row; consumers use column 0.
            pl.write(sink_gradient, [head, 0], pl.read(total, [0, 0]))
        return sink_gradient

    @pl.jit.host
    def entry(sink_rows: pl.Tensor, sink_gradient: pl.Out[pl.Tensor]):
        sink_reduce(sink_rows, sink_gradient, device=0)

    return sink_reduce if inline else entry


def make_block_extract(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Discard frozen-history dK/dV and retain the temporary K32 rows."""
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    begin, block, dimension = geometry.block_begin, geometry.block, geometry.dimension

    @kernel
    def block_extract(joined: pl.Tensor, block_gradient: pl.Out[pl.Tensor]):
        with pl.spmd(block, name_hint="orthrus_dsv41_c1a_block_gradient"):
            row = pl.tile.get_block_idx()
            block_gradient[row:row + 1, 0:dimension] = pl.slice(
                joined, [1, dimension], [begin + row, 0],
            )
        return block_gradient

    @pl.jit.host
    def entry(joined: pl.Tensor, block_gradient: pl.Out[pl.Tensor]):
        block_extract(joined, block_gradient, device=0)

    return block_extract if inline else entry


def make_head_reduce(geometry=C1AAttentionVJPGeometry(), *, inline=False):
    """Sum per-query-head K/V gradients into the shared latent rows.

    C1A is MQA on the latent K/V side: all local query heads read the same
    history and temporary block.  The two GEMMs therefore first produce one
    gradient matrix per query head.  This kernel performs the missing exact
    head reduction before frozen history is discarded by ``block_extract``.
    """
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    heads, rows, dimension = (
        geometry.local_heads, geometry.token_columns, geometry.dimension,
    )
    tile = 64
    tasks = rows * (dimension // tile)

    @kernel
    def head_reduce(per_head: pl.Tensor, shared: pl.Out[pl.Tensor]):
        with pl.spmd(tasks, name_hint="orthrus_dsv41_c1a_mqa_gradient_reduce"):
            task = pl.tile.get_block_idx()
            row = task // (dimension // tile)
            column = task % (dimension // tile) * tile
            total = pl.full([1, tile], dtype=pl.FP32, value=0.)
            for head in pl.range(heads):
                total = pl.add(
                    total,
                    pl.slice(per_head, [1, tile], [head * rows + row, column]),
                )
            shared[row:row + 1, column:column + tile] = total
        return shared

    @pl.jit.host
    def entry(per_head: pl.Tensor, shared: pl.Out[pl.Tensor]):
        head_reduce(per_head, shared, device=0)

    return head_reduce if inline else entry


def c1a_vjp_factories(geometry=C1AAttentionVJPGeometry()):
    """Return the C1A-specific leaves; GEMMs come from the shared trainer."""
    from .pypto_dsv41_online_programs import make_linear

    # Matrix stages are configured here so production-shape compilation tests
    # cover the exact TP4 dimensions used by the model-specific assembly.
    return {
        "probability_vjp": lambda: make_softmax_vjp(geometry),
        "sink_reduce": lambda: make_sink_reduce(geometry),
        "head_reduce": lambda: make_head_reduce(geometry),
        "block_extract": lambda: make_block_extract(geometry),
        "d_probability": lambda: make_linear(
            (geometry.block, geometry.dimension),
            (geometry.token_columns, geometry.dimension),
            transpose_b=True, batches=geometry.local_heads, b_repeats=geometry.local_heads,
        ),
        "d_query": lambda: make_linear(
            (geometry.block, geometry.token_columns),
            (geometry.token_columns, geometry.dimension),
            transpose_b=False, batches=geometry.local_heads, b_repeats=geometry.local_heads,
        ),
        "d_key_per_head": lambda: make_linear(
            (geometry.block, geometry.token_columns),
            (geometry.block, geometry.dimension),
            transpose_a=True, transpose_b=False, gradient_weight=True,
            batches=geometry.local_heads,
        ),
        "d_value_per_head": lambda: make_linear(
            (geometry.block, geometry.token_columns),
            (geometry.block, geometry.dimension),
            transpose_a=True, transpose_b=False, gradient_weight=True,
            batches=geometry.local_heads,
        ),
    }
