"""Merge per-query compressed history into the V4.1 SWA and draft-block output."""

from .pypto_dsv41_swa import BLOCK, HEAD_DIM

TOPK = 512
TILE = 128
HEAD_TILE = 16
PARTS = TOPK // TILE


def make_sparse_program():
    import pypto.language as pl

    @pl.jit
    def attend(
        query: pl.Tensor, pages: pl.Tensor, rows: pl.Tensor, lengths: pl.Tensor,
        bias: pl.Tensor, zeros: pl.Tensor, local_output: pl.Tensor, local_lse: pl.Tensor,
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        segment_output = pl.create_tensor([BLOCK * PARTS * HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
        segment_lse = pl.create_tensor([BLOCK * PARTS * HEAD_TILE, 1], dtype=pl.FP32)

        # Each query has its own Top-K. Heads within that query reuse the same KV tile.
        with pl.spmd(BLOCK * PARTS, name_hint="orthrus_v41_compressed"):
            task = pl.tile.get_block_idx()
            token, offset = task // PARTS, task % PARTS * TILE
            count = pl.cast(pl.read(lengths, [token]), pl.INDEX)
            if offset < count:
                kv = pl.create_l1([TILE, HEAD_DIM], pl.BF16)
                kv = pl.gather_row(kv, zeros, [0, 0], [0, 0], [TILE, HEAD_DIM])
                for row in pl.range(pl.min(TILE, count - offset)):
                    physical = pl.cast(pl.read(rows, [token, offset + row]), pl.INDEX)
                    kv = pl.gather_row(kv, pages, [row, 0], [physical, 0], [1, HEAD_DIM])
                q = pl.slice(query, [HEAD_TILE, HEAD_DIM], [token * HEAD_TILE, 0])
                scores = pl.mul(pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32), HEAD_DIM ** -0.5)
                scores = pl.col_expand_add(scores, pl.slice(bias, [1, TILE], [token, offset]))
                ref = pl.row_max(scores)
                numerator = pl.exp(pl.row_expand_sub(scores, ref))
                denom = pl.row_sum(numerator)
                probability = pl.row_expand_div(numerator, denom)
                high = pl.cast(probability, pl.BF16, mode="rint")
                low = pl.cast(pl.sub(probability, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
                dst = task * HEAD_TILE
                value = pl.matmul(high, kv, out_dtype=pl.FP32)
                segment_output[dst:dst + HEAD_TILE, 0:HEAD_DIM] = pl.matmul_acc(value, low, kv)
                segment_lse[dst:dst + HEAD_TILE, 0:1] = pl.add(ref, pl.log(denom))

        with pl.spmd(BLOCK, name_hint="orthrus_v41_compressed_merge"):
            token = pl.tile.get_block_idx()
            start = token * HEAD_TILE
            count = pl.cast(pl.read(lengths, [token]), pl.INDEX)
            merged = pl.slice(local_output, [HEAD_TILE, HEAD_DIM], [start, 0])
            merged_lse = pl.slice(local_lse, [HEAD_TILE, 1], [start, 0])
            for part in pl.range(PARTS):
                if part * TILE < count:
                    source = (token * PARTS + part) * HEAD_TILE
                    value = pl.slice(segment_output, [HEAD_TILE, HEAD_DIM], [source, 0])
                    value_lse = pl.slice(segment_lse, [HEAD_TILE, 1], [source, 0])
                    # Local LSE already accounts for the sink, exactly once.
                    common = pl.maximum(merged_lse, value_lse)
                    left = pl.exp(pl.sub(merged_lse, common))
                    right = pl.exp(pl.sub(value_lse, common))
                    total = pl.add(left, right)
                    merged = pl.row_expand_div(
                        pl.add(pl.row_expand_mul(merged, left), pl.row_expand_mul(value, right)), total,
                    )
                    merged_lse = pl.add(common, pl.log(total))
            output[start:start + HEAD_TILE, 0:HEAD_DIM] = merged
            lse[start:start + HEAD_TILE, 0:1] = merged_lse
        return output, lse

    @pl.jit.host
    def entry(
        query: pl.Tensor, pages: pl.Tensor, rows: pl.Tensor, lengths: pl.Tensor,
        bias: pl.Tensor, zeros: pl.Tensor, local_output: pl.Tensor, local_lse: pl.Tensor,
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        attend(query, pages, rows, lengths, bias, zeros, local_output, local_lse, output, lse, device=0)

    return entry
