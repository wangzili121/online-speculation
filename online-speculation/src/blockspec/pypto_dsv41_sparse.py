"""Merge per-query compressed history into the V4.1 SWA and draft-block output."""

from .pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW, make_swa_attention

TOPK = 512
TILE = 128
HEAD_TILE = 16
PARTS = TOPK // TILE


def compressed_physical_rows(selected, page_table, *, anchor, ratio, physical_pages, stride_rows, offset_rows=0):
    """Lower packed Top-K selections against the committed history and raw page stride."""
    import torch

    if (ratio not in (1, 2) or anchor < 0 or physical_pages < 1
            or not 0 <= offset_rows <= stride_rows - WINDOW // ratio):
        raise ValueError("invalid compressed history geometry")
    if (selected.ndim != 2 or selected.shape[0] not in (8, 16, BLOCK)
            or selected.shape[1] != TOPK or selected.dtype not in (torch.int32, torch.int64)):
        raise ValueError("require one padded Top-K row per draft position")
    if page_table.ndim != 1 or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("require an integer logical-to-physical page table")
    live = selected != -1
    lengths = live.sum(dim=-1, dtype=torch.int32)
    if not torch.equal(live, torch.arange(TOPK, device=selected.device) < lengths[:, None]):
        raise ValueError("valid Top-K selections must precede padding")
    logical = selected[live]
    if ((logical < 0) | (logical >= anchor // ratio)).any():
        raise ValueError("compressed row extends beyond committed history")
    page_rows = WINDOW // ratio
    page, offset = logical // page_rows, logical % page_rows
    if (page >= page_table.numel()).any():
        raise ValueError("compressed row has no logical page")
    physical = page_table[page]
    if ((physical < 0) | (physical >= physical_pages)).any():
        raise ValueError("compressed row has no physical page")
    rows = torch.full_like(selected, -1, dtype=torch.int32)
    rows[live] = (physical * stride_rows + offset_rows + offset).int()
    return rows, lengths


def make_sparse_attention(*, head_major_swa=False, block_size=BLOCK):
    import pypto.language as pl

    @pl.jit
    def attend(
        query: pl.Tensor, pages: pl.Tensor, rows: pl.Tensor, lengths: pl.Tensor,
        bias: pl.Tensor, zeros: pl.Tensor, local_output: pl.Tensor, local_lse: pl.Tensor,
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        segment_output = pl.create_tensor([block_size * PARTS * HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
        segment_lse = pl.create_tensor([block_size * PARTS * HEAD_TILE, 1], dtype=pl.FP32)
        if head_major_swa:
            # SWA emits one block per head. Merge consumes all heads of one token.
            local_values = pl.create_tensor([block_size * HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
            lse_rows = pl.create_tensor([block_size, HEAD_TILE], dtype=pl.FP32)
            with pl.spmd(block_size, name_hint="orthrus_v41_swa_layout"):
                token = pl.tile.get_block_idx()
                lse_row = pl.tile.full([1, HEAD_TILE], pl.FP32, value=0.0)
                for head in pl.range(HEAD_TILE):
                    source = head * block_size + token
                    dest = token * HEAD_TILE + head
                    local_values[dest:dest + 1, :] = local_output[source:source + 1, :]
                    pl.tile.write(lse_row, [0, head], pl.read(local_lse, [source, 0]))
                pl.store(lse_row, [token, 0], lse_rows)
            local_logsumexp = pl.reshape(lse_rows, [block_size * HEAD_TILE, 1])
        else:
            local_values = local_output
            local_logsumexp = local_lse

        # Each query has its own Top-K. Heads within that query reuse the same KV tile.
        with pl.spmd(block_size * PARTS, name_hint="orthrus_v41_compressed"):
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

        with pl.spmd(block_size, name_hint="orthrus_v41_compressed_merge"):
            token = pl.tile.get_block_idx()
            start = token * HEAD_TILE
            count = pl.cast(pl.read(lengths, [token]), pl.INDEX)
            merged = pl.slice(local_values, [HEAD_TILE, HEAD_DIM], [start, 0])
            merged_lse = pl.slice(local_logsumexp, [HEAD_TILE, 1], [start, 0])
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

    return attend


def make_sparse_program(*, head_major_swa=False, block_size=BLOCK):
    import pypto.language as pl

    attend = make_sparse_attention(head_major_swa=head_major_swa, block_size=block_size)

    @pl.jit.host
    def entry(
        query: pl.Tensor, pages: pl.Tensor, rows: pl.Tensor, lengths: pl.Tensor,
        bias: pl.Tensor, zeros: pl.Tensor, local_output: pl.Tensor, local_lse: pl.Tensor,
        output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        attend(query, pages, rows, lengths, bias, zeros, local_output, local_lse, output, lse, device=0)

    return entry


def make_joined_attention(*, block_size=BLOCK):
    """Dispatch staged SWA and compressed Attention together for the TP4 layout."""
    import pypto.language as pl

    swa = pl.jit.inline(make_swa_attention(HEAD_TILE, token_major_query=True, block_size=block_size)._func)
    sparse = pl.jit.inline(make_sparse_attention(head_major_swa=True, block_size=block_size)._func)

    @pl.jit
    def attend(
        query: pl.Tensor, swa_pages: pl.Tensor, spans: pl.Tensor, temporary: pl.Tensor,
        history_bias: pl.Tensor, sink: pl.Tensor, swa_zeros: pl.Tensor,
        query_flat: pl.Tensor, pages: pl.Tensor, rows: pl.Tensor, lengths: pl.Tensor,
        bias: pl.Tensor, zeros: pl.Tensor, output: pl.Out[pl.Tensor], lse: pl.Out[pl.Tensor],
    ):
        local = pl.create_tensor([block_size * HEAD_TILE, HEAD_DIM], dtype=pl.FP32)
        local_lse = pl.create_tensor([block_size * HEAD_TILE, 1], dtype=pl.FP32)
        local, local_lse = swa(query, swa_pages, spans, temporary, history_bias, sink, swa_zeros,
                               local, local_lse)
        output, lse = sparse(query_flat, pages, rows, lengths, bias, zeros, local, local_lse, output, lse)
        return output, lse

    return attend
