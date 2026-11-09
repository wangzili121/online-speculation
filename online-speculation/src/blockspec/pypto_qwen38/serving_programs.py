"""推理收尾的 PyPTO 计算：有限值检查及首个最大值 token。"""


def make_greedy(*, source_rows=32, vocab=248320, tile=1024):
    if any(type(n) is not int or n < 1 for n in (source_rows, vocab, tile)) or tile % 256:
        raise ValueError("positive dimensions and a 256-aligned vocabulary tile required")
    import pypto.language as pl
    CHUNKS, FULL, TAIL = (vocab + tile - 1) // tile, vocab // tile, vocab % tile
    PAD = (CHUNKS + 7) // 8 * 8
    V = float(vocab)

    def make_part(columns):
        @pl.jit.inline
        def part(logits: pl.Tensor, maxima: pl.Tensor, indices: pl.Tensor, finite: pl.Tensor,
                 begin: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
                 chunk: pl.Scalar[pl.INT32], group: pl.Scalar[pl.INT32]):
            row = group * 8
            if row < width:
                count = pl.min(8, width - row)
                values = pl.cast(pl.slice(logits, [8, columns], [begin + row, chunk * tile],
                                           valid_shape=[count, columns], pad_value=0.), pl.FP32)
                best = pl.row_max(values)
                token = pl.add(pl.row_argmax(values), pl.cast(chunk * tile, pl.INT32))
                # NaN 比较返回 false；正负 Inf 也不满足有限上界。
                valid = pl.cast(pl.row_min(pl.cmp(pl.abs(values), 3.4028234663852886e38, cmp_type=3)), pl.INT32)
                maxima[chunk:chunk + 1, row:row + 8] = pl.reshape(best, [1, 8])
                indices[chunk:chunk + 1, row:row + 8] = pl.reshape(token, [1, 8])
                finite[chunk:chunk + 1, row:row + 8] = pl.reshape(valid, [1, 8])
            return maxima, indices, finite
        return part

    whole = make_part(tile)
    tail = make_part(TAIL or tile)

    @pl.jit
    def greedy(source: pl.Tensor, result: pl.Out[pl.Tensor],
               begin: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        logits = pl.reshape(source, [source_rows, vocab])
        maxima = pl.create_tensor([PAD, 32], dtype=pl.FP32)
        indices = pl.create_tensor([PAD, 32], dtype=pl.INT32)
        finite = pl.create_tensor([PAD, 32], dtype=pl.INT32)
        with pl.spmd(4, name_hint="orthrus_greedy_clear"):
            row = pl.tile.get_block_idx() * 8
            maxima[0:PAD, row:row + 8] = pl.full([PAD, 8], dtype=pl.FP32, value=-3.4028234663852886e38)
            indices[0:PAD, row:row + 8] = pl.full([PAD, 8], dtype=pl.INT32, value=vocab)
            finite[0:PAD, row:row + 8] = pl.full([PAD, 8], dtype=pl.INT32, value=1)
        with pl.spmd(FULL * 4, name_hint="orthrus_greedy_parts"):
            task = pl.tile.get_block_idx()
            chunk = task // 4
            group = task % 4
            maxima, indices, finite = whole(logits, maxima, indices, finite, begin, width, chunk, group)
        if TAIL != 0:
            with pl.spmd(4, name_hint="orthrus_greedy_tail"):
                group = pl.tile.get_block_idx()
                maxima, indices, finite = tail(logits, maxima, indices, finite, begin, width, FULL, group)
        with pl.spmd(4, name_hint="orthrus_greedy_merge"):
            row = pl.tile.get_block_idx() * 8
            values = pl.transpose(pl.slice(maxima, [PAD, 8], [0, row]), 0, 1)
            ids = pl.cast(pl.transpose(pl.slice(indices, [PAD, 8], [0, row]), 0, 1), pl.FP32)
            flags = pl.cast(pl.transpose(pl.slice(finite, [PAD, 8], [0, row]), 0, 1), pl.FP32)
            best = pl.row_max(values)
            tied = pl.cmp(pl.row_expand_sub(values, best), 0., cmp_type=0)
            candidates = pl.add(pl.mul(ids, tied), pl.mul(pl.add(pl.neg(tied), 1.), V))
            selected = pl.cast(pl.row_min(candidates), pl.INT32)
            valid = pl.cast(pl.row_min(flags), pl.INT32)
            result[0:1, row:row + 8] = pl.reshape(selected, [1, 8])
            result[1:2, row:row + 8] = pl.reshape(valid, [1, 8])
        return result

    @pl.jit.host
    def entry(source: pl.Tensor, result: pl.Out[pl.Tensor],
              begin: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        greedy(source, result, begin, width, device=0)

    return greedy, entry


def make_state_copy(*, conv_elements=48 * 4 * 10240, recurrent_elements=48 * 48 * 128 * 128,
                    source_slots=8, destination_slots=2):
    """逐字节等价的设备内状态复制；不经过主机或改动源状态。"""
    if any(type(n) is not int or n < 1 for n in (conv_elements, recurrent_elements, source_slots, destination_slots)):
        raise ValueError("positive state dimensions required")
    if conv_elements % 4096 or recurrent_elements % 4096:
        raise ValueError("state dimensions must contain complete 4096-element tiles")
    import pypto.language as pl
    C, R = conv_elements // 4096, recurrent_elements // 4096

    @pl.jit
    def copy(conv: pl.Tensor, recurrent: pl.Tensor, out_conv: pl.Out[pl.Tensor], out_recurrent: pl.Out[pl.Tensor],
             source_slot: pl.Scalar[pl.INT32], destination_slot: pl.Scalar[pl.INT32]):
        c = pl.reshape(conv, [source_slots * C, 4096])
        r = pl.reshape(recurrent, [source_slots * R, 4096])
        oc = pl.reshape(out_conv, [destination_slots * C, 4096])
        ore = pl.reshape(out_recurrent, [destination_slots * R, 4096])
        with pl.spmd(C, name_hint="orthrus_snapshot_conv"):
            row = pl.tile.get_block_idx()
            oc[destination_slot * C + row:destination_slot * C + row + 1, 0:4096] = pl.slice(c, [1, 4096], [source_slot * C + row, 0])
        with pl.spmd(R, name_hint="orthrus_snapshot_recurrent"):
            row = pl.tile.get_block_idx()
            ore[destination_slot * R + row:destination_slot * R + row + 1, 0:4096] = pl.slice(r, [1, 4096], [source_slot * R + row, 0])
        return out_conv, out_recurrent

    @pl.jit.host
    def entry(conv: pl.Tensor, recurrent: pl.Tensor, out_conv: pl.Out[pl.Tensor], out_recurrent: pl.Out[pl.Tensor],
              source_slot: pl.Scalar[pl.INT32], destination_slot: pl.Scalar[pl.INT32]):
        copy(conv, recurrent, out_conv, out_recurrent, source_slot, destination_slot, device=0)

    return copy, entry


def make_draft_inputs(*, mask_token):
    """根据 anchor、历史长度和页表生成 K32 输入；RoPE 表常驻设备。"""
    if type(mask_token) is not int or not 0 <= mask_token < 248320:
        raise ValueError("mask token must belong to the model vocabulary")
    import pypto.language as pl

    @pl.jit
    def prepare(rope_cos: pl.Tensor, rope_sin: pl.Tensor, columns: pl.Tensor, pages: pl.Tensor,
                ids: pl.Out[pl.Tensor], cos: pl.Out[pl.Tensor], sin: pl.Out[pl.Tensor],
                table: pl.Out[pl.Tensor], mask: pl.Out[pl.Tensor],
                anchor: pl.Scalar[pl.INT32], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        ids_row = pl.reshape(ids, [1, 32])
        page_row = pl.reshape(table, [1, 32])
        with pl.spmd(32, name_hint="orthrus_prepare_inputs"):
            row = pl.tile.get_block_idx()
            position = pl.min(past + row, 511)
            cos[row:row + 1, 0:64] = pl.slice(rope_cos, [1, 64], [position, 0])
            sin[row:row + 1, 0:64] = pl.slice(rope_sin, [1, 64], [position, 0])
            if row == 0:
                index = pl.slice(columns, [1, 32], [0, 0])
                first = pl.cmp(index, 0, cmp_type=0)
                ids_row[0:1, 0:32] = pl.add(pl.mul(first, pl.cast(anchor - mask_token, pl.INT32)), mask_token)
                page_row[0:1, 0:32] = pl.reshape(pages, [1, 32])
            if row < 16:
                relative = pl.cast(pl.sub(pl.slice(columns, [1, 512], [0, 0]), pl.cast(past + width, pl.INT32)), pl.FP32)
                visible = pl.cmp(relative, 0., cmp_type=2)
                mask[row:row + 1, 0:512] = pl.add(pl.mul(visible, 10000.), -10000.)
        return ids, cos, sin, table, mask

    @pl.jit.host
    def entry(rope_cos: pl.Tensor, rope_sin: pl.Tensor, columns: pl.Tensor, pages: pl.Tensor,
              ids: pl.Out[pl.Tensor], cos: pl.Out[pl.Tensor], sin: pl.Out[pl.Tensor],
              table: pl.Out[pl.Tensor], mask: pl.Out[pl.Tensor],
              anchor: pl.Scalar[pl.INT32], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        prepare(rope_cos, rope_sin, columns, pages, ids, cos, sin, table, mask, anchor, past, width, device=0)

    return prepare, entry


def make_target_inputs():
    """主模型的因果 mask 和位置查表；页号、输入 token 由调度器提供。"""
    import pypto.language as pl

    @pl.jit
    def prepare(rope_cos: pl.Tensor, rope_sin: pl.Tensor, columns: pl.Tensor, counts: pl.Tensor, offsets: pl.Tensor,
                cos: pl.Out[pl.Tensor], sin: pl.Out[pl.Tensor], masks: pl.Out[pl.Tensor]):
        co = pl.reshape(cos, [128, 64])
        si = pl.reshape(sin, [128, 64])
        ma = pl.reshape(masks, [128, 512])
        with pl.spmd(128, name_hint="orthrus_target_inputs"):
            task = pl.tile.get_block_idx()
            batch = task // 16
            row = task % 16
            count = pl.read(counts, [batch])
            past = pl.read(offsets, [batch])
            if row < count:
                co[task:task + 1, 0:64] = pl.slice(rope_cos, [1, 64], [past + row, 0])
                si[task:task + 1, 0:64] = pl.slice(rope_sin, [1, 64], [past + row, 0])
            else:
                co[task:task + 1, 0:64] = pl.full([1, 64], dtype=pl.BF16, value=1.)
                si[task:task + 1, 0:64] = pl.full([1, 64], dtype=pl.BF16, value=0.)
            boundary = pl.min(past + row + 1, past + count)
            relative = pl.cast(pl.sub(pl.slice(columns, [1, 512], [0, 0]), pl.cast(boundary, pl.INT32)), pl.FP32)
            visible = pl.cmp(relative, 0., cmp_type=2)
            ma[task:task + 1, 0:512] = pl.add(pl.mul(visible, 10000.), -10000.)
        return cos, sin, masks

    @pl.jit.host
    def entry(rope_cos: pl.Tensor, rope_sin: pl.Tensor, columns: pl.Tensor, counts: pl.Tensor, offsets: pl.Tensor,
              cos: pl.Out[pl.Tensor], sin: pl.Out[pl.Tensor], masks: pl.Out[pl.Tensor]):
        prepare(rope_cos, rope_sin, columns, counts, offsets, cos, sin, masks, device=0)

    return prepare, entry
