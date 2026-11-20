"""TP2 draft 的只读 GDN 递推；供推理与独立检查共用。"""


def make_recurrence(batch_size, *, valid_rows=False, history_slots=32):
    if type(batch_size) is not int or batch_size not in (1, 2):
        raise ValueError("one or two independent draft blocks required")
    if type(valid_rows) is not bool:
        raise ValueError("valid_rows must be a boolean")
    if type(history_slots) is not int or not batch_size <= history_slots <= 32:
        raise ValueError("history capacity must contain the requested slots")
    import pypto.language as pl
    recurrence_blocks = min(48 * batch_size, 64)
    gdn_valid_rows = valid_rows

    @pl.jit.inline
    def recurrent(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor,
                  history: pl.Tensor, out: pl.Tensor, slot: pl.Scalar[pl.INT32],
                  slot_second: pl.Scalar[pl.INT32], layer: pl.Scalar[pl.INT32],
                  width: pl.Scalar[pl.INT32], width_second: pl.Scalar[pl.INT32]):
        states = pl.reshape(history, [history_slots * 48 * 24 * 128, 128])
        with pl.spmd(recurrence_blocks, name_hint="orthrus_tp2_recurrent"):
            for task in pl.range(pl.tile.get_block_idx(), 48 * batch_size, recurrence_blocks):
                seq = task // 48
                head = task % 48 // 2
                first = seq * 32
                col = task % 2 * 64
                request_slot = slot + seq * (slot_second - slot)
                start = ((request_slot * 48 + layer) * 24 + head) * 128 + col
                state = pl.slice(states, [64, 128], [start, 0])
                count = width + seq * (width_second - width)
                if not gdn_valid_rows:
                    count = 32
                for token in pl.range(count):
                    query = pl.slice(q, [1, 128], [first + token, head * 128])
                    key = pl.slice(k, [1, 128], [first + token, head * 128])
                    decay_value = pl.read(decay, [first + token, head])
                    strength = pl.read(beta, [first + token, head])
                    factor = pl.exp(pl.mul(pl.full([1, 128], pl.FP32, value=1.), decay_value))
                    faded = pl.col_expand_mul(state, factor)
                    remembered = pl.row_sum(pl.col_expand_mul(faded, key))
                    value = pl.cast(pl.slice(v, [1, 64], [first + token, 2048 + head * 128 + col]), pl.FP32)
                    delta = pl.mul(pl.sub(pl.reshape(value, [64, 1]), remembered), strength)
                    state = pl.add(faded, pl.col_expand_mul(
                        pl.row_expand_mul(pl.full([64, 128], pl.FP32, value=1.), delta), key))
                    y = pl.reshape(pl.row_sum(pl.col_expand_mul(state, query)), [1, 64])
                    out[first + token:first + token + 1, head * 128 + col:head * 128 + col + 64] = pl.cast(
                        y, pl.BF16, mode="rint")
                if gdn_valid_rows:
                    # 无效行仍初始化，后续向量/矩阵核不会读取未定义的工作区。
                    for token in pl.range(count, 32):
                        out[first + token:first + token + 1, head * 128 + col:head * 128 + col + 64] = pl.full(
                            [1, 64], pl.BF16, value=0.)
        return out

    return recurrent


def make_entry(batch_size, *, valid_rows=False, history_slots=32):
    import pypto.language as pl
    recurrent = make_recurrence(batch_size, valid_rows=valid_rows, history_slots=history_slots)

    @pl.jit
    def kernel(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor,
               history: pl.Tensor, out: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32],
               slot_second: pl.Scalar[pl.INT32], layer: pl.Scalar[pl.INT32],
               width: pl.Scalar[pl.INT32], width_second: pl.Scalar[pl.INT32]):
        return recurrent(q, k, v, decay, beta, history, out, slot, slot_second, layer, width, width_second)

    @pl.jit.host
    def entry(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor,
              history: pl.Tensor, out: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32],
              slot_second: pl.Scalar[pl.INT32], layer: pl.Scalar[pl.INT32],
              width: pl.Scalar[pl.INT32], width_second: pl.Scalar[pl.INT32]):
        kernel(q, k, v, decay, beta, history, out, slot, slot_second, layer, width, width_second, device=0)
    return entry
