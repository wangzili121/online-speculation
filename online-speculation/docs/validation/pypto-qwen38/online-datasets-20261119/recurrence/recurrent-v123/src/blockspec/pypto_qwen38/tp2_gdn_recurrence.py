"""只读目标状态的 K32 GDN 递推；预取只改变搬运粒度。"""


def make_recurrence(batch_size, *, prefetch=False, state_tile=64, precompute_gates=False):
    if batch_size not in (1, 2) or type(prefetch) is not bool or state_tile not in (64, 128):
        raise ValueError("one/two blocks, boolean prefetch and 64/128 state rows required")
    if type(precompute_gates) is not bool or precompute_gates and prefetch:
        raise ValueError("gate precomputation and input prefetch are separate candidates")
    import pypto.language as pl
    pieces = 128 // state_tile
    tasks = 24 * pieces * batch_size
    blocks = min(tasks, 64)
    T = state_tile

    @pl.jit.inline
    def recurrent(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor,
                  history: pl.Tensor, out: pl.Tensor, slot: pl.Scalar[pl.INT32],
                  slot_second: pl.Scalar[pl.INT32], layer: pl.Scalar[pl.INT32]):
        states = pl.reshape(history, [32 * 48 * 24 * 128, 128])
        with pl.spmd(blocks, name_hint="orthrus_tp2_recurrent"):
            for task in pl.range(pl.tile.get_block_idx(), tasks, blocks):
                seq = task // (24 * pieces)
                head = task % (24 * pieces) // pieces
                first = seq * 32
                col = task % pieces * T
                request_slot = slot + seq * (slot_second - slot)
                start = ((request_slot * 48 + layer) * 24 + head) * 128 + col
                state = pl.slice(states, [T, 128], [start, 0])
                if precompute_gates:
                    # 一次读取整块门控值，递推中不再逐位置读取标量或重复展开 exp。
                    logs = pl.slice(decay, [32, 32], [first, 0])
                    strengths = pl.slice(beta, [32, 32], [first, 0])
                    factors = pl.exp(pl.row_expand_mul(pl.full([32, 128], pl.FP32, value=1.),
                                                      pl.slice(logs, [32, 1], [0, head])))
                    for token in pl.range(32):
                        query = pl.slice(q, [1, 128], [first + token, head * 128])
                        key = pl.slice(k, [1, 128], [first + token, head * 128])
                        value = pl.cast(pl.slice(v, [1, T], [first + token, 2048 + head * 128 + col]), pl.FP32)
                        factor = pl.slice(factors, [1, 128], [token, 0])
                        gate_strength = pl.slice(strengths, [1, 1], [token, head])
                        faded = pl.col_expand_mul(state, factor)
                        remembered = pl.row_sum(pl.col_expand_mul(faded, key))
                        delta = pl.col_expand_mul(pl.sub(pl.reshape(value, [T, 1]), remembered), gate_strength)
                        state = pl.add(faded, pl.col_expand_mul(
                            pl.row_expand_mul(pl.full([T, 128], pl.FP32, value=1.), delta), key))
                        y = pl.reshape(pl.row_sum(pl.col_expand_mul(state, query)), [1, T])
                        out[first + token:first + token + 1, head * 128 + col:head * 128 + col + T] = pl.cast(
                            y, pl.BF16, mode="rint")
                elif prefetch:
                    queries = pl.slice(q, [32, 128], [first, head * 128])
                    keys = pl.slice(k, [32, 128], [first, head * 128])
                    values = pl.cast(pl.slice(v, [32, T], [first, 2048 + head * 128 + col]), pl.FP32)
                    for token in pl.range(32):
                        query = pl.slice(queries, [1, 128], [token, 0])
                        key = pl.slice(keys, [1, 128], [token, 0])
                        value = pl.slice(values, [1, T], [token, 0])
                        decay_value = pl.read(decay, [first + token, head])
                        strength = pl.read(beta, [first + token, head])
                        factor = pl.exp(pl.mul(pl.full([1, 128], pl.FP32, value=1.), decay_value))
                        faded = pl.col_expand_mul(state, factor)
                        remembered = pl.row_sum(pl.col_expand_mul(faded, key))
                        delta = pl.mul(pl.sub(pl.reshape(value, [T, 1]), remembered), strength)
                        state = pl.add(faded, pl.col_expand_mul(
                            pl.row_expand_mul(pl.full([T, 128], pl.FP32, value=1.), delta), key))
                        y = pl.reshape(pl.row_sum(pl.col_expand_mul(state, query)), [1, T])
                        out[first + token:first + token + 1, head * 128 + col:head * 128 + col + T] = pl.cast(
                            y, pl.BF16, mode="rint")
                else:
                    for token in pl.range(32):
                        query = pl.slice(q, [1, 128], [first + token, head * 128])
                        key = pl.slice(k, [1, 128], [first + token, head * 128])
                        value = pl.cast(pl.slice(v, [1, T], [first + token, 2048 + head * 128 + col]), pl.FP32)
                        decay_value = pl.read(decay, [first + token, head])
                        strength = pl.read(beta, [first + token, head])
                        factor = pl.exp(pl.mul(pl.full([1, 128], pl.FP32, value=1.), decay_value))
                        faded = pl.col_expand_mul(state, factor)
                        remembered = pl.row_sum(pl.col_expand_mul(faded, key))
                        delta = pl.mul(pl.sub(pl.reshape(value, [T, 1]), remembered), strength)
                        state = pl.add(faded, pl.col_expand_mul(
                            pl.row_expand_mul(pl.full([T, 128], pl.FP32, value=1.), delta), key))
                        y = pl.reshape(pl.row_sum(pl.col_expand_mul(state, query)), [1, T])
                        out[first + token:first + token + 1, head * 128 + col:head * 128 + col + T] = pl.cast(
                            y, pl.BF16, mode="rint")
        return out

    return recurrent


def make_entry(batch_size, *, prefetch=False, state_tile=64, precompute_gates=False):
    import pypto.language as pl
    recurrent = make_recurrence(batch_size, prefetch=prefetch, state_tile=state_tile, precompute_gates=precompute_gates)

    @pl.jit
    def kernel(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor,
               history: pl.Tensor, out: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32],
               slot_second: pl.Scalar[pl.INT32], layer: pl.Scalar[pl.INT32]):
        return recurrent(q, k, v, decay, beta, history, out, slot, slot_second, layer)

    @pl.jit.host
    def entry(q: pl.Tensor, k: pl.Tensor, v: pl.Tensor, decay: pl.Tensor, beta: pl.Tensor,
              history: pl.Tensor, out: pl.Out[pl.Tensor], slot: pl.Scalar[pl.INT32],
              slot_second: pl.Scalar[pl.INT32], layer: pl.Scalar[pl.INT32]):
        kernel(q, k, v, decay, beta, history, out, slot, slot_second, layer, device=0)

    return entry
