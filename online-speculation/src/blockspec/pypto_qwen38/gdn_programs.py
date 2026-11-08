"""只读历史初态的整块 GDN：原生混合核与纯 Vector 递推对照。"""

from .baseline import native_module


def make_vector_step(*, rows=16, value_tile=64, history_matrices=1, packed=False):
    """给独立验证和整网共用的 Vector 递推；输入、输出状态可为同一私有缓冲。"""
    if type(rows) is not int or rows not in {16, 32}:
        raise ValueError("rows must be 16 or 32")
    if type(value_tile) is not int or value_tile not in {16, 32, 64}:
        raise ValueError("value_tile must be 16, 32 or 64")
    if type(history_matrices) is not int or history_matrices < 1:
        raise ValueError("positive history_matrices required")
    if type(packed) is not bool:
        raise ValueError("packed must be a boolean")
    import pypto.language as pl
    H, D, T = 48, 128, rows
    CHANNELS, GROUP = (10240, 3) if packed else (H * D, 1)
    K_OFFSET, V_OFFSET = (2048, 4096) if packed else (0, 0)

    @pl.jit.inline
    def vector(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
               history: pl.Tensor, result: pl.Tensor, final: pl.Tensor, width: pl.Scalar[pl.INT32],
               history_base: pl.Scalar[pl.INT32]):
        source_rows = pl.reshape(history, [history_matrices * H * D, D])
        final_rows = pl.reshape(final, [H * D, D])
        output = pl.reshape(result, [T, H * D])
        queries = pl.reshape(query, [T, CHANNELS])
        keys = pl.reshape(key, [T, CHANNELS])
        values = pl.reshape(value, [T, CHANNELS])
        with pl.spmd(H * (D // value_tile), name_hint="orthrus_gdn_inline_norm_recurrence"):
            task = pl.tile.get_block_idx()
            head, begin = task // (D // value_tile), task % (D // value_tile) * value_tile
            state_row = head * D + begin
            state = pl.slice(source_rows, [value_tile, D], [history_base + state_row, 0])
            # Q/K 直接在本递推任务内归一化，不经全局中间张量。
            q_tile = pl.cast(pl.slice(queries, [T, D], [0, head // GROUP * D]), pl.FP32)
            k_tile = pl.cast(pl.slice(keys, [T, D], [0, K_OFFSET + head // GROUP * D]), pl.FP32)
            q_norm = pl.mul(pl.row_expand_mul(q_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(q_tile, q_tile)), 1e-6), high_precision=True)), D ** -.5)
            k_norm = pl.row_expand_mul(k_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(k_tile, k_tile)), 1e-6), high_precision=True))
            for token in pl.range(width):
                q = q_norm[token:token + 1, 0:D]
                k = k_norm[token:token + 1, 0:D]
                g = pl.read(log_decay, [token, head])
                b = pl.read(beta, [token, head])
                decay = pl.exp(pl.mul(pl.full([1, D], dtype=pl.FP32, value=1.), g))
                decayed = pl.col_expand_mul(state, decay)
                remembered = pl.row_sum(pl.col_expand_mul(decayed, k))
                v = pl.cast(pl.slice(values, [1, value_tile], [token, V_OFFSET + head * D + begin]), pl.FP32)
                # V 已连续存储，行转列用视图即可，不复制或重复求和。
                v_col = pl.reshape(v, [value_tile, 1])
                delta = pl.mul(pl.sub(v_col, remembered), b)
                state = pl.add(decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([value_tile, D], dtype=pl.FP32, value=1.), delta), k))
                # q @ state.T 等价于每个 value 行与 q 点乘；保持在 Vector 上。
                prediction = pl.reshape(pl.row_sum(pl.col_expand_mul(state, q)), [1, value_tile])
                output[token:token + 1, head * D + begin:head * D + begin + value_tile] = pl.cast(prediction, pl.BF16, mode="rint")
            for token in pl.range(width, T):
                output[token:token + 1, head * D + begin:head * D + begin + value_tile] = pl.full([1, value_tile], dtype=pl.BF16, value=0.)
            final_rows[state_row:state_row + value_tile, 0:D] = state
        return result, final

    return vector


def make_recurrence(source, *, backend="native", value_tile=32, packed=False):
    if backend not in {"native", "vector16", "vector32"}:
        raise ValueError("GDN backend must be native, vector16 or vector32")
    if type(value_tile) is not int or value_tile not in {16, 32, 64}:
        raise ValueError("value_tile must be 16, 32 or 64")
    if type(packed) is not bool or packed and backend != "vector32":
        raise ValueError("packed input requires vector32")
    import pypto.language as pl
    H, D = 48, 128
    NATIVE, DIRECT = backend == "native", backend == "vector32"
    native = native_module(source, "model").delta_window16_recurrence_window16_inline
    vector = make_vector_step(rows=32 if DIRECT else 16, value_tile=value_tile, packed=packed)

    if DIRECT:
        @pl.jit
        def recurrence(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
                       history: pl.Tensor, result: pl.Out[pl.Tensor], final: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
            return vector(query, key, value, log_decay, beta, history, result, final, width, 0)
    else:
        if NATIVE:
            @pl.jit.inline
            def step(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
                     state: pl.Tensor, result: pl.Tensor, width: pl.Scalar[pl.INT32]):
                return native(query, key, value, log_decay, beta, state, result, width)
        else:
            @pl.jit.inline
            def step(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
                     state: pl.Tensor, result: pl.Tensor, width: pl.Scalar[pl.INT32]):
                return vector(query, key, value, log_decay, beta, state, result, state, width, 0)

        @pl.jit
        def recurrence(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
                       history: pl.Tensor, result: pl.Out[pl.Tensor], final: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
            source_rows = pl.reshape(history, [H * D, D])
            final_rows = pl.reshape(final, [H * D, D])
            with pl.spmd(H * D // 16, name_hint="orthrus_gdn_copy_initial"):
                row = pl.tile.get_block_idx() * 16
                final_rows[row:row + 16, 0:D] = pl.slice(source_rows, [16, D], [row, 0])
            for half in pl.range(2):
                start = half * 16
                count = pl.min(16, pl.max(0, width - start))
                q: pl.Tensor[[16, H, D], pl.BF16] = pl.slice(query, [16, H, D], [start, 0, 0])
                k: pl.Tensor[[16, H, D], pl.BF16] = pl.slice(key, [16, H, D], [start, 0, 0])
                v: pl.Tensor[[16, H, D], pl.BF16] = pl.slice(value, [16, H, D], [start, 0, 0])
                g: pl.Tensor[[16, H], pl.FP32] = pl.slice(log_decay, [16, H], [start, 0])
                b: pl.Tensor[[16, H], pl.FP32] = pl.slice(beta, [16, H], [start, 0])
                out = pl.create_tensor([16, H, D], dtype=pl.BF16)
                out, final = step(q, k, v, g, b, final, out, count)
                result[start:start + 16, 0:H, 0:D] = out
            return result, final

    @pl.jit.host
    def entry(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
              history: pl.Tensor, result: pl.Out[pl.Tensor], final: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
        recurrence(query, key, value, log_decay, beta, history, result, final, width, device=0)

    return recurrence, entry
