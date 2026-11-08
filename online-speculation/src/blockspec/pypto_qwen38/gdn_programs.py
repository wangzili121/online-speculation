"""只读历史初态的整块 GDN：原生混合核与纯 Vector 递推对照。"""

from .baseline import native_module


def make_recurrence(source, *, backend="native", value_tile=32):
    if backend not in {"native", "vector16", "vector32"}:
        raise ValueError("GDN backend must be native, vector16 or vector32")
    if type(value_tile) is not int or value_tile not in {16, 32, 64}:
        raise ValueError("value_tile must be 16, 32 or 64")
    import pypto.language as pl

    native = native_module(source, "model").delta_window16_recurrence_window16_inline
    H, D = 48, 128
    T = 32 if backend == "vector32" else 16
    NATIVE, DIRECT = backend == "native", backend == "vector32"

    @pl.jit.inline
    def vector(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
               history: pl.Tensor, result: pl.Tensor, final: pl.Tensor, width: pl.Scalar[pl.INT32]):
        q_rows = pl.reshape(query, [T * H, D])
        k_rows = pl.reshape(key, [T * H, D])
        q_norm = pl.create_tensor([T * H, D], dtype=pl.FP32)
        k_norm = pl.create_tensor([T * H, D], dtype=pl.FP32)
        with pl.spmd(T * H // 16, name_hint="orthrus_gdn_qk_norm") as normalized:
            row = pl.tile.get_block_idx() * 16
            q_tile = pl.cast(pl.slice(q_rows, [16, D], [row, 0]), pl.FP32)
            k_tile = pl.cast(pl.slice(k_rows, [16, D], [row, 0]), pl.FP32)
            q_norm[row:row + 16, 0:D] = pl.mul(pl.row_expand_mul(q_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(q_tile, q_tile)), 1e-6), high_precision=True)), D ** -.5)
            k_norm[row:row + 16, 0:D] = pl.row_expand_mul(k_tile, pl.rsqrt(pl.add(pl.row_sum(pl.mul(k_tile, k_tile)), 1e-6), high_precision=True))
        source_rows = pl.reshape(history, [H * D, D])
        final_rows = pl.reshape(final, [H * D, D])
        output = pl.reshape(result, [T, H * D])
        with pl.spmd(H * (D // value_tile), name_hint="orthrus_gdn_vector_recurrence", deps=[normalized]):
            task = pl.tile.get_block_idx()
            head, begin = task // (D // value_tile), task % (D // value_tile) * value_tile
            state_row = head * D + begin
            state = pl.slice(source_rows, [value_tile, D], [state_row, 0])
            for token in pl.range(width):
                row = token * H + head
                q = q_norm[row:row + 1, 0:D]
                k = k_norm[row:row + 1, 0:D]
                g = pl.read(log_decay, [token, head])
                b = pl.read(beta, [token, head])
                decay = pl.exp(pl.mul(pl.full([1, D], dtype=pl.FP32, value=1.), g))
                decayed = pl.col_expand_mul(state, decay)
                remembered = pl.row_sum(pl.col_expand_mul(decayed, k))
                v = pl.cast(pl.reshape(pl.slice(value, [1, 1, value_tile], [token, head, begin], drop_dims=[0]), [1, value_tile]), pl.FP32)
                # 保留原生 V 行到列的布局变换，单独比较移除 Cube matmul 的效果。
                v_col = pl.mul(pl.row_sum(pl.transpose(pl.col_expand_mul(pl.full([16, value_tile], dtype=pl.FP32, value=1.), v), 0, 1)), 1. / 16)
                delta = pl.mul(pl.sub(v_col, remembered), b)
                state = pl.add(decayed, pl.col_expand_mul(pl.row_expand_mul(pl.full([value_tile, D], dtype=pl.FP32, value=1.), delta), k))
                # q @ state.T 等价于每个 value 行与 q 点乘；保持在 Vector 上。
                prediction = pl.reshape(pl.row_sum(pl.col_expand_mul(state, q)), [1, value_tile])
                output[token:token + 1, head * D + begin:head * D + begin + value_tile] = pl.cast(prediction, pl.BF16, mode="rint")
            for token in pl.range(width, T):
                output[token:token + 1, head * D + begin:head * D + begin + value_tile] = pl.full([1, value_tile], dtype=pl.BF16, value=0.)
            final_rows[state_row:state_row + value_tile, 0:D] = state
        return result, final

    if DIRECT:
        @pl.jit
        def recurrence(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, log_decay: pl.Tensor, beta: pl.Tensor,
                       history: pl.Tensor, result: pl.Out[pl.Tensor], final: pl.Out[pl.Tensor], width: pl.Scalar[pl.INT32]):
            return vector(query, key, value, log_decay, beta, history, result, final, width)
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
                return vector(query, key, value, log_decay, beta, state, result, state, width)

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
