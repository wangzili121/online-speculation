"""Orthrus 分页双向 Attention：历史和临时块直接搬入 L1，不生成连续 KV 副本。"""

from .attention import AttentionShape


def _make_fixed_attention(shape, *, physical_pages, pool_layers, capacity):
    import pypto.language as pl

    K, HQ, HK, D = shape.block_size, shape.q_heads, shape.kv_heads, shape.head_dim
    M, C, PAGE, SPAN = shape.query_tile, capacity, shape.page_size, 128
    ROWS = pool_layers * HK * physical_pages * PAGE
    SLOTS = 1 if M == 32 else 2

    @pl.jit.inline
    def gather(pool: pl.Tensor, block: pl.Tensor, zero: pl.Tensor, table: pl.Tensor,
               layer: pl.Scalar[pl.INT32], head: pl.Scalar[pl.INT32],
               past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32], start: pl.Scalar[pl.INT32]):
        local = pl.create_l1([SPAN, D], pl.BF16)
        local = pl.gather_row(local, zero, [0, 0], [0, 0], [SPAN, D])
        # 每次只搬同一物理页的有效连续行；末页尾部和 inactive MASK 行不读。
        for page in pl.range(SPAN // PAGE):
            begin = start + page * PAGE
            if begin < past:
                count = pl.min(PAGE, past - begin)
                physical = pl.read(table, [begin // PAGE])
                source = ((layer * HK + head) * physical_pages + physical) * PAGE
                local = pl.gather_row(local, pool, [page * PAGE, 0], [source, 0], [PAGE, D],
                                      valid_shape=[count, D])
            lo = pl.max(begin, past)
            hi = pl.min(begin + PAGE, past + width)
            if hi > lo:
                local = pl.gather_row(local, block, [lo - start, 0], [lo - past, head * D], [PAGE, D],
                                      valid_shape=[hi - lo, D])
        return local

    @pl.jit.inline
    def scores(q: pl.Tensor, block: pl.Tensor, pool: pl.Tensor, zero: pl.Tensor,
               table: pl.Tensor, bias: pl.Tensor, layer: pl.Scalar[pl.INT32], head: pl.Scalar[pl.INT32],
               past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32], start: pl.Scalar[pl.INT32]):
        keys = gather(pool, block, zero, table, layer, head, past, width, start)
        product = pl.matmul(q, keys, b_trans=True, out_dtype=pl.FP32)
        rounded = pl.cast(product, pl.BF16, mode="rint")
        scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), D ** -.5), pl.BF16, mode="rint")
        result = pl.col_expand_add(pl.cast(scaled, pl.FP32), pl.slice(bias, [1, SPAN], [0, start]))
        return result

    @pl.jit.inline
    def pair(q: pl.Tensor, block: pl.Tensor, pool: pl.Tensor, zero: pl.Tensor,
             table: pl.Tensor, bias: pl.Tensor, layer: pl.Scalar[pl.INT32], head: pl.Scalar[pl.INT32],
             past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32], start: pl.Scalar[pl.INT32]):
        first = scores(q, block, pool, zero, table, bias, layer, head, past, width, start)
        second = scores(q, block, pool, zero, table, bias, layer, head, past, width, start + SPAN)
        return pl.concat(first, second)

    @pl.jit.inline
    def quad(q: pl.Tensor, block: pl.Tensor, pool: pl.Tensor, zero: pl.Tensor,
             table: pl.Tensor, bias: pl.Tensor, layer: pl.Scalar[pl.INT32], head: pl.Scalar[pl.INT32],
             past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32], start: pl.Scalar[pl.INT32]):
        first = pair(q, block, pool, zero, table, bias, layer, head, past, width, start)
        second = pair(q, block, pool, zero, table, bias, layer, head, past, width, start + 2 * SPAN)
        return pl.concat(first, second)

    score_kernel = scores if capacity == 128 else pair if capacity == 256 else quad

    @pl.jit.inline
    def paged(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
              key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
              output: pl.Tensor, past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
              layer: pl.Scalar[pl.INT32]):
        key_flat = pl.reshape(key_pool, [ROWS, D])
        value_flat = pl.reshape(value_pool, [ROWS, D])
        zero = pl.create_tensor([SPAN, D], dtype=pl.BF16)
        with pl.spmd(1, name_hint="orthrus_l1_zero_source"):
            zero_row = pl.tile.get_block_idx() * SPAN
            zero[zero_row:zero_row + SPAN, 0:D] = pl.full([SPAN, D], dtype=pl.BF16, value=0.)
        with pl.spmd(HQ * (K // M), name_hint="orthrus_direct_paged_attention",
                     optimizations=[pl.cross_core_slot(slot_num=SLOTS)]):
            task = pl.tile.get_block_idx()
            head, row = task // (K // M), task % (K // M) * M
            kv = head // (HQ // HK)
            q = pl.slice(query, [M, D], [row, head * D])
            score = score_kernel(q, key, key_flat, zero, table, bias, layer, kv, past, width, 0)
            exponential = pl.exp(pl.row_expand_sub(score, pl.row_max(score)))
            probability = pl.cast(pl.row_expand_div(exponential, pl.row_sum(exponential)), pl.BF16, mode="rint")
            total = pl.create_tensor([M, D], dtype=pl.FP32)
            # 先保持固定区间累加，避免跨 Cube/Vector 的条件输出携带状态。
            for part in pl.range(C // SPAN):
                start = part * SPAN
                values = gather(value_flat, value, zero, table, layer, kv, past, width, start)
                p = pl.slice(probability, [M, SPAN], [0, start])
                total = pl.matmul_acc(total, p, values, init_cond=(part == 0))
            attended = pl.cast(total, pl.BF16, mode="rint")
            raw_gate = pl.cast(pl.slice(gate, [M, D], [row, head * D]), pl.FP32)
            sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(raw_gate)), 1.)), pl.BF16, mode="rint")
            output[row:row + M, head * D:(head + 1) * D] = pl.cast(
                pl.mul(pl.cast(attended, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode="rint")
        return output

    @pl.jit
    def attention(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
                  key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
                  output: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
                  layer: pl.Scalar[pl.INT32]):
        return paged(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, layer)

    @pl.jit.host
    def entry(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
              key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
              output: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
              layer: pl.Scalar[pl.INT32]):
        attention(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, layer, device=0)

    return paged, entry


def make_paged_attention(shape=AttentionShape(), *, physical_pages=256, pool_layers=1, capacity=None):
    """按有效历史与块长选择固定形状 kernel；控制流在编排层，不进入混合核。"""
    if shape.context != 512 or shape.page_size != 16 or shape.query_tile not in {16, 32}:
        raise ValueError("the pinned paged backend requires context=512, page_size=16 and query_tile=16 or 32")
    if any(type(n) is not int or n < 1 for n in (physical_pages, pool_layers)):
        raise ValueError("positive physical page and cache-layer counts required")
    if capacity is not None:
        if type(capacity) is not int or capacity not in {128, 256, 512}:
            raise ValueError("capacity must be 128, 256 or 512")
        return _make_fixed_attention(shape, physical_pages=physical_pages, pool_layers=pool_layers, capacity=capacity)
    import pypto.language as pl
    short, _ = _make_fixed_attention(shape, physical_pages=physical_pages, pool_layers=pool_layers, capacity=128)
    medium, _ = _make_fixed_attention(shape, physical_pages=physical_pages, pool_layers=pool_layers, capacity=256)
    long, _ = _make_fixed_attention(shape, physical_pages=physical_pages, pool_layers=pool_layers, capacity=512)

    @pl.jit.inline
    def paged(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
              key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
              output: pl.Tensor, past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
              layer: pl.Scalar[pl.INT32]):
        if past + width <= 128:
            output = short(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, layer)
        elif past + width <= 256:
            output = medium(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, layer)
        else:
            output = long(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, layer)
        return output

    @pl.jit
    def attention(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
                  key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
                  output: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
                  layer: pl.Scalar[pl.INT32]):
        return paged(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, layer)

    @pl.jit.host
    def entry(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
              key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
              output: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
              layer: pl.Scalar[pl.INT32]):
        attention(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, layer, device=0)

    return paged, entry
