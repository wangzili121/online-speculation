"""只读历史的分页双向 Attention；融合核心与原生 backbone 解耦。"""

from .attention import AttentionShape


def make_attention(shape=AttentionShape(), *, backend="fused", physical_pages=256, pool_layers=1):
    if backend not in {"fused", "staged"}:
        raise ValueError("attention implementation must be fused or staged")
    if any(type(n) is not int or n < 1 for n in (physical_pages, pool_layers)):
        raise ValueError("physical page and cache-layer counts must be positive")
    import pypto.language as pl

    K, HQ, HK, D = shape.block_size, shape.q_heads, shape.kv_heads, shape.head_dim
    C, PAGE, M = shape.context, shape.page_size, shape.query_tile
    PHYSICAL = physical_pages
    LAYERS = pool_layers
    PAGES, TILES = C // PAGE, K // M
    KV_TILE = min(C, 128)
    if C % KV_TILE:
        raise ValueError("context must be divisible by the KV compute tile")
    if C != 512:
        raise ValueError("fused attention currently matches the native 512-position capacity")

    @pl.jit.inline
    def score_part(q: pl.Tensor, keys: pl.Tensor, bias: pl.Tensor,
                   head: pl.Scalar[pl.INT32], column: pl.Scalar[pl.INT32]):
        k = pl.reshape(pl.slice(keys, [1, KV_TILE, D], [head, column, 0]), [KV_TILE, D])
        score = pl.matmul(q, k, out_dtype=pl.FP32, b_trans=True)
        rounded = pl.cast(score, pl.BF16, mode="rint")
        scaled = pl.cast(pl.mul(pl.cast(rounded, pl.FP32), D ** -.5), pl.BF16, mode="rint")
        return pl.col_expand_add(pl.cast(scaled, pl.FP32), pl.slice(bias, [1, KV_TILE], [0, column]))

    @pl.jit.inline
    def fused_compute(query: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, gate: pl.Tensor,
                      bias: pl.Tensor, output: pl.Tensor):
        with pl.spmd(HQ * TILES, name_hint="orthrus_bidirectional_attention"):
            task = pl.tile.get_block_idx()
            head = task // TILES
            row = task % TILES * M
            kv_head = head // (HQ // HK)
            q = pl.slice(query, [M, D], [row, head * D])
            # K/V 分块进入 L1，完整分数保留在片上；避免同时放入两份 512×D。
            first = score_part(q, keys, bias, kv_head, 0)
            second = score_part(q, keys, bias, kv_head, KV_TILE)
            third = score_part(q, keys, bias, kv_head, 2 * KV_TILE)
            fourth = score_part(q, keys, bias, kv_head, 3 * KV_TILE)
            scores = pl.concat(pl.concat(first, second), pl.concat(third, fourth))
            exponential = pl.exp(pl.row_expand_sub(scores, pl.row_max(scores)))
            probability = pl.cast(pl.row_expand_div(exponential, pl.row_sum(exponential)), pl.BF16, mode="rint")
            v = pl.reshape(pl.slice(values, [1, C, D], [kv_head, 0, 0]), [C, D])
            attended = pl.cast(pl.matmul(probability, v, out_dtype=pl.FP32), pl.BF16, mode="rint")
            raw_gate = pl.cast(pl.slice(gate, [M, D], [row, head * D]), pl.FP32)
            sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(raw_gate)), 1.0)), pl.BF16, mode="rint")
            output[row:row + M, head * D:(head + 1) * D] = pl.cast(
                pl.mul(pl.cast(attended, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode="rint")
        return output

    @pl.jit.inline
    def staged_compute(query: pl.Tensor, keys: pl.Tensor, values: pl.Tensor, gate: pl.Tensor,
                       bias: pl.Tensor, output: pl.Tensor):
        # 与原生 QK/softmax/PV 分阶段方式对照；历史准备与融合候选完全相同。
        scores = pl.create_tensor([HQ * K, C], dtype=pl.FP32)
        probabilities = pl.create_tensor([HQ * K, C], dtype=pl.BF16)
        with pl.spmd(HQ * TILES, name_hint="orthrus_staged_qk"):
            task = pl.tile.get_block_idx()
            head = task // TILES
            row = task % TILES * M
            q = pl.slice(query, [M, D], [row, head * D])
            for part in pl.range(C // KV_TILE):
                column = part * KV_TILE
                scores[head * K + row:head * K + row + M, column:column + KV_TILE] = score_part(
                    q, keys, bias, head // (HQ // HK), column)
        with pl.spmd(HQ * TILES, name_hint="orthrus_staged_softmax"):
            row = pl.tile.get_block_idx() * M
            score = pl.slice(scores, [M, C], [row, 0])
            exponential = pl.exp(pl.row_expand_sub(score, pl.row_max(score)))
            probabilities[row:row + M, 0:C] = pl.cast(
                pl.row_expand_div(exponential, pl.row_sum(exponential)), pl.BF16, mode="rint")
        with pl.spmd(HQ * TILES, name_hint="orthrus_staged_pv"):
            task = pl.tile.get_block_idx()
            head = task // TILES
            row = task % TILES * M
            total = pl.full([M, D], dtype=pl.FP32, value=0.0)
            for part in pl.range(C // PAGE):
                column = part * PAGE
                probability = pl.slice(probabilities, [M, PAGE], [head * K + row, column])
                v = pl.reshape(pl.slice(values, [1, PAGE, D], [head // (HQ // HK), column, 0]), [PAGE, D])
                total = pl.add(total, pl.matmul(probability, v, out_dtype=pl.FP32))
            attended = pl.cast(total, pl.BF16, mode="rint")
            raw_gate = pl.cast(pl.slice(gate, [M, D], [row, head * D]), pl.FP32)
            sigmoid = pl.cast(pl.recip(pl.add(pl.exp(pl.neg(raw_gate)), 1.0)), pl.BF16, mode="rint")
            output[row:row + M, head * D:(head + 1) * D] = pl.cast(
                pl.mul(pl.cast(attended, pl.FP32), pl.cast(sigmoid, pl.FP32)), pl.BF16, mode="rint")
        return output

    compute = fused_compute if backend == "fused" else staged_compute

    @pl.jit.inline
    def paged(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
              key_pool: pl.Tensor[[LAYERS, HK, PHYSICAL, PAGE, D], pl.BF16],
              value_pool: pl.Tensor[[LAYERS, HK, PHYSICAL, PAGE, D], pl.BF16], table: pl.Tensor, bias: pl.Tensor,
              output: pl.Tensor, past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
              cache_layer: pl.Scalar[pl.INT32]):
        keys = pl.create_tensor([HK, C, D], dtype=pl.BF16)
        values = pl.create_tensor([HK, C, D], dtype=pl.BF16)
        with pl.spmd(HK * PAGES, name_hint="orthrus_history_gather"):
            task = pl.tile.get_block_idx()
            head = task // PAGES
            logical = task % PAGES
            row = logical * PAGE
            # 未使用页不查表、不读取；补零避免无效 cache 中的 NaN 进入 QK。
            keys[head:head + 1, row:row + PAGE, 0:D] = pl.full([1, PAGE, D], dtype=pl.BF16, value=0.0)
            values[head:head + 1, row:row + PAGE, 0:D] = pl.full([1, PAGE, D], dtype=pl.BF16, value=0.0)
            if row < past:
                physical = pl.read(table, [logical])
                if row + PAGE <= past:
                    keys[head:head + 1, row:row + PAGE, 0:D] = pl.slice(
                        key_pool, [1, 1, 1, PAGE, D], [cache_layer, head, physical, 0, 0], drop_dims=[0, 1])
                    values[head:head + 1, row:row + PAGE, 0:D] = pl.slice(
                        value_pool, [1, 1, 1, PAGE, D], [cache_layer, head, physical, 0, 0], drop_dims=[0, 1])
                else:
                    for token in pl.range(past - row):
                        keys[head:head + 1, row + token:row + token + 1, 0:D] = pl.slice(
                            key_pool, [1, 1, 1, 1, D], [cache_layer, head, physical, token, 0], drop_dims=[0, 1])
                        values[head:head + 1, row + token:row + token + 1, 0:D] = pl.slice(
                            value_pool, [1, 1, 1, 1, D], [cache_layer, head, physical, token, 0], drop_dims=[0, 1])
        with pl.spmd(HK, name_hint="orthrus_temporary_kv"):
            head = pl.tile.get_block_idx()
            for token in pl.range(width):
                keys[head:head + 1, past + token:past + token + 1, 0:D] = pl.reshape(
                    pl.slice(key, [1, D], [token, head * D]), [1, 1, D])
                values[head:head + 1, past + token:past + token + 1, 0:D] = pl.reshape(
                    pl.slice(value, [1, D], [token, head * D]), [1, 1, D])
        output = compute(query, keys, values, gate, bias, output)
        return output

    @pl.jit
    def attention(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
                  key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
                  output: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
                  cache_layer: pl.Scalar[pl.INT32]):
        return paged(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, cache_layer)

    @pl.jit.host
    def entry(query: pl.Tensor, key: pl.Tensor, value: pl.Tensor, gate: pl.Tensor,
              key_pool: pl.Tensor, value_pool: pl.Tensor, table: pl.Tensor, bias: pl.Tensor,
              output: pl.Out[pl.Tensor], past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32],
              cache_layer: pl.Scalar[pl.INT32]):
        attention(query, key, value, gate, key_pool, value_pool, table, bias, output, past, width, cache_layer, device=0)

    return paged, entry
