"""K32 Full Attention 准备；保留逐 head 的原生 norm/RoPE 舍入。"""

from .baseline import native_module
from .projection_programs import make_norm, make_projection


def make_prepare(source):
    import pypto.language as pl
    model = native_module(source, "model")
    norm, _ = make_norm(source)
    q_linear, _ = make_projection(source, 5120, 12288)
    kv_linear, _ = make_projection(source, 5120, 1024)
    normalize = model.attention3_prepare_normalize_head_inline
    rotary = model.attention3_prepare_rotary_head_inline

    @pl.jit
    def prepare(hidden: pl.Tensor, input_gamma: pl.Tensor,
                q_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor,
                q_gamma: pl.Tensor, k_gamma: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
                query: pl.Out[pl.Tensor], key: pl.Out[pl.Tensor], value: pl.Out[pl.Tensor],
                gate: pl.Out[pl.Tensor], layer: pl.Scalar[pl.INT32]):
        gamma: pl.Tensor[[1, 5120], pl.FP32] = pl.slice(input_gamma, [1, 5120], [layer, 0])
        normalized = pl.create_tensor([32, 5120], dtype=pl.BF16)
        normalized = norm(hidden, gamma, normalized)
        q_projected = pl.create_tensor([32, 12288], dtype=pl.BF16)
        k_projected = pl.create_tensor([32, 1024], dtype=pl.BF16)
        q_projected = q_linear(normalized, q_weight, q_projected)
        k_projected = kv_linear(normalized, k_weight, k_projected)
        value = kv_linear(normalized, v_weight, value)
        # Q 投影按 head 交错存放 query 和 gate；两半各自应用原生 RoPE。
        with pl.spmd(48, name_hint="orthrus_query_norm_rope32"):
            task = pl.tile.get_block_idx()
            row = task // 24 * 16
            head = task % 24
            raw: pl.Tensor[[16, 256], pl.BF16] = pl.slice(q_projected, [16, 256], [row, head * 512])
            qn = pl.create_tensor([16, 256], dtype=pl.BF16)
            qn = normalize(raw, q_gamma, qn)
            qr = pl.create_tensor([16, 256], dtype=pl.BF16)
            ch: pl.Tensor[[16, 64], pl.BF16] = pl.slice(cos, [16, 64], [row, 0])
            sh: pl.Tensor[[16, 64], pl.BF16] = pl.slice(sin, [16, 64], [row, 0])
            qr = rotary(qn, ch, sh, qr)
            query[row:row + 16, head * 256:head * 256 + 256] = qr
            gate[row:row + 16, head * 256:head * 256 + 256] = pl.slice(q_projected, [16, 256], [row, head * 512 + 256])
        with pl.spmd(8, name_hint="orthrus_key_norm_rope32"):
            task = pl.tile.get_block_idx()
            row = task // 4 * 16
            head = task % 4
            raw: pl.Tensor[[16, 256], pl.BF16] = pl.slice(k_projected, [16, 256], [row, head * 256])
            kn = pl.create_tensor([16, 256], dtype=pl.BF16)
            kn = normalize(raw, k_gamma, kn)
            kr = pl.create_tensor([16, 256], dtype=pl.BF16)
            ch: pl.Tensor[[16, 64], pl.BF16] = pl.slice(cos, [16, 64], [row, 0])
            sh: pl.Tensor[[16, 64], pl.BF16] = pl.slice(sin, [16, 64], [row, 0])
            kr = rotary(kn, ch, sh, kr)
            key[row:row + 16, head * 256:head * 256 + 256] = kr
        return query, key, value, gate

    return prepare


def make_post(source, paged, shared_tail):
    import pypto.language as pl
    output_linear, _ = make_projection(source, 6144, 5120)

    @pl.jit
    def full(hidden: pl.Tensor, post_gamma: pl.Tensor,
             mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
             out_weight: pl.Tensor, query: pl.Tensor, key: pl.Tensor, value: pl.Tensor,
             gate: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor,
             block_table: pl.Tensor, mask: pl.Tensor, result: pl.Out[pl.Tensor],
             layer: pl.Scalar[pl.INT32], offset: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        index = layer // 4
        gated = pl.create_tensor([32, 6144], dtype=pl.BF16)
        gated = paged(query, key, value, gate, key_pool, value_pool, block_table, mask, gated, offset, width, index)
        projected = pl.create_tensor([32, 5120], dtype=pl.BF16)
        projected = output_linear(gated, out_weight, projected)
        result = shared_tail(hidden, projected, post_gamma, mlp_gate, mlp_up, mlp_down, result, layer)
        return result

    return full
