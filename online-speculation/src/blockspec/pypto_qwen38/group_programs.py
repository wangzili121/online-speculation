"""将 3 个 GDN 层和 1 个 Full Attention 层编排为一次 HOST 提交。"""


def make_group_host(gdn, prepare, full):
    import pypto.language as pl

    @pl.jit.host
    def group_host(hidden: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor,
                   mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
                   qkv0: pl.Tensor, z0: pl.Tensor, a0: pl.Tensor, b0: pl.Tensor,
                   conv0: pl.Tensor, log0: pl.Tensor, dt0: pl.Tensor, norm0: pl.Tensor, out0: pl.Tensor,
                   qkv1: pl.Tensor, z1: pl.Tensor, a1: pl.Tensor, b1: pl.Tensor,
                   conv1: pl.Tensor, log1: pl.Tensor, dt1: pl.Tensor, norm1: pl.Tensor, out1: pl.Tensor,
                   qkv2: pl.Tensor, z2: pl.Tensor, a2: pl.Tensor, b2: pl.Tensor,
                   conv2: pl.Tensor, log2: pl.Tensor, dt2: pl.Tensor, norm2: pl.Tensor, out2: pl.Tensor,
                   q_weight: pl.Tensor, k_weight: pl.Tensor, v_weight: pl.Tensor,
                   q_gamma: pl.Tensor, k_gamma: pl.Tensor, out_weight: pl.Tensor,
                   cos: pl.Tensor, sin: pl.Tensor,
                   query: pl.Out[pl.Tensor], key: pl.Out[pl.Tensor], value: pl.Out[pl.Tensor], gate: pl.Out[pl.Tensor],
                   conv_history: pl.Tensor, recurrent_history: pl.Tensor, key_pool: pl.Tensor, value_pool: pl.Tensor,
                   table: pl.Tensor, mask: pl.Tensor, scratch0: pl.Out[pl.Tensor], scratch1: pl.Out[pl.Tensor],
                   result: pl.Out[pl.Tensor], base: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
                   past: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INT32]):
        # 每次都引用调用后的返回值，建立跨调用的数据依赖；历史 cache 只作输入。
        first = gdn(hidden, input_gamma, post_gamma, mlp_gate, mlp_up, mlp_down,
                    qkv0, z0, a0, b0, conv0, log0, dt0, norm0, out0,
                    conv_history, recurrent_history, scratch0, base, slot, device=0)
        second = gdn(first, input_gamma, post_gamma, mlp_gate, mlp_up, mlp_down,
                     qkv1, z1, a1, b1, conv1, log1, dt1, norm1, out1,
                     conv_history, recurrent_history, scratch1, base + 1, slot, device=0)
        third = gdn(second, input_gamma, post_gamma, mlp_gate, mlp_up, mlp_down,
                    qkv2, z2, a2, b2, conv2, log2, dt2, norm2, out2,
                    conv_history, recurrent_history, first, base + 2, slot, device=0)
        query, key, value, gate = prepare(third, input_gamma, q_weight, k_weight, v_weight, q_gamma, k_gamma,
                                          cos, sin, query, key, value, gate, base + 3, device=0)
        full(third, post_gamma, mlp_gate, mlp_up, mlp_down, out_weight, query, key, value, gate,
             key_pool, value_pool, table, mask, result, base + 3, past, width, device=0)

    return group_host


def make_group_prefix(gdn):
    """采集轮次将末组拆为 3 个 GDN + Full，最后一层输入直接写私有缓冲。"""
    import pypto.language as pl

    @pl.jit.host
    def prefix(hidden: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor,
               mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
               qkv0: pl.Tensor, z0: pl.Tensor, a0: pl.Tensor, b0: pl.Tensor,
               conv0: pl.Tensor, log0: pl.Tensor, dt0: pl.Tensor, norm0: pl.Tensor, out0: pl.Tensor,
               qkv1: pl.Tensor, z1: pl.Tensor, a1: pl.Tensor, b1: pl.Tensor,
               conv1: pl.Tensor, log1: pl.Tensor, dt1: pl.Tensor, norm1: pl.Tensor, out1: pl.Tensor,
               qkv2: pl.Tensor, z2: pl.Tensor, a2: pl.Tensor, b2: pl.Tensor,
               conv2: pl.Tensor, log2: pl.Tensor, dt2: pl.Tensor, norm2: pl.Tensor, out2: pl.Tensor,
               conv_history: pl.Tensor, recurrent_history: pl.Tensor,
               scratch0: pl.Out[pl.Tensor], scratch1: pl.Out[pl.Tensor], boundary: pl.Out[pl.Tensor],
               base: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32]):
        first = gdn(hidden, input_gamma, post_gamma, mlp_gate, mlp_up, mlp_down,
                    qkv0, z0, a0, b0, conv0, log0, dt0, norm0, out0,
                    conv_history, recurrent_history, scratch0, base, slot, device=0)
        second = gdn(first, input_gamma, post_gamma, mlp_gate, mlp_up, mlp_down,
                     qkv1, z1, a1, b1, conv1, log1, dt1, norm1, out1,
                     conv_history, recurrent_history, scratch1, base + 1, slot, device=0)
        gdn(second, input_gamma, post_gamma, mlp_gate, mlp_up, mlp_down,
            qkv2, z2, a2, b2, conv2, log2, dt2, norm2, out2,
            conv_history, recurrent_history, boundary, base + 2, slot, device=0)
    return prefix
