"""先准备 K32 的卷积和门控，再一次递推；历史矩阵只作为输入。"""

from .baseline import native_module
from .gdn_aux_programs import make_auxiliary
from .gdn_programs import make_vector_step
from .gdn_convolution_programs import make_convolution


def make_block_layer(source, shared_tail):
    import pypto.language as pl
    model = native_module(source, "model")
    stem = model.layer0_stem_layer0_stem_inline
    convolution = make_convolution(source, history_matrices=8 * 48)
    gates, tail = make_auxiliary(source)
    recurrence = make_vector_step(rows=32, value_tile=64, history_matrices=8 * 48, packed=True)

    @pl.jit
    def gdn(hidden: pl.Tensor, input_gamma: pl.Tensor, post_gamma: pl.Tensor,
            mlp_gate: pl.Tensor, mlp_up: pl.Tensor, mlp_down: pl.Tensor,
            qkv_weight: pl.Tensor, z_weight: pl.Tensor, a_weight: pl.Tensor, b_weight: pl.Tensor,
            conv_weight: pl.Tensor, a_log: pl.Tensor, dt_bias: pl.Tensor,
            norm_weight: pl.Tensor, out_weight: pl.Tensor,
            conv_history: pl.Tensor, recurrent_history: pl.Tensor, result: pl.Out[pl.Tensor],
            layer: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32]):
        index = layer - layer // 4
        gamma = pl.slice(input_gamma, [1, 5120], [layer, 0])
        projected_full = pl.create_tensor([32, 10240], dtype=pl.BF16)
        log_decay = pl.create_tensor([32, 48], dtype=pl.FP32)
        beta = pl.create_tensor([32, 48], dtype=pl.FP32)
        z_full = pl.create_tensor([32, 6144], dtype=pl.BF16)
        for half in pl.range(2):
            row = half * 16
            current = pl.slice(hidden, [16, 5120], [row, 0])
            normalized = pl.create_tensor([16, 5120], dtype=pl.BF16)
            qkv = pl.create_tensor([16, 10240], dtype=pl.BF16)
            z = pl.create_tensor([16, 6144], dtype=pl.BF16)
            a = pl.create_tensor([16, 128], dtype=pl.BF16)
            b = pl.create_tensor([16, 128], dtype=pl.BF16)
            normalized, qkv, z, a, b = stem(current, gamma, qkv_weight, z_weight, a_weight, b_weight,
                                           normalized, qkv, z, a, b)
            g = pl.create_tensor([16, 48], dtype=pl.FP32)
            b16 = pl.create_tensor([16, 48], dtype=pl.FP32)
            g, b16 = gates(a, b, a_log, dt_bias, g, b16)
            projected_full[row:row + 16, 0:10240] = qkv
            log_decay[row:row + 16, 0:48] = g
            beta[row:row + 16, 0:48] = b16
            z_full[row:row + 16, 0:6144] = z
        convolved_full = pl.create_tensor([32, 10240], dtype=pl.BF16)
        convolved_full = convolution(projected_full, conv_weight, conv_history, convolved_full, (slot * 48 + index) * 4)
        core = pl.create_tensor([32, 48, 128], dtype=pl.BF16)
        final = pl.create_tensor([48, 128, 128], dtype=pl.FP32)
        # 一次读取 AR 初态，不复制到可写目标，也不在两段之间回写临时矩阵。
        core, final = recurrence(convolved_full, convolved_full, convolved_full, log_decay, beta, recurrent_history, core, final,
                                 32, (slot * 48 + index) * 6144)
        for half in pl.range(2):
            row = half * 16
            current = pl.slice(hidden, [16, 5120], [row, 0])
            local_core: pl.Tensor[[16, 48, 128], pl.BF16] = pl.slice(core, [16, 48, 128], [row, 0, 0])
            local_z: pl.Tensor[[16, 6144], pl.BF16] = pl.slice(z_full, [16, 6144], [row, 0])
            gated = pl.create_tensor([768, 128], dtype=pl.BF16)
            attn = pl.create_tensor([16, 5120], dtype=pl.BF16)
            gated, attn = tail(local_core, local_z, norm_weight, out_weight, gated, attn)
            following = pl.create_tensor([16, 5120], dtype=pl.BF16)
            following = shared_tail(current, attn, post_gamma, mlp_gate, mlp_up, mlp_down, following, layer)
            result[row:row + 16, 0:5120] = following
        return result

    return gdn
