"""K32 GDN 层：投影、只读卷积、临时递推和共享 MLP。"""

from .gdn_aux_programs import make_auxiliary
from .gdn_programs import make_vector_step
from .gdn_convolution_programs import make_convolution
from .projection_programs import make_projection, make_norm


def make_block_layer(source, shared_tail):
    import pypto.language as pl
    norm, _ = make_norm(source)
    qkv_linear, _ = make_projection(source, 5120, 10240)
    z_linear, _ = make_projection(source, 5120, 6144)
    a_linear, _ = make_projection(source, 5120, 128)
    b_linear, _ = make_projection(source, 5120, 128)
    gates, tail = make_auxiliary(source, rows=32)
    convolution = make_convolution(source, history_matrices=8 * 48)
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
        gamma: pl.Tensor[[1, 5120], pl.FP32] = pl.slice(input_gamma, [1, 5120], [layer, 0])
        normalized = pl.create_tensor([32, 5120], dtype=pl.BF16)
        normalized = norm(hidden, gamma, normalized)
        qkv = pl.create_tensor([32, 10240], dtype=pl.BF16)
        z = pl.create_tensor([32, 6144], dtype=pl.BF16)
        a = pl.create_tensor([32, 128], dtype=pl.BF16)
        b = pl.create_tensor([32, 128], dtype=pl.BF16)
        qkv = qkv_linear(normalized, qkv_weight, qkv)
        z = z_linear(normalized, z_weight, z)
        a = a_linear(normalized, a_weight, a)
        b = b_linear(normalized, b_weight, b)
        decay = pl.create_tensor([32, 48], dtype=pl.FP32)
        beta = pl.create_tensor([32, 48], dtype=pl.FP32)
        decay, beta = gates(a, b, a_log, dt_bias, decay, beta)
        convolved = pl.create_tensor([32, 10240], dtype=pl.BF16)
        convolved = convolution(qkv, conv_weight, conv_history, convolved, (slot * 48 + index) * 4)
        core = pl.create_tensor([32, 48, 128], dtype=pl.BF16)
        final = pl.create_tensor([48, 128, 128], dtype=pl.FP32)
        core, final = recurrence(convolved, convolved, convolved, decay, beta, recurrent_history, core, final,
                                 32, (slot * 48 + index) * 6144)
        gated = pl.create_tensor([1536, 128], dtype=pl.BF16)
        attention = pl.create_tensor([32, 5120], dtype=pl.BF16)
        gated, attention = tail(core, z, norm_weight, out_weight, gated, attention)
        result = shared_tail(hidden, attention, post_gamma, mlp_gate, mlp_up, mlp_down, result, layer)
        return result

    return gdn
