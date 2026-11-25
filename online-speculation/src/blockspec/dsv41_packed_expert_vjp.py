"""Input and route-weight VJPs for the packed A3 W8A8 routed expert."""


def packed_expert_vjp_reference(value, gradient, route, gate, gate_scale, up, up_scale, down, down_scale, limit=10.0):
    """Replay the native INT8 forward with straight-through quantization."""
    import torch
    import torch.nn.functional as F

    def project(source, weight, scale):
        amax = source.double().abs().amax(-1, keepdim=True).clamp_min(float(torch.tensor(127e-30)))
        codes = (source.double() * 127 / amax).round().clamp(-127, 127)
        integer_dot = codes @ weight.double().T
        exact = integer_dot.float() * (amax.float() * (1 / 127)) * scale
        surrogate = (source @ weight.float().T) * scale
        return exact.detach() + (surrogate - surrogate.detach())

    x = value.detach().float().requires_grad_(True)
    factor = route.detach().float().requires_grad_(True)
    # Round primal values at native BF16 boundaries and preserve the FP32 tangent.
    def narrow(source):
        return source.detach().bfloat16().float() + (source - source.detach())

    hidden = F.silu(narrow(project(x, gate, gate_scale)).clamp(max=limit)) * narrow(project(x, up, up_scale)).clamp(-limit, limit)
    backward_output = project(narrow(hidden * factor), down, down_scale)
    dx, d_route = torch.autograd.grad(backward_output, (x, factor), gradient)
    return narrow(backward_output).detach(), dx, d_route


def make_packed_expert_vjp(rows, hidden, intermediate, limit, *, inline=False):
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.activation_quant import quantize_projection_activation
    from models.deepseek_v4_1_flash_a3.gemm import gemm_int8_tiled

    def reverse_projection(source_width, target_width, hint):
        mt, kt, nt = 16, 128, 64
        panels = target_width // nt

        @pl.jit.inline(auto_scope=False)
        def project(gradient: pl.Tensor, weight: pl.Tensor, scale: pl.Tensor, output: pl.Tensor):
            with pl.spmd(rows // mt * panels, name_hint=hint):
                task = pl.tile.get_block_idx()
                row, col = task // panels * mt, task % panels * nt
                accum = pl.create_tensor([mt, nt], dtype=pl.FP32)
                for step in pl.range(source_width // kt):
                    offset = step * kt
                    incoming = pl.slice(gradient, [mt, kt], [row, offset])
                    factor = pl.slice(scale, [1, kt], [0, offset])
                    incoming = pl.cast(pl.col_expand_mul(incoming, factor), pl.BF16, mode="rint")
                    # The frozen matrix is already output-major [K, N] for this VJP.
                    # Materialize one tile at a time and fold scales into its left operand.
                    codes_f16 = pl.cast(pl.slice(weight, [kt, nt], [offset, col]), pl.FP16)
                    codes = pl.cast(pl.cast(codes_f16, pl.FP32), pl.BF16, mode="rint")
                    accum = pl.matmul_acc(accum, incoming, codes, init_cond=(step == 0))
                output[row:row + mt, col:col + nt] = accum
            return output

        return project

    reverse_down = reverse_projection(hidden, intermediate, "orthrus_packed_expert_down_vjp")
    reverse_gate = reverse_projection(intermediate, hidden, "orthrus_packed_expert_gate_vjp")
    reverse_up = reverse_projection(intermediate, hidden, "orthrus_packed_expert_up_vjp")

    @pl.jit.inline(auto_scope=False)
    def activation(
        gate: pl.Tensor, up: pl.Tensor, dy: pl.Tensor, route: pl.Tensor,
        d_gate: pl.Tensor, d_up: pl.Tensor, d_route: pl.Tensor, active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(rows, name_hint="orthrus_packed_expert_weighted_swiglu_vjp"):
            row = pl.tile.get_block_idx()
            zero = pl.full([1, intermediate], dtype=pl.FP32, value=0.)
            route_gradient = pl.tile.full([1, 16], dtype=pl.FP32, value=0.)
            if row < active:
                gate_raw = pl.cast(pl.slice(gate, [1, intermediate], [row, 0]), pl.FP32)
                up_raw = pl.cast(pl.slice(up, [1, intermediate], [row, 0]), pl.FP32)
                g = pl.minimum(gate_raw, limit)
                u = pl.minimum(pl.maximum(up_raw, -limit), limit)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(g)), 1.))
                silu = pl.mul(g, sigmoid)
                silu_derivative = pl.mul(sigmoid, pl.add(pl.mul(g, pl.add(pl.neg(sigmoid), 1.)), 1.))
                upstream = pl.slice(dy, [1, intermediate], [row, 0])
                scaled = pl.mul(upstream, pl.read(route, [row, 0]))
                gate_mask = pl.add(pl.neg(pl.cmp(gate_raw, limit, cmp_type=4)), 1.)
                up_mask = pl.mul(pl.cmp(up_raw, -limit, cmp_type=5),
                                 pl.add(pl.neg(pl.cmp(up_raw, limit, cmp_type=4)), 1.))
                d_gate[row:row + 1, :] = pl.mul(pl.mul(pl.mul(scaled, u), silu_derivative), gate_mask)
                d_up[row:row + 1, :] = pl.mul(pl.mul(scaled, silu), up_mask)
                product = pl.reshape(pl.mul(upstream, pl.mul(silu, u)), [8, intermediate // 8])
                ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
                partial = pl.row_expand_mul(ones, pl.row_sum(product))
                total = pl.row_sum(pl.transpose(partial, 0, 1))
                pl.tile.write(route_gradient, [0, 0], pl.read(total, [0, 0]))
            else:
                d_gate[row:row + 1, :] = zero
                d_up[row:row + 1, :] = zero
            pl.store(route_gradient, [row, 0], d_route)
        return d_gate, d_up, d_route

    kernel = pl.jit.inline(auto_scope=False) if inline else pl.jit

    @kernel
    def expert(
        value: pl.Tensor[[rows, hidden], pl.BF16],
        dy: pl.Tensor[[rows, hidden], pl.FP32],
        route: pl.Tensor[[rows, 16], pl.FP32],
        w1: pl.Tensor[[intermediate, hidden], pl.INT8],
        s1: pl.Tensor[[1, intermediate], pl.FP32],
        w3: pl.Tensor[[intermediate, hidden], pl.INT8],
        s3: pl.Tensor[[1, intermediate], pl.FP32],
        w2: pl.Tensor[[hidden, intermediate], pl.INT8],
        s2: pl.Tensor[[1, hidden], pl.FP32],
        dx: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        d_route: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        x_int8, factor = quantize_projection_activation(value, active, hidden, True, False)
        gate = pl.create_tensor([rows, intermediate], dtype=pl.BF16)
        up = pl.create_tensor([rows, intermediate], dtype=pl.BF16)
        gemm_int8_tiled(x_int8, factor, w1, s1, gate, active, hidden, intermediate,
                        "orthrus_packed_expert_gate_replay", 128, 256, True, 16)
        gemm_int8_tiled(x_int8, factor, w3, s3, up, active, hidden, intermediate,
                        "orthrus_packed_expert_up_replay", 128, 256, True, 16)
        hidden_vjp = pl.create_tensor([rows, intermediate], dtype=pl.FP32)
        d_gate = pl.create_tensor([rows, intermediate], dtype=pl.FP32)
        d_up = pl.create_tensor([rows, intermediate], dtype=pl.FP32)
        reverse_down(dy, w2, s2, hidden_vjp)
        activation(gate, up, hidden_vjp, route, d_gate, d_up, d_route, active)
        dx_gate = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        dx_up = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        reverse_gate(d_gate, w1, s1, dx_gate)
        reverse_up(d_up, w3, s3, dx_up)
        with pl.spmd(rows, name_hint="orthrus_packed_expert_input_gradient"):
            row = pl.tile.get_block_idx()
            for col in pl.range(0, hidden, 512):
                grad = pl.add(pl.slice(dx_gate, [1, 512], [row, col]), pl.slice(dx_up, [1, 512], [row, col]))
                dx[row:row + 1, col:col + 512] = grad
        return dx, d_route

    if inline:
        return expert

    @pl.jit.host
    def entry(
        value: pl.Tensor[[rows, hidden], pl.BF16],
        dy: pl.Tensor[[rows, hidden], pl.FP32],
        route: pl.Tensor[[rows, 16], pl.FP32],
        w1: pl.Tensor[[intermediate, hidden], pl.INT8],
        s1: pl.Tensor[[1, intermediate], pl.FP32],
        w3: pl.Tensor[[intermediate, hidden], pl.INT8],
        s3: pl.Tensor[[1, intermediate], pl.FP32],
        w2: pl.Tensor[[hidden, intermediate], pl.INT8],
        s2: pl.Tensor[[1, hidden], pl.FP32],
        dx: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        d_route: pl.Out[pl.Tensor[[rows, 16], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        expert(value, dy, route, w1, s1, w3, s3, w2, s2, dx, d_route, active, device=0)

    return entry
