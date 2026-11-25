"""Shared expert input VJP reading the native A3 INT8 NZ banks directly."""


def make_packed_shared_expert_vjp(rows, hidden, intermediate, limit, *, inline=False):
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.activation_quant import quantize_projection_activation

    # The packed gate/up alternates 64 gate and 64 up columns. The logical
    # input row strides across each 32-column INT8 Cube fractal are unchanged.
    panel, c0 = 64, 32
    paired = 2 * intermediate
    gate_raw_shape = (paired // c0 * hidden, c0)
    down_raw_shape = (hidden // c0 * intermediate, c0)

    def reverse_projection(source_width, target_width, hint):
        kt, nt = 128, 64
        blocks = target_width // nt

        @pl.jit.inline(auto_scope=False)
        def project(gradient: pl.Tensor, weight: pl.Tensor, scale: pl.Tensor, output: pl.Tensor):
            with pl.spmd(rows // 16 * blocks, name_hint=hint):
                replay_task = pl.tile.get_block_idx()
                replay_row, replay_col = replay_task // blocks * 16, replay_task % blocks * nt
                acc = pl.create_tensor([16, nt], dtype=pl.FP32)
                for step in pl.range(source_width // kt):
                    begin = step * kt
                    dy = pl.slice(gradient, [16, kt], [replay_row, begin])
                    factors = pl.slice(scale, [1, kt], [0, begin])
                    lhs = pl.cast(pl.col_expand_mul(dy, factors), pl.BF16, mode="rint")
                    tile = pl.create_tensor([nt, kt], dtype=pl.BF16)
                    for part in pl.range(kt // c0):
                        base = (begin // c0 + part) * target_width + replay_col
                        codes = pl.cast(pl.slice(weight, [nt, c0], [base, 0]), pl.FP16)
                        tile[:, part * c0:(part + 1) * c0] = pl.cast(pl.cast(codes, pl.FP32), pl.BF16, mode="rint")
                    acc = pl.matmul_acc(acc, lhs, tile, b_trans=True, init_cond=(step == 0))
                output[replay_row:replay_row + 16, replay_col:replay_col + nt] = acc
            return output

        return project

    down_vjp = reverse_projection(hidden, intermediate, "orthrus_shared_down_nz_vjp")
    paired_vjp = reverse_projection(paired, hidden, "orthrus_shared_gate_up_nz_vjp")

    kernel = pl.jit.inline(auto_scope=False) if inline else pl.jit

    @kernel
    def shared(
        value: pl.Tensor[[rows, hidden], pl.BF16],
        dy: pl.Tensor[[rows, hidden], pl.FP32],
        w13: pl.Tensor[gate_raw_shape, pl.INT8],
        s1: pl.Tensor[[1, intermediate], pl.FP32],
        s3: pl.Tensor[[1, intermediate], pl.FP32],
        w2: pl.Tensor[down_raw_shape, pl.INT8],
        s2: pl.Tensor[[1, hidden], pl.FP32],
        dx: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        qx, qs = quantize_projection_activation(value, active, hidden, True, False)
        scales = pl.create_tensor([1, paired], dtype=pl.FP32)
        with pl.spmd(intermediate // panel, name_hint="orthrus_shared_gate_up_scales"):
            scale_col = pl.tile.get_block_idx() * panel
            scales[:, 2 * scale_col:2 * scale_col + panel] = s1[:, scale_col:scale_col + panel]
            scales[:, 2 * scale_col + panel:2 * scale_col + 2 * panel] = s3[:, scale_col:scale_col + panel]
        gate_up = pl.create_tensor([rows, paired], dtype=pl.BF16)
        with pl.spmd(rows // 16 * (paired // c0), name_hint="orthrus_shared_gate_up_int8_replay"):
            replay_task = pl.tile.get_block_idx()
            replay_row, group = replay_task // (paired // c0) * 16, replay_task % (paired // c0)
            acc = pl.create_tensor([16, c0], dtype=pl.INT32)
            for k in pl.range(0, hidden, 256):
                lhs = pl.slice(qx, [16, 256], [replay_row, k])
                rhs = pl.slice(w13, [256, c0], [group * hidden + k, 0])
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k == 0))
            replay_col = group * c0
            x_scale = pl.slice(qs, [16, 1], [replay_row, 0])
            w_scale = pl.slice(scales, [1, c0], [0, replay_col])
            rounded = pl.col_expand_mul(pl.row_expand_mul(pl.cast(acc, pl.FP32), x_scale), w_scale)
            gate_up[replay_row:replay_row + 16, replay_col:replay_col + c0] = pl.cast(rounded, pl.BF16, mode="rint")

        grad_hidden = pl.create_tensor([rows, intermediate], dtype=pl.FP32)
        down_vjp(dy, w2, s2, grad_hidden)
        grad_paired = pl.create_tensor([rows, paired], dtype=pl.FP32)
        with pl.spmd(rows * (intermediate // panel), name_hint="orthrus_shared_swiglu_vjp"):
            task = pl.tile.get_block_idx()
            row, col = task // (intermediate // panel), task % (intermediate // panel) * panel
            dgate = pl.full([1, panel], dtype=pl.FP32, value=0.)
            dup = pl.full([1, panel], dtype=pl.FP32, value=0.)
            if row < active:
                gate_raw = pl.cast(pl.slice(gate_up, [1, panel], [row, 2 * col]), pl.FP32)
                up_raw = pl.cast(pl.slice(gate_up, [1, panel], [row, 2 * col + panel]), pl.FP32)
                gate = pl.minimum(gate_raw, limit)
                up = pl.minimum(pl.maximum(up_raw, -limit), limit)
                sig = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.))
                silu = pl.mul(gate, sig)
                dsilu = pl.mul(sig, pl.add(pl.mul(gate, pl.add(pl.neg(sig), 1.)), 1.))
                grad = pl.slice(grad_hidden, [1, panel], [row, col])
                active_gate = pl.add(pl.neg(pl.cmp(gate_raw, limit, cmp_type=4)), 1.)
                active_up = pl.mul(pl.cmp(up_raw, -limit, cmp_type=5),
                                   pl.add(pl.neg(pl.cmp(up_raw, limit, cmp_type=4)), 1.))
                dgate = pl.mul(pl.mul(pl.mul(grad, up), dsilu), active_gate)
                dup = pl.mul(pl.mul(grad, silu), active_up)
            grad_paired[row:row + 1, 2 * col:2 * col + panel] = dgate
            grad_paired[row:row + 1, 2 * col + panel:2 * col + 2 * panel] = dup
        paired_vjp(grad_paired, w13, scales, dx)
        return dx

    if inline:
        return shared

    @pl.jit.host
    def entry(
        value: pl.Tensor[[rows, hidden], pl.BF16],
        dy: pl.Tensor[[rows, hidden], pl.FP32],
        w13: pl.Tensor[gate_raw_shape, pl.INT8],
        s1: pl.Tensor[[1, intermediate], pl.FP32],
        s3: pl.Tensor[[1, intermediate], pl.FP32],
        w2: pl.Tensor[down_raw_shape, pl.INT8],
        s2: pl.Tensor[[1, hidden], pl.FP32],
        dx: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        shared(value, dy, w13, s1, s3, w2, s2, dx, active, device=0)
    return entry
