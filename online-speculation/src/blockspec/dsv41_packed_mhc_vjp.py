"""Common resident FP32 projection and gates VJP for both mHC boundaries."""


def make_packed_mhc_projection_vjp(block, model, column_tile=128):
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.hc import DECODE_LINEAR_OK, LINEAR_T_TILE, MIX_PAD, hc_project_decode

    from .pypto_dsv41_tail_vjp import make_hc_pre_gate_vjp

    hidden, copies = model.hidden_size, model.hc_mult
    flat, channels = hidden * copies, (copies + 2) * copies
    linear_rows = (block + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    gate_vjp = make_hc_pre_gate_vjp(block, copies, model.hc_sinkhorn_iters, model.hc_eps, inline=True)

    @pl.jit.inline(auto_scope=False)
    def project(
        saved: pl.Tensor[[block * copies, hidden], pl.FP32],
        mix_grad: pl.Tensor[[block, 32], pl.FP32],
        function: pl.Tensor[[channels, flat], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        bias: pl.Tensor[[channels], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        x = pl.reshape(saved, [block, copies, hidden])
        inverse = pl.create_tensor([linear_rows, 1], dtype=pl.FP32)
        partials = pl.create_tensor([DECODE_LINEAR_OK * linear_rows, MIX_PAD], dtype=pl.FP32)
        hc_project_decode(x, function, inverse, partials, short_attention=False, one_token=False)
        projection = pl.create_tensor([block, channels], dtype=pl.FP32)
        incoming_pre = pl.create_tensor([block, 8], dtype=pl.FP32)
        incoming_post = pl.create_tensor([block, 8], dtype=pl.FP32)
        incoming_mix = pl.create_tensor([block * copies, 8], dtype=pl.FP32)
        with pl.spmd(block, name_hint="orthrus_hc_projected_and_gate_seeds"):
            token = pl.tile.get_block_idx()
            combined = pl.tile.full([1, channels], dtype=pl.FP32, value=0.)
            for split in pl.range(DECODE_LINEAR_OK):
                combined = pl.add(combined, pl.load(partials, [split * linear_rows + token, 0], [1, channels]))
            pl.store(pl.mul(combined, pl.read(inverse, [token, 0])), [token, 0], projection)
            seed_pre = pl.tile.full([1, 8], dtype=pl.FP32, value=0.)
            seed_post = pl.tile.full([1, 8], dtype=pl.FP32, value=0.)
            for stream in pl.unroll(copies):
                if token < active:
                    pl.tile.write(seed_pre, [0, stream], pl.read(mix_grad, [token, stream]))
                    pl.tile.write(seed_post, [0, stream], pl.read(mix_grad, [token, copies + stream]))
                seed_mix = pl.tile.full([1, 8], dtype=pl.FP32, value=0.)
                if token < active:
                    for target in pl.unroll(copies):
                        pl.tile.write(seed_mix, [0, target],
                                      pl.read(mix_grad, [token, 2 * copies + stream * copies + target]))
                pl.store(seed_mix, [token * copies + stream, 0], incoming_mix)
            pl.store(seed_pre, [token, 0], incoming_pre)
            pl.store(seed_post, [token, 0], incoming_post)

        projected_grad = pl.create_tensor([block, channels], dtype=pl.FP32)
        gate_vjp(projection, scale, bias, incoming_pre, incoming_post, incoming_mix, projected_grad)
        correction = pl.create_tensor([block, 8], dtype=pl.FP32)
        with pl.spmd(block, name_hint="orthrus_hc_rms_vjp_coefficient"):
            tok = pl.tile.get_block_idx()
            dot = pl.cast(0., pl.FP32)
            for channel in pl.range(channels):
                dot = dot + pl.read(projection, [tok, channel]) * pl.read(projected_grad, [tok, channel])
            inv = pl.read(inverse, [tok, 0])
            value = dot * inv * inv * (1. / flat)
            pl.store(pl.add(pl.tile.full([1, 8], dtype=pl.FP32, value=0.), value), [tok, 0], correction)

        return projected_grad, inverse, correction

    @pl.jit.inline
    def tile_vjp(
        saved: pl.Tensor, function: pl.Tensor, dp: pl.Tensor, inverse: pl.Tensor, correction: pl.Tensor,
        row: pl.Scalar[pl.INDEX], col: pl.Scalar[pl.INDEX],
    ):
        token, source = row // copies, row % copies
        total = pl.full([1, column_tile], dtype=pl.FP32, value=0.)
        for feature in pl.range(channels):
            weight = pl.slice(function, [1, column_tile], [feature, source * hidden + col])
            total = pl.add(total, pl.mul(weight, pl.read(dp, [token, feature])))
        orig = pl.slice(saved, [1, column_tile], [row, col])
        projected_dx = pl.mul(total, pl.read(inverse, [token, 0]))
        return pl.sub(projected_dx, pl.mul(orig, pl.read(correction, [token, 0])))

    return project, tile_vjp
