"""PyPTO leaves for the fixed-route DSV4.1 MoE reverse pass."""


def make_dequantize_blocked_weight(input_width, output_width, *, inline=False):
    """Restore the BF16 weight used by a frozen W8A8 input VJP.

    Native DSV4.1 stores an INT8 ``[K, N]`` matrix as
    ``[N / 64, K, 64]``.  The frozen target weight needs no parameter
    gradient; online learning only needs ``dY @ W.T``.  This leaf restores
    ``W * channel_scale`` on device so the shared training GEMM can compute
    that input VJP.  It also makes the activation-quantization derivative
    explicit: forward values stay quantized, while the input VJP uses the
    dequantized frozen weight (the usual straight-through boundary).
    """
    if (
        type(input_width) is not int or type(output_width) is not int
        or input_width <= 0 or output_width <= 0
        or input_width % 256 or output_width % 64
    ):
        raise ValueError("blocked W8A8 weight requires K/256 and N/64 alignment")
    import pypto.language as pl

    blocks = output_width // 64
    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def dequantize(
        packed: pl.Tensor,
        channel_scale: pl.Tensor,
        logical: pl.Out[pl.Tensor],
    ):
        with pl.spmd(input_width * blocks, name_hint="orthrus_dsv41_w8a8_vjp_weight"):
            task = pl.tile.get_block_idx()
            row = task // blocks
            block = task % blocks
            # A2/A3 has no direct INT8->FP32 vector conversion.
            codes = pl.cast(
                pl.cast(
                    pl.slice(packed, [1, 64], [block * input_width + row, 0]),
                    pl.FP16,
                ),
                pl.FP32,
            )
            scales = pl.reshape(
                pl.slice(channel_scale, [64], [block * 64]), [1, 64],
            )
            logical[row:row + 1, block * 64:block * 64 + 64] = pl.cast(
                pl.mul(codes, scales), pl.BF16, mode="rint",
            )
        return logical

    @pl.jit.host
    def entry(
        packed: pl.Tensor,
        channel_scale: pl.Tensor,
        logical: pl.Out[pl.Tensor],
    ):
        dequantize(packed, channel_scale, logical, device=0)

    return dequantize if inline else entry


def make_moe_combine_vjp(rows=32, hidden=5120, topk=6, *, inline=False):
    """Return gradients for shared output, selected expert rows and weights."""
    if rows < 1 or hidden < 64 or hidden % 64 or topk != 6:
        raise ValueError("expected DSV4.1 rows, 64-aligned hidden and Top-K=6")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    tile = 64

    @kernel
    def moe_combine_vjp(
        output_gradient: pl.Tensor,
        expert_outputs: pl.Tensor,
        route_weights: pl.Tensor,
        shared_gradient: pl.Out[pl.Tensor],
        expert_gradient: pl.Out[pl.Tensor],
        weight_gradient: pl.Out[pl.Tensor],
    ):
        # One block owns a complete token row. Route gradients occupy six
        # values in a 16-FP32 row, avoiding concurrent whole-cache-line stores.
        with pl.spmd(rows, name_hint="orthrus_dsv41_moe_combine_backward"):
            token = pl.tile.get_block_idx()
            for slot in pl.range(topk):
                task = token * topk + slot
                total = pl.full([8, 8], dtype=pl.FP32, value=0.)
                weight = pl.read(route_weights, [token, slot])
                for part in pl.range(hidden // tile):
                    dy = pl.cast(
                        pl.slice(output_gradient, [1, tile], [token, part * tile]), pl.FP32,
                    )
                    expert = pl.cast(
                        pl.slice(expert_outputs, [1, tile], [task, part * tile]), pl.FP32,
                    )
                    if slot == 0:
                        shared_gradient[token:token + 1, part * tile:(part + 1) * tile] = pl.cast(
                            dy, pl.BF16, mode="rint",
                        )
                    expert_gradient[task:task + 1, part * tile:(part + 1) * tile] = pl.cast(
                        pl.mul(dy, weight), pl.BF16, mode="rint",
                    )
                    product = pl.reshape(pl.mul(dy, expert), [8, 8])
                    partial = pl.row_expand_mul(
                        pl.full([8, 8], dtype=pl.FP32, value=1.), pl.row_sum(product),
                    )
                    total = pl.add(
                        total,
                        pl.row_expand_mul(
                            pl.full([8, 8], dtype=pl.FP32, value=1.),
                            pl.row_sum(pl.transpose(partial, 0, 1)),
                        ),
                    )
                pl.write(weight_gradient, [token, slot], pl.read(total, [0, 0]))
        return shared_gradient, expert_gradient, weight_gradient

    @pl.jit.host
    def entry(
        output_gradient: pl.Tensor,
        expert_outputs: pl.Tensor,
        route_weights: pl.Tensor,
        shared_gradient: pl.Out[pl.Tensor],
        expert_gradient: pl.Out[pl.Tensor],
        weight_gradient: pl.Out[pl.Tensor],
    ):
        moe_combine_vjp(
            output_gradient, expert_outputs, route_weights,
            shared_gradient, expert_gradient, weight_gradient, device=0,
        )

    return moe_combine_vjp if inline else entry


def make_normalized_route_vjp(rows=32, topk=6, *, inline=False):
    """Differentiate normalized selected scores while keeping Top-K IDs fixed."""
    if rows < 1 or topk != 6:
        raise ValueError("expected DSV4.1 rows and Top-K=6")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit

    @kernel
    def normalized_route_vjp(
        selected_scores: pl.Tensor,
        weight_gradient: pl.Tensor,
        score_gradient: pl.Out[pl.Tensor],
        route_scale: pl.Scalar[pl.FP32],
    ):
        # ``score_gradient`` is [rows, 16], so each token owns one complete
        # 64-byte FP32 cache line even though only six values are meaningful.
        with pl.spmd(rows, name_hint="orthrus_dsv41_route_normalize_backward"):
            token = pl.tile.get_block_idx()
            total = pl.cast(0., pl.FP32)
            weighted = pl.cast(0., pl.FP32)
            for slot in pl.range(topk):
                score = pl.read(selected_scores, [token, slot])
                total = total + score
                weighted = weighted + score * pl.read(weight_gradient, [token, slot])
            for slot in pl.range(topk):
                result = route_scale / total * (
                    pl.read(weight_gradient, [token, slot]) - weighted / total
                )
                pl.write(score_gradient, [token, slot], result)
        return score_gradient

    @pl.jit.host
    def entry(
        selected_scores: pl.Tensor,
        weight_gradient: pl.Tensor,
        score_gradient: pl.Out[pl.Tensor],
        route_scale: pl.Scalar[pl.FP32],
    ):
        normalized_route_vjp(
            selected_scores, weight_gradient, score_gradient, route_scale, device=0,
        )

    return normalized_route_vjp if inline else entry


def make_clipped_swiglu_vjp(rows=32, intermediate=2304, limit=10.0, *, inline=False):
    """Differentiate the routed expert activation with fixed W8A8 values.

    ``gate_up`` is the FP32 value reconstructed from the native INT32 GEMM,
    token scale and channel scale.  The returned BF16 gradient feeds the
    dequantized frozen gate/up input VJP; the activation requantizer is treated
    as a straight-through boundary and no frozen parameter gradient is formed.
    """
    if (
        type(rows) is not int or type(intermediate) is not int
        or rows < 1 or intermediate < 64 or intermediate % 64
        or not isinstance(limit, (int, float)) or limit <= 0
    ):
        raise ValueError("aligned routed-expert rows and a positive clamp limit required")
    import pypto.language as pl

    kernel = pl.jit.inline if inline else pl.jit
    tile = 256 if intermediate % 256 == 0 else 128 if intermediate % 128 == 0 else 64

    @kernel
    def swiglu_vjp(
        gate_up: pl.Tensor,
        output_gradient: pl.Tensor,
        gate_up_gradient: pl.Out[pl.Tensor],
    ):
        with pl.spmd(rows * (intermediate // tile), name_hint="orthrus_dsv41_clipped_swiglu_backward"):
            task = pl.tile.get_block_idx()
            token = task // (intermediate // tile)
            column = task % (intermediate // tile) * tile
            gate_raw = pl.slice(gate_up, [1, tile], [token, column])
            up_raw = pl.slice(gate_up, [1, tile], [token, intermediate + column])
            dy = pl.cast(pl.slice(output_gradient, [1, tile], [token, column]), pl.FP32)
            gate = pl.minimum(gate_raw, limit)
            up = pl.maximum(pl.minimum(up_raw, limit), -limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            silu = pl.mul(gate, sigmoid)
            derivative = pl.mul(
                sigmoid,
                pl.add(pl.mul(gate, pl.add(pl.neg(sigmoid), 1.0)), 1.0),
            )
            dgate = pl.mul(pl.mul(dy, up), derivative)
            dup = pl.mul(dy, silu)
            # PyPTO cmp: 2 is < and 5 is >=.  The equality convention is
            # irrelevant for normal floating values but follows Torch clamp.
            gate_mask = pl.cmp(gate_raw, limit, cmp_type=5)
            gate_mask = pl.add(pl.neg(gate_mask), 1.0)
            up_low = pl.cmp(up_raw, -limit, cmp_type=5)
            up_high = pl.add(pl.neg(pl.cmp(up_raw, limit, cmp_type=4)), 1.0)
            dgate = pl.mul(dgate, gate_mask)
            dup = pl.mul(dup, pl.mul(up_low, up_high))
            gate_up_gradient[token:token + 1, column:column + tile] = pl.cast(
                dgate, pl.BF16, mode="rint",
            )
            gate_up_gradient[
                token:token + 1, intermediate + column:intermediate + column + tile
            ] = pl.cast(dup, pl.BF16, mode="rint")
        return gate_up_gradient

    @pl.jit.host
    def entry(
        gate_up: pl.Tensor,
        output_gradient: pl.Tensor,
        gate_up_gradient: pl.Out[pl.Tensor],
    ):
        swiglu_vjp(gate_up, output_gradient, gate_up_gradient, device=0)

    return swiglu_vjp if inline else entry
