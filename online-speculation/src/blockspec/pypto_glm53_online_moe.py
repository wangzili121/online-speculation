"""Activation derivatives for the frozen GLM expert layers."""


def make_swiglu_backward(rows, width, limit):
    """Differentiate the clamped SwiGLU used in GLM shared and routed experts."""
    import pypto.language as pl

    tile = 64
    if rows < 1 or width < tile or width % tile or limit < 0:
        raise ValueError("invalid GLM SwiGLU backward geometry")
    parts = width // tile
    clamped = limit > 0

    @pl.jit
    def backward(
        gate: pl.Tensor[[rows, width], pl.BF16], up: pl.Tensor[[rows, width], pl.BF16],
        upstream: pl.Tensor[[rows, width], pl.FP32],
        d_gate: pl.Out[pl.Tensor[[rows, width], pl.FP32]], d_up: pl.Out[pl.Tensor[[rows, width], pl.FP32]],
    ):
        for task in pl.spmd(rows * parts, name_hint="orthrus_glm53_swiglu_backward"):
            row = task // parts
            column = task % parts * tile
            gate_raw = pl.cast(gate[row:row + 1, column:column + tile], pl.FP32)
            up_raw = pl.cast(up[row:row + 1, column:column + tile], pl.FP32)
            if clamped:
                g = pl.minimum(gate_raw, limit)
                u = pl.minimum(pl.maximum(up_raw, -limit), limit)
            else:
                g, u = gate_raw, up_raw
            # exp(-abs(g)) stays finite at both tails of SiLU.
            e = pl.exp(pl.neg(pl.abs(g)))
            nonnegative = pl.cast(pl.cast(pl.cmp(g, pl.abs(g), cmp_type=0), pl.INT32), pl.FP32)
            numerator = pl.add(e, pl.mul(nonnegative, pl.add(pl.neg(e), 1.0)))
            sigmoid = pl.div(numerator, pl.add(e, 1.0))
            d_silu = pl.mul(sigmoid, pl.add(pl.mul(g, pl.add(pl.neg(sigmoid), 1.0)), 1.0))
            dy = upstream[row:row + 1, column:column + tile]
            dg = pl.mul(pl.mul(dy, u), d_silu)
            du = pl.mul(pl.mul(dy, g), sigmoid)
            if clamped:
                gate_pass = pl.cast(pl.cast(pl.cmp(gate_raw, g, cmp_type=0), pl.INT32), pl.FP32)
                up_pass = pl.cast(pl.cast(pl.cmp(up_raw, u, cmp_type=0), pl.INT32), pl.FP32)
                dg = pl.mul(dg, gate_pass)
                du = pl.mul(du, up_pass)
            d_gate[row:row + 1, column:column + tile] = dg
            d_up[row:row + 1, column:column + tile] = du
        return d_gate, d_up

    @pl.jit.host
    def entry(gate: pl.Tensor, up: pl.Tensor, upstream: pl.Tensor, d_gate: pl.Out[pl.Tensor], d_up: pl.Out[pl.Tensor]):
        backward(gate, up, upstream, d_gate, d_up, device=0)

    return entry


def make_frozen_int8_linear_backward(rows, in_features, out_features, *, round_scale=False, device_entry=False):
    """Compute dX with frozen row-scaled weights and an activation-quantization STE."""
    import pypto.language as pl

    columns, reduction = 64, 128
    if rows < 16 or rows % 16 or in_features < columns or in_features % columns:
        raise ValueError("invalid GLM expert input gradient geometry")
    if out_features < reduction or out_features % reduction:
        raise ValueError("expert reduction width must be 128-aligned")

    @pl.jit
    def linear_input_backward(
        upstream: pl.Tensor[[rows, out_features], pl.FP32],
        weight: pl.Tensor[[out_features, in_features], pl.INT8], scale: pl.Tensor[[out_features], pl.FP32],
        gradient: pl.Out[pl.Tensor[[rows, in_features], pl.FP32]],
    ):
        for piece in pl.spmd(in_features // columns, name_hint="orthrus_glm53_frozen_linear_backward"):
            col = piece * columns
            accum = pl.create_tensor([rows, columns], dtype=pl.FP32)
            for start in pl.pipeline(0, out_features, reduction, stage=2):
                dy = pl.slice(upstream, [rows, reduction], [0, start])
                channel_scale = pl.reshape(scale[start:start + reduction], [1, reduction])
                if round_scale:
                    # The shared and routed expert kernels round these row scales before dequantization.
                    channel_scale = pl.cast(pl.cast(channel_scale, pl.BF16, mode="rint"), pl.FP32)
                scaled = pl.col_expand_mul(dy, channel_scale)
                high = pl.cast(scaled, pl.BF16, mode="rint")
                low = pl.cast(pl.sub(scaled, pl.cast(high, pl.FP32)), pl.BF16, mode="rint")
                packed = pl.slice(weight, [reduction, columns], [start, col])
                unpacked = pl.cast(pl.cast(pl.cast(packed, pl.FP16), pl.FP32), pl.BF16, mode="rint")
                accum = pl.matmul_acc(accum, high, unpacked, init_cond=start == 0)
                accum = pl.matmul_acc(accum, low, unpacked)
            gradient[:, col:col + columns] = accum
        return gradient

    @pl.jit.host
    def entry(upstream: pl.Tensor, weight: pl.Tensor, scale: pl.Tensor, gradient: pl.Out[pl.Tensor]):
        linear_input_backward(upstream, weight, scale, gradient, device=0)

    return linear_input_backward if device_entry else entry


def _make_expert_projection(rows, in_features, out_features, *, round_output=False):
    import pypto.language as pl

    tile, reduction = 128, min(512, in_features)

    @pl.jit.inline
    def int8_project(
        x: pl.Tensor[[rows, in_features], pl.INT8], token_scale: pl.Tensor[[rows, 8], pl.FP32],
        weight: pl.Tensor[[out_features, in_features], pl.INT8], scale: pl.Tensor[[out_features], pl.FP32],
        output: pl.Tensor[[rows, out_features], pl.FP32],
    ):
        accumulators = pl.create_tensor([rows, out_features], dtype=pl.INT32)
        for stripe in pl.spmd(out_features // tile, name_hint="glm_expert_replay_matmul"):
            col = stripe * tile
            acc = pl.create_tensor([rows, tile], dtype=pl.INT32)
            for inner in pl.pipeline(0, in_features, reduction, stage=2):
                xs = pl.slice(x, [rows, reduction], [0, inner])
                ws = pl.slice(weight, [tile, reduction], [col, inner])
                acc = pl.matmul_acc(acc, xs, ws, b_trans=True, init_cond=inner == 0)
            accumulators[:, col:col + tile] = acc
        for task in pl.spmd(rows * (out_features // tile), name_hint="glm_expert_replay_dequant"):
            row = task // (out_features // tile)
            col = task % (out_features // tile) * tile
            y = pl.cast(pl.slice(accumulators, [1, tile], [row, col]), pl.FP32)
            channel = pl.reshape(scale[col:col + tile], [1, tile])
            channel = pl.cast(pl.cast(channel, pl.BF16, mode="rint"), pl.FP32)
            dequantized = pl.mul(pl.mul(y, pl.read(token_scale, [row, 0])), channel)
            if round_output:
                dequantized = pl.cast(pl.cast(dequantized, pl.BF16, mode="rint"), pl.FP32)
            output[row:row + 1, col:col + tile] = dequantized
        return output

    return int8_project


def _make_expert_replay(rows, width, hidden, limit):
    import pypto.language as pl
    from models.glm5_3_flash.elementwise import accurate_divide, accurate_sigmoid
    from models.glm5_3_flash.quantization import INT8_SCALE_MAX

    tile, reduction = 128, 512
    if rows != 16 or width % reduction or hidden % reduction:
        raise ValueError("GLM routed replay requires 16 padded rows and 512-aligned features")

    project_gate_up = _make_expert_projection(rows, width, 2 * hidden)
    project_down = _make_expert_projection(rows, hidden, width)

    @pl.jit
    def expert_replay(
        x: pl.Tensor[[rows, width], pl.INT8], x_scale: pl.Tensor[[rows, 8], pl.FP32],
        w_gate_up: pl.Tensor[[2 * hidden, width], pl.INT8], s_gate_up: pl.Tensor[[2 * hidden], pl.FP32],
        w_down: pl.Tensor[[width, hidden], pl.INT8], s_down: pl.Tensor[[width], pl.FP32],
        raw: pl.Out[pl.Tensor[[rows, 2 * hidden], pl.FP32]], output: pl.Out[pl.Tensor[[rows, width], pl.BF16]],
    ):
        activation = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        quantized = pl.create_tensor([rows, hidden], dtype=pl.INT8)
        activation_scale = pl.create_tensor([rows, 8], dtype=pl.FP32)
        y = pl.create_tensor([rows, width], dtype=pl.FP32)
        raw = project_gate_up(x, x_scale, w_gate_up, s_gate_up, raw)
        for part in pl.spmd(hidden // tile, name_hint="glm_expert_replay_swiglu"):
            col = part * tile
            gate = pl.create_tensor([rows, tile], dtype=pl.FP32)
            gate[:, :] = pl.minimum(raw[:, col:col + tile], limit)
            up_col = col + hidden
            up = pl.minimum(pl.maximum(raw[:, up_col:up_col + tile], -limit), limit)
            sigmoid = accurate_sigmoid(gate, rows, tile)
            activation[:, col:col + tile] = pl.mul(pl.mul(gate, sigmoid), up)
        for row in pl.spmd(rows, name_hint="glm_expert_replay_quant"):
            peak = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, hidden, tile):
                peak = pl.maximum(peak, pl.abs(activation[row:row + 1, col:col + tile]))
            expanded = pl.col_expand(pl.full([8, tile], dtype=pl.FP32, value=0.0), peak)
            amax = pl.read(pl.row_max(expanded), [0, 0])
            inverse = pl.cast(0.0, pl.FP32)
            if amax > 0.0:
                one = pl.create_tensor([1, 8], dtype=pl.FP32)
                denominator = pl.create_tensor([1, 8], dtype=pl.FP32)
                one[:, :] = pl.full([1, 8], dtype=pl.FP32, value=1.0)
                denominator[:, :] = pl.add(pl.full([1, 8], dtype=pl.FP32, value=0.0), amax)
                inverse = pl.read(accurate_divide(one, denominator), [0, 0]) * INT8_SCALE_MAX
            scales = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            pl.write(scales, [0, 0], amax * (1.0 / INT8_SCALE_MAX))
            activation_scale[row:row + 1, :] = scales
            for col in pl.range(0, hidden, tile):
                h = pl.mul(activation[row:row + 1, col:col + tile], inverse)
                i = pl.cast(h, pl.INT32, mode="rint")
                quantized[row:row + 1, col:col + tile] = pl.cast(pl.cast(i, pl.FP16, mode="round"), pl.INT8, mode="trunc")
        y = project_down(quantized, activation_scale, w_down, s_down, y)
        for row in pl.spmd(rows, name_hint="glm_expert_replay_round"):
            for col in pl.range(0, width, tile):
                output[row:row + 1, col:col + tile] = pl.cast(y[row:row + 1, col:col + tile], pl.BF16, mode="rint")
        return raw, output

    return expert_replay


def _make_expert_swiglu_backward(rows, hidden, limit):
    import pypto.language as pl

    tile = 128

    @pl.jit
    def expert_swiglu_backward(
        raw: pl.Tensor[[rows, 2 * hidden], pl.FP32], upstream: pl.Tensor[[rows, hidden], pl.FP32],
        output: pl.Out[pl.Tensor[[rows, 2 * hidden], pl.FP32]],
    ):
        for task in pl.spmd(rows * (hidden // tile), name_hint="glm_expert_swiglu_vjp"):
            row = task // (hidden // tile)
            col = task % (hidden // tile) * tile
            gate_raw = pl.slice(raw, [1, tile], [row, col])
            up_raw = pl.slice(raw, [1, tile], [row, hidden + col])
            g = pl.minimum(gate_raw, limit)
            u = pl.minimum(pl.maximum(up_raw, -limit), limit)
            e = pl.exp(pl.neg(pl.abs(g)))
            one_minus_e = pl.add(pl.neg(e), 1.0)
            nonnegative = pl.cast(pl.cast(pl.cmp(g, pl.abs(g), cmp_type=0), pl.INT32), pl.FP32)
            sigmoid = pl.div(pl.add(e, pl.mul(nonnegative, one_minus_e)), pl.add(e, 1.0))
            slope = pl.mul(sigmoid, pl.add(pl.mul(g, pl.add(pl.neg(sigmoid), 1.0)), 1.0))
            dy = pl.slice(upstream, [1, tile], [row, col])
            gate_pass = pl.cast(pl.cast(pl.cmp(gate_raw, g, cmp_type=0), pl.INT32), pl.FP32)
            up_pass = pl.cast(pl.cast(pl.cmp(up_raw, u, cmp_type=0), pl.INT32), pl.FP32)
            output[row:row + 1, col:col + tile] = pl.mul(pl.mul(pl.mul(dy, u), slope), gate_pass)
            up_col = col + hidden
            output[row:row + 1, up_col:up_col + tile] = pl.mul(pl.mul(pl.mul(dy, g), sigmoid), up_pass)
        return output

    return expert_swiglu_backward


def make_expert_backward(rows, width, hidden, limit):
    """Replay and differentiate a frozen routed W8A8 expert."""
    import pypto.language as pl

    if limit <= 0:
        raise ValueError("the GLM expert backward expects the checkpoint's SwiGLU limit")
    down = make_frozen_int8_linear_backward(rows, hidden, width, round_scale=True, device_entry=True)
    gate_up = make_frozen_int8_linear_backward(rows, width, 2 * hidden, round_scale=True, device_entry=True)
    replay = _make_expert_replay(rows, width, hidden, limit)
    tile = 128

    @pl.jit
    def apply_route(
        upstream: pl.Tensor[[rows, width], pl.FP32], unweighted: pl.Tensor[[rows, width], pl.BF16],
        route: pl.Tensor[[rows, 8], pl.FP32],
        weighted: pl.Out[pl.Tensor[[rows, width], pl.FP32]], route_grad: pl.Out[pl.Tensor[[rows, 8], pl.FP32]],
    ):
        for row in pl.spmd(rows, name_hint="glm_expert_route_vjp"):
            coefficient = pl.read(route, [row, 0])
            products = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for col in pl.range(0, width, tile):
                g = pl.slice(upstream, [1, tile], [row, col])
                y = pl.cast(pl.slice(unweighted, [1, tile], [row, col]), pl.FP32)
                weighted[row:row + 1, col:col + tile] = pl.mul(g, coefficient)
                products = pl.add(products, pl.mul(g, y))
            zero = pl.full([8, tile], dtype=pl.FP32, value=0.0)
            dot = pl.row_sum(pl.col_expand(zero, products))
            result = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            pl.write(result, [0, 0], pl.read(dot, [0, 0]))
            route_grad[row:row + 1, :] = result
        return weighted, route_grad

    expert_swiglu_backward = _make_expert_swiglu_backward(rows, hidden, limit)

    @pl.jit.host
    def entry(
        upstream: pl.Tensor, x: pl.Tensor, x_scale: pl.Tensor, route: pl.Tensor,
        gate_up_weight: pl.Tensor, gate_up_scale: pl.Tensor, down_weight: pl.Tensor, down_scale: pl.Tensor,
        grad_input: pl.Out[pl.Tensor], grad_route: pl.Out[pl.Tensor],
    ):
        weighted = pl.create_tensor([rows, width], dtype=pl.FP32)
        activation_grad = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        preactivation_grad = pl.create_tensor([rows, 2 * hidden], dtype=pl.FP32)
        gate_up_tape = pl.create_tensor([rows, 2 * hidden], dtype=pl.FP32)
        unweighted = pl.create_tensor([rows, width], dtype=pl.BF16)
        raw, unweighted_out = replay(x, x_scale, gate_up_weight, gate_up_scale, down_weight, down_scale,
                                     gate_up_tape, unweighted, device=0)
        dy, _route = apply_route(upstream, unweighted_out, route, weighted, grad_route, device=0)
        d_activation = down(dy, down_weight, down_scale, activation_grad, device=0)
        d_raw = expert_swiglu_backward(raw, d_activation, preactivation_grad, device=0)
        gate_up(d_raw, gate_up_weight, gate_up_scale, grad_input, device=0)

    return entry


def _make_expert_nz_unpack(out_features, in_features):
    import pypto.language as pl

    rows, cols, workers = 16, 32, 48
    stripes = in_features // cols
    blocks = out_features // rows * stripes

    @pl.jit
    def unpack(
        packed: pl.Tensor[[out_features, in_features], pl.INT8],
        logical: pl.Out[pl.Tensor[[out_features, in_features], pl.INT8]],
    ):
        raw = pl.reshape(packed, [stripes, out_features * cols])
        for worker in pl.spmd(workers, name_hint="glm_expert_nz_to_nd"):
            for block in pl.range(worker, blocks, workers):
                row = block // stripes * rows
                stripe = block % stripes
                column = stripe * cols
                tile = pl.slice(raw, [1, rows * cols], [stripe, row * cols])
                logical[row:row + rows, column:column + cols] = pl.reshape(tile, [rows, cols])
        return logical

    return unpack


def make_shared_expert_backward(rows, width, hidden, limit):
    """Backpropagate through one TP shard of the resident NZ shared expert."""
    import pypto.language as pl

    if rows != 16 or width % 512 or hidden % 128 or hidden <= 0 or limit <= 0:
        raise ValueError("invalid GLM shared expert geometry")

    unpack_gate_up = _make_expert_nz_unpack(2 * hidden, width)
    unpack_down = _make_expert_nz_unpack(width, hidden)
    gate_up = make_frozen_int8_linear_backward(rows, width, 2 * hidden, round_scale=True, device_entry=True)
    down = make_frozen_int8_linear_backward(rows, hidden, width, round_scale=True, device_entry=True)
    swiglu = _make_expert_swiglu_backward(rows, hidden, limit)
    project = _make_expert_projection(rows, width, 2 * hidden, round_output=True)

    @pl.jit
    def replay(x: pl.Tensor, x_scale: pl.Tensor, weights: pl.Tensor, weight_scale: pl.Tensor,
               raw: pl.Out[pl.Tensor[[rows, 2 * hidden], pl.FP32]]):
        raw = project(x, x_scale, weights, weight_scale, raw)
        return raw

    @pl.jit.host
    def entry(
        upstream: pl.Tensor, x: pl.Tensor, x_scale: pl.Tensor,
        gate_up_nz: pl.Tensor, gate_up_scale: pl.Tensor, down_nz: pl.Tensor, down_scale: pl.Tensor,
        grad_input: pl.Out[pl.Tensor],
    ):
        gate_up_nd = pl.create_tensor([2 * hidden, width], dtype=pl.INT8)
        down_nd = pl.create_tensor([width, hidden], dtype=pl.INT8)
        gate_up_tape = pl.create_tensor([rows, 2 * hidden], dtype=pl.FP32)
        activation_grad = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        preactivation_grad = pl.create_tensor([rows, 2 * hidden], dtype=pl.FP32)
        w_gate_up = unpack_gate_up(gate_up_nz, gate_up_nd, device=0)
        w_down = unpack_down(down_nz, down_nd, device=0)
        raw = replay(x, x_scale, w_gate_up, gate_up_scale, gate_up_tape, device=0)
        d_activation = down(upstream, w_down, down_scale, activation_grad, device=0)
        d_raw = swiglu(raw, d_activation, preactivation_grad, device=0)
        gate_up(d_raw, w_gate_up, gate_up_scale, grad_input, device=0)

    return entry


def make_router_replay():
    """Recover the actual expert routes and the unnormalized selected sigmoid scores."""
    import pypto.language as pl
    from models.glm5_3_flash.config import D, N_EXPERTS, TOPK
    from models.glm5_3_flash.elementwise import accurate_sigmoid
    from models.glm5_3_flash.gate import GATE_N_TILE, moe_gate
    from models.glm5_3_flash.linear import bf16_linear_row
    from models.glm5_3_flash.rmsnorm import rmsnorm

    rows = 16

    @pl.jit
    def replay(
        x: pl.Tensor[[rows, D], pl.BF16], norm_weight: pl.Tensor[[D], pl.BF16],
        weight: pl.Tensor[[N_EXPERTS, D], pl.BF16], correction: pl.Tensor[[N_EXPERTS], pl.FP32],
        selected: pl.Out[pl.Tensor[[rows, TOPK], pl.INT32]],
        coefficients: pl.Out[pl.Tensor[[rows, TOPK], pl.FP32]],
        sigmoid: pl.Out[pl.Tensor[[rows, TOPK], pl.FP32]],
        quantized: pl.Out[pl.Tensor[[rows, D], pl.INT8]],
        scale: pl.Out[pl.Tensor[[rows, 1], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
        moe_gate(x, norm_weight, weight, correction, coefficients, selected, quantized, scale, active)
        rmsnorm(x, norm_weight, normalized)
        # One task owns an entire output row. The selected weights are
        # differentiated with the exact IDs produced by the deployed router.
        for row in pl.spmd(rows, name_hint="glm_moe_selected_sigmoid"):
            output = pl.full([1, TOPK], dtype=pl.FP32, value=0.0)
            if row < active:
                for route in pl.range(TOPK):
                    expert = pl.cast(pl.read(selected, [row, route]), pl.INDEX)
                    start = expert // GATE_N_TILE * GATE_N_TILE
                    logits = pl.create_tensor([1, GATE_N_TILE], dtype=pl.FP32)
                    logits[:, :] = bf16_linear_row(normalized, weight, row, start, GATE_N_TILE, D, 128, 2)
                    p = accurate_sigmoid(logits, 1, GATE_N_TILE)
                    pl.write(output, [0, route], pl.read(p, [0, expert % GATE_N_TILE]))
            sigmoid[row:row + 1, :] = output
        return selected, coefficients, sigmoid, quantized, scale

    @pl.jit.host
    def entry(x: pl.Tensor, norm_weight: pl.Tensor, weight: pl.Tensor, correction: pl.Tensor,
              selected: pl.Out[pl.Tensor], coefficients: pl.Out[pl.Tensor], sigmoid: pl.Out[pl.Tensor],
              quantized: pl.Out[pl.Tensor], scale: pl.Out[pl.Tensor], active: pl.Scalar[pl.INT32]):
        replay(x, norm_weight, weight, correction, selected, coefficients, sigmoid, quantized, scale, active, device=0)

    return entry


def make_router_backward(rows, width, experts, route_scale):
    """Differentiate the selected sigmoid weights with the forward Top-K IDs fixed."""
    import pypto.language as pl

    routes, tile = 8, 128
    if rows < 1 or width < tile or width % tile or experts < routes or route_scale <= 0:
        raise ValueError("invalid GLM router gradient geometry")
    pieces = width // tile

    @pl.jit
    def router_backward(
        sigmoid: pl.Tensor[[rows, routes], pl.FP32],
        indices: pl.Tensor[[rows, routes], pl.INT32],
        route_gradient: pl.Tensor[[rows, routes], pl.FP32],
        gate_weight: pl.Tensor[[experts, width], pl.BF16],
        grad_input: pl.Out[pl.Tensor[[rows, width], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        logits_grad = pl.create_tensor([rows, routes], dtype=pl.FP32)
        for row in pl.spmd(rows, name_hint="glm_router_weights_vjp"):
            if row < active:
                score = sigmoid[row:row + 1, :]
                upstream = route_gradient[row:row + 1, :]
                zeros = pl.full([8, routes], dtype=pl.FP32, value=0.0)
                normalizer = pl.read(pl.row_sum(pl.col_expand(zeros, score)), [0, 0]) + 1e-20
                coupling = pl.read(pl.row_sum(pl.col_expand(zeros, pl.mul(upstream, score))), [0, 0]) / normalizer
                d_score = pl.mul(pl.sub(upstream, coupling), route_scale / normalizer)
                logit_gradient = pl.mul(d_score, pl.mul(score, pl.add(pl.neg(score), 1.0)))
                logits_grad[row:row + 1, :] = logit_gradient
            else:
                logits_grad[row:row + 1, :] = pl.full([1, routes], dtype=pl.FP32, value=0.0)
        for task in pl.spmd(rows * pieces, name_hint="glm_router_input_vjp"):
            row = task // pieces
            col = task % pieces * tile
            dx = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            if row < active:
                for route in pl.unroll(routes):
                    expert = pl.cast(pl.read(indices, [row, route]), pl.INDEX)
                    coefficient = pl.read(logits_grad, [row, route])
                    weight = pl.cast(pl.slice(gate_weight, [1, tile], [expert, col]), pl.FP32)
                    dx = pl.add(dx, pl.mul(weight, coefficient))
            grad_input[row:row + 1, col:col + tile] = dx
        return grad_input

    @pl.jit.host
    def entry(sigmoid: pl.Tensor, indices: pl.Tensor, route_gradient: pl.Tensor, gate_weight: pl.Tensor,
              grad_input: pl.Out[pl.Tensor], active: pl.Scalar[pl.INT32]):
        router_backward(sigmoid, indices, route_gradient, gate_weight, grad_input, active, device=0)

    return entry
