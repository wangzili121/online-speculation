"""Differentiate GLM's routed W8A8 experts for a replicated Orthrus block."""

from .pypto_glm53_online_head import make_final_norm_backward
from .pypto_glm53_online_moe import (
    make_expert_backward, make_router_backward, make_router_replay, make_shared_expert_backward,
)
from .pypto_glm53_online_suffix import make_fp32_tp_sum


ROWS = 16
ROUTES = 8
ROUTE_STRIDE = 512


def make_routed_ep_backward(ranks, local_experts):
    """Replay the selected local experts and sum input and routing gradients across EP."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, FLASH

    if ranks < 1 or local_experts < 1 or ranks * local_experts > FLASH.n_routed_experts:
        raise ValueError("invalid GLM routed expert backward geometry")
    hidden = FLASH.moe_intermediate_size
    rows, routes = ROWS, ROUTES
    merged_width = D + ROUTE_STRIDE
    expert_pad = (local_experts + 7) // 8 * 8
    tile = 128
    expert_vjp = make_expert_backward(rows, D, hidden, FLASH.swiglu_limit, inline=True)
    ep_sum = make_fp32_tp_sum(ranks, rows, merged_width)

    @pl.jit(auto_scope=False)
    def local_vjp(
        tokens: pl.Tensor[[rows, D], pl.INT8],
        scale: pl.Tensor[[rows, 1], pl.FP32],
        upstream: pl.Tensor[[rows, D], pl.FP32],
        experts: pl.Tensor[[rows, routes], pl.INT32],
        weights: pl.Tensor[[rows, routes], pl.FP32],
        gate_up: pl.Tensor[[local_experts, 2 * hidden, D], pl.INT8],
        gate_up_scale: pl.Tensor[[local_experts, 2 * hidden], pl.FP32],
        down: pl.Tensor[[local_experts, D, hidden], pl.INT8],
        down_scale: pl.Tensor[[local_experts, D], pl.FP32],
        partial: pl.Out[pl.Tensor[[rows, merged_width], pl.FP32]],
        rank: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
    ):
        present = pl.create_tensor([1, expert_pad], dtype=pl.INT32)
        row_scale = pl.create_tensor([rows, routes], dtype=pl.FP32)
        first_expert = rank * local_experts
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_ep_selected_experts"):
            present[:, :] = pl.full([1, expert_pad], dtype=pl.INT32, value=0)
            for token in pl.range(active):
                for slot in pl.range(routes):
                    local_id = pl.read(experts, [token, slot]) - first_expert
                    if 0 <= local_id and local_id < local_experts:
                        pl.write(present, [0, local_id], pl.cast(1, pl.INT32))
        for scale_row in pl.spmd(rows, name_hint="glm_ep_pad_activation_scale"):
            expanded = pl.full([1, routes], dtype=pl.FP32, value=0.0)
            if scale_row < active:
                pl.write(expanded, [0, 0], pl.read(scale, [scale_row, 0]))
            row_scale[scale_row:scale_row + 1, :] = expanded
        for zero_task in pl.spmd(rows * (merged_width // tile), name_hint="glm_ep_zero_gradient"):
            zero_row = zero_task // (merged_width // tile)
            zero_col = zero_task % (merged_width // tile) * tile
            partial[zero_row:zero_row + 1, zero_col:zero_col + tile] = pl.full([1, tile], dtype=pl.FP32, value=0.0)

        for expert in pl.range(local_experts):
            if pl.read(present, [0, expert]) != 0:
                with pl.scope():
                    coefficients = pl.create_tensor([rows, routes], dtype=pl.FP32)
                    slots = pl.create_tensor([rows], dtype=pl.INT32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_ep_expert_rows"):
                        for select_row in pl.range(rows):
                            index = pl.cast(-1, pl.INT32)
                            coefficient = pl.cast(0.0, pl.FP32)
                            if select_row < active:
                                for route in pl.range(routes):
                                    if pl.read(experts, [select_row, route]) == first_expert + expert:
                                        index = pl.cast(route, pl.INT32)
                                        coefficient = pl.read(weights, [select_row, route])
                            packed = pl.full([1, routes], dtype=pl.FP32, value=0.0)
                            pl.write(packed, [0, 0], coefficient)
                            coefficients[select_row:select_row + 1, :] = packed
                            pl.write(slots, [select_row], index)

                    dx = pl.create_tensor([rows, D], dtype=pl.FP32)
                    dr = pl.create_tensor([rows, routes], dtype=pl.FP32)
                    x_part, route_part = expert_vjp(
                        upstream, tokens, row_scale, coefficients, gate_up[expert], gate_up_scale[expert],
                        down[expert], down_scale[expert], dx, dr,
                    )
                    for task in pl.spmd(rows * (D // tile), name_hint="glm_ep_accumulate_input"):
                        accum_row = task // (D // tile)
                        accum_col = task % (D // tile) * tile
                        if pl.read(slots, [accum_row]) >= 0:
                            previous = partial[accum_row:accum_row + 1, accum_col:accum_col + tile]
                            dx_tile = x_part[accum_row:accum_row + 1, accum_col:accum_col + tile]
                            partial[accum_row:accum_row + 1, accum_col:accum_col + tile] = pl.add(previous, dx_tile)
                    for scatter_row in pl.spmd(rows, name_hint="glm_ep_scatter_route"):
                        selected_slot = pl.read(slots, [scatter_row])
                        if selected_slot >= 0:
                            route_row = pl.create_tensor([1, 16], dtype=pl.FP32)
                            route_row[:, :] = partial[scatter_row:scatter_row + 1, D:D + 16]
                            pl.write(route_row, [0, selected_slot], pl.read(route_part, [scatter_row, 0]))
                            partial[scatter_row:scatter_row + 1, D:D + 16] = route_row
        return partial

    @pl.jit.host
    def entry(
        tokens: pl.Tensor[[ranks, rows, D], pl.INT8],
        scale: pl.Tensor[[ranks, rows, 1], pl.FP32],
        upstream: pl.Tensor[[ranks, rows, D], pl.FP32],
        experts: pl.Tensor[[ranks, rows, routes], pl.INT32],
        weights: pl.Tensor[[ranks, rows, routes], pl.FP32],
        gate_up: pl.Tensor[[ranks, local_experts, 2 * hidden, D], pl.INT8],
        gate_up_scale: pl.Tensor[[ranks, local_experts, 2 * hidden], pl.FP32],
        down: pl.Tensor[[ranks, local_experts, D, hidden], pl.INT8],
        down_scale: pl.Tensor[[ranks, local_experts, D], pl.FP32],
        partial: pl.InOut[pl.Tensor[[ranks, rows, merged_width], pl.FP32]],
        combined: pl.Out[pl.Tensor[[ranks, rows, merged_width], pl.FP32]],
        active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        exchange_buffer = pld.alloc_window_buffer([rows, merged_width], dtype=pl.FP32)
        arrived_buffer = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            exchange = pld.window(exchange_buffer, [rows, merged_width], dtype=pl.FP32)
            arrived = pld.window(arrived_buffer, [ranks, 1], dtype=pl.INT32)
            rank_local = partial[rank]
            rank_combined = combined[rank]
            local = local_vjp(tokens[rank], scale[rank], upstream[rank], experts[rank], weights[rank],
                              gate_up[rank], gate_up_scale[rank], down[rank], down_scale[rank],
                              rank_local, rank, active, device=rank)
            ep_sum(local, rank_combined, exchange, arrived, rank, epoch, device=rank)

    return entry


def make_frozen_moe_backward(ranks, local_experts):
    """Build the resident routed and shared/normalization stages for a draft block."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import D, FLASH

    rows, routes, tile = ROWS, ROUTES, 128
    hidden, n_experts = FLASH.moe_intermediate_size, FLASH.n_routed_experts
    shared_hidden = hidden // ranks
    merged_width = D + ROUTE_STRIDE
    router_fwd = make_router_replay(device_entry=True)
    router_bwd = make_router_backward(rows, D, n_experts, FLASH.routed_scaling_factor, device_entry=True)
    routed_stage = make_routed_ep_backward(ranks, local_experts)
    unpack_up, unpack_down, replay, down_vjp, swiglu_vjp, up_vjp = make_shared_expert_backward(
        rows, D, shared_hidden, FLASH.swiglu_limit, device_stages=True,
    )
    inline_unpack_up = pl.jit.inline(unpack_up._func)
    inline_unpack_down = pl.jit.inline(unpack_down._func)
    inline_replay = pl.jit.inline(replay._func)
    inline_down_vjp = pl.jit.inline(down_vjp._func)
    inline_swiglu_vjp = pl.jit.inline(swiglu_vjp._func)
    inline_up_vjp = pl.jit.inline(up_vjp._func)
    shared_sum = make_fp32_tp_sum(ranks, rows, D)
    norm_bwd = make_final_norm_backward(D, FLASH.rms_norm_eps, rows, streams=1, device_entry=True)

    @pl.jit
    def shared_backward(
        tokens: pl.Tensor[[rows, D], pl.INT8],
        scale: pl.Tensor[[rows, 1], pl.FP32],
        upstream: pl.Tensor[[rows, D], pl.FP32],
        gate_up: pl.Tensor[[2 * shared_hidden, D], pl.INT8],
        gate_up_scale: pl.Tensor[[2 * shared_hidden], pl.FP32],
        down: pl.Tensor[[D, shared_hidden], pl.INT8],
        down_scale: pl.Tensor[[D], pl.FP32],
        output: pl.Out[pl.Tensor[[rows, D], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        row_scale = pl.create_tensor([rows, routes], dtype=pl.FP32)
        masked = pl.create_tensor([rows, D], dtype=pl.FP32)
        gate_up_nd = pl.create_tensor([2 * shared_hidden, D], dtype=pl.INT8)
        down_nd = pl.create_tensor([D, shared_hidden], dtype=pl.INT8)
        tape = pl.create_tensor([rows, 2 * shared_hidden], dtype=pl.FP32)
        d_swiglu = pl.create_tensor([rows, shared_hidden], dtype=pl.FP32)
        d_gu = pl.create_tensor([rows, 2 * shared_hidden], dtype=pl.FP32)
        for scale_row in pl.spmd(rows, name_hint="glm_shared_pad_scale"):
            aligned = pl.full([1, routes], dtype=pl.FP32, value=0.0)
            if scale_row < active:
                pl.write(aligned, [0, 0], pl.read(scale, [scale_row, 0]))
            row_scale[scale_row:scale_row + 1, :] = aligned
        for mask_task in pl.spmd(rows * (D // tile), name_hint="glm_shared_mask_inactive"):
            mask_row = mask_task // (D // tile)
            mask_col = mask_task % (D // tile) * tile
            if mask_row < active:
                masked[mask_row:mask_row + 1, mask_col:mask_col + tile] = upstream[
                    mask_row:mask_row + 1, mask_col:mask_col + tile
                ]
            else:
                masked[mask_row:mask_row + 1, mask_col:mask_col + tile] = pl.full([1, tile], dtype=pl.FP32, value=0.0)
        w_gu = inline_unpack_up(gate_up, gate_up_nd)
        w_down = inline_unpack_down(down, down_nd)
        raw = inline_replay(tokens, row_scale, w_gu, gate_up_scale, tape)
        dh = inline_down_vjp(masked, w_down, down_scale, d_swiglu)
        draw = inline_swiglu_vjp(raw, dh, d_gu)
        dx = pl.create_tensor([rows, D], dtype=pl.FP32)
        gradient = inline_up_vjp(draw, w_gu, gate_up_scale, dx)
        for task in pl.spmd(rows * (D // tile), name_hint="glm_shared_store_gradient"):
            row = task // (D // tile)
            col = task % (D // tile) * tile
            output[row:row + 1, col:col + tile] = gradient[row:row + 1, col:col + tile]
        return output

    @pl.jit
    def extract_routing(combined: pl.Tensor[[rows, merged_width], pl.FP32],
                        output: pl.Out[pl.Tensor[[rows, routes], pl.FP32]]):
        for row in pl.spmd(rows, name_hint="glm_ep_route_gradients"):
            output[row:row + 1, :] = combined[row:row + 1, D:D + routes]
        return output

    @pl.jit
    def add_router(
        combined: pl.Tensor[[rows, merged_width], pl.FP32],
        shared: pl.Tensor[[rows, D], pl.FP32],
        gate: pl.InOut[pl.Tensor[[rows, D], pl.FP32]],
    ):
        for task in pl.spmd(rows * (D // tile), name_hint="glm_ep_add_router"):
            row = task // (D // tile)
            col = task % (D // tile) * tile
            routed = combined[row:row + 1, col:col + tile]
            experts = pl.add(routed, shared[row:row + 1, col:col + tile])
            gate[row:row + 1, col:col + tile] = pl.add(experts, gate[row:row + 1, col:col + tile])
        return gate

    @pl.jit.host
    def router_stage(
        activation: pl.Tensor[[ranks, rows, D], pl.BF16],
        norm_weight: pl.Tensor[[ranks, D], pl.BF16],
        router_weight: pl.Tensor[[ranks, n_experts, D], pl.BF16],
        router_correction: pl.Tensor[[ranks, n_experts], pl.FP32],
        selected: pl.Out[pl.Tensor[[ranks, rows, routes], pl.INT32]],
        coefficients: pl.Out[pl.Tensor[[ranks, rows, routes], pl.FP32]],
        sigmoid: pl.Out[pl.Tensor[[ranks, rows, routes], pl.FP32]],
        quantized: pl.Out[pl.Tensor[[ranks, rows, D], pl.INT8]],
        scale: pl.Out[pl.Tensor[[ranks, rows, 1], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(pld.world_size()):
            router_fwd(activation[rank], norm_weight[rank], router_weight[rank], router_correction[rank],
                       selected[rank], coefficients[rank], sigmoid[rank], quantized[rank], scale[rank],
                       active, device=rank)

    @pl.jit.host
    def finish_stage(
        activation: pl.Tensor[[ranks, rows, D], pl.BF16],
        upstream: pl.Tensor[[ranks, rows, D], pl.FP32],
        norm_weight: pl.Tensor[[ranks, D], pl.BF16],
        router_weight: pl.Tensor[[ranks, n_experts, D], pl.BF16],
        shared_gate_up: pl.Tensor[[ranks, 2 * shared_hidden, D], pl.INT8],
        shared_gate_up_scale: pl.Tensor[[ranks, 2 * shared_hidden], pl.FP32],
        shared_down: pl.Tensor[[ranks, D, shared_hidden], pl.INT8],
        shared_down_scale: pl.Tensor[[ranks, D], pl.FP32],
        selected: pl.Tensor[[ranks, rows, routes], pl.INT32],
        sigmoid: pl.Tensor[[ranks, rows, routes], pl.FP32],
        quantized: pl.Tensor[[ranks, rows, D], pl.INT8],
        scale: pl.Tensor[[ranks, rows, 1], pl.FP32],
        combined: pl.Tensor[[ranks, rows, merged_width], pl.FP32],
        shared_partial: pl.InOut[pl.Tensor[[ranks, rows, D], pl.FP32]],
        shared_combined: pl.InOut[pl.Tensor[[ranks, rows, D], pl.FP32]],
        route_gradient: pl.InOut[pl.Tensor[[ranks, rows, routes], pl.FP32]],
        norm_gradient: pl.InOut[pl.Tensor[[ranks, rows, D], pl.FP32]],
        output: pl.Out[pl.Tensor[[ranks, rows, D], pl.FP32]],
        active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        exchange_buffer = pld.alloc_window_buffer([rows, D], dtype=pl.FP32)
        arrived_buffer = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            exchange = pld.window(exchange_buffer, [rows, D], dtype=pl.FP32)
            arrived = pld.window(arrived_buffer, [ranks, 1], dtype=pl.INT32)
            local_shared = shared_partial[rank]
            total_shared = shared_combined[rank]
            local = shared_backward(quantized[rank], scale[rank], upstream[rank], shared_gate_up[rank],
                                    shared_gate_up_scale[rank], shared_down[rank], shared_down_scale[rank],
                                    local_shared, active, device=rank)
            total = shared_sum(local, total_shared, exchange, arrived, rank, epoch, device=rank)
            routes_grad = route_gradient[rank]
            d_route = extract_routing(combined[rank], routes_grad, device=rank)
            scores = sigmoid[rank]
            experts = selected[rank]
            d_norm = norm_gradient[rank]
            d_router = router_bwd(scores, experts, d_route, router_weight[rank], d_norm, active, device=rank)
            d_input = add_router(combined[rank], total, d_router, device=rank)
            norm_bwd(activation[rank], norm_weight[rank], d_input, output[rank], device=rank)

    return router_stage, routed_stage, finish_stage
