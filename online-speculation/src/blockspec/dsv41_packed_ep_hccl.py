"""Pack 1122 W8A8 expert dispatch buffers for a host-issued EP AllToAll."""


def make_ep_forward_packets(tokens, hidden, world, local_experts, topk):
    """Build rank-local pack and restore stages around the two EP AllToAll calls."""
    import pypto.language as pl

    source_routes = tokens * topk
    packets = world * source_routes
    expert_rows = packets + local_experts * 16
    count_width = (local_experts + 15) // 16 * 16
    aux_width = 8

    @pl.jit
    def pack(
        x: pl.Tensor[[tokens, hidden], pl.INT8],
        scale: pl.Tensor[[1, tokens], pl.FP32],
        ids: pl.Tensor[[tokens, topk], pl.INT32],
        weight: pl.Tensor[[tokens, topk], pl.FP32],
        counts: pl.Out[pl.Tensor[[world, count_width], pl.INT32]],
        payload: pl.Out[pl.Tensor[[packets, hidden], pl.INT8]],
        auxiliary: pl.Out[pl.Tensor[[packets, aux_width], pl.FP32]],
        live: pl.Scalar[pl.INT32],
    ):
        # Each destination owns a separate bucket, even for a rank with zero active rows.
        routes = pl.create_tensor([world, source_routes], dtype=pl.INT32)
        with pl.spmd(world, name_hint="hccl_ep_count_routes") as sorted_routes:
            peer = pl.tile.get_block_idx()
            lengths = pl.array.create(local_experts, pl.INT32)
            cursor = pl.array.create(local_experts, pl.INT32)
            for expert in pl.range(local_experts):
                lengths[expert] = 0
            route_tile = pl.tile.full([1, source_routes], dtype=pl.INT32, value=-1)
            for row in pl.range(live):
                for choice in pl.range(topk):
                    expert_id = pl.read(ids, [row, choice])
                    if expert_id // local_experts == peer:
                        local_id = expert_id % local_experts
                        lengths[local_id] = lengths[local_id] + 1

            count_row = pl.tile.full([1, count_width], dtype=pl.INT32, value=0)
            offset = pl.const(0, pl.INT32)
            for expert in pl.range(local_experts):
                cursor[expert] = offset
                offset = offset + lengths[expert]
                pl.tile.write(count_row, [0, expert], lengths[expert])
            pl.store(count_row, [peer, 0], counts)

            for row in pl.range(live):
                for choice in pl.range(topk):
                    expert_id = pl.read(ids, [row, choice])
                    if expert_id // local_experts == peer:
                        local_id = expert_id % local_experts
                        flat_route = pl.cast(row * topk + choice, pl.INT32)
                        pl.tile.write(route_tile, [0, cursor[local_id]], flat_route)
                        cursor[local_id] = cursor[local_id] + 1
            pl.store(route_tile, [peer, 0], routes)

        with pl.spmd(packets, name_hint="hccl_ep_pack_routes", deps=[sorted_routes]):
            packet = pl.tile.get_block_idx()
            peer = packet // source_routes
            bucket_row = packet % source_routes
            route = pl.cast(pl.read(routes, [peer, bucket_row]), pl.INDEX)
            row_aux = pl.tile.full([1, aux_width], dtype=pl.FP32, value=0.)
            if route >= 0:
                token = route // topk
                choice = route % topk
                values = pl.tile.load(x, [token, 0], [1, hidden])
                pl.store(values, [packet, 0], payload)
                pl.tile.write(row_aux, [0, 0], pl.read(weight, [token, choice]))
                pl.tile.write(row_aux, [0, 1], pl.read(scale, [0, token]))
                pl.tile.write(row_aux, [0, 2], pl.cast(pl.cast(route, pl.INT32), pl.FP32))
            else:
                zeros_fp16 = pl.tile.full([1, hidden], dtype=pl.FP16, value=0.)
                zero = pl.tile.cast(zeros_fp16, target_type=pl.INT8, mode="rint")
                pl.store(zero, [packet, 0], payload)
            pl.store(row_aux, [packet, 0], auxiliary)

    @pl.jit
    def unpack(
        counts: pl.Tensor[[world, count_width], pl.INT32],
        payload: pl.Tensor[[packets, hidden], pl.INT8],
        auxiliary: pl.Tensor[[packets, aux_width], pl.FP32],
        expert_counts: pl.Out[pl.Tensor[[local_experts, 2], pl.INT32]],
        expert_x: pl.Out[pl.Tensor[[expert_rows, hidden], pl.INT8]],
        expert_scale: pl.Out[pl.Tensor[[1, expert_rows], pl.FP32]],
        expert_weight: pl.Out[pl.Tensor[[1, expert_rows], pl.FP32]],
        expert_route: pl.Out[pl.Tensor[[1, expert_rows], pl.INT32]],
    ):
        # Match native gather order within each expert so quantized GEMM sees the same rows.
        packet_for_row = pl.create_tensor([expert_rows], dtype=pl.INT32)
        total = pl.create_tensor([1], dtype=pl.INT32)
        with pl.spmd(1, name_hint="hccl_ep_expert_offsets") as mapped:
            control_row = pl.tile.get_block_idx()
            cursor = pl.array.create(world, pl.INT32)
            for source in pl.range(world):
                cursor[source] = 0
            start = pl.const(0, pl.INT32)
            for expert in pl.range(local_experts):
                length = pl.const(0, pl.INT32)
                for source in pl.range(world):
                    count = pl.read(counts, [source, expert])
                    packet = source * source_routes + cursor[source]
                    for offset in pl.range(count):
                        pl.write(packet_for_row, [start + length + offset], pl.cast(packet + offset, pl.INT32))
                    cursor[source] = cursor[source] + count
                    length = length + count
                pl.write(expert_counts, [expert, 0], length)
                pl.write(expert_counts, [expert, 1], start)
                padded = pl.cast((length + 15) // 16 * 16, pl.INT32)
                for offset in pl.range(length, padded):
                    pl.write(packet_for_row, [start + offset], pl.cast(-1, pl.INT32))
                start = start + padded
            pl.write(total, [control_row], start)

        end = pl.cast(pl.read(total, [0]), pl.INDEX)
        with pl.spmd(pl.max(1, end // 16), name_hint="hccl_ep_compact_experts", deps=[mapped]):
            begin = pl.tile.get_block_idx() * 16
            if begin < end:
                scales = pl.tile.full([1, 16], dtype=pl.FP32, value=0.)
                weights = pl.tile.full([1, 16], dtype=pl.FP32, value=0.)
                return_addresses = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
                for lane in pl.range(16):
                    row = begin + lane
                    packet = pl.cast(pl.read(packet_for_row, [row]), pl.INDEX)
                    if packet >= 0:
                        values = pl.tile.load(payload, [packet, 0], [1, hidden])
                        pl.store(values, [row, 0], expert_x)
                        pl.tile.write(weights, [0, lane], pl.read(auxiliary, [packet, 0]))
                        pl.tile.write(scales, [0, lane], pl.read(auxiliary, [packet, 1]))
                        source = packet // source_routes
                        route = pl.cast(pl.read(auxiliary, [packet, 2]), pl.INT32)
                        address = pl.cast(source * source_routes, pl.INT32) + route
                        pl.tile.write(return_addresses, [0, lane], address)
                    else:
                        zeros_fp16 = pl.tile.full([1, hidden], dtype=pl.FP16, value=0.)
                        zero = pl.tile.cast(zeros_fp16, target_type=pl.INT8, mode="rint")
                        pl.store(zero, [row, 0], expert_x)
                pl.store(scales, [0, begin], expert_scale)
                pl.store(weights, [0, begin], expert_weight)
                pl.store(return_addresses, [0, begin], expert_route)

    @pl.jit
    def pack_return(
        expert_output: pl.Tensor[[expert_rows, hidden], pl.BF16],
        expert_route: pl.Tensor[[1, expert_rows], pl.INT32],
        expert_counts: pl.Tensor[[local_experts, 2], pl.INT32],
        payload: pl.Out[pl.Tensor[[packets, hidden], pl.BF16]],
    ):
        with pl.spmd(packets, name_hint="hccl_ep_clear_return") as cleared:
            packet = pl.tile.get_block_idx()
            zero = pl.tile.full([1, hidden], dtype=pl.BF16, value=0.)
            pl.store(zero, [packet, 0], payload)
        with pl.spmd(local_experts, name_hint="hccl_ep_pack_return", deps=[cleared]):
            expert = pl.tile.get_block_idx()
            start = pl.read(expert_counts, [expert, 1])
            length = pl.read(expert_counts, [expert, 0])
            for offset in pl.range(length):
                row = start + offset
                return_slot = pl.read(expert_route, [0, row])
                values = pl.tile.load(expert_output, [row, 0], [1, hidden])
                pl.store(values, [return_slot, 0], payload)

    @pl.jit
    def combine(
        payload: pl.Tensor[[packets, hidden], pl.BF16],
        ids: pl.Tensor[[tokens, topk], pl.INT32],
        shared: pl.Tensor[[tokens, hidden], pl.BF16],
        output: pl.Out[pl.Tensor[[tokens, hidden], pl.BF16]],
        live: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(tokens, name_hint="hccl_ep_combine"):
            if token < live:
                initial = pl.tile.load(shared, [token, 0], [1, hidden])
                acc = pl.tile.cast(initial, target_type=pl.FP32)
                for choice in pl.range(topk):
                    owner = pl.read(ids, [token, choice]) // local_experts
                    packet = owner * source_routes + token * topk + choice
                    routed = pl.tile.load(payload, [packet, 0], [1, hidden])
                    contribution = pl.tile.cast(routed, target_type=pl.FP32)
                    acc = pl.tile.add(acc, contribution)
                result = pl.tile.cast(acc, target_type=pl.BF16, mode="rint")
                pl.store(result, [token, 0], output)

    return pack, unpack, pack_return, combine


def make_ep_forward_experts(tokens, world, local_experts, *, prefill, fuse_return=False):
    """Apply native W8A8 experts to the compact rows received by one EP rank."""
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.expert_routed import expert_routed_compute

    hidden, inter = FLASH.hidden_size, FLASH.moe_intermediate_size
    packets = world * tokens * FLASH.num_experts_per_tok
    rows = packets + local_experts * 16
    output_rows = packets if fuse_return else rows

    if fuse_return:
        @pl.jit.inline(auto_scope=False)
        def initialize(output: pl.Tensor):
            with pl.spmd(packets, name_hint="hccl_ep_clear_fused_return") as cleared:
                packet = pl.tile.get_block_idx()
                zero = pl.tile.full([1, hidden], dtype=pl.BF16, value=0.)
                pl.store(zero, [packet, 0], output)
            return cleared
    else:
        @pl.jit.inline(auto_scope=False)
        def initialize(output: pl.Tensor):
            return pl.system.task_dummy(deps=[])

    @pl.jit
    def experts(
        expert_counts: pl.Tensor[[local_experts, 2], pl.INT32],
        x: pl.Tensor[[rows, hidden], pl.INT8],
        x_scale: pl.Tensor[[1, rows], pl.FP32],
        route_weight: pl.Tensor[[1, rows], pl.FP32],
        route_address: pl.Tensor[[1, rows], pl.INT32],
        w1: pl.Tensor[[local_experts, inter, hidden], pl.INT8],
        s1: pl.Tensor[[local_experts, inter], pl.FP32],
        w3: pl.Tensor[[local_experts, inter, hidden], pl.INT8],
        s3: pl.Tensor[[local_experts, inter], pl.FP32],
        w2: pl.Tensor[[local_experts, hidden, inter], pl.INT8],
        s2: pl.Tensor[[local_experts, hidden], pl.FP32],
        output: pl.Out[pl.Tensor[[output_rows, hidden], pl.BF16]],
    ):
        scales = pl.reshape(x_scale, [rows, 1])
        cleared = initialize(output)
        for expert in pl.parallel(local_experts):
            count = pl.read(expert_counts, [expert, 0])
            if count > 0:
                padded = (count + 15) // 16 * 16
                offset = pl.cast(pl.read(expert_counts, [expert, 1]), pl.INDEX)
                source = pl.slice(x, [padded, hidden], [offset, 0])
                source_scale = pl.slice(scales, [padded, 1], [offset, 0])
                weights = pl.slice(route_weight, [1, padded], [0, offset])
                gate_scale = pl.slice(s1, [1, inter], [expert, 0])
                up_scale = pl.slice(s3, [1, inter], [expert, 0])
                down_scale = pl.slice(s2, [1, hidden], [expert, 0])
                if fuse_return:
                    compact = pl.create_tensor([padded, hidden], dtype=pl.BF16)
                    expert_routed_compute(
                        source, source_scale, weights,
                        w1[expert], gate_scale, w3[expert], up_scale, w2[expert], down_scale,
                        compact, count, True, prefill,
                    )
                    with pl.spmd(count, name_hint="hccl_ep_store_fused_return", deps=[cleared]):
                        row = pl.tile.get_block_idx()
                        slot = pl.read(route_address, [0, offset + row])
                        value = pl.tile.load(compact, [row, 0], [1, hidden])
                        pl.store(value, [slot, 0], output)
                else:
                    destination = pl.slice(output, [padded, hidden], [offset, 0])
                    expert_routed_compute(
                        source, source_scale, weights,
                        w1[expert], gate_scale, w3[expert], up_scale, w2[expert], down_scale,
                        destination, count, True, prefill,
                    )

    return experts
