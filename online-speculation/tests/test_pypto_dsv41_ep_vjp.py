import inspect

import torch

from blockspec import pypto_dsv41_ep_vjp as ep


def _route_ids():
    # Six distinct experts per token, deliberately crossing both nodes and all
    # sixteen global owners.
    token = torch.arange(ep.K32_ROWS, dtype=torch.int64).reshape(-1, 1)
    slot = torch.arange(ep.TOPK, dtype=torch.int64).reshape(1, -1)
    return ((token * 37 + slot * 61) % ep.EXPERTS).to(torch.int32)


def _cpu_pack(x, dy, ids, active, source_rank):
    metadata = torch.zeros(ep.PACKET_ROWS, ep.PACKET_FIELDS, dtype=torch.int64)
    packet_x = torch.zeros(ep.PACKET_ROWS, x.shape[1], dtype=x.dtype)
    packet_dy = torch.zeros_like(packet_x)
    counts = torch.zeros(ep.EP_WORLD, ep.MAP_PAD, dtype=torch.int32)
    for owner in range(ep.EP_WORLD):
        rows = []
        for token in range(active):
            for slot in range(ep.TOPK):
                expert = int(ids[token, slot])
                if expert // ep.LOCAL_EXPERTS == owner:
                    rows.append((token, slot, expert))
        counts[owner, 0] = len(rows)
        for row, (token, slot, expert) in enumerate(rows):
            packet = owner * ep.ROUTES + row
            metadata[packet] = torch.tensor([source_rank, token, slot, expert])
            packet_x[packet] = x[token]
            packet_dy[packet] = dy[token * ep.TOPK + slot]
    return metadata, packet_x, packet_dy, counts


def _all_to_all_dispatch(packed, owner):
    hidden = packed[0][1].shape[1]
    metadata = torch.zeros(ep.PACKET_ROWS, ep.PACKET_FIELDS, dtype=torch.int64)
    packet_x = torch.zeros(ep.PACKET_ROWS, hidden, dtype=packed[0][1].dtype)
    packet_dy = torch.zeros_like(packet_x)
    counts = torch.zeros(ep.EP_WORLD, ep.MAP_PAD, dtype=torch.int32)
    for source, (src_meta, src_x, src_dy, src_counts) in enumerate(packed):
        count = int(src_counts[owner, 0])
        source_slice = slice(owner * ep.ROUTES, owner * ep.ROUTES + count)
        target_slice = slice(source * ep.ROUTES, source * ep.ROUTES + count)
        metadata[target_slice] = src_meta[source_slice]
        packet_x[target_slice] = src_x[source_slice]
        packet_dy[target_slice] = src_dy[source_slice]
        counts[source, 0] = count
    return metadata, packet_x, packet_dy, counts


def _cpu_unpack(received, owner):
    metadata, packet_x, packet_dy, counts = received
    hidden = packet_x.shape[1]
    expert_x = torch.zeros(ep.SLAB_ROWS, hidden, dtype=packet_x.dtype)
    expert_dy = torch.zeros_like(expert_x)
    expert_counts = torch.zeros(ep.LOCAL_EXPERTS, dtype=torch.int32)
    packet_map = torch.zeros(ep.PACKET_ROWS, ep.MAP_PAD, dtype=torch.int32)
    for local in range(ep.LOCAL_EXPERTS):
        expert = owner * ep.LOCAL_EXPERTS + local
        count = 0
        for source in range(ep.EP_WORLD):
            for row in range(int(counts[source, 0])):
                packet = source * ep.ROUTES + row
                if int(metadata[packet, 3]) == expert:
                    slab = local * ep.EXPERT_CAPACITY + count
                    expert_x[slab] = packet_x[packet]
                    expert_dy[slab] = packet_dy[packet]
                    packet_map[packet] = torch.tensor([
                        slab, 1, *metadata[packet].tolist(), source, row,
                    ], dtype=torch.int32)
                    count += 1
        expert_counts[local] = count
    return expert_x, expert_dy, expert_counts, packet_map


def _cpu_return(expert_dx, counts, packet_map):
    returned = torch.zeros(ep.PACKET_ROWS, expert_dx.shape[1], dtype=expert_dx.dtype)
    for source in range(ep.EP_WORLD):
        for row in range(int(counts[source, 0])):
            packet = source * ep.ROUTES + row
            if int(packet_map[packet, 1]):
                returned[packet] = expert_dx[int(packet_map[packet, 0])]
    return returned


def _all_to_all_return(owner_packets, packed, source):
    hidden = owner_packets[0].shape[1]
    returned = torch.zeros(ep.PACKET_ROWS, hidden, dtype=owner_packets[0].dtype)
    for owner, packet in enumerate(owner_packets):
        count = int(packed[source][3][owner, 0])
        owner_slice = slice(source * ep.ROUTES, source * ep.ROUTES + count)
        source_slice = slice(owner * ep.ROUTES, owner * ep.ROUTES + count)
        returned[source_slice] = packet[owner_slice]
    return returned


def _cpu_restore(returned, metadata, counts, active, source_rank):
    route_dx = torch.zeros(ep.ROUTES, returned.shape[1], dtype=returned.dtype)
    for owner in range(ep.EP_WORLD):
        for row in range(int(counts[owner, 0])):
            packet = owner * ep.ROUTES + row
            assert int(metadata[packet, 0]) == source_rank
            token, slot = metadata[packet, 1:3].tolist()
            route_dx[token * ep.TOPK + slot] = returned[packet]
    token_dx = torch.zeros(ep.K32_ROWS, returned.shape[1], dtype=returned.dtype)
    token_dx[:active] = route_dx.reshape(ep.K32_ROWS, ep.TOPK, -1)[:active].float().sum(1).to(
        returned.dtype
    )
    return route_dx, token_dx


def test_production_geometry_is_fixed_k32_ep16():
    shape = ep.EPVJPGeometry()
    assert (shape.rows, shape.topk, shape.experts, shape.world, shape.hidden) == (
        32, 6, 384, 16, 5120,
    )
    assert shape.routes == 192
    assert shape.packet_rows == 3072
    assert shape.local_experts == 24
    assert shape.expert_capacity == 512
    assert shape.slab_rows == 12288


def test_node_local_rank_maps_to_global_ep16_rank():
    assert [ep.global_ep_rank(0, rank) for rank in range(8)] == list(range(8))
    assert [ep.global_ep_rank(1, rank) for rank in range(8)] == list(range(8, 16))


def test_cpu_ep16_round_trip_restores_and_sums_routes_with_inactive_zero():
    torch.manual_seed(7)
    hidden, active = 7, 19
    ids = _route_ids()
    source_x = [torch.randn(ep.K32_ROWS, hidden, dtype=torch.bfloat16) for _ in range(ep.EP_WORLD)]
    source_dy = [torch.randn(ep.ROUTES, hidden, dtype=torch.bfloat16) for _ in range(ep.EP_WORLD)]
    packed = [
        _cpu_pack(source_x[rank], source_dy[rank], ids, active, rank)
        for rank in range(ep.EP_WORLD)
    ]

    owner_packets = []
    expected_route_dx = []
    for owner in range(ep.EP_WORLD):
        received = _all_to_all_dispatch(packed, owner)
        expert_x, expert_dy, expert_counts, packet_map = _cpu_unpack(received, owner)
        assert int(expert_counts.sum()) == sum(int(pack[3][owner, 0]) for pack in packed)
        # Stand in for the frozen expert VJP with a deterministic owner-local
        # function.  The packet tests care about placement, not expert math.
        expert_dx = (expert_x.float() + 2 * expert_dy.float() + owner).to(torch.bfloat16)
        owner_packets.append(_cpu_return(expert_dx, received[3], packet_map))

    for source in range(ep.EP_WORLD):
        returned = _all_to_all_return(owner_packets, packed, source)
        route_dx, token_dx = _cpu_restore(
            returned, packed[source][0], packed[source][3], active, source,
        )
        expected = torch.zeros_like(route_dx)
        for token in range(active):
            for slot in range(ep.TOPK):
                expert = int(ids[token, slot])
                owner = expert // ep.LOCAL_EXPERTS
                expected[token * ep.TOPK + slot] = (
                    source_x[source][token].float()
                    + 2 * source_dy[source][token * ep.TOPK + slot].float()
                    + owner
                ).to(torch.bfloat16)
        torch.testing.assert_close(route_dx, expected)
        torch.testing.assert_close(
            token_dx[:active], expected.reshape(ep.K32_ROWS, ep.TOPK, hidden)[:active].float().sum(1).to(
                torch.bfloat16
            ),
        )
        assert torch.count_nonzero(token_dx[active:]) == 0
        assert torch.count_nonzero(route_dx[active * ep.TOPK:]) == 0


def test_packet_map_preserves_source_token_slot_and_global_expert():
    hidden = 1
    ids = _route_ids()
    packed = [
        _cpu_pack(
            torch.full((ep.K32_ROWS, hidden), rank, dtype=torch.bfloat16),
            torch.zeros(ep.ROUTES, hidden, dtype=torch.bfloat16), ids, 3, rank,
        )
        for rank in range(ep.EP_WORLD)
    ]
    owner = 11
    received = _all_to_all_dispatch(packed, owner)
    _, _, _, packet_map = _cpu_unpack(received, owner)
    valid = packet_map[:, 1].bool()
    assert valid.any()
    assert torch.equal(packet_map[valid, 2:6].long(), received[0][valid].long())
    assert bool(((packet_map[valid, 5] // ep.LOCAL_EXPERTS) == owner).all())


def test_pypto_source_defines_four_fixed_shape_packet_kernels():
    source = inspect.getsource(ep)
    assert set(ep.ep_vjp_factories()) == {
        "dispatch_pack", "dispatch_unpack", "return_pack", "source_restore",
    }
    for name in (
        "orthrus_ep_vjp_pack_payload",
        "orthrus_ep_vjp_unpack_payload",
        "orthrus_ep_vjp_return_pack",
        "orthrus_ep_vjp_sum_routes",
    ):
        assert name in source
    assert "pl.Tensor[[K32_ROWS, HIDDEN], pl.BF16]" in source
    assert "pl.Tensor[[PACKET_ROWS, PACKET_FIELDS], pl.INT64]" in source
    assert "pl.Tensor[[SLAB_ROWS, HIDDEN], pl.BF16]" in source
    assert "deps=[slabs_zeroed]" in source
    assert "deps=[routes_restored]" in source
