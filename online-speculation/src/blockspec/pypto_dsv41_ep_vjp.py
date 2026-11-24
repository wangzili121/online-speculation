"""Fixed-shape PyPTO packets for the Orthrus DSV4.1 EP16 MoE VJP.

The forward router is replayed, so backward dispatches the normalized expert
input and one expert-output gradient for every recorded route.  HCCL remains
outside these leaves: each packet tensor is laid out as sixteen fixed peer
buckets and can be passed directly through the resident EP all-to-all.
"""

from dataclasses import dataclass


K32_ROWS = 32
TOPK = 6
EXPERTS = 384
EP_WORLD = 16
HIDDEN = 5120
LOCAL_EXPERTS = EXPERTS // EP_WORLD
ROUTES = K32_ROWS * TOPK
PACKET_ROWS = EP_WORLD * ROUTES
# One online update routes a single K32 replay, partitioned over the owner TP
# group. Top-K selects each expert at most once per token.
EXPERT_CAPACITY = K32_ROWS
SLAB_ROWS = LOCAL_EXPERTS * EXPERT_CAPACITY
PACKET_FIELDS = 4
MAP_PAD = 8
HIDDEN_TILE = 512


@dataclass(frozen=True)
class EPVJPGeometry:
    """The production packet ABI shared by both nodes of the EP16 service."""

    rows: int = K32_ROWS
    topk: int = TOPK
    experts: int = EXPERTS
    world: int = EP_WORLD
    hidden: int = HIDDEN

    def __post_init__(self):
        if (
            self.rows != K32_ROWS or self.topk != TOPK
            or self.experts != EXPERTS or self.world != EP_WORLD
            or self.hidden != HIDDEN
        ):
            raise ValueError("Orthrus DSV4.1 EP VJP uses the fixed K32/EP16 ABI")

    @property
    def routes(self):
        return self.rows * self.topk

    @property
    def packet_rows(self):
        return self.world * self.routes

    @property
    def local_experts(self):
        return self.experts // self.world

    @property
    def expert_capacity(self):
        return self.rows

    @property
    def slab_rows(self):
        return self.local_experts * self.expert_capacity


def global_ep_rank(node_rank, local_rank, *, ranks_per_node=8):
    """Translate a node-local device rank to the global EP16 rank."""
    if (
        type(node_rank) is not int or node_rank not in (0, 1)
        or type(local_rank) is not int or not 0 <= local_rank < ranks_per_node
        or ranks_per_node != 8
    ):
        raise ValueError("expected node 0/1 and a local rank in [0, 8)")
    return node_rank * ranks_per_node + local_rank


def make_ep_vjp_dispatch_pack():
    """Pack normalized x and route dY into global-owner EP16 buckets."""
    import pypto.language as pl

    @pl.jit
    def ep_vjp_dispatch_pack(
        normalized_x: pl.Tensor[[K32_ROWS, HIDDEN], pl.BF16],
        route_output_gradient: pl.Tensor[[ROUTES, HIDDEN], pl.BF16],
        route_ids: pl.Tensor[[K32_ROWS, TOPK], pl.INT32],
        sent_metadata: pl.InOut[pl.Tensor[[PACKET_ROWS, PACKET_FIELDS], pl.INT64]],
        sent_x: pl.InOut[pl.Tensor[[PACKET_ROWS, HIDDEN], pl.BF16]],
        sent_dy: pl.InOut[pl.Tensor[[PACKET_ROWS, HIDDEN], pl.BF16]],
        sent_counts: pl.InOut[pl.Tensor[[EP_WORLD, MAP_PAD], pl.INT32]],
        active: pl.Scalar[pl.INT32],
        source_rank: pl.Scalar[pl.INT32],
    ):
        # Metadata and counts are serialized because narrow stores from several
        # cores may otherwise share one cache line.  The owner is always a
        # global EP rank: expert // 24, independent of node-local device IDs.
        with pl.spmd(1, name_hint="orthrus_ep_vjp_pack_metadata") as metadata_ready:
            _block = pl.tile.get_block_idx()
            for peer in pl.range(EP_WORLD):
                for field in pl.range(MAP_PAD):
                    pl.write(sent_counts, [peer, field], pl.cast(0, pl.INT32))
                for row in pl.range(ROUTES):
                    packet = peer * ROUTES + row
                    for field in pl.range(PACKET_FIELDS):
                        pl.write(sent_metadata, [packet, field], pl.cast(0, pl.INT64))
                count = pl.cast(0, pl.INDEX)
                for token in pl.range(active):
                    for slot in pl.range(TOPK):
                        expert = pl.read(route_ids, [token, slot])
                        if expert // LOCAL_EXPERTS == peer:
                            packet = peer * ROUTES + count
                            pl.write(sent_metadata, [packet, 0], pl.cast(source_rank, pl.INT64))
                            pl.write(sent_metadata, [packet, 1], pl.cast(token, pl.INT64))
                            pl.write(sent_metadata, [packet, 2], pl.cast(slot, pl.INT64))
                            pl.write(sent_metadata, [packet, 3], pl.cast(expert, pl.INT64))
                            count = count + 1
                pl.write(sent_counts, [peer, 0], pl.cast(count, pl.INT32))

        with pl.spmd(
            PACKET_ROWS,
            name_hint="orthrus_ep_vjp_pack_payload",
            deps=[metadata_ready],
        ):
            packet = pl.tile.get_block_idx()
            peer = packet // ROUTES
            bucket_row = packet % ROUTES
            peer_count = pl.read(sent_counts, [peer, 0])
            for part in pl.range(HIDDEN // HIDDEN_TILE):
                column = part * HIDDEN_TILE
                if bucket_row < peer_count:
                    token = pl.cast(pl.read(sent_metadata, [packet, 1]), pl.INDEX)
                    slot = pl.cast(pl.read(sent_metadata, [packet, 2]), pl.INDEX)
                    route = token * TOPK + slot
                    sent_x[packet:packet + 1, column:column + HIDDEN_TILE] = normalized_x[
                        token:token + 1, column:column + HIDDEN_TILE
                    ]
                    sent_dy[packet:packet + 1, column:column + HIDDEN_TILE] = route_output_gradient[
                        route:route + 1, column:column + HIDDEN_TILE
                    ]
                else:
                    zero = pl.tile.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.)
                    pl.store(zero, [packet, column], sent_x)
                    pl.store(zero, [packet, column], sent_dy)
        return sent_metadata, sent_x, sent_dy, sent_counts

    return ep_vjp_dispatch_pack


def make_ep_vjp_dispatch_unpack():
    """Regroup received packets into fixed slabs for 24 owned experts."""
    import pypto.language as pl

    @pl.jit
    def ep_vjp_dispatch_unpack(
        packet_metadata: pl.Tensor[[PACKET_ROWS, PACKET_FIELDS], pl.INT64],
        packet_x: pl.Tensor[[PACKET_ROWS, HIDDEN], pl.BF16],
        packet_dy: pl.Tensor[[PACKET_ROWS, HIDDEN], pl.BF16],
        packet_counts: pl.Tensor[[EP_WORLD, MAP_PAD], pl.INT32],
        expert_x: pl.InOut[pl.Tensor[[SLAB_ROWS, HIDDEN], pl.BF16]],
        expert_dy: pl.InOut[pl.Tensor[[SLAB_ROWS, HIDDEN], pl.BF16]],
        expert_counts: pl.InOut[pl.Tensor[[LOCAL_EXPERTS], pl.INT32]],
        packet_map: pl.InOut[pl.Tensor[[PACKET_ROWS, MAP_PAD], pl.INT32]],
        received_total: pl.InOut[pl.Tensor[[MAP_PAD], pl.INT32]],
        owner_rank: pl.Scalar[pl.INT32],
    ):
        # packet_map is indexed by the received packet.  Field 0 is its slab
        # row; field 1 is validity; fields 2..5 preserve source/token/slot/id.
        with pl.spmd(1, name_hint="orthrus_ep_vjp_unpack_metadata") as metadata_ready:
            _block = pl.tile.get_block_idx()
            for packet in pl.range(PACKET_ROWS):
                for field in pl.range(MAP_PAD):
                    pl.write(packet_map, [packet, field], pl.cast(0, pl.INT32))
            total = pl.cast(0, pl.INDEX)
            for local in pl.range(LOCAL_EXPERTS):
                expert = owner_rank * LOCAL_EXPERTS + local
                count = pl.cast(0, pl.INDEX)
                for source in pl.range(EP_WORLD):
                    source_count = pl.cast(pl.read(packet_counts, [source, 0]), pl.INDEX)
                    for row in pl.range(source_count):
                        packet = source * ROUTES + row
                        if pl.read(packet_metadata, [packet, 3]) == expert:
                            slab = local * EXPERT_CAPACITY + count
                            pl.write(packet_map, [packet, 0], pl.cast(slab, pl.INT32))
                            pl.write(packet_map, [packet, 1], pl.cast(1, pl.INT32))
                            for field in pl.range(PACKET_FIELDS):
                                value = pl.read(packet_metadata, [packet, field])
                                pl.write(packet_map, [packet, field + 2], pl.cast(value, pl.INT32))
                            pl.write(packet_map, [packet, 6], pl.cast(source, pl.INT32))
                            pl.write(packet_map, [packet, 7], pl.cast(row, pl.INT32))
                            count = count + 1
                pl.write(expert_counts, [local], pl.cast(count, pl.INT32))
                total = total + count
            for field in pl.range(MAP_PAD):
                pl.write(received_total, [field], pl.cast(0, pl.INT32))
            pl.write(received_total, [0], pl.cast(total, pl.INT32))

        with pl.spmd(
            SLAB_ROWS,
            name_hint="orthrus_ep_vjp_zero_slabs",
            deps=[metadata_ready],
        ) as slabs_zeroed:
            slab = pl.tile.get_block_idx()
            zero = pl.tile.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.)
            for part in pl.range(HIDDEN // HIDDEN_TILE):
                column = part * HIDDEN_TILE
                pl.store(zero, [slab, column], expert_x)
                pl.store(zero, [slab, column], expert_dy)

        with pl.spmd(
            PACKET_ROWS,
            name_hint="orthrus_ep_vjp_unpack_payload",
            deps=[slabs_zeroed],
        ):
            packet = pl.tile.get_block_idx()
            if pl.read(packet_map, [packet, 1]) != 0:
                slab = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                for part in pl.range(HIDDEN // HIDDEN_TILE):
                    column = part * HIDDEN_TILE
                    expert_x[slab:slab + 1, column:column + HIDDEN_TILE] = packet_x[
                        packet:packet + 1, column:column + HIDDEN_TILE
                    ]
                    expert_dy[slab:slab + 1, column:column + HIDDEN_TILE] = packet_dy[
                        packet:packet + 1, column:column + HIDDEN_TILE
                    ]
        return expert_x, expert_dy, expert_counts, packet_map, received_total

    return ep_vjp_dispatch_unpack


def make_ep_vjp_return_pack():
    """Pack owner-computed expert dX into the original source buckets."""
    import pypto.language as pl

    @pl.jit
    def ep_vjp_return_pack(
        expert_dx: pl.Tensor[[SLAB_ROWS, HIDDEN], pl.BF16],
        packet_counts: pl.Tensor[[EP_WORLD, MAP_PAD], pl.INT32],
        packet_map: pl.Tensor[[PACKET_ROWS, MAP_PAD], pl.INT32],
        sent_dx: pl.InOut[pl.Tensor[[PACKET_ROWS, HIDDEN], pl.BF16]],
    ):
        with pl.spmd(
            PACKET_ROWS,
            name_hint="orthrus_ep_vjp_return_pack",
        ):
            packet = pl.tile.get_block_idx()
            source = packet // ROUTES
            bucket_row = packet % ROUTES
            for part in pl.range(HIDDEN // HIDDEN_TILE):
                column = part * HIDDEN_TILE
                if (
                    bucket_row < pl.read(packet_counts, [source, 0])
                    and pl.read(packet_map, [packet, 1]) != 0
                ):
                    slab = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                    sent_dx[packet:packet + 1, column:column + HIDDEN_TILE] = expert_dx[
                        slab:slab + 1, column:column + HIDDEN_TILE
                    ]
                else:
                    zero = pl.tile.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.)
                    pl.store(zero, [packet, column], sent_dx)
        return sent_dx

    return ep_vjp_return_pack


def make_ep_vjp_source_restore():
    """Restore route order and deterministically sum six dX rows per token."""
    import pypto.language as pl

    @pl.jit
    def ep_vjp_source_restore(
        returned_dx: pl.Tensor[[PACKET_ROWS, HIDDEN], pl.BF16],
        sent_metadata: pl.Tensor[[PACKET_ROWS, PACKET_FIELDS], pl.INT64],
        sent_counts: pl.Tensor[[EP_WORLD, MAP_PAD], pl.INT32],
        route_dx: pl.InOut[pl.Tensor[[ROUTES, HIDDEN], pl.BF16]],
        token_dx: pl.InOut[pl.Tensor[[K32_ROWS, HIDDEN], pl.BF16]],
        active: pl.Scalar[pl.INT32],
        source_rank: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(
            ROUTES * (HIDDEN // HIDDEN_TILE),
            name_hint="orthrus_ep_vjp_zero_routes",
        ) as routes_zeroed:
            task = pl.tile.get_block_idx()
            zero = pl.tile.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.)
            route = task // (HIDDEN // HIDDEN_TILE)
            column = task % (HIDDEN // HIDDEN_TILE) * HIDDEN_TILE
            pl.store(zero, [route, column], route_dx)

        with pl.spmd(
            PACKET_ROWS * (HIDDEN // HIDDEN_TILE),
            name_hint="orthrus_ep_vjp_restore_routes",
            deps=[routes_zeroed],
        ) as routes_restored:
            task = pl.tile.get_block_idx()
            packet = task // (HIDDEN // HIDDEN_TILE)
            column = task % (HIDDEN // HIDDEN_TILE) * HIDDEN_TILE
            owner = packet // ROUTES
            bucket_row = packet % ROUTES
            if (
                bucket_row < pl.read(sent_counts, [owner, 0])
                and pl.read(sent_metadata, [packet, 0]) == source_rank
            ):
                token = pl.cast(pl.read(sent_metadata, [packet, 1]), pl.INDEX)
                slot = pl.cast(pl.read(sent_metadata, [packet, 2]), pl.INDEX)
                route = token * TOPK + slot
                route_dx[route:route + 1, column:column + HIDDEN_TILE] = returned_dx[
                    packet:packet + 1, column:column + HIDDEN_TILE
                ]

        with pl.spmd(
            K32_ROWS * (HIDDEN // HIDDEN_TILE),
            name_hint="orthrus_ep_vjp_sum_routes",
            deps=[routes_restored],
        ):
            task = pl.tile.get_block_idx()
            zero = pl.tile.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.)
            token = task // (HIDDEN // HIDDEN_TILE)
            column = task % (HIDDEN // HIDDEN_TILE) * HIDDEN_TILE
            if token < active:
                total = pl.tile.full([1, HIDDEN_TILE], dtype=pl.FP32, value=0.)
                for slot in pl.range(TOPK):
                    value = pl.cast(
                        pl.load(route_dx, [token * TOPK + slot, column], [1, HIDDEN_TILE]),
                        pl.FP32,
                    )
                    total = pl.add(total, value)
                pl.store(
                    pl.cast(total, pl.BF16, mode="rint"),
                    [token, column], token_dx,
                )
            else:
                pl.store(zero, [token, column], token_dx)
        return route_dx, token_dx

    return ep_vjp_source_restore


def ep_vjp_factories():
    """Named production leaves used by resident owner/peer orchestration."""
    return {
        "dispatch_pack": make_ep_vjp_dispatch_pack,
        "dispatch_unpack": make_ep_vjp_dispatch_unpack,
        "return_pack": make_ep_vjp_return_pack,
        "source_restore": make_ep_vjp_source_restore,
    }
