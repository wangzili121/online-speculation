"""Fixed-shape PyPTO packets for the Orthrus DSV4.1 EP16 MoE VJP.

The forward router is replayed, so backward dispatches the normalized expert
input and one expert-output gradient for every recorded route.  HCCL remains
outside these leaves: each packet tensor is laid out as sixteen equal-width
peer buckets sized from the captured route IDs for that layer.
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
EP_BUCKET_WIDTHS = (32, 64, 128, ROUTES)
# One online update routes a single K32 replay, partitioned over the owner TP
# group. Top-K selects each expert at most once per token.
EXPERT_CAPACITY = K32_ROWS
EXPERT_SMALL_ROWS = 16
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
    bucket_rows: int = ROUTES

    def __post_init__(self):
        if (
            self.rows != K32_ROWS or self.topk != TOPK
            or self.experts != EXPERTS or self.world != EP_WORLD
            or self.hidden != HIDDEN
            or type(self.bucket_rows) is not int or self.bucket_rows not in EP_BUCKET_WIDTHS
        ):
            raise ValueError("Orthrus DSV4.1 EP VJP uses the fixed K32/EP16 ABI")

    @property
    def routes(self):
        return self.rows * self.topk

    @property
    def packet_rows(self):
        return self.world * self.bucket_rows

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


def route_bucket_width(source_ids):
    """Choose one wire width for the largest source-to-owner bucket."""
    import torch

    maximum = 0
    for ids in source_ids:
        if (ids.ndim != 2 or ids.shape[0] > K32_ROWS or ids.shape[1] != TOPK
                or ids.dtype != torch.int32):
            raise ValueError("EP route IDs must be [rows, top-k] INT32 controls")
        if ((ids < 0) | (ids >= EXPERTS)).any():
            raise ValueError("EP route ID has no expert owner")
        owners = ids.flatten().to(torch.int64) // LOCAL_EXPERTS
        maximum = max(maximum, int(torch.bincount(owners, minlength=EP_WORLD).max()))
    return next(width for width in EP_BUCKET_WIDTHS if width >= maximum)


def make_ep_vjp_dispatch_pack(bucket_rows=ROUTES):
    """Pack normalized x and route dY into global-owner EP16 buckets."""
    import pypto.language as pl
    P = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows

    @pl.jit
    def ep_vjp_dispatch_pack(
        normalized_x: pl.Tensor[[K32_ROWS, HIDDEN], pl.BF16],
        route_output_gradient: pl.Tensor[[ROUTES, HIDDEN], pl.BF16],
        route_ids: pl.Tensor[[K32_ROWS, TOPK], pl.INT32],
        sent_metadata: pl.InOut[pl.Tensor[[P, PACKET_FIELDS], pl.INT64]],
        sent_x: pl.InOut[pl.Tensor[[P, HIDDEN], pl.BF16]],
        sent_dy: pl.InOut[pl.Tensor[[P, HIDDEN], pl.BF16]],
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
                for row in pl.range(bucket_rows):
                    packet = peer * bucket_rows + row
                    for field in pl.range(PACKET_FIELDS):
                        pl.write(sent_metadata, [packet, field], pl.cast(0, pl.INT64))
                count = pl.cast(0, pl.INDEX)
                for token in pl.range(active):
                    for slot in pl.range(TOPK):
                        expert = pl.read(route_ids, [token, slot])
                        if expert // LOCAL_EXPERTS == peer:
                            packet = peer * bucket_rows + count
                            pl.write(sent_metadata, [packet, 0], pl.cast(source_rank, pl.INT64))
                            pl.write(sent_metadata, [packet, 1], pl.cast(token, pl.INT64))
                            pl.write(sent_metadata, [packet, 2], pl.cast(slot, pl.INT64))
                            pl.write(sent_metadata, [packet, 3], pl.cast(expert, pl.INT64))
                            count = count + 1
                pl.write(sent_counts, [peer, 0], pl.cast(count, pl.INT32))

        with pl.spmd(
            P,
            name_hint="orthrus_ep_vjp_pack_payload",
            deps=[metadata_ready],
        ):
            packet = pl.tile.get_block_idx()
            peer = packet // bucket_rows
            bucket_row = packet % bucket_rows
            peer_count = pl.read(sent_counts, [peer, 0])
            if bucket_row < peer_count:
                token = pl.cast(pl.read(sent_metadata, [packet, 1]), pl.INDEX)
                slot = pl.cast(pl.read(sent_metadata, [packet, 2]), pl.INDEX)
                route = token * TOPK + slot
                x = pl.load(normalized_x, [token, 0], [1, HIDDEN])
                dy = pl.load(route_output_gradient, [route, 0], [1, HIDDEN])
                pl.store(x, [packet, 0], sent_x)
                pl.store(dy, [packet, 0], sent_dy)
            else:
                zero = pl.tile.full([1, HIDDEN], dtype=pl.BF16, value=0.)
                pl.store(zero, [packet, 0], sent_x)
                pl.store(zero, [packet, 0], sent_dy)
        return sent_metadata, sent_x, sent_dy, sent_counts

    return ep_vjp_dispatch_pack


def make_ep_vjp_empty_payload(bucket_rows=ROUTES):
    """Fill the two BF16 payloads for a source rank with no replay tokens."""
    import pypto.language as pl

    packet_rows = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows
    chunk = 4

    @pl.jit
    def empty_payload(
        sent_x: pl.Out[pl.Tensor[[packet_rows, HIDDEN], pl.BF16]],
        sent_dy: pl.Out[pl.Tensor[[packet_rows, HIDDEN], pl.BF16]],
    ):
        for task in pl.spmd(packet_rows // chunk, name_hint="orthrus_ep_vjp_empty_payload"):
            offset = task * chunk
            zero = pl.tile.full([chunk, HIDDEN], dtype=pl.BF16, value=0.)
            pl.store(zero, [offset, 0], sent_x)
            pl.store(zero, [offset, 0], sent_dy)
        return sent_x, sent_dy

    return empty_payload


def make_ep_vjp_dispatch_unpack(bucket_rows=ROUTES):
    """Regroup received packets and initialize only rows read by the expert VJP."""
    import pypto.language as pl
    P = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows

    @pl.jit
    def ep_vjp_dispatch_unpack(
        packet_metadata: pl.Tensor[[P, PACKET_FIELDS], pl.INT64],
        packet_x: pl.Tensor[[P, HIDDEN], pl.BF16],
        packet_dy: pl.Tensor[[P, HIDDEN], pl.BF16],
        packet_counts: pl.Tensor[[EP_WORLD, MAP_PAD], pl.INT32],
        expert_x: pl.InOut[pl.Tensor[[SLAB_ROWS, HIDDEN], pl.BF16]],
        expert_dy: pl.InOut[pl.Tensor[[SLAB_ROWS, HIDDEN], pl.BF16]],
        expert_counts: pl.InOut[pl.Tensor[[LOCAL_EXPERTS], pl.INT32]],
        packet_map: pl.InOut[pl.Tensor[[P, MAP_PAD], pl.INT32]],
        received_total: pl.InOut[pl.Tensor[[MAP_PAD], pl.INT32]],
        owner_rank: pl.Scalar[pl.INT32],
    ):
        # packet_map is indexed by the received packet.  Field 0 is its slab
        # row; field 1 is validity; fields 2..5 preserve source/token/slot/id.
        slab_packet = pl.create_tensor([LOCAL_EXPERTS, EXPERT_CAPACITY], dtype=pl.INT32)
        with pl.spmd(1, name_hint="orthrus_ep_vjp_unpack_metadata") as metadata_ready:
            _block = pl.tile.get_block_idx()
            for packet in pl.range(P):
                for field in pl.range(MAP_PAD):
                    pl.write(packet_map, [packet, field], pl.cast(0, pl.INT32))
            counts = pl.tile.full([1, LOCAL_EXPERTS], dtype=pl.INT32, value=0)
            total = pl.cast(0, pl.INDEX)
            owner_begin = owner_rank * LOCAL_EXPERTS
            # Scanning in source/bucket order preserves the order inside every expert.
            for source in pl.range(EP_WORLD):
                source_count = pl.cast(pl.read(packet_counts, [source, 0]), pl.INDEX)
                for row in pl.range(source_count):
                    packet = source * bucket_rows + row
                    expert = pl.cast(pl.read(packet_metadata, [packet, 3]), pl.INDEX) - owner_begin
                    if expert >= 0 and expert < LOCAL_EXPERTS:
                        count = pl.cast(pl.read(counts, [0, expert]), pl.INDEX)
                        slab = expert * EXPERT_CAPACITY + count
                        pl.write(slab_packet, [expert, count], pl.cast(packet, pl.INT32))
                        pl.write(packet_map, [packet, 0], pl.cast(slab, pl.INT32))
                        pl.write(packet_map, [packet, 1], pl.cast(1, pl.INT32))
                        for field in pl.range(PACKET_FIELDS):
                            value = pl.read(packet_metadata, [packet, field])
                            pl.write(packet_map, [packet, field + 2], pl.cast(value, pl.INT32))
                        pl.write(packet_map, [packet, 6], pl.cast(source, pl.INT32))
                        pl.write(packet_map, [packet, 7], pl.cast(row, pl.INT32))
                        pl.write(counts, [0, expert], pl.cast(count + 1, pl.INT32))
                        total = total + 1
            for local in pl.range(LOCAL_EXPERTS):
                pl.write(expert_counts, [local], pl.read(counts, [0, local]))
            for field in pl.range(MAP_PAD):
                pl.write(received_total, [field], pl.cast(0, pl.INT32))
            pl.write(received_total, [0], pl.cast(total, pl.INT32))

        with pl.spmd(
            SLAB_ROWS,
            name_hint="orthrus_ep_vjp_unpack_slabs",
            deps=[metadata_ready],
        ):
            slab = pl.tile.get_block_idx()
            local, row = slab // EXPERT_CAPACITY, slab % EXPERT_CAPACITY
            live_rows = pl.read(expert_counts, [local])
            if row < live_rows:
                packet = pl.cast(pl.read(slab_packet, [local, row]), pl.INDEX)
                x = pl.load(packet_x, [packet, 0], [1, HIDDEN])
                dy = pl.load(packet_dy, [packet, 0], [1, HIDDEN])
                pl.store(x, [slab, 0], expert_x)
                pl.store(dy, [slab, 0], expert_dy)
            elif live_rows > 0 and (row < EXPERT_SMALL_ROWS or live_rows > EXPERT_SMALL_ROWS):
                zero = pl.tile.full([1, HIDDEN], dtype=pl.BF16, value=0.)
                pl.store(zero, [slab, 0], expert_x)
                pl.store(zero, [slab, 0], expert_dy)
        return expert_x, expert_dy, expert_counts, packet_map, received_total

    return ep_vjp_dispatch_unpack


def make_ep_vjp_return_pack(bucket_rows=ROUTES):
    """Pack owner-computed expert dX into the original source buckets."""
    import pypto.language as pl
    P = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows

    @pl.jit
    def ep_vjp_return_pack(
        expert_dx: pl.Tensor[[SLAB_ROWS, HIDDEN], pl.BF16],
        packet_counts: pl.Tensor[[EP_WORLD, MAP_PAD], pl.INT32],
        packet_map: pl.Tensor[[P, MAP_PAD], pl.INT32],
        sent_dx: pl.InOut[pl.Tensor[[P, HIDDEN], pl.BF16]],
    ):
        with pl.spmd(
            P,
            name_hint="orthrus_ep_vjp_return_pack",
        ):
            packet = pl.tile.get_block_idx()
            source = packet // bucket_rows
            bucket_row = packet % bucket_rows
            if (
                bucket_row < pl.read(packet_counts, [source, 0])
                and pl.read(packet_map, [packet, 1]) != 0
            ):
                slab = pl.cast(pl.read(packet_map, [packet, 0]), pl.INDEX)
                dx = pl.load(expert_dx, [slab, 0], [1, HIDDEN])
                pl.store(dx, [packet, 0], sent_dx)
            else:
                zero = pl.tile.full([1, HIDDEN], dtype=pl.BF16, value=0.)
                pl.store(zero, [packet, 0], sent_dx)
        return sent_dx

    return ep_vjp_return_pack


def make_ep_vjp_source_restore(bucket_rows=ROUTES):
    """Restore route order and deterministically sum six dX rows per token."""
    import pypto.language as pl
    P = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows

    @pl.jit
    def ep_vjp_source_restore(
        returned_dx: pl.Tensor[[P, HIDDEN], pl.BF16],
        sent_metadata: pl.Tensor[[P, PACKET_FIELDS], pl.INT64],
        sent_counts: pl.Tensor[[EP_WORLD, MAP_PAD], pl.INT32],
        route_dx: pl.InOut[pl.Tensor[[ROUTES, HIDDEN], pl.BF16]],
        token_dx: pl.InOut[pl.Tensor[[K32_ROWS, HIDDEN], pl.BF16]],
        active: pl.Scalar[pl.INT32],
        source_rank: pl.Scalar[pl.INT32],
    ):
        route_packet = pl.create_tensor([K32_ROWS, MAP_PAD], dtype=pl.INT32)
        with pl.spmd(1, name_hint="orthrus_ep_vjp_index_routes") as indexed:
            _block = pl.tile.get_block_idx()
            for token in pl.range(K32_ROWS):
                for slot in pl.range(TOPK):
                    pl.write(route_packet, [token, slot], pl.cast(-1, pl.INT32))
            for owner in pl.range(EP_WORLD):
                count = pl.cast(pl.read(sent_counts, [owner, 0]), pl.INDEX)
                for row in pl.range(count):
                    packet = owner * bucket_rows + row
                    if pl.read(sent_metadata, [packet, 0]) == source_rank:
                        token = pl.cast(pl.read(sent_metadata, [packet, 1]), pl.INDEX)
                        slot = pl.cast(pl.read(sent_metadata, [packet, 2]), pl.INDEX)
                        pl.write(route_packet, [token, slot], pl.cast(packet, pl.INT32))

        with pl.spmd(
            K32_ROWS * (HIDDEN // HIDDEN_TILE),
            name_hint="orthrus_ep_vjp_sum_routes",
            deps=[indexed],
        ):
            task = pl.tile.get_block_idx()
            zero = pl.tile.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.)
            token = task // (HIDDEN // HIDDEN_TILE)
            column = task % (HIDDEN // HIDDEN_TILE) * HIDDEN_TILE
            if token < active:
                total = pl.tile.full([1, HIDDEN_TILE], dtype=pl.FP32, value=0.)
                for slot in pl.range(TOPK):
                    route = token * TOPK + slot
                    entry = pl.read(route_packet, [token, slot])
                    value = zero
                    if entry >= 0:
                        packet = pl.cast(entry, pl.INDEX)
                        value = pl.load(returned_dx, [packet, column], [1, HIDDEN_TILE])
                    pl.store(value, [route, column], route_dx)
                    total = pl.add(total, pl.cast(value, pl.FP32))
                pl.store(
                    pl.cast(total, pl.BF16, mode="rint"),
                    [token, column], token_dx,
                )
            else:
                for slot in pl.range(TOPK):
                    pl.store(zero, [token * TOPK + slot, column], route_dx)
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
