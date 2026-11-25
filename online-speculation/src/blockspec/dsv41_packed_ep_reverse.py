"""Batch routed-expert input VJPs by expert and return them to their source lanes."""

from dataclasses import dataclass
from time import perf_counter

from .dsv41_packed_expert_vjp import make_packed_expert_vjp


def make_packed_ep_reverse(block, hidden, intermediate, limit, topk, experts, dp, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld

    world, local, packets = dp * tp, block // tp, block * topk
    local_experts = experts // world
    slab = max(16, block)
    route_pad = (topk + 7) // 8 * 8
    slot_pad = (block + 7) // 8 * 8

    def build_compute(rows):
        vjp = make_packed_expert_vjp(rows, hidden, intermediate, limit, inline=True)

        @pl.jit.inline(auto_scope=False)
        def compute(
            slot_map: pl.Tensor, expert: pl.Scalar[pl.INT32], count: pl.Scalar[pl.INT32],
            meta: pl.Tensor, value: pl.Tensor, dy: pl.Tensor, route: pl.Tensor,
            w1: pl.Tensor, s1: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor,
            returned_dx: pld.DistributedTensor[[local * topk, hidden], pl.FP32],
            returned_droute: pld.DistributedTensor[[local * topk, 16], pl.FP32],
        ):
            staged_x = pl.create_tensor([rows, hidden], dtype=pl.BF16)
            staged_dy = pl.create_tensor([rows, hidden], dtype=pl.FP32)
            staged_route = pl.create_tensor([rows, 16], dtype=pl.FP32)
            dx = pl.create_tensor([rows, hidden], dtype=pl.FP32)
            droute = pl.create_tensor([rows, 16], dtype=pl.FP32)
            with pl.spmd(rows, name_hint="orthrus_ep_expert_gather"):
                row = pl.tile.get_block_idx()
                if row < count:
                    packet = pl.read(slot_map, [expert, row])
                    staged_x[row:row + 1, :] = value[packet:packet + 1, :]
                    staged_dy[row:row + 1, :] = dy[packet:packet + 1, :]
                    staged_route[row:row + 1, :] = route[packet:packet + 1, :]
                else:
                    staged_dy[row:row + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
            vjp(staged_x, staged_dy, staged_route, w1, s1, w3, s3, w2, s2, dx, droute, count)
            with pl.spmd(count, name_hint="orthrus_ep_expert_return") as sent:
                row = pl.tile.get_block_idx()
                packet = pl.read(slot_map, [expert, row])
                source_rank = pl.read(meta, [packet, 2])
                address = pl.read(meta, [packet, 3]) * topk + pl.read(meta, [packet, 4])
                pld.tensor.put(dst=returned_dx, peer=source_rank, src=dx,
                               dst_offsets=[address, 0], src_offsets=[row, 0], shape=[1, hidden])
                pld.tensor.put(dst=returned_droute, peer=source_rank, src=droute,
                               dst_offsets=[address, 0], src_offsets=[row, 0], shape=[1, 16])
            return sent
        return compute

    compute_small = build_compute(16)
    compute_full = build_compute(slab)

    @pl.jit
    def reverse_owned(
        meta: pl.Tensor[[packets, 16], pl.INT32],
        value: pl.Tensor[[packets, hidden], pl.BF16],
        dy: pl.Tensor[[packets, hidden], pl.FP32],
        route: pl.Tensor[[packets, 16], pl.FP32],
        w1: pl.Tensor[[local_experts, intermediate, hidden], pl.INT8],
        s1: pl.Tensor[[local_experts, intermediate], pl.FP32],
        w3: pl.Tensor[[local_experts, intermediate, hidden], pl.INT8],
        s3: pl.Tensor[[local_experts, intermediate], pl.FP32],
        w2: pl.Tensor[[local_experts, hidden, intermediate], pl.INT8],
        s2: pl.Tensor[[local_experts, hidden], pl.FP32],
        dx: pl.Out[pl.Tensor[[local, hidden], pl.FP32]],
        droute: pl.Out[pl.Tensor[[local, route_pad], pl.FP32]],
        returned_dx: pld.DistributedTensor[[local * topk, hidden], pl.FP32],
        returned_droute: pld.DistributedTensor[[local * topk, 16], pl.FP32],
        complete: pld.DistributedTensor[[1, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
        packet_epoch: pl.Scalar[pl.INT32], return_epoch: pl.Scalar[pl.INT32],
    ):
        slot_map = pl.create_tensor([local_experts, slot_pad], dtype=pl.INT32)
        counts = pl.create_tensor([local_experts, 8], dtype=pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="orthrus_ep_expert_groups", allow_early_resolve=True) as grouped:
            cursor = pl.array.create(local_experts, pl.INT32)
            for expert in pl.range(local_experts):
                cursor[expert] = 0
            for packet in pl.range(active * topk):
                if pl.read(meta, [packet, 0]) == packet_epoch:
                    destination = pl.read(meta, [packet, 1])
                    slot = cursor[destination]
                    pl.write(slot_map, [destination, slot], pl.cast(packet, pl.INT32))
                    cursor[destination] = slot + 1
            for expert in pl.range(local_experts):
                pl.write(counts, [expert, 0], cursor[expert])

        sends = pl.array.create(local_experts, pl.TASK_ID)
        for expert in pl.parallel(local_experts):
            sends[expert] = grouped
            count = pl.read(counts, [expert, 0])
            if count > 0:
                gate_scale = pl.slice(s1, [1, intermediate], [expert, 0])
                up_scale = pl.slice(s3, [1, intermediate], [expert, 0])
                down_scale = pl.slice(s2, [1, hidden], [expert, 0])
                if slab > 16 and count > 16:
                    sends[expert] = compute_full(slot_map, pl.cast(expert, pl.INT32), count, meta, value, dy, route,
                                                 w1[expert], gate_scale, w3[expert], up_scale, w2[expert], down_scale,
                                                 returned_dx, returned_droute)
                else:
                    sends[expert] = compute_small(slot_map, pl.cast(expert, pl.INT32), count, meta, value, dy, route,
                                                  w1[expert], gate_scale, w3[expert], up_scale, w2[expert], down_scale,
                                                  returned_dx, returned_droute)
        with pl.at(level=pl.Level.CORE_GROUP, deps=[sends], allow_early_resolve=False) as published:
            for lane in pl.range(tp):
                pld.system.notify(target=complete, peer=owner + lane, offsets=[0, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)

        if rank >= owner and rank < owner + tp:
            with pl.at(level=pl.Level.CORE_GROUP, deps=[published], allow_early_resolve=False) as available:
                pld.system.wait(signal=complete, offsets=[0, 0], expected=return_epoch * world, cmp=pld.WaitCmp.Ge)
            with pl.spmd(local, deps=[available], name_hint="orthrus_ep_reverse_combine"):
                row = pl.tile.get_block_idx()
                values = pl.full([1, hidden], dtype=pl.FP32, value=0.)
                scores = pl.tile.full([1, route_pad], dtype=pl.FP32, value=0.)
                if row * tp + rank - owner < active:
                    for choice in pl.range(topk):
                        address = row * topk + choice
                        values = pl.add(values, returned_dx[address:address + 1, :])
                        pl.tile.write(scores, [0, choice], pl.read(returned_droute, [address, 0]))
                dx[row:row + 1, :] = values
                pl.store(scores, [row, 0], droute)
        return dx, droute

    @pl.jit.host
    def dispatch(
        meta: pl.Tensor[[world, packets, 16], pl.INT32],
        value: pl.Tensor[[world, packets, hidden], pl.BF16],
        dy: pl.Tensor[[world, packets, hidden], pl.FP32],
        route: pl.Tensor[[world, packets, 16], pl.FP32],
        w1: pl.Tensor[[world, local_experts, intermediate, hidden], pl.INT8],
        s1: pl.Tensor[[world, local_experts, intermediate], pl.FP32],
        w3: pl.Tensor[[world, local_experts, intermediate, hidden], pl.INT8],
        s3: pl.Tensor[[world, local_experts, intermediate], pl.FP32],
        w2: pl.Tensor[[world, local_experts, hidden, intermediate], pl.INT8],
        s2: pl.Tensor[[world, local_experts, hidden], pl.FP32],
        dx: pl.Out[pl.Tensor[[world, local, hidden], pl.FP32]],
        droute: pl.Out[pl.Tensor[[world, local, route_pad], pl.FP32]],
        owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
        packet_epoch: pl.Scalar[pl.INT32], return_epoch: pl.Scalar[pl.INT32],
    ):
        dx_buf = pld.alloc_window_buffer([local * topk, hidden], dtype=pl.FP32)
        droute_buf = pld.alloc_window_buffer([local * topk, 16], dtype=pl.FP32)
        address_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            dx_window = pld.window(dx_buf, [local * topk, hidden], dtype=pl.FP32)
            droute_window = pld.window(droute_buf, [local * topk, 16], dtype=pl.FP32)
            complete = pld.window(address_buf, [1, 1], dtype=pl.INT32)
            reverse_owned(meta[rank], value[rank], dy[rank], route[rank], w1[rank], s1[rank], w3[rank], s3[rank],
                          w2[rank], s2[rank], dx[rank], droute[rank], dx_window, droute_window, complete,
                          rank, owner, active, packet_epoch, return_epoch, device=rank)
    return dispatch


@dataclass(frozen=True)
class PackedExpertGradients:
    owner: int
    layer: int
    rows: int
    input: object
    route: object
    seconds: float


class PackedExpertReverse:
    def __init__(self, runtime, config, graph, banks, block, hidden, intermediate, topk, local_experts, tp):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = runtime, config, graph
        self.shape = ((local_experts, intermediate, hidden), (local_experts, intermediate),
                      (local_experts, intermediate, hidden), (local_experts, intermediate),
                      (local_experts, hidden, intermediate), (local_experts, hidden))
        self.ranks = tuple(range(len(banks)))
        self.banks = tuple(tuple(rank[name] for rank in banks)
                           for name in ("routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
                                        "routed_w2", "routed_w2_scale"))
        self.device_tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.epochs = {owner: 0 for owner in range(0, len(banks), tp)}
        output_shapes = ((block // tp, hidden), (block // tp, (topk + 7) // 8 * 8))
        self.outputs = tuple(self._stack([runtime.alloc_tensor(shape, torch.float32, worker_id=rank)
                                         for rank in self.ranks]) for shape in output_shapes)

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.ranks)

    def _layer_weights(self, layer):
        from math import prod

        for shape, shards in zip(self.shape, self.banks, strict=True):
            views = []
            for shard in shards:
                offset = layer * prod(shape) * shard.dtype.itemsize
                views.append(self.device_tensor(shard.data_ptr + offset, shape, shard.dtype, buffer=shard.buffer,
                                                byte_offset=shard.byte_offset + offset))
            yield self._stack(views)

    def __call__(self, packets):
        weights = tuple(self._layer_weights(packets.layer))
        self.epochs[packets.owner] += 1
        start = perf_counter()
        self.runtime.run(self.graph, packets.metadata, packets.input, packets.gradient, packets.weights,
                         *weights, *self.outputs, packets.owner, packets.rows, packets.epoch, self.epochs[packets.owner],
                         config=self.config)
        return PackedExpertGradients(packets.owner, packets.layer, packets.rows, *self.outputs, perf_counter() - start)
