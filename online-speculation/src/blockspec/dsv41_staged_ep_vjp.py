"""Route resident draft FFN gradients over the staged EP transport."""

from dataclasses import dataclass
from time import perf_counter

from .dsv41_online_profile import timed_stage
from .dsv41_packed_expert_vjp import make_packed_expert_vjp
from .dsv41_packed_ffn_tape import packed_ffn_tape_shapes


def make_staged_expert_vjp(model, block, last_layers, *, world, local_ranks, tp):
    """Build dispatch, frozen W8A8 expert VJP, and source-order reduction graphs."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    hidden, inter, copies = model.hidden_size, model.moe_intermediate_size, model.hc_mult
    topk, experts = model.num_experts_per_tok, model.n_routed_experts // world
    owned = block // tp
    slots = owned * topk
    packets = world * slots
    header = (slots + 15) // 16 * 16
    choices = (topk + 7) // 8 * 8
    shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    vectors_shape = shapes["ffn_tape_vectors"][0]
    routes_shape = shapes["ffn_tape_route_ids"][0]
    clip = model.swiglu_limit

    @pl.jit(auto_scope=False)
    def pack_rank(
        gradient: pl.Tensor[[block, hidden], pl.FP32],
        vectors: pl.Tensor[vectors_shape, pl.BF16],
        ids: pl.Tensor[routes_shape, pl.INT32],
        weights: pl.Tensor[routes_shape, pl.FP32],
        tags: pl.Tensor[[world, header], pl.INT32],
        factors: pl.Tensor[[world, header], pl.FP32],
        values: pl.Tensor[[packets, hidden], pl.BF16],
        seeds: pl.Tensor[[packets, hidden], pl.FP32],
        local_seed: pl.Tensor[[owned, hidden], pl.FP32],
        seed_window: pld.DistributedTensor[[owned, hidden], pl.FP32],
        ready_window: pld.DistributedTensor[[1, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], owner: pl.Scalar[pl.INT32], live: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        if rank == owner:
            with pl.spmd(block, name_hint="staged_ep_vjp_share_seed") as scattered:
                row = pl.tile.get_block_idx()
                if row < live:
                    pld.tensor.put(dst=seed_window, peer=owner + row % tp, src=gradient,
                                   dst_offsets=[row // tp, 0], src_offsets=[row, 0], shape=[1, hidden])
            with pl.at(level=pl.Level.CORE_GROUP, deps=[scattered], allow_early_resolve=False):
                for lane in pl.range(tp):
                    pld.system.notify(target=ready_window, peer=owner + lane, offsets=[0, 0],
                                      value=epoch, op=pld.NotifyOp.Set)
        if owner >= 0 and rank >= owner and rank < owner + tp:
            with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as ready:
                pld.system.wait(signal=ready_window, offsets=[0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
            with pl.spmd(owned, deps=[ready], name_hint="staged_ep_vjp_local_seed") as seeded:
                row = pl.tile.get_block_idx()
                if row * tp + rank - owner < live:
                    local_seed[row:row + 1, :] = seed_window[row:row + 1, :]
                else:
                    local_seed[row:row + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
            with pl.spmd(world, deps=[seeded], name_hint="staged_ep_vjp_pack_route"):
                peer = pl.tile.get_block_idx()
                tag_row = pl.tile.full([1, header], dtype=pl.INT32, value=0)
                factor_row = pl.tile.full([1, header], dtype=pl.FP32, value=0.)
                for row in pl.range(owned):
                    if row * tp + rank - owner < live:
                        tape_row = slot * owned + row
                        vector_row = slot * (2 * block + owned) + 2 * block + row
                        for choice in pl.range(topk):
                            expert = pl.read(ids, [tape_row, choice])
                            if expert // experts == peer:
                                position = row * topk + choice
                                packet = peer * slots + position
                                pl.tile.write(tag_row, [0, position], pl.cast(expert % experts + 1, pl.INT32))
                                pl.tile.write(factor_row, [0, position], pl.read(weights, [tape_row, choice]))
                                values[packet:packet + 1, :] = vectors[vector_row:vector_row + 1, :]
                                seeds[packet:packet + 1, :] = local_seed[row:row + 1, :]
                pl.store(tag_row, [peer, 0], tags)
                pl.store(factor_row, [peer, 0], factors)

    @pl.jit.host
    def pack(
        gradient: pl.Tensor[[local_ranks, block, hidden], pl.FP32],
        vectors: pl.Tensor[[local_ranks, *vectors_shape], pl.BF16],
        ids: pl.Tensor[[local_ranks, *routes_shape], pl.INT32],
        weights: pl.Tensor[[local_ranks, *routes_shape], pl.FP32],
        tags: pl.Out[pl.Tensor[[local_ranks, world, header], pl.INT32]],
        factors: pl.Out[pl.Tensor[[local_ranks, world, header], pl.FP32]],
        values: pl.Out[pl.Tensor[[local_ranks, packets, hidden], pl.BF16]],
        seeds: pl.Out[pl.Tensor[[local_ranks, packets, hidden], pl.FP32]],
        local_seed: pl.Out[pl.Tensor[[local_ranks, owned, hidden], pl.FP32]],
        owner: pl.Scalar[pl.INT32], live: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        seed_buffer = pld.alloc_window_buffer([owned, hidden], dtype=pl.FP32)
        ready_buffer = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            seed_window = pld.window(seed_buffer, [owned, hidden], dtype=pl.FP32)
            ready_window = pld.window(ready_buffer, [1, 1], dtype=pl.INT32)
            pack_rank(gradient[rank], vectors[rank], ids[rank], weights[rank], tags[rank], factors[rank],
                      values[rank], seeds[rank], local_seed[rank], seed_window, ready_window,
                      rank, owner, live, slot, epoch, device=rank)

    def expert_group(rows):
        vjp = make_packed_expert_vjp(rows, hidden, inter, clip, inline=True)

        @pl.jit.inline(auto_scope=False)
        def apply_group(
            mapping: pl.Tensor, expert: pl.Scalar[pl.INT32], count: pl.Scalar[pl.INT32],
            factors: pl.Tensor, values: pl.Tensor, seeds: pl.Tensor,
            w1: pl.Tensor, s1: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor,
            result: pl.Tensor, route: pl.Tensor,
        ):
            x = pl.create_tensor([rows, hidden], dtype=pl.BF16)
            dy = pl.create_tensor([rows, hidden], dtype=pl.FP32)
            weights = pl.create_tensor([rows, 16], dtype=pl.FP32)
            dx = pl.create_tensor([rows, hidden], dtype=pl.FP32)
            dr = pl.create_tensor([rows, 16], dtype=pl.FP32)
            with pl.spmd(rows, name_hint="staged_ep_vjp_gather"):
                row = pl.tile.get_block_idx()
                if row < count:
                    packet = pl.read(mapping, [expert, row])
                    x[row:row + 1, :] = values[packet:packet + 1, :]
                    dy[row:row + 1, :] = seeds[packet:packet + 1, :]
                    coeff = pl.tile.full([1, 16], dtype=pl.FP32, value=0.)
                    pl.tile.write(coeff, [0, 0], pl.read(factors, [packet // slots, packet % slots]))
                    pl.store(coeff, [row, 0], weights)
                else:
                    dy[row:row + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
            vjp(x, dy, weights, w1, s1, w3, s3, w2, s2, dx, dr, count)
            with pl.spmd(count, name_hint="staged_ep_vjp_return"):
                row = pl.tile.get_block_idx()
                packet = pl.read(mapping, [expert, row])
                result[packet:packet + 1, :] = dx[row:row + 1, :]
                route[packet:packet + 1, :] = dr[row:row + 1, :]
        return apply_group

    short = expert_group(16)
    wide = expert_group(block) if block > 16 else short

    @pl.jit(auto_scope=False)
    def compute_rank(
        tags: pl.Tensor, factors: pl.Tensor, values: pl.Tensor, seeds: pl.Tensor,
        w1: pl.Tensor, s1: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor,
        result: pl.Tensor, route: pl.Tensor, source: pl.Scalar[pl.INT32],
    ):
        mapping = pl.create_tensor([experts, block], dtype=pl.INT32)
        counts = pl.create_tensor([experts, 16], dtype=pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="staged_ep_vjp_group", allow_early_resolve=True):
            cursor = pl.array.create(experts, pl.INT32)
            for expert in pl.range(experts):
                cursor[expert] = 0
            for lane in pl.range(tp):
                for slot in pl.range(slots):
                    tag = pl.read(tags, [source + lane, slot])
                    if tag != 0:
                        expert = tag - 1
                        index = cursor[expert]
                        pl.write(mapping, [expert, index], pl.cast((source + lane) * slots + slot, pl.INT32))
                        cursor[expert] = index + 1
            for expert in pl.range(experts):
                pl.write(counts, [expert, 0], cursor[expert])
        for expert in pl.parallel(experts):
            count = pl.read(counts, [expert, 0])
            if count != 0:
                gate = pl.slice(s1, [1, inter], [expert, 0])
                up = pl.slice(s3, [1, inter], [expert, 0])
                down = pl.slice(s2, [1, hidden], [expert, 0])
                if block > 16 and count > 16:
                    wide(mapping, pl.cast(expert, pl.INT32), count, factors, values, seeds,
                         w1[expert], gate, w3[expert], up, w2[expert], down, result, route)
                else:
                    short(mapping, pl.cast(expert, pl.INT32), count, factors, values, seeds,
                          w1[expert], gate, w3[expert], up, w2[expert], down, result, route)

    @pl.jit.host
    def compute(
        tags: pl.Tensor[[local_ranks, world, header], pl.INT32],
        factors: pl.Tensor[[local_ranks, world, header], pl.FP32],
        values: pl.Tensor[[local_ranks, packets, hidden], pl.BF16],
        seeds: pl.Tensor[[local_ranks, packets, hidden], pl.FP32],
        w1: pl.Tensor[[local_ranks, experts, inter, hidden], pl.INT8],
        s1: pl.Tensor[[local_ranks, experts, inter], pl.FP32],
        w3: pl.Tensor[[local_ranks, experts, inter, hidden], pl.INT8],
        s3: pl.Tensor[[local_ranks, experts, inter], pl.FP32],
        w2: pl.Tensor[[local_ranks, experts, hidden, inter], pl.INT8],
        s2: pl.Tensor[[local_ranks, experts, hidden], pl.FP32],
        result: pl.Out[pl.Tensor[[local_ranks, packets, hidden], pl.FP32]],
        route: pl.Out[pl.Tensor[[local_ranks, packets, 16], pl.FP32]],
        source: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(pld.world_size()):
            compute_rank(tags[rank], factors[rank], values[rank], seeds[rank], w1[rank], s1[rank],
                         w3[rank], s3[rank], w2[rank], s2[rank], result[rank], route[rank], source, device=rank)

    @pl.jit(auto_scope=False)
    def combine_rank(
        ids: pl.Tensor, values: pl.Tensor, weights: pl.Tensor, dx: pl.Tensor, dr: pl.Tensor,
        rank: pl.Scalar[pl.INT32], owner: pl.Scalar[pl.INT32],
        live: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        if owner >= 0 and rank >= owner and rank < owner + tp:
            with pl.spmd(owned, name_hint="staged_ep_vjp_combine"):
                row = pl.tile.get_block_idx()
                result = pl.full([1, hidden], dtype=pl.FP32, value=0.)
                route = pl.tile.full([1, choices], dtype=pl.FP32, value=0.)
                if row * tp + rank - owner < live:
                    for choice in pl.range(topk):
                        peer = pl.read(ids, [slot * owned + row, choice]) // experts
                        packet = peer * slots + row * topk + choice
                        result = pl.add(result, values[packet:packet + 1, :])
                        pl.tile.write(route, [0, choice], pl.read(weights, [packet, 0]))
                dx[row:row + 1, :] = result
                pl.store(route, [row, 0], dr)

    @pl.jit.host
    def combine(
        ids: pl.Tensor[[local_ranks, *routes_shape], pl.INT32],
        values: pl.Tensor[[local_ranks, packets, hidden], pl.FP32],
        weights: pl.Tensor[[local_ranks, packets, 16], pl.FP32],
        dx: pl.Out[pl.Tensor[[local_ranks, owned, hidden], pl.FP32]],
        dr: pl.Out[pl.Tensor[[local_ranks, owned, choices], pl.FP32]],
        owner: pl.Scalar[pl.INT32], live: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(pld.world_size()):
            combine_rank(ids[rank], values[rank], weights[rank], dx[rank], dr[rank], rank, owner, live, slot, device=rank)

    return pack, compute, combine


@dataclass(frozen=True)
class StagedExpertBackward:
    """Resident gradients, valid until the next layer dispatch."""

    local_gradient: object
    gradients: object


class StagedExpertVJP:
    """Run frozen experts on both hosts and return gradients to the student owner."""

    def __init__(self, runtime, transport, programs, banks, model, *, block, last_layers):
        import torch
        from pypto.runtime import StackedDeviceTensor

        from .dsv41_packed_ep_ipc import ResidentHcclMappings

        self.runtime, self.transport = runtime, transport
        self.pack, self.compute, self.combine = programs
        self.banks = banks
        self.plan = transport.plan
        self.global_ranks = tuple(self.plan.local_ranks)
        self.workers = tuple(range(len(self.global_ranks)))
        self.stacked = StackedDeviceTensor
        self.first_layer = model.num_hidden_layers - last_layers
        self.epoch = 0
        self.tape_names = ("ffn_tape_vectors", "ffn_tape_route_ids", "ffn_tape_route_weights")

        if transport.mappings is None:
            transport.mappings = ResidentHcclMappings(runtime, transport.memory)
        if transport.mappings.runtime is not runtime:
            raise ValueError("EP reverse requires the graph owner's HCCL IPC mappings")
        world, tp = sum(self.plan.counts), self.plan.tp
        topk, hidden = model.num_experts_per_tok, model.hidden_size
        owned = block // tp
        slots = owned * topk
        packets = world * slots
        header = (slots + 15) // 16 * 16

        def allocate(shape, dtype, exportable=False):
            shards = tuple(runtime.alloc_tensor(shape, dtype, worker_id=rank, exportable=exportable)
                           for rank in self.workers)
            if exportable:
                for rank, tensor in enumerate(shards):
                    transport.mappings.register(rank, tensor)
            return self._stack(shards)

        def packets_pair(specs):
            def allocate_set():
                return tuple(allocate(shape, dtype, True) for _, shape, dtype in specs)

            send, recv = allocate_set(), allocate_set()
            transfers = [
                tuple((before.shards[rank], after.shards[rank], name)
                      for before, after, (name, _, _) in zip(send, recv, specs, strict=True))
                for rank in self.workers
            ]
            return send, recv, transfers

        self.empty_seed = allocate((block, hidden), torch.float32)
        self.local_seed = allocate((owned, hidden), torch.float32)
        self.dx = allocate((owned, hidden), torch.float32)
        self.dr = allocate((owned, (topk + 7) // 8 * 8), torch.float32)
        self.send, self.recv, self.dispatch = packets_pair((
            ("expert_ids", (world, header), torch.int32),
            ("route_factors", (world, header), torch.float32),
            ("activations", (packets, hidden), torch.bfloat16),
            ("gradients", (packets, hidden), torch.float32),
        ))
        self.return_send, self.return_recv, self.returned = packets_pair((
            ("input_vjp", (packets, hidden), torch.float32),
            ("route_vjp", (packets, 16), torch.float32),
        ))

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.workers)

    def run_layer(self, stage, *, owner, layer, rows, weights, student=None, gradient=None):
        """All hosts enter this call; only the node holding the request has the tape."""
        from .dsv41_packed_ep_reverse import PackedExpertGradients

        local_owner = self.global_ranks.index(owner) if owner in self.global_ranks else -1
        if local_owner >= 0:
            if (student is None or gradient is None or gradient.owner != local_owner
                    or gradient.layer != layer or gradient.rows != rows):
                raise ValueError("The local owner must supply its saved FFN tape and gradient")
        elif student is not None or gradient is not None:
            raise ValueError("Remote expert ranks receive their gradients through EP")
        tape = student.ffn_tape if student is not None else {}
        buffers = [self._stack(tuple(tape.get(rank, bank)[name] for rank, bank in enumerate(self.banks)))
                   for name in self.tape_names]
        values = list(self.empty_seed.shards)
        if gradient is not None:
            values[local_owner] = gradient.value
        self.epoch += 1
        slot = layer - self.first_layer

        start = perf_counter()
        with timed_stage("expert_vjp_pack"):
            self.runtime.run(self.pack, self._stack(tuple(values)), *buffers, *self.send,
                             self.local_seed, local_owner, rows, slot, self.epoch)
        self.transport.exchange_resident(f"{stage}/gradient-dispatch", self.dispatch)
        with timed_stage("expert_vjp_compute"):
            self.runtime.run(self.compute, *self.recv, *weights, *self.return_send, owner)
        self.transport.exchange_resident(f"{stage}/gradient-return", self.returned)
        if local_owner < 0:
            return None
        with timed_stage("expert_vjp_combine"):
            self.runtime.run(self.combine, buffers[1], *self.return_recv, self.dx, self.dr, local_owner, rows, slot)
        gradients = PackedExpertGradients(local_owner, layer, rows, self.dx, self.dr, perf_counter() - start)
        return StagedExpertBackward(self.local_seed, gradients)
