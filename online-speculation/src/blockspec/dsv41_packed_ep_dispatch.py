"""Route saved draft rows and unweighted FFN gradients to their expert owners."""

from dataclasses import dataclass
from time import perf_counter

from .dsv41_packed_ffn_tape import packed_ffn_tape_shapes


def make_packed_ep_dispatch(block, last_layers, hidden, copies, topk, experts, dp, tp, *, fuse_tp=True):
    import pypto.language as pl
    import pypto.language.distributed as pld

    world, local, packets = dp * tp, block // tp, block * topk
    local_experts = experts // world
    shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    vector_shape = shapes["ffn_tape_vectors"][0]
    route_shape = shapes["ffn_tape_route_ids"][0]

    def scatter_tp(
        value: pl.Tensor[[block, hidden], pl.FP32],
        grad_window: pld.DistributedTensor[[local, hidden], pl.FP32],
        grad_ready: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(block, name_hint="orthrus_moe_tp_gradient_send") as sent:
            row = pl.tile.get_block_idx()
            if row < active:
                pld.tensor.put(dst=grad_window, peer=owner + row % tp, src=value,
                               dst_offsets=[row // tp, 0], src_offsets=[row, 0], shape=[1, hidden])
        with pl.at(level=pl.Level.CORE_GROUP, deps=[sent], allow_early_resolve=False):
            for lane in pl.range(tp):
                pld.system.notify(target=grad_ready, peer=owner + lane, offsets=[0, 0],
                                  value=epoch, op=pld.NotifyOp.Set)

    def dispatch_ep(
        vectors: pl.Tensor[vector_shape, pl.BF16],
        ids: pl.Tensor[route_shape, pl.INT32],
        weights: pl.Tensor[route_shape, pl.FP32],
        out_metadata: pl.Out[pl.Tensor[[packets, 16], pl.INT32]],
        out_weights: pl.Out[pl.Tensor[[packets, 16], pl.FP32]],
        out_x: pl.Out[pl.Tensor[[packets, hidden], pl.BF16]],
        out_dy: pl.Out[pl.Tensor[[packets, hidden], pl.FP32]],
        grad_window: pld.DistributedTensor[[local, hidden], pl.FP32],
        grad_ready: pld.DistributedTensor[[1, 1], pl.INT32],
        ep_metadata: pld.DistributedTensor[[packets, 16], pl.INT32],
        ep_weights: pld.DistributedTensor[[packets, 16], pl.FP32],
        ep_x: pld.DistributedTensor[[packets, hidden], pl.BF16],
        ep_dy: pld.DistributedTensor[[packets, hidden], pl.FP32],
        ep_ready: pld.DistributedTensor[[1, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], owner: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        if rank >= owner and rank < owner + tp:
            lane = rank - owner
            local_dy = pl.create_tensor([local, hidden], dtype=pl.FP32)
            headers = pl.create_tensor([local * topk, 16], dtype=pl.INT32)
            coefficients = pl.create_tensor([local * topk, 16], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as ready:
                pld.system.wait(signal=grad_ready, offsets=[0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
            with pl.spmd(local, deps=[ready], name_hint="orthrus_moe_route_header") as packed:
                row = pl.tile.get_block_idx()
                token = row * tp + lane
                if token < active:
                    local_dy[row:row + 1, :] = grad_window[row:row + 1, :]
                    for choice in pl.range(topk):
                        route = pl.read(ids, [slot * local + row, choice])
                        factor = pl.read(weights, [slot * local + row, choice])
                        header = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
                        coefficient = pl.tile.full([1, 16], dtype=pl.FP32, value=0.)
                        pl.tile.write(header, [0, 0], epoch)
                        pl.tile.write(header, [0, 1], pl.cast(route % local_experts, pl.INT32))
                        pl.tile.write(header, [0, 2], rank)
                        pl.tile.write(header, [0, 3], pl.cast(row, pl.INT32))
                        pl.tile.write(header, [0, 4], pl.cast(choice, pl.INT32))
                        pl.tile.write(coefficient, [0, 0], factor)
                        pl.store(header, [row * topk + choice, 0], headers)
                        pl.store(coefficient, [row * topk + choice, 0], coefficients)

            with pl.spmd(local, deps=[packed], name_hint="orthrus_moe_ep_packet_send") as sent:
                row = pl.tile.get_block_idx()
                token = row * tp + lane
                if token < active:
                    vector_row = slot * (2 * block + local) + 2 * block + row
                    for choice in pl.range(topk):
                        route = pl.read(ids, [slot * local + row, choice])
                        peer = route // local_experts
                        packet, sending = token * topk + choice, row * topk + choice
                        pld.tensor.put(dst=ep_x, peer=peer, src=vectors,
                                       dst_offsets=[packet, 0], src_offsets=[vector_row, 0], shape=[1, hidden])
                        pld.tensor.put(dst=ep_dy, peer=peer, src=local_dy,
                                       dst_offsets=[packet, 0], src_offsets=[row, 0], shape=[1, hidden])
                        pld.tensor.put(dst=ep_weights, peer=peer, src=coefficients,
                                       dst_offsets=[packet, 0], src_offsets=[sending, 0], shape=[1, 16])
                        pld.tensor.put(dst=ep_metadata, peer=peer, src=headers,
                                       dst_offsets=[packet, 0], src_offsets=[sending, 0], shape=[1, 16])
            with pl.at(level=pl.Level.CORE_GROUP, deps=[sent], allow_early_resolve=False):
                for peer in pl.range(world):
                    pld.system.notify(target=ep_ready, peer=peer, offsets=[0, 0],
                                      value=1, op=pld.NotifyOp.AtomicAdd)

        with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as available:
            pld.system.wait(signal=ep_ready, offsets=[0, 0], expected=epoch * tp, cmp=pld.WaitCmp.Ge)
        with pl.spmd(block, deps=[available], name_hint="orthrus_moe_ep_packet_receive"):
            row = pl.tile.get_block_idx()
            for choice in pl.range(topk):
                packet = row * topk + choice
                zero_header = pl.full([1, 16], dtype=pl.INT32, value=0)
                if row < active and pl.read(ep_metadata, [packet, 0]) == epoch:
                    out_metadata[packet:packet + 1, :] = ep_metadata[packet:packet + 1, :]
                    out_weights[packet:packet + 1, :] = ep_weights[packet:packet + 1, :]
                    out_x[packet:packet + 1, :] = ep_x[packet:packet + 1, :]
                    out_dy[packet:packet + 1, :] = ep_dy[packet:packet + 1, :]
                else:
                    out_metadata[packet:packet + 1, :] = zero_header
        return out_metadata, out_weights, out_x, out_dy

    tp_send = pl.jit(scatter_tp)
    tp_inline = pl.jit.inline(auto_scope=False)(scatter_tp)
    ep_send = pl.jit(dispatch_ep)
    ep_inline = pl.jit.inline(auto_scope=False)(dispatch_ep)

    @pl.jit
    def owner_dispatch(
        value: pl.Tensor[[block, hidden], pl.FP32],
        vectors: pl.Tensor[vector_shape, pl.BF16],
        ids: pl.Tensor[route_shape, pl.INT32],
        weights: pl.Tensor[route_shape, pl.FP32],
        out_metadata: pl.Out[pl.Tensor[[packets, 16], pl.INT32]],
        out_weights: pl.Out[pl.Tensor[[packets, 16], pl.FP32]],
        out_x: pl.Out[pl.Tensor[[packets, hidden], pl.BF16]],
        out_dy: pl.Out[pl.Tensor[[packets, hidden], pl.FP32]],
        grad_window: pld.DistributedTensor[[local, hidden], pl.FP32],
        grad_ready: pld.DistributedTensor[[1, 1], pl.INT32],
        ep_metadata: pld.DistributedTensor[[packets, 16], pl.INT32],
        ep_weights: pld.DistributedTensor[[packets, 16], pl.FP32],
        ep_x: pld.DistributedTensor[[packets, hidden], pl.BF16],
        ep_dy: pld.DistributedTensor[[packets, hidden], pl.FP32],
        ep_ready: pld.DistributedTensor[[1, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], owner: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        tp_inline(value, grad_window, grad_ready, owner, active, epoch)
        return ep_inline(vectors, ids, weights, out_metadata, out_weights, out_x, out_dy,
                         grad_window, grad_ready, ep_metadata, ep_weights, ep_x, ep_dy,
                         ep_ready, rank, owner, active, epoch, slot)

    @pl.jit.host
    def dispatch(
        value: pl.Tensor[[dp, block, hidden], pl.FP32],
        vectors: pl.Tensor[[world, *vector_shape], pl.BF16],
        ids: pl.Tensor[[world, *route_shape], pl.INT32],
        weights: pl.Tensor[[world, *route_shape], pl.FP32],
        out_metadata: pl.Out[pl.Tensor[[world, packets, 16], pl.INT32]],
        out_weights: pl.Out[pl.Tensor[[world, packets, 16], pl.FP32]],
        out_x: pl.Out[pl.Tensor[[world, packets, hidden], pl.BF16]],
        out_dy: pl.Out[pl.Tensor[[world, packets, hidden], pl.FP32]],
        owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        grad_buf = pld.alloc_window_buffer([local, hidden], dtype=pl.FP32)
        grad_ready_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        ep_metadata_buf = pld.alloc_window_buffer([packets, 16], dtype=pl.INT32)
        ep_weights_buf = pld.alloc_window_buffer([packets, 16], dtype=pl.FP32)
        ep_x_buf = pld.alloc_window_buffer([packets, hidden], dtype=pl.BF16)
        ep_dy_buf = pld.alloc_window_buffer([packets, hidden], dtype=pl.FP32)
        ep_ready_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            grad_window = pld.window(grad_buf, [local, hidden], dtype=pl.FP32)
            grad_ready = pld.window(grad_ready_buf, [1, 1], dtype=pl.INT32)
            ep_metadata = pld.window(ep_metadata_buf, [packets, 16], dtype=pl.INT32)
            ep_weights = pld.window(ep_weights_buf, [packets, 16], dtype=pl.FP32)
            ep_x = pld.window(ep_x_buf, [packets, hidden], dtype=pl.BF16)
            ep_dy = pld.window(ep_dy_buf, [packets, hidden], dtype=pl.FP32)
            ep_ready = pld.window(ep_ready_buf, [1, 1], dtype=pl.INT32)
            if fuse_tp:
                if rank == owner:
                    replica = rank // tp
                    owner_dispatch(value[replica], vectors[rank], ids[rank], weights[rank],
                                   out_metadata[rank], out_weights[rank], out_x[rank], out_dy[rank],
                                   grad_window, grad_ready, ep_metadata, ep_weights, ep_x, ep_dy,
                                   ep_ready, rank, owner, active, epoch, slot, device=rank)
                else:
                    ep_send(vectors[rank], ids[rank], weights[rank], out_metadata[rank], out_weights[rank],
                            out_x[rank], out_dy[rank], grad_window, grad_ready, ep_metadata, ep_weights, ep_x, ep_dy,
                            ep_ready, rank, owner, active, epoch, slot, device=rank)
            else:
                if rank == owner:
                    replica = rank // tp
                    tp_send(value[replica], grad_window, grad_ready, owner, active, epoch, device=rank)
                ep_send(vectors[rank], ids[rank], weights[rank], out_metadata[rank], out_weights[rank],
                        out_x[rank], out_dy[rank], grad_window, grad_ready, ep_metadata, ep_weights, ep_x, ep_dy,
                        ep_ready, rank, owner, active, epoch, slot, device=rank)

    return dispatch


@dataclass(frozen=True)
class PackedExpertPackets:
    """Output shards indexed by EP rank; only rows with matching metadata epoch are live."""

    layer: int
    epoch: int
    metadata: object
    weights: object
    input: object
    gradient: object
    seconds: float


class PackedExpertDispatch:
    def __init__(self, proposer, post_vjp, graph, topk, hidden):
        import torch
        from pypto.runtime import StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.post = post_vjp
        self.epoch = 0
        self.ranks = tuple(range(proposer.world))
        self.stacked = StackedDeviceTensor
        self.inputs = {
            name: [proposer.ranks[rank][name] for rank in self.ranks]
            for name in ("ffn_tape_vectors", "ffn_tape_route_ids", "ffn_tape_route_weights")
        }
        packets = proposer.block * topk

        def allocate(shape, dtype):
            return self._stack([self.runtime.alloc_tensor(shape, dtype, worker_id=rank) for rank in self.ranks])

        self.outputs = (allocate((packets, 16), torch.int32), allocate((packets, 16), torch.float32),
                        allocate((packets, hidden), torch.bfloat16), allocate((packets, hidden), torch.float32))

    def _stack(self, shards):
        return self.stacked(shards, (len(shards), *shards[0].shape), self.ranks)

    def __call__(self, student, gradient):
        owner = student.owner
        replica = self.post.owners.index(owner)
        if student.ffn_tape is None or gradient.owner != owner or gradient.value is not self.post.outputs[0].shards[replica]:
            raise ValueError("Expert dispatch requires the live tape and FFN output gradient")
        buffers = []
        for name, template in self.inputs.items():
            values = list(template)
            for rank, tape in student.ffn_tape.items():
                values[rank] = tape[name]
            buffers.append(self._stack(values))
        self.epoch += 1
        start = perf_counter()
        self.runtime.run(self.graph, self.post.outputs[0], *buffers, *self.outputs, owner, gradient.rows,
                         self.epoch, gradient.layer - self.post.first_layer, config=self.config)
        return PackedExpertPackets(gradient.layer, self.epoch, *self.outputs, perf_counter() - start)
