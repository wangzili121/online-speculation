"""Finish the packed MoE VJP and return the collapsed FFN gradient to its owner."""

from dataclasses import dataclass
from math import prod
from time import perf_counter

from .dsv41_packed_ffn_tape import packed_ffn_tape_shapes
from .dsv41_packed_router_vjp import make_packed_router_vjp
from .dsv41_packed_shared_expert import make_packed_shared_expert_vjp


def make_packed_ffn_input_vjp(block, last_layers, model, dp, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld

    hidden, inter, copies = model.hidden_size, model.moe_intermediate_size, model.hc_mult
    topk, experts = model.num_experts_per_tok, model.n_routed_experts
    epsilon = model.rms_norm_eps
    world, local = dp * tp, block // tp
    padded = max(16, local)
    route_pad = (topk + 7) // 8 * 8
    tape = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    vectors_shape = tape["ffn_tape_vectors"][0]
    routes_shape = tape["ffn_tape_route_ids"][0]
    gate_shape = (2 * inter // 32 * hidden, 32)
    down_shape = (hidden // 32 * inter, 32)
    shared_vjp = make_packed_shared_expert_vjp(padded, hidden, inter, model.swiglu_limit, inline=True)
    router_vjp = make_packed_router_vjp(local, hidden, experts, topk, model.gate_temperature,
                                        model.routed_scaling_factor, inline=True)

    @pl.jit
    def finish(
        tape_vectors: pl.Tensor[vectors_shape, pl.BF16],
        tape_ids: pl.Tensor[routes_shape, pl.INT32],
        dy: pl.Tensor[[local, hidden], pl.FP32],
        routed_dx: pl.Tensor[[local, hidden], pl.FP32],
        droutes: pl.Tensor[[local, route_pad], pl.FP32],
        shared_w13: pl.Tensor[gate_shape, pl.INT8],
        shared_s1: pl.Tensor[[1, inter], pl.FP32],
        shared_s3: pl.Tensor[[1, inter], pl.FP32],
        shared_w2: pl.Tensor[down_shape, pl.INT8],
        shared_s2: pl.Tensor[[1, hidden], pl.FP32],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        norm: pl.Tensor[[1, hidden], pl.BF16],
        gradient: pl.Out[pl.Tensor[[block, hidden], pl.FP32]],
        window: pld.DistributedTensor[[block, hidden], pl.FP32],
        signal: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], rank: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        lane = rank - owner
        active_local = pl.cast(pl.max(0, (active + tp - 1 - lane) // tp), pl.INT32)
        tape_start = slot * (2 * block + local)
        normalized = pl.slice(tape_vectors, [local, hidden], [tape_start + 2 * block, 0])
        ids = pl.slice(tape_ids, [local, 16], [slot * local, 0])
        sends = pl.array.create(1, pl.TASK_ID)
        sends[0] = pl.system.task_dummy(deps=[])
        if active_local > 0:
            shared_x = pl.create_tensor([padded, hidden], dtype=pl.BF16)
            shared_dy = pl.create_tensor([padded, hidden], dtype=pl.FP32)
            shared_dx = pl.create_tensor([padded, hidden], dtype=pl.FP32)
            router_dx = pl.create_tensor([local, hidden], dtype=pl.FP32)
            with pl.spmd(padded, name_hint="orthrus_shared_expert_pad"):
                index = pl.tile.get_block_idx()
                if index < active_local:
                    shared_x[index:index + 1, :] = normalized[index:index + 1, :]
                    shared_dy[index:index + 1, :] = dy[index:index + 1, :]
                else:
                    shared_x[index:index + 1, :] = pl.full([1, hidden], dtype=pl.BF16, value=0.)
                    shared_dy[index:index + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
            shared_vjp(shared_x, shared_dy, shared_w13, shared_s1, shared_s3, shared_w2, shared_s2,
                       shared_dx, active_local)
            router_vjp(normalized, ids, droutes, gate_weight, routed_dx, router_dx, active_local)
            dx_local = pl.create_tensor([local, hidden], dtype=pl.FP32)
            with pl.spmd(active_local, name_hint="orthrus_ffn_norm_input_vjp"):
                row = pl.tile.get_block_idx()
                global_row = row * tp + lane
                collapsed = pl.slice(tape_vectors, [1, hidden], [tape_start + global_row, 0])
                x = pl.reshape(pl.cast(collapsed, pl.FP32), [8, hidden // 8])
                mixed_dx = pl.add(shared_dx[row:row + 1, :], router_dx[row:row + 1, :])
                scaled = pl.reshape(pl.mul(mixed_dx, pl.cast(norm, pl.FP32)), [8, hidden // 8])
                ones = pl.full([8, 8], dtype=pl.FP32, value=1.)
                squares = pl.row_expand_mul(ones, pl.row_sum(pl.mul(x, x)))
                inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.transpose(squares, 0, 1)), 1. / hidden), epsilon),
                               high_precision=True)
                partials = pl.row_expand_mul(ones, pl.row_sum(pl.mul(x, scaled)))
                dot = pl.row_sum(pl.transpose(partials, 0, 1))
                coefficient = pl.mul(pl.mul(pl.mul(inv, inv), inv), pl.mul(dot, 1. / hidden))
                dx = pl.sub(pl.row_expand_mul(scaled, inv), pl.row_expand_mul(x, coefficient))
                dx_local[row:row + 1, :] = pl.reshape(dx, [1, hidden])
            with pl.spmd(active_local, name_hint="orthrus_ffn_norm_gradient_return") as sent:
                sending = pl.tile.get_block_idx()
                pld.tensor.put(dst=window, peer=owner, src=dx_local,
                               dst_offsets=[sending * tp + lane, 0], src_offsets=[sending, 0], shape=[1, hidden])
            sends[0] = sent
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="orthrus_ffn_norm_gradient_ready", deps=[sends],
                   allow_early_resolve=False):
            pld.system.notify(target=signal, peer=owner, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        if rank == owner:
            with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as done:
                pld.system.wait(signal=signal, offsets=[0, 0], expected=epoch * tp, cmp=pld.WaitCmp.Ge)
            with pl.spmd(block, deps=[done], name_hint="orthrus_ffn_norm_gradient_collect"):
                out_row = pl.tile.get_block_idx()
                if out_row < active:
                    gradient[out_row:out_row + 1, :] = window[out_row:out_row + 1, :]
                else:
                    gradient[out_row:out_row + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
        return gradient

    @pl.jit.host
    def dispatch(
        vectors: pl.Tensor[[world, *vectors_shape], pl.BF16],
        ids: pl.Tensor[[world, *routes_shape], pl.INT32],
        dy: pl.Tensor[[world, local, hidden], pl.FP32],
        routed_dx: pl.Tensor[[world, local, hidden], pl.FP32],
        droutes: pl.Tensor[[world, local, route_pad], pl.FP32],
        shared_w13: pl.Tensor[[world, *gate_shape], pl.INT8],
        shared_s1: pl.Tensor[[world, 1, inter], pl.FP32],
        shared_s3: pl.Tensor[[world, 1, inter], pl.FP32],
        shared_w2: pl.Tensor[[world, *down_shape], pl.INT8],
        shared_s2: pl.Tensor[[world, 1, hidden], pl.FP32],
        gate_weight: pl.Tensor[[world, experts, hidden], pl.FP32],
        norm: pl.Tensor[[world, 1, hidden], pl.BF16],
        gradient: pl.Out[pl.Tensor[[world, block, hidden], pl.FP32]],
        owner: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32], slot: pl.Scalar[pl.INT32],
    ):
        returns = pld.alloc_window_buffer([block, hidden], dtype=pl.FP32)
        complete = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            if rank >= owner and rank < owner + tp:
                window = pld.window(returns, [block, hidden], dtype=pl.FP32)
                signal = pld.window(complete, [1, 1], dtype=pl.INT32)
                finish(vectors[rank], ids[rank], dy[rank], routed_dx[rank], droutes[rank], shared_w13[rank],
                       shared_s1[rank], shared_s3[rank], shared_w2[rank], shared_s2[rank], gate_weight[rank], norm[rank],
                       gradient[rank], window, signal, owner, rank, active, epoch, slot, device=rank)
    return dispatch


@dataclass(frozen=True)
class PackedFFNInputGradient:
    owner: int
    layer: int
    rows: int
    collapsed: object
    seconds: float


class PackedFFNInputVJP:
    def __init__(self, proposer, graph, banks, model):
        import torch
        from pypto.runtime import DeviceTensor, StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.ranks = tuple(range(proposer.world))
        self.owners = self.ranks[::proposer.tp]
        self.epochs = {owner: 0 for owner in self.owners}
        self.device_tensor, self.stacked = DeviceTensor, StackedDeviceTensor
        self.first_layer = model.num_hidden_layers - proposer.last_layers
        hidden, inter = model.hidden_size, model.moe_intermediate_size
        self.weights = tuple((shape, tuple(rank[key] for rank in banks)) for key, shape in (
            ("shared_w13", (2 * inter // 32 * hidden, 32)),
            ("shared_w1_scale", (1, inter)), ("shared_w3_scale", (1, inter)),
            ("shared_w2", (hidden // 32 * inter, 32)), ("shared_w2_scale", (1, hidden)),
            ("gate_weight", (model.n_routed_experts, hidden)), ("ffn_norm_weight", (1, hidden)),
        ))
        self.tape = {key: tuple(rank[key] for rank in proposer.ranks)
                     for key in ("ffn_tape_vectors", "ffn_tape_route_ids")}
        self.output = self._stack([self.runtime.alloc_tensor((proposer.block, hidden), torch.float32, worker_id=rank)
                                   for rank in self.ranks], self.ranks)

    def _stack(self, shards, ranks):
        return self.stacked(shards, (len(shards), *shards[0].shape), ranks)

    def _layer_weights(self, layer):
        for shape, shards in self.weights:
            views = []
            for bank in shards:
                offset = layer * prod(shape) * bank.dtype.itemsize
                views.append(self.device_tensor(bank.data_ptr + offset, shape, bank.dtype, buffer=bank.buffer,
                                                byte_offset=bank.byte_offset + offset))
            yield self._stack(views, self.ranks)

    def __call__(self, student, packets, gradients):
        owner, layer, rows = gradients.owner, gradients.layer, gradients.rows
        tape = tuple(self._stack([student.ffn_tape[rank][key] if rank in student.ffn_tape else fallback[rank]
                                 for rank in self.ranks], self.ranks)
                     for key, fallback in self.tape.items())
        self.epochs[owner] += 1
        start = perf_counter()
        self.runtime.run(self.graph, *tape, packets.local_gradient, gradients.input, gradients.route,
                         *self._layer_weights(layer), self.output, owner, rows, self.epochs[owner],
                         layer - self.first_layer, config=self.config)
        return PackedFFNInputGradient(owner, layer, rows, self.output.shards[owner],
                                      perf_counter() - start)
