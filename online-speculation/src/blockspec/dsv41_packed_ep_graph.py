"""Bind the split native FFN to the node's existing PyPTO graph worker."""

from .dsv41_packed_ep_hccl import (
    make_ep_forward_experts,
    make_ep_forward_front,
    make_ep_forward_packets,
    make_ep_forward_tp_rows,
)
from .dsv41_online_profile import timed_stage
from .dsv41_packed_ep_ipc import ResidentHcclMappings


def make_hccl_ep_graphs(tokens, rows, local_ranks, *, prefill):
    """Build resident prepare, expert and restore graphs around the two host collectives."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.config import EP_SIZE, FLASH, TP_SIZE

    world, tp = EP_SIZE, TP_SIZE
    hidden, inter, topk = FLASH.hidden_size, FLASH.moe_intermediate_size, FLASH.num_experts_per_tok
    experts = FLASH.n_routed_experts
    local_experts = experts // world
    packets = world * tokens * topk
    counts_width = (local_experts + 15) // 16 * 16
    expert_rows = packets + local_experts * 16
    prepare_local = make_ep_forward_front(tokens, rows, prefill=prefill, inline=True)
    pack_local, unpack_local, _, combine_local = make_ep_forward_packets(
        tokens, hidden, world, local_experts, topk, inline=True,
    )
    expert_local = make_ep_forward_experts(tokens, world, local_experts, prefill=prefill,
                                          fuse_return=True, inline=True)
    scatter_local = make_ep_forward_tp_rows(tokens, hidden, tp, rows, inline=True)

    @pl.jit(auto_scope=False)
    def prepare_rank(
        ffn: pl.Tensor, norm: pl.Tensor, gate: pl.Tensor, bias: pl.Tensor,
        shared_w13: pl.Tensor, scale1: pl.Tensor, scale3: pl.Tensor,
        shared_w2: pl.Tensor, scale2: pl.Tensor, meta: pl.Tensor,
        normed: pl.Tensor, ids: pl.Tensor, weights: pl.Tensor, shared: pl.Tensor,
        send_counts: pl.Tensor, send_payload: pl.Tensor, send_aux: pl.Tensor,
    ):
        active = pl.read(meta, [0])
        lane = pl.read(meta, [1])
        live = pl.cast((active + tp - 1 - lane) // tp, pl.INT32)
        payload = pl.create_tensor([tokens, hidden], dtype=pl.INT8)
        scales = pl.create_tensor([1, tokens], dtype=pl.FP32)
        prepare_local(ffn, norm, gate, bias, shared_w13, scale1, scale3, shared_w2, scale2,
                      normed, payload, scales, ids, weights, shared, active, lane)
        return pack_local(payload, scales, ids, weights, send_counts, send_payload, send_aux, live)

    @pl.jit(auto_scope=False)
    def compute_rank(
        recv_counts: pl.Tensor, recv_payload: pl.Tensor, recv_aux: pl.Tensor,
        w1: pl.Tensor, s1: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor,
        w2: pl.Tensor, s2: pl.Tensor, output: pl.Tensor,
    ):
        compact_counts = pl.create_tensor([local_experts, 2], dtype=pl.INT32)
        compact_x = pl.create_tensor([expert_rows, hidden], dtype=pl.INT8)
        compact_scale = pl.create_tensor([1, expert_rows], dtype=pl.FP32)
        compact_weight = pl.create_tensor([1, expert_rows], dtype=pl.FP32)
        compact_route = pl.create_tensor([1, expert_rows], dtype=pl.INT32)
        unpack_local(recv_counts, recv_payload, recv_aux, compact_counts,
                     compact_x, compact_scale, compact_weight, compact_route)
        return expert_local(compact_counts, compact_x, compact_scale, compact_weight,
                            compact_route, w1, s1, w3, s3, w2, s2, output)

    @pl.jit(auto_scope=False)
    def restore_rank(received: pl.Tensor, ids: pl.Tensor, shared: pl.Tensor,
                     meta: pl.Tensor, output: pl.Tensor):
        active = pl.read(meta, [0])
        lane = pl.read(meta, [1])
        live = pl.cast((active + tp - 1 - lane) // tp, pl.INT32)
        compact = pl.create_tensor([tokens, hidden], dtype=pl.BF16)
        combine_local(received, ids, shared, compact, live)
        return scatter_local(compact, output, active, lane)

    @pl.jit.host
    def prepare(
        ffn: pl.Tensor[[local_ranks, rows, hidden], pl.BF16],
        norm: pl.Tensor[[local_ranks, hidden], pl.BF16],
        gate: pl.Tensor[[local_ranks, experts, hidden], pl.FP32],
        bias: pl.Tensor[[local_ranks, experts], pl.FP32],
        shared_w13: pl.Tensor[[local_ranks, hidden, 2 * inter], pl.INT8, pl.NZ],
        scale1: pl.Tensor[[local_ranks, 1, inter], pl.FP32],
        scale3: pl.Tensor[[local_ranks, 1, inter], pl.FP32],
        shared_w2: pl.Tensor[[local_ranks, inter, hidden], pl.INT8, pl.NZ],
        scale2: pl.Tensor[[local_ranks, 1, hidden], pl.FP32],
        meta: pl.Tensor[[local_ranks, 2], pl.INT32],
        normed: pl.Out[pl.Tensor[[local_ranks, tokens, hidden], pl.BF16]],
        ids: pl.Out[pl.Tensor[[local_ranks, tokens, topk], pl.INT32]],
        weights: pl.Out[pl.Tensor[[local_ranks, tokens, topk], pl.FP32]],
        shared: pl.Out[pl.Tensor[[local_ranks, tokens, hidden], pl.BF16]],
        send_counts: pl.Out[pl.Tensor[[local_ranks, world, counts_width], pl.INT32]],
        send_payload: pl.Out[pl.Tensor[[local_ranks, packets, hidden], pl.INT8]],
        send_aux: pl.Out[pl.Tensor[[local_ranks, packets, 8], pl.FP32]],
    ):
        for rank in pl.range(pld.world_size()):
            prepare_rank(ffn[rank], norm[rank], gate[rank], bias[rank], shared_w13[rank],
                         scale1[rank], scale3[rank], shared_w2[rank], scale2[rank], meta[rank],
                         normed[rank], ids[rank], weights[rank], shared[rank],
                         send_counts[rank], send_payload[rank], send_aux[rank], device=rank)

    @pl.jit.host
    def compute(
        recv_counts: pl.Tensor[[local_ranks, world, counts_width], pl.INT32],
        recv_payload: pl.Tensor[[local_ranks, packets, hidden], pl.INT8],
        recv_aux: pl.Tensor[[local_ranks, packets, 8], pl.FP32],
        w1: pl.Tensor[[local_ranks, local_experts, inter, hidden], pl.INT8],
        s1: pl.Tensor[[local_ranks, local_experts, inter], pl.FP32],
        w3: pl.Tensor[[local_ranks, local_experts, inter, hidden], pl.INT8],
        s3: pl.Tensor[[local_ranks, local_experts, inter], pl.FP32],
        w2: pl.Tensor[[local_ranks, local_experts, hidden, inter], pl.INT8],
        s2: pl.Tensor[[local_ranks, local_experts, hidden], pl.FP32],
        output: pl.Out[pl.Tensor[[local_ranks, packets, hidden], pl.BF16]],
    ):
        for rank in pl.range(pld.world_size()):
            compute_rank(recv_counts[rank], recv_payload[rank], recv_aux[rank], w1[rank], s1[rank],
                         w3[rank], s3[rank], w2[rank], s2[rank], output[rank], device=rank)

    @pl.jit.host
    def restore(
        received: pl.Tensor[[local_ranks, packets, hidden], pl.BF16],
        ids: pl.Tensor[[local_ranks, tokens, topk], pl.INT32],
        shared: pl.Tensor[[local_ranks, tokens, hidden], pl.BF16],
        meta: pl.Tensor[[local_ranks, 2], pl.INT32],
        output: pl.Out[pl.Tensor[[local_ranks, rows, hidden], pl.BF16]],
    ):
        for rank in pl.range(pld.world_size()):
            restore_rank(received[rank], ids[rank], shared[rank], meta[rank], output[rank], device=rank)

    return prepare, compute, restore


class GraphResidentHcclMoE:
    """Execute FFN stages on the native graph worker with mapped HCCL buffers."""

    def __init__(self, runtime, transport, programs, *, tokens, rows):
        import torch
        from models.deepseek_v4_1_flash_a3.config import FLASH
        from pypto.runtime import StackedDeviceTensor

        self.runtime, self.transport, self.programs = runtime, transport, programs
        self.rows = rows
        plan = transport.plan
        self.global_ranks = tuple(plan.local_ranks)
        ranks = tuple(range(len(self.global_ranks)))
        world = sum(plan.counts)
        hidden, topk = FLASH.hidden_size, FLASH.num_experts_per_tok
        local_experts = FLASH.n_routed_experts // world
        packets = world * tokens * topk
        count_width = (local_experts + 15) // 16 * 16
        if transport.mappings is None:
            transport.mappings = ResidentHcclMappings(runtime, transport.memory)
        if transport.mappings.runtime is not runtime:
            raise ValueError("The HCCL and MoE buffers must share the same graph owner")
        self.mappings = transport.mappings

        def allocate(shape, dtype, *, exportable=False):
            shards = [runtime.alloc_tensor(shape, dtype, worker_id=rank, exportable=exportable) for rank in ranks]
            if exportable:
                for rank, shard in enumerate(shards):
                    self.mappings.register(rank, shard)
            return StackedDeviceTensor(shards, (len(ranks), *shape), ranks)

        self.meta = allocate((2,), torch.int32)
        self.normed = allocate((tokens, hidden), torch.bfloat16)
        self.ids = allocate((tokens, topk), torch.int32)
        self.weights = allocate((tokens, topk), torch.float32)
        self.shared = allocate((tokens, hidden), torch.bfloat16)
        self.owned = allocate((rows, hidden), torch.bfloat16)

        def packets_for_dispatch():
            return (
                allocate((world, count_width), torch.int32, exportable=True),
                allocate((packets, hidden), torch.int8, exportable=True),
                allocate((packets, 8), torch.float32, exportable=True),
            )

        self.dispatch_send, self.dispatch_recv = packets_for_dispatch(), packets_for_dispatch()
        shape = (packets, hidden)
        self.return_send = allocate(shape, torch.bfloat16, exportable=True)
        self.return_recv = allocate(shape, torch.bfloat16, exportable=True)

        def transfers(send, recv, names):
            return [tuple((a.shards[rank], b.shards[rank], name)
                          for a, b, name in zip(send, recv, names, strict=True)) for rank in ranks]

        self.dispatch = transfers(self.dispatch_send, self.dispatch_recv, ("counts", "hidden", "aux"))
        self.returned = transfers((self.return_send,), (self.return_recv,), ("expert_output",))
        self.active = None

    def bind_batch(self, active):
        """Install the owner token counts once before running the batch's layers."""
        import torch

        counts = tuple(active)
        tp = self.transport.plan.tp
        dp = sum(self.transport.plan.counts) // tp
        if len(counts) != dp or any(not 0 <= value <= self.rows for value in counts):
            raise ValueError("Pass the active token count of each global DP owner")
        if counts != self.active:
            for rank, global_rank in enumerate(self.global_ranks):
                owner, lane = divmod(global_rank, tp)
                metadata = torch.tensor([counts[owner], lane], dtype=torch.int32)
                self.runtime.copy_to(self.meta.shards[rank].data_ptr, metadata.data_ptr(),
                                     metadata.nbytes, worker_id=rank)
            self.active = counts

    def run(self, stage, *, prefill, ffn_input, shared_weights, routed_weights):
        """Deliver resident owner rows for the native TP reduction and mHC post."""
        if self.active is None:
            raise ValueError("Bind owner counts before executing FFN")
        prepare, experts, restore = self.programs["prefill" if prefill else "decode"]
        with timed_stage("ep_prepare_routes"):
            self.runtime.run(prepare, ffn_input, *shared_weights, self.meta,
                             self.normed, self.ids, self.weights, self.shared, *self.dispatch_send)
        self.transport.exchange_resident(f"{stage}/dispatch", self.dispatch)
        with timed_stage("ep_expert_compute"):
            self.runtime.run(experts, *self.dispatch_recv, *routed_weights, self.return_send)
        self.transport.exchange_resident(f"{stage}/return", self.returned)
        with timed_stage("ep_restore"):
            self.runtime.run(restore, self.return_recv, self.ids, self.shared, self.meta, self.owned)
        return self.owned
