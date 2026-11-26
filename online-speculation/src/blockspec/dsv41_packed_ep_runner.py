"""Execute one native 1122 FFN slab with resident device buffers and global HCCL."""

from .dsv41_online_profile import timed_stage
from .dsv41_packed_ep_hccl import (
    make_ep_forward_experts,
    make_ep_forward_front,
    make_ep_forward_packets,
    make_ep_forward_tp_rows,
)


def compile_hccl_moe(config, *, tokens, rows):
    """Compile both layer paths against the local worker's native A3 geometry."""
    from models.deepseek_v4_1_flash_a3.config import EP_SIZE, FLASH, TP_SIZE

    hidden, topk = FLASH.hidden_size, FLASH.num_experts_per_tok
    local = FLASH.n_routed_experts // EP_SIZE
    pack, unpack, _, combine = make_ep_forward_packets(tokens, hidden, EP_SIZE, local, topk)
    factories = dict(pack=pack, unpack=unpack, combine=combine,
                     scatter=make_ep_forward_tp_rows(tokens, hidden, TP_SIZE, rows))
    for phase in ("prefill", "decode"):
        prefill = phase == "prefill"
        factories[f"{phase}_front"] = make_ep_forward_front(tokens, rows, prefill=prefill)
        factories[f"{phase}_expert"] = make_ep_forward_experts(
            tokens, EP_SIZE, local, prefill=prefill, fuse_return=True,
        )
    return {name: factory.compile(config=config) for name, factory in factories.items()}


class ResidentHcclMoE:
    """Keep an EP slab and both AllToAll buffers resident on each local worker."""

    def __init__(self, workers, configs, transport, *, tokens, rows):
        import torch
        from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE

        self.plan = transport.plan
        self.workers, self.configs, self.transport = tuple(workers), tuple(configs), transport
        self.rows = rows
        tp, world = self.plan.tp, sum(self.plan.counts)
        if tp != TP_SIZE or world != 16 or len(self.workers) != len(self.plan.local_ranks):
            raise ValueError("The native HCCL FFN slab requires the local ranks of DP4/TP4/EP16")
        if len(self.configs) != len(self.workers) or not 0 < rows <= tokens * tp:
            raise ValueError("Each worker needs a config and the slab must fit its TP token rows")
        hidden, topk, local = FLASH.hidden_size, FLASH.num_experts_per_tok, FLASH.n_routed_experts // world
        packets = world * tokens * topk
        expert_rows = packets + local * 16
        count_width = (local + 15) // 16 * 16
        fp32, bf16, i8, i32 = torch.float32, torch.bfloat16, torch.int8, torch.int32

        def allocate(*specs):
            return [tuple(worker.alloc_tensor(shape, dtype) for shape, dtype in specs) for worker in self.workers]

        self.front = allocate(((tokens, hidden), bf16), ((tokens, hidden), i8), ((1, tokens), fp32),
                              ((tokens, topk), i32), ((tokens, topk), fp32), ((tokens, hidden), bf16))
        dispatch = (((world, count_width), i32), ((packets, hidden), i8), ((packets, 8), fp32))
        self.dispatch_send, self.dispatch_recv = allocate(*dispatch), allocate(*dispatch)
        self.expert_input = allocate(((local, 2), i32), ((expert_rows, hidden), i8), ((1, expert_rows), fp32),
                                     ((1, expert_rows), fp32), ((1, expert_rows), i32))
        self.return_send = allocate(((packets, hidden), bf16))
        self.return_recv = allocate(((packets, hidden), bf16))
        self.restored = allocate(((tokens, hidden), bf16), ((rows, hidden), bf16))
        self.programs = compile_hccl_moe(self.configs[0], tokens=tokens, rows=rows)
        self.dispatch_transfers = self._transfers(self.dispatch_send, self.dispatch_recv, ("counts", "hidden", "aux"))
        self.return_transfers = self._transfers(self.return_send, self.return_recv, ("expert_output",))

    @staticmethod
    def _transfers(source, destination, names):
        return [tuple(zip(send, receive, names, strict=True))
                for send, receive in zip(source, destination, strict=True)]

    def run(self, stage, *, prefill, active, ffn_inputs, shared_weights, routed_weights):
        """Return interleaved BF16 owner rows for the native TP reduction and mHC post."""
        phase = "prefill" if prefill else "decode"
        tp = self.plan.tp
        ranks = tuple(self.plan.local_ranks)
        if len(active) != sum(self.plan.counts) // tp or any(not 0 <= value <= self.rows for value in active):
            raise ValueError("Pass the global active token count of each DP owner")
        if not len(ffn_inputs) == len(shared_weights) == len(routed_weights) == len(ranks):
            raise ValueError("Pass input and weight views for every local rank")
        counts = [active[rank // tp] for rank in ranks]
        lanes = [rank % tp for rank in ranks]
        live = [(count + tp - 1 - lane) // tp for count, lane in zip(counts, lanes, strict=True)]
        map_ranks = self.transport.runtime.map_ranks

        def launch(rank, task, *arguments):
            self.workers[rank].run(self.programs[task], *arguments, config=self.configs[rank])

        def prepare(rank):
            front = self.front[rank]
            if live[rank]:
                launch(rank, f"{phase}_front", ffn_inputs[rank], *shared_weights[rank],
                       *front, counts[rank], lanes[rank])
            launch(rank, "pack", *front[1:5], *self.dispatch_send[rank], live[rank])

        def execute_experts(rank):
            # Idle source ranks still own and execute experts selected by other ranks.
            launch(rank, "unpack", *self.dispatch_recv[rank], *self.expert_input[rank])
            launch(rank, f"{phase}_expert", *self.expert_input[rank], *routed_weights[rank], *self.return_send[rank])

        def restore(rank):
            _, _, _, indices, _, shared = self.front[rank]
            compact, owned = self.restored[rank]
            if live[rank]:
                launch(rank, "combine", *self.return_recv[rank], indices, shared, compact, live[rank])
            launch(rank, "scatter", compact, owned, counts[rank], lanes[rank])
            return owned

        with timed_stage("ep_prepare_routes"):
            map_ranks(prepare)
        self.transport.exchange_resident(f"{stage}/dispatch", self.dispatch_transfers)
        with timed_stage("ep_expert_compute"):
            map_ranks(execute_experts)
        self.transport.exchange_resident(f"{stage}/return", self.return_transfers)
        with timed_stage("ep_restore"):
            return map_ranks(restore)
