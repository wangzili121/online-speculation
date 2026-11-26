"""Execute resident 1122 CED Prefill around the shared HCCL expert exchange."""

from .dsv41_online_profile import timed_stage
from .dsv41_packed_decode import resident_ffn_weights, resident_program_args
from .dsv41_packed_ep_graph import GraphResidentHcclMoE


def prefill_slabs(capacity, tokens, tp):
    """Partition a Prefill layer into complete interleaved TP owner slabs."""
    rows = tokens * tp
    return tuple((start, min(rows, capacity - start)) for start in range(0, capacity, rows))


def make_prefill_ffn_restore(rows, local_ranks):
    """Combine EP owner rows inside each TP group and apply the native mHC post."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.moe import prefill_ffn_restore
    from models.deepseek_v4_1_flash_a3.o_proj import retire_prefill_signals

    hidden, copies, tp = FLASH.hidden_size, FLASH.hc_mult, TP_SIZE

    @pl.jit(auto_scope=False)
    def restore_rank(
        attention: pl.Tensor[[rows, copies, hidden], pl.FP32],
        owned: pl.Tensor[[rows, hidden], pl.BF16],
        post: pl.Tensor[[rows, copies], pl.FP32],
        residual: pl.Tensor[[rows, copies, copies], pl.FP32],
        meta: pl.Tensor[[2], pl.INT32],
        output: pl.Tensor[[rows, copies, hidden], pl.FP32],
        window: pld.DistributedTensor[[rows, hidden], pl.BF16],
        arrived: pld.DistributedTensor[[tp, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        mixed = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        active = pl.read(meta, [0])
        prefill_ffn_restore(
            attention, owned, post, residual, window, arrived, mixed, output,
            rank // tp * tp, rank % tp, active, 1, prefill=True,
        )
        return retire_prefill_signals(output, arrived, rank // tp * tp, rank % tp)

    @pl.jit.host
    def restore(
        attention: pl.Tensor[[local_ranks, rows, copies, hidden], pl.FP32],
        owned: pl.Tensor[[local_ranks, rows, hidden], pl.BF16],
        post: pl.Tensor[[local_ranks, rows, copies], pl.FP32],
        residual: pl.Tensor[[local_ranks, rows, copies, copies], pl.FP32],
        meta: pl.Tensor[[local_ranks, 2], pl.INT32],
        output: pl.Out[pl.Tensor[[local_ranks, rows, copies, hidden], pl.FP32]],
        execute_layers: pl.Scalar[pl.INT32],
    ):
        window_buffer = pld.alloc_window_buffer([rows, hidden], dtype=pl.BF16)
        signal_buffer = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            window = pld.window(window_buffer, [rows, hidden], dtype=pl.BF16)
            arrived = pld.window(signal_buffer, [tp, 1], dtype=pl.INT32)
            if execute_layers != 0:
                restore_rank(attention[rank], owned[rank], post[rank], residual[rank],
                             meta[rank], output[rank], window, arrived, rank, device=rank)

    return restore


class ResidentPackedPrefill:
    """Reuse weights, CED state and physical pages across a packed Prefill wave."""

    def __init__(self, runtime, transport, *, attention, experts, restores, head, buffers, capacity, tokens):
        import torch
        from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE, FLASH, TP_SIZE
        from pypto.runtime import StackedDeviceTensor

        self.runtime, self.attention, self.head = runtime, attention, head
        self.capacity = capacity
        widths = prefill_slabs(capacity, tokens, TP_SIZE)
        devices = buffers["embedding"].worker_ids
        hidden, copies = FLASH.hidden_size, FLASH.hc_mult
        pages = buffers["c1a_index_block_table"].shards[0].shape[-1]
        blocks = (pages + 1) * (BLOCK_SIZE // FLASH.candidate_block_size)
        pool = min(FLASH.candidate_topk_blocks, (blocks + 31) // 32 * 32)

        def alloc(shape, dtype):
            shards = [runtime.alloc_tensor(shape, dtype, worker_id=dev) for dev in devices]
            return StackedDeviceTensor(shards, (len(shards), *shape), devices)

        def rows_from(bank, start, rows):
            dims = bank.shards[0].shape
            shape = (rows, *dims[1:])
            shards = [runtime.view_tensor(shard, shape, worker_id=dev,
                                          byte_offset=start * (shard.nbytes // dims[0]))
                      for dev, shard in zip(devices, bank.shards, strict=True)]
            return StackedDeviceTensor(shards, (len(shards), *shape), devices)

        values = dict(
            buffers,
            hidden_bank=alloc((2 * capacity, copies, hidden), torch.float32),
            mix_bank=alloc((2 * capacity, copies), torch.float32),
            rope_cos_bank=alloc((4, capacity, FLASH.qk_rope_head_dim // 2), torch.float32),
            rope_sin_bank=alloc((4, capacity, FLASH.qk_rope_head_dim // 2), torch.float32),
            compressed_lengths=alloc((2, capacity), torch.int32),
            attn_hidden=alloc((capacity, copies, hidden), torch.float32),
            ffn_input=alloc((capacity, hidden), torch.bfloat16),
            ffn_post=alloc((capacity, copies), torch.float32),
            ffn_residual=alloc((capacity, copies, copies), torch.float32),
            topk_indices=alloc((capacity, FLASH.index_topk), torch.int32),
            pool_block_ids=alloc((capacity, pool), torch.int32),
            pool_block_counts=alloc((capacity,), torch.int32),
        )
        values["x_hc"] = rows_from(values["hidden_bank"], 0, capacity)
        values["pre_mix"] = rows_from(values["mix_bank"], 0, capacity)
        values["logit_row_indices"] = values["logit_rows"]
        self.attention_args = resident_program_args(attention, values)
        self.head_args = resident_program_args(head, values)
        self.logits, self.sampled_ids = values["logits"], values["sampled_ids"]
        self.ffn_weights = tuple(resident_ffn_weights(runtime, values, layer, FLASH.num_hidden_layers)[:2]
                                 for layer in range(FLASH.num_hidden_layers))
        self.moe = {
            width: GraphResidentHcclMoE(runtime, transport, {"prefill": experts[width]}, tokens=tokens, rows=width)
            for width in dict.fromkeys(rows for _, rows in widths)
        }
        self.slabs = []
        reserve_args = {}
        for begin, width in widths:
            attn, ffn, post, residual, output = (
                rows_from(values[name], begin, width)
                for name in ("attn_hidden", "ffn_input", "ffn_post", "ffn_residual", "hidden_bank")
            )
            moe = self.moe[width]
            restore_args = (attn, moe.owned, post, residual, moe.meta, output)
            self.slabs.append((begin, width, ffn, restores[width], restore_args))
            reserve_args.setdefault(width, restore_args)
        runtime.run(attention, *self.attention_args, 0, 0)
        for width, args in reserve_args.items():
            runtime.run(restores[width], *args, 0)
        runtime.run(head, *self.head_args, 0)

    def step(self, stage, encoder, decoder):
        """Advance one chunk with the counts supplied in its device metadata."""
        for layer, (shared, routed) in enumerate(self.ffn_weights):
            with timed_stage("prefill_attention"):
                self.runtime.run(self.attention, *self.attention_args, layer, 1)
            counts = encoder if layer < 20 else decoder
            for begin, width, ffn, restore, restore_args in self.slabs:
                moe = self.moe[width]
                moe.bind_batch(tuple(min(width, max(0, count - begin)) for count in counts))
                moe.run(f"{stage}/layer-{layer}/{begin}", prefill=True, ffn_input=ffn,
                        shared_weights=shared, routed_weights=routed)
                with timed_stage("prefill_ffn_tp_mhc_post"):
                    self.runtime.run(restore, *restore_args, 1)
        with timed_stage("prefill_final_head"):
            self.runtime.run(self.head, *self.head_args, 1)
        return self.logits, self.sampled_ids
