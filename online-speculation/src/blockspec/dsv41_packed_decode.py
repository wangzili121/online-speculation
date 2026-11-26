"""Resident Decode boundaries for the 1122 cross-host expert path."""

from .dsv41_online_profile import timed_stage
from .dsv41_packed_ep_graph import GraphResidentHcclMoE


def make_decode_ffn_tail(rows, local_ranks):
    """Apply native FFN hyper-connections after external EP dispatch and combine."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.hc import hc_ffn_decode_mixes
    from models.deepseek_v4_1_flash_a3.moe import prefill_ffn_restore
    from models.deepseek_v4_1_flash_a3.o_proj import retire_prefill_signals

    hidden, copies, mix = FLASH.hidden_size, FLASH.hc_mult, FLASH.mix_hc
    tp = TP_SIZE
    if local_ranks < tp or local_ranks % tp:
        raise ValueError("Decode FFN restore requires complete local TP groups")

    @pl.jit(auto_scope=False)
    def restore_rank(
        attention: pl.Tensor[[rows, copies, hidden], pl.FP32],
        owned: pl.Tensor[[rows, hidden], pl.BF16],
        function: pl.Tensor[[mix, copies * hidden], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[mix], pl.FP32],
        meta: pl.Tensor[[2], pl.INT32],
        output: pl.Tensor[[rows, copies, hidden], pl.FP32],
        pre_mix: pl.Tensor[[rows, copies], pl.FP32],
        window: pld.DistributedTensor[[rows, hidden], pl.BF16],
        arrived: pld.DistributedTensor[[tp, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        active = pl.read(meta, [0])
        post = pl.create_tensor([rows, copies], dtype=pl.FP32)
        residual = pl.create_tensor([rows, copies, copies], dtype=pl.FP32)
        mixed = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        if active == 1:
            hc_ffn_decode_mixes(attention, function, scale, base, pre_mix, post, residual, one_token=True)
        else:
            hc_ffn_decode_mixes(attention, function, scale, base, pre_mix, post, residual, one_token=False)
        prefill_ffn_restore(
            attention, owned, post, residual, window, arrived, mixed, output,
            rank // tp * tp, rank % tp, active, 1, prefill=False,
        )
        return retire_prefill_signals(output, arrived, rank // tp * tp, rank % tp)

    @pl.jit.host
    def restore(
        attention: pl.Tensor[[local_ranks, rows, copies, hidden], pl.FP32],
        owned: pl.Tensor[[local_ranks, rows, hidden], pl.BF16],
        function: pl.Tensor[[local_ranks, mix, copies * hidden], pl.FP32],
        scale: pl.Tensor[[local_ranks, 3], pl.FP32],
        base: pl.Tensor[[local_ranks, mix], pl.FP32],
        meta: pl.Tensor[[local_ranks, 2], pl.INT32],
        output: pl.Out[pl.Tensor[[local_ranks, rows, copies, hidden], pl.FP32]],
        pre_mix: pl.Out[pl.Tensor[[local_ranks, rows, copies], pl.FP32]],
        execute: pl.Scalar[pl.INT32],
    ):
        window_buffer = pld.alloc_window_buffer([rows, hidden], dtype=pl.BF16)
        signal_buffer = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            window = pld.window(window_buffer, [rows, hidden], dtype=pl.BF16)
            arrived = pld.window(signal_buffer, [tp, 1], dtype=pl.INT32)
            if execute != 0:
                restore_rank(attention[rank], owned[rank], function[rank], scale[rank], base[rank],
                             meta[rank], output[rank], pre_mix[rank], window, arrived, rank, device=rank)

    return restore


class ResidentDecodeFFN:
    """Run the FFN of a Decode layer using the attention and next-layer HBM buffers."""

    def __init__(self, moe, restore):
        self.moe = moe
        self.restore = restore

    def reserve_windows(self, attention, mixes, output, pre_mix):
        self._restore(attention, self.moe.owned, mixes, output, pre_mix, execute=0)

    def run_layer(self, stage, *, attention, ffn_input, shared_weights, routed_weights,
                  mixes, output, pre_mix):
        owned = self.moe.run(stage, prefill=False, ffn_input=ffn_input,
                             shared_weights=shared_weights, routed_weights=routed_weights)
        with timed_stage("decode_ffn_tp_mhc_post"):
            self._restore(attention, owned, mixes, output, pre_mix, execute=1)

    def _restore(self, attention, owned, mixes, output, pre_mix, *, execute):
        self.moe.runtime.run(self.restore, attention, owned, *mixes, self.moe.meta, output, pre_mix, execute)


def make_decode_final_head(local_ranks):
    """Compute the native vocabulary scores using each node's complete TP groups."""
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.deepseek_v4_1_flash_a3.config import FLASH, LOCAL_T_DYN, TP_SIZE
    from models.deepseek_v4_1_flash_a3.lm_head import (
        GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD, VOCAB_PER_TP, l2_final_output,
    )

    hidden, copies, vocab, tp = FLASH.hidden_size, FLASH.hc_mult, FLASH.vocab_size, TP_SIZE
    if local_ranks < tp or local_ranks % tp:
        raise ValueError("Decode logits require complete local TP groups")

    @pl.jit.host
    def head(
        x_hc: pl.Tensor[[local_ranks, LOCAL_T_DYN, copies, hidden], pl.FP32],
        pre_mix: pl.Tensor[[local_ranks, LOCAL_T_DYN, copies], pl.FP32],
        final_norm_weight: pl.Tensor[[local_ranks, hidden], pl.BF16],
        lm_head_weight: pl.Tensor[[local_ranks, VOCAB_PER_TP, hidden], pl.BF16],
        logit_row_indices: pl.Tensor[[local_ranks, MAX_LOGIT_ROWS], pl.INT32],
        logits: pl.Out[pl.Tensor[[local_ranks, MAX_LOGIT_ROWS, vocab], pl.FP32]],
        sampled_ids: pl.Out[pl.Tensor[[local_ranks, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
        execute_layers: pl.Scalar[pl.INT32],
    ):
        x_hc.bind_dynamic(1, LOCAL_T_DYN)
        hidden_buffer = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, hidden], dtype=pl.BF16)
        hidden_done_buffer = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        logits_buffer = pld.alloc_window_buffer([MAX_LOGIT_ROWS, vocab], dtype=pl.FP32)
        logits_done_buffer = pld.alloc_window_buffer([tp, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            hidden_window = pld.window(hidden_buffer, [GROUP_LOGIT_ROWS, hidden], dtype=pl.BF16)
            hidden_done = pld.window(hidden_done_buffer, [tp, 1], dtype=pl.INT32)
            logits_window = pld.window(logits_buffer, [MAX_LOGIT_ROWS, vocab], dtype=pl.FP32)
            logits_done = pld.window(logits_done_buffer, [tp, 1], dtype=pl.INT32)
            if execute_layers != 0:
                l2_final_output(
                    x_hc[rank], pre_mix[rank], final_norm_weight[rank], lm_head_weight[rank],
                    logit_row_indices[rank], logits[rank], sampled_ids[rank],
                    hidden_window, hidden_done, logits_window, logits_done,
                    rank // tp * tp, rank % tp, 1, 1, device=rank,
                )

    return head


def decode_attention_scratch(rows, table_pages):
    """Return the per-rank scratch shared across one Decode wave's attention stages."""
    import torch
    from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE, FLASH

    hidden, hc = FLASH.hidden_size, FLASH.hc_mult
    candidates = table_pages * (BLOCK_SIZE // FLASH.candidate_block_size)
    pool_width = min(FLASH.candidate_topk_blocks, max(32, (candidates + 31) // 32 * 32))
    return {
        "rope_rows": ((6, rows, FLASH.qk_rope_head_dim // 2), torch.float32),
        "compressed_lens": ((2, rows), torch.int32),
        "topk_indices": ((rows, FLASH.index_topk), torch.int32),
        "pool_block_ids": ((rows, pool_width), torch.int32),
        "pool_block_counts": ((rows,), torch.int32),
        "attention_hidden": ((rows, hc, hidden), torch.float32),
        "attention_pre_mix": ((rows, hc), torch.float32),
        "ffn_input": ((rows, hidden), torch.bfloat16),
    }


def resident_program_args(program, buffers):
    names = (parameter.name.removesuffix("__ssa_v0") for parameter in program._get_metadata()[0])
    return tuple(buffers[name] for name in names if name not in ("layer_index", "execute_layers"))


def resident_ffn_weights(runtime, buffers, layer, layer_count):
    """Borrow the native FFN checkpoint views for a layer on each local rank."""
    from pypto.runtime import StackedDeviceTensor

    def view(name):
        bank = buffers[name]
        local_shape = bank.shards[0].shape
        leading = local_shape[0] // layer_count
        # The NZ matrices each have a leading matrix axis. The standalone
        # MoE graphs take a single matrix per rank.
        matrix = name in ("shared_w13", "shared_w2")
        shape = local_shape[1:] if matrix else (leading, *local_shape[1:])
        shards = tuple(
            runtime.view_tensor(shard, shape, byte_offset=layer * (shard.nbytes // layer_count),
                                     worker_id=worker_id)
            for shard, worker_id in zip(bank.shards, bank.worker_ids, strict=True)
        )
        return StackedDeviceTensor(shards, (len(shards), *shape), bank.worker_ids)

    shared = tuple(view(name) for name in (
        "ffn_norm_weight", "gate_weight", "correction_bias", "shared_w13", "shared_w1_scale",
        "shared_w3_scale", "shared_w2", "shared_w2_scale",
    ))
    routed = tuple(view(f"routed_{name}") for name in (
        "w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale",
    ))
    mixes = tuple(view(f"hc_ffn_{name}") for name in ("fn", "scale", "base"))
    return shared, routed, mixes


class ResidentPackedDecode:
    """Schedule the 40 native Decode attention layers around external EP16."""

    def __init__(self, attention, ffn, head, buffers):
        from models.deepseek_v4_1_flash_a3.config import FLASH

        self.attention, self.ffn, self.head = attention, ffn, head
        self.runtime = ffn.moe.runtime
        self.buffers = dict(buffers, count=ffn.moe.meta)
        self.attention_arguments = resident_program_args(attention, self.buffers)
        self.head_arguments = resident_program_args(head, self.buffers)
        self.layer_count = FLASH.num_hidden_layers
        self.layer_weights = tuple(resident_ffn_weights(self.runtime, self.buffers, layer, self.layer_count)
                                   for layer in range(self.layer_count))

    @classmethod
    def bind(cls, runtime, transport, *, attention, ep_programs, restore, head, buffers, tokens):
        """Allocate reusable layer scratch alongside the supplied resident model state."""
        from pypto.runtime import StackedDeviceTensor

        workers = buffers["x_hc"].worker_ids
        rows = buffers["x_hc"].shards[0].shape[0]
        table_pages = buffers["c1a_index_block_table"].shards[0].shape[-1]

        def scratch(shape, dtype):
            shards = [runtime.alloc_tensor(shape, dtype, worker_id=worker) for worker in workers]
            return StackedDeviceTensor(shards, (len(workers), *shape), workers)

        buffers = dict(buffers, **{name: scratch(shape, dtype)
                                 for name, (shape, dtype) in decode_attention_scratch(rows, table_pages).items()})
        moe = GraphResidentHcclMoE(runtime, transport, ep_programs, tokens=tokens, rows=rows)
        result = cls(attention, ResidentDecodeFFN(moe, restore), head, buffers)
        result.reserve_windows()
        return result

    def reserve_windows(self):
        self.runtime.run(self.attention, *self.attention_arguments, 0, 0)
        values = self.buffers
        self.ffn.reserve_windows(values["attention_hidden"], self.layer_weights[0][2],
                                 values["x_hc"], values["pre_mix"])
        self.runtime.run(self.head, *self.head_arguments, 0)

    def step(self, stage, active):
        """Advance one packed wave using caller-owned ABI inputs and caches."""
        self.ffn.moe.bind_batch(active)
        values = self.buffers
        for layer, (shared, routed, mixes) in enumerate(self.layer_weights):
            with timed_stage("decode_attention"):
                self.runtime.run(self.attention, *self.attention_arguments, layer, 1)
            self.ffn.run_layer(f"{stage}/layer-{layer}", attention=values["attention_hidden"],
                               ffn_input=values["ffn_input"], shared_weights=shared, routed_weights=routed,
                               mixes=mixes, output=values["x_hc"], pre_mix=values["pre_mix"])
        with timed_stage("decode_final_head"):
            self.runtime.run(self.head, *self.head_arguments, 1)
        return values["logits"], values["sampled_ids"]
