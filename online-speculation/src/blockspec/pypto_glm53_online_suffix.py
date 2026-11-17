"""Backpropagate distributed GLM vocabulary distillation to the final mHC state."""

from .pypto_glm53_online_head import make_final_norm_backward, make_vocab_backward
from .pypto_glm53_online_shards import make_sharded_kl


def make_fp32_tp_sum(ranks, rows, width):
    """Replicate the summed activation gradients without BF16 rounding."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    tile = 512
    workers = 8
    columns = width // tile
    if ranks < 1 or rows < 1 or columns < 1 or width % tile:
        raise ValueError("invalid GLM tensor-parallel gradient geometry")

    @pl.jit(auto_scope=False)
    def reduce(
        partial: pl.Tensor[[rows, width], pl.FP32],
        output: pl.Out[pl.Tensor[[rows, width], pl.FP32]],
        exchange: pld.DistributedTensor[[rows, width], pl.FP32],
        arrived: pld.DistributedTensor[[ranks, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_grad_reuse", allow_early_resolve=False) as reusable:
            for peer in pl.range(ranks):
                pld.system.wait(arrived, offsets=[peer, 0], expected=2 * (epoch - 1), cmp=pld.WaitCmp.Ge)
        with pl.spmd(workers, name_hint="glm_grad_publish", deps=[reusable], allow_early_resolve=False) as published:
            lane = pl.tile.get_block_idx()
            for index in pl.range(lane, rows * columns, workers):
                row, col = index // columns, index % columns * tile
                pl.store(pl.load(partial, [row, col], [1, tile]), [row, col], exchange)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_grad_ready", deps=[published]) as ready:
            for peer in pl.range(ranks):
                pld.system.notify(arrived, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_grad_wait", deps=[ready], allow_early_resolve=False) as available:
            for peer in pl.range(ranks):
                pld.system.wait(arrived, offsets=[peer, 0], expected=2 * epoch - 1, cmp=pld.WaitCmp.Ge)
        with pl.spmd(workers, name_hint="glm_grad_sum", deps=[available]) as computed:
            lane = pl.tile.get_block_idx()
            for index in pl.range(lane, rows * columns, workers):
                row, col = index // columns, index % columns * tile
                total = pl.tile.full([1, tile], dtype=pl.FP32, value=0.0)
                for peer in pl.range(ranks):
                    value = pld.tile.remote_load(exchange, peer=peer, offsets=[row, col], shape=[1, tile])
                    total = pl.tile.add(total, value)
                output = pl.store(total, [row, col], output)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_grad_retire", deps=[computed]) as done:
            for peer in pl.range(ranks):
                pld.system.notify(arrived, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_grad_done", deps=[done], allow_early_resolve=False):
            for peer in pl.range(ranks):
                pld.system.wait(arrived, offsets=[peer, 0], expected=2 * epoch, cmp=pld.WaitCmp.Ge)
        return output

    return reduce


def make_distillation_suffix(ranks, rows, local_vocab, width, epsilon, streams=4):
    """One device dispatch from vocabulary shards to the final per-stream gradient."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    padded_vocab = (local_vocab + 63) // 64 * 64
    kl = make_sharded_kl(ranks, rows, local_vocab, device_entry=True)
    vocab = make_vocab_backward(local_vocab, width, rows, device_entry=True)
    reduce = make_fp32_tp_sum(ranks, rows, width)
    norm = make_final_norm_backward(width, epsilon, rows, streams, device_entry=True)

    @pl.jit.host
    def entry(
        student: pl.Tensor[[ranks, rows, padded_vocab], pl.FP32],
        teacher: pl.Tensor[[ranks, rows, padded_vocab], pl.FP32],
        weight: pl.Tensor[[ranks, padded_vocab, width], pl.BF16],
        activation: pl.Tensor[[ranks, rows, width], pl.BF16],
        scale: pl.Tensor[[ranks, width], pl.BF16],
        dlogit: pl.InOut[pl.Tensor[[ranks, rows, padded_vocab], pl.FP32]],
        local: pl.InOut[pl.Tensor[[ranks, rows, width], pl.FP32]],
        combined: pl.InOut[pl.Tensor[[ranks, rows, width], pl.FP32]],
        loss: pl.Out[pl.Tensor[[ranks, rows, 8], pl.FP32]],
        gradient: pl.Out[pl.Tensor[[ranks, rows * streams, width], pl.FP32]],
        valid: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        stats_buffer = pld.alloc_window_buffer([rows, 32], dtype=pl.FP32)
        stats_ready_buffer = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        hidden_buffer = pld.alloc_window_buffer([rows, width], dtype=pl.FP32)
        hidden_ready_buffer = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            stats = pld.window(stats_buffer, [rows, 32], dtype=pl.FP32)
            stats_ready = pld.window(stats_ready_buffer, [ranks, 1], dtype=pl.INT32)
            hidden = pld.window(hidden_buffer, [rows, width], dtype=pl.FP32)
            hidden_ready = pld.window(hidden_ready_buffer, [ranks, 1], dtype=pl.INT32)
            dlogit_rank = dlogit[rank]
            loss_rank = loss[rank]
            local_rank = local[rank]
            combined_rank = combined[rank]
            # Keep explicit outputs before the mutable communication windows in each L2 signature.
            dlogit_out, _ = kl(student[rank], teacher[rank], dlogit_rank, loss_rank, stats, stats_ready,
                               valid, rank, epoch, device=rank)
            local_out = vocab(dlogit_out, weight[rank], local_rank, device=rank)
            combined_out = reduce(local_out, combined_rank, hidden, hidden_ready, rank, epoch, device=rank)
            norm(activation[rank], scale[rank], combined_out, gradient[rank], device=rank)

    return entry
