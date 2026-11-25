"""Input VJP of the packed draft's frozen, TP-sharded vocabulary projection."""

from dataclasses import dataclass
from time import perf_counter

from .pypto_dsv41_online_programs import make_linear


def make_packed_lm_head_vjp(rows, vocabulary, hidden, dp, tp):
    import pypto.language as pl
    import pypto.language.distributed as pld

    world, shard = dp * tp, vocabulary // tp
    transfer_tile, hidden_tile = 2048, 256
    full_tiles, tail = divmod(shard, transfer_tile)
    tail_tile = max(tail, 32)
    transfer_blocks = min(24 // tp, full_tiles + bool(tail))
    project = make_linear((rows, shard), (shard, hidden), transpose_b=False,
                          inline=True, reduction_limit=256, output_fp32=True)

    @pl.jit.inline(auto_scope=False)
    def scatter(
        source: pl.Tensor[[rows, vocabulary], pl.BF16],
        window: pld.DistributedTensor[[rows, shard], pl.BF16],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], valid: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(tp * transfer_blocks, name_hint="orthrus_logits_vjp_send") as sent:
            block = pl.tile.get_block_idx()
            lane = block // transfer_blocks
            offset = block % transfer_blocks
            for row in pl.range(valid):
                for part in pl.range(offset, full_tiles, transfer_blocks):
                    column = part * transfer_tile
                    pld.tensor.put(dst=window, peer=owner + lane, src=source,
                                   dst_offsets=[row, column], src_offsets=[row, lane * shard + column],
                                   shape=[1, transfer_tile])
                if tail != 0:
                    if offset == full_tiles % transfer_blocks:
                        column = full_tiles * transfer_tile
                        pld.tensor.put(dst=window, peer=owner + lane, src=source,
                                       dst_offsets=[row, column], src_offsets=[row, lane * shard + column],
                                       shape=[1, tail_tile])
        with pl.at(level=pl.Level.CORE_GROUP, deps=[sent], allow_early_resolve=False):
            for lane in pl.range(tp):
                pld.system.notify(target=ready, peer=owner + lane, offsets=[0, 0],
                                  value=epoch, op=pld.NotifyOp.Set)

    def project_and_reduce(
        weight: pl.Tensor[[shard, hidden], pl.BF16],
        output: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        window: pld.DistributedTensor[[rows, shard], pl.BF16],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        partials: pld.DistributedTensor[[tp * rows, hidden], pl.FP32],
        done: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], lane: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
        valid: pl.Scalar[pl.INT32],
    ):
        local = pl.create_tensor([rows, shard], dtype=pl.BF16)
        partial = pl.create_tensor([rows, hidden], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=False) as available:
            pld.system.wait(signal=ready, offsets=[0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        with pl.spmd(rows, deps=[available], name_hint="orthrus_logits_vjp_load"):
            # The persistent window retains old rows when the teacher block shrinks.
            row = pl.tile.get_block_idx()
            for part in pl.range(full_tiles):
                column = part * transfer_tile
                if row < valid:
                    local[row:row + 1, column:column + transfer_tile] = window[row:row + 1, column:column + transfer_tile]
                else:
                    local[row:row + 1, column:column + transfer_tile] = pl.full([1, transfer_tile], dtype=pl.BF16, value=0.)
            if tail != 0:
                column = full_tiles * transfer_tile
                if row < valid:
                    local[row:row + 1, column:column + tail_tile] = window[row:row + 1, column:column + tail_tile]
                else:
                    local[row:row + 1, column:column + tail_tile] = pl.full([1, tail_tile], dtype=pl.BF16, value=0.)
        partial = project(local, weight, partial)
        with pl.spmd(rows, name_hint="orthrus_lm_head_vjp_send") as sent:
            row = pl.tile.get_block_idx()
            pld.tensor.put(dst=partials, peer=owner, src=partial,
                           dst_offsets=[lane * rows + row, 0], src_offsets=[row, 0], shape=[1, hidden])
        with pl.at(level=pl.Level.CORE_GROUP, deps=[sent], allow_early_resolve=False) as published:
            # Peers publish concurrently to the owner's signal window.
            pld.system.notify(target=done, peer=owner, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        if lane == 0:
            with pl.at(level=pl.Level.CORE_GROUP, deps=[published], allow_early_resolve=False) as gathered:
                pld.system.wait(signal=done, offsets=[0, 0], expected=epoch * tp, cmp=pld.WaitCmp.Ge)
            with pl.spmd(hidden // hidden_tile, deps=[gathered], name_hint="orthrus_lm_head_vjp_sum"):
                column = pl.tile.get_block_idx() * hidden_tile
                total = pl.full([rows, hidden_tile], dtype=pl.FP32, value=0.)
                for peer in pl.range(tp):
                    base = peer * rows
                    total = pl.add(total, partials[base:base + rows, column:column + hidden_tile])
                output[:, column:column + hidden_tile] = total
        return output

    rank_vjp = pl.jit(project_and_reduce)
    owner_projection = pl.jit.inline(auto_scope=False)(project_and_reduce)

    @pl.jit
    def owner_vjp(
        source: pl.Tensor[[rows, vocabulary], pl.BF16],
        weight: pl.Tensor[[shard, hidden], pl.BF16],
        output: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        window: pld.DistributedTensor[[rows, shard], pl.BF16],
        ready: pld.DistributedTensor[[1, 1], pl.INT32],
        partials: pld.DistributedTensor[[tp * rows, hidden], pl.FP32],
        done: pld.DistributedTensor[[1, 1], pl.INT32],
        owner: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], valid: pl.Scalar[pl.INT32],
    ):
        scatter(source, window, ready, owner, epoch, valid)
        return owner_projection(weight, output, window, ready, partials, done, owner, 0, epoch, valid)

    @pl.jit.host
    def dispatch(
        gradient: pl.Tensor[[dp, rows, vocabulary], pl.BF16],
        weights: pl.Tensor[[world, shard, hidden], pl.BF16],
        output: pl.Out[pl.Tensor[[world, rows, hidden], pl.FP32]],
        owner: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32], valid: pl.Scalar[pl.INT32],
    ):
        gradients_buf = pld.alloc_window_buffer([rows, shard], dtype=pl.BF16)
        ready_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        partials_buf = pld.alloc_window_buffer([tp * rows, hidden], dtype=pl.FP32)
        done_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            window = pld.window(gradients_buf, [rows, shard], dtype=pl.BF16)
            ready = pld.window(ready_buf, [1, 1], dtype=pl.INT32)
            partials = pld.window(partials_buf, [tp * rows, hidden], dtype=pl.FP32)
            done = pld.window(done_buf, [1, 1], dtype=pl.INT32)
            if rank == owner:
                replica = rank // tp
                owner_vjp(gradient[replica], weights[rank], output[rank], window, ready, partials, done,
                          owner, epoch, valid, device=rank)
            elif rank > owner and rank < owner + tp:
                lane = rank - owner
                rank_vjp(weights[rank], output[rank], window, ready, partials, done,
                         owner, lane, epoch, valid, device=rank)

    return dispatch


@dataclass(frozen=True)
class PackedLMHeadGradient:
    """FP32 hidden gradient on the DP owner, valid until its next VJP call."""

    owner: int
    rows: int
    gradient: object
    seconds: float


class PackedLMHeadVJP:
    def __init__(self, proposer, loss, graph):
        import torch
        from pypto.runtime import StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.loss = loss
        # The runtime retains communication windows across synchronous updates.
        self.epochs = [0] * len(loss.owners)
        weights = [rank["lm_head_weight"] for rank in proposer.ranks]
        shape = loss.rows, weights[0].shape[1]
        ranks = tuple(range(proposer.world))
        outputs = [self.runtime.alloc_tensor(shape, torch.float32, worker_id=rank) for rank in ranks]
        self.weights = StackedDeviceTensor(weights, (proposer.world, *weights[0].shape), ranks)
        self.output = StackedDeviceTensor(outputs, (proposer.world, *shape), ranks)

    def __call__(self, result):
        replica = self.loss.owners.index(result.owner)
        if result.gradient is not self.loss.gradient.shards[replica]:
            raise ValueError("Packed LM head VJP requires the resident loss gradient")
        self.epochs[replica] += 1
        start = perf_counter()
        self.runtime.run(self.graph, self.loss.gradient, self.weights, self.output,
                         result.owner, self.epochs[replica], result.rows, config=self.config)
        return PackedLMHeadGradient(result.owner, result.rows, self.output.shards[result.owner],
                                    perf_counter() - start)
