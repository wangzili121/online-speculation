"""Global KL over TP vocabulary shards, without concatenating logits on the host."""


def make_sharded_kl(ranks, rows, local_vocab, *, device_entry=False):
    """Return per-rank logit gradients and additive KL contributions."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    if ranks < 1 or rows < 1 or local_vocab < 1:
        raise ValueError("invalid GLM vocabulary shard geometry")
    tile = 64
    padded_vocab = (local_vocab + tile - 1) // tile * tile
    neg_large = -1e30

    @pl.jit(auto_scope=False)
    def kl(
        student: pl.Tensor[[rows, padded_vocab], pl.FP32], teacher: pl.Tensor[[rows, padded_vocab], pl.FP32],
        gradient: pl.Out[pl.Tensor[[rows, padded_vocab], pl.FP32]],
        loss: pl.Out[pl.Tensor[[rows, 8], pl.FP32]],
        stats: pld.DistributedTensor[[rows, 32], pl.FP32], ready: pld.DistributedTensor[[ranks, 1], pl.INT32],
        valid: pl.Scalar[pl.INT32], rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_kl_reuse", allow_early_resolve=False) as reusable:
            for peer in pl.range(ranks):
                pld.system.wait(ready, offsets=[peer, 0], expected=2 * (epoch - 1), cmp=pld.WaitCmp.Ge)
        with pl.spmd(rows, name_hint="glm_kl_publish", deps=[reusable], allow_early_resolve=False) as published:
            row = pl.tile.get_block_idx()
            if row < valid:
                mx_s = pl.tile.full([1, tile], dtype=pl.FP32, value=neg_large)
                mx_t = pl.tile.full([1, tile], dtype=pl.FP32, value=neg_large)
                ramp = pl.tile.arange(0, [1, tile], dtype=pl.INT32)
                lo = pl.tile.full([1, tile], dtype=pl.INT32, value=0)
                hi = pl.tile.full([1, tile], dtype=pl.INT32, value=1)
                for column in pl.range(0, padded_vocab, tile):
                    pos = pl.tile.adds(ramp, pl.cast(column + 1 - local_vocab, pl.INT32))
                    padding = pl.tile.cast(pl.tile.minimum(pl.tile.maximum(pos, lo), hi), pl.FP32)
                    floor = pl.tile.muls(padding, neg_large)
                    local_s = pl.tile.add(pl.load(student, [row, column], [1, tile]), floor)
                    local_t = pl.tile.add(pl.load(teacher, [row, column], [1, tile]), floor)
                    mx_s = pl.tile.maximum(mx_s, local_s)
                    mx_t = pl.tile.maximum(mx_t, local_t)
                ms = pl.cast(neg_large, pl.FP32)
                mt = pl.cast(neg_large, pl.FP32)
                for lane in pl.range(tile):
                    ms = pl.max(ms, pl.tile.read(mx_s, [0, lane]))
                    mt = pl.max(mt, pl.tile.read(mx_t, [0, lane]))
                sum_s = pl.tile.full([1, tile], dtype=pl.FP32, value=0.0)
                sum_t = pl.tile.full([1, tile], dtype=pl.FP32, value=0.0)
                lower = pl.tile.full([1, tile], dtype=pl.FP32, value=-80.0)
                upper = pl.tile.full([1, tile], dtype=pl.FP32, value=0.0)
                for column in pl.range(0, padded_vocab, tile):
                    pos = pl.tile.adds(ramp, pl.cast(column + 1 - local_vocab, pl.INT32))
                    present = pl.tile.cast(pl.tile.sub(hi, pl.tile.minimum(pl.tile.maximum(pos, lo), hi)), pl.FP32)
                    local_s = pl.tile.subs(pl.load(student, [row, column], [1, tile]), ms)
                    local_t = pl.tile.subs(pl.load(teacher, [row, column], [1, tile]), mt)
                    s = pl.tile.exp(pl.tile.minimum(pl.tile.maximum(local_s, lower), upper))
                    t = pl.tile.exp(pl.tile.minimum(pl.tile.maximum(local_t, lower), upper))
                    sum_s = pl.tile.add(sum_s, pl.tile.mul(s, present))
                    sum_t = pl.tile.add(sum_t, pl.tile.mul(t, present))
                zs = pl.cast(0.0, pl.FP32)
                zt = pl.cast(0.0, pl.FP32)
                for lane in pl.range(tile):
                    zs = zs + pl.tile.read(sum_s, [0, lane])
                    zt = zt + pl.tile.read(sum_t, [0, lane])
                # Each row owns a complete cache line before publishing it to peers.
                local_stats = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
                pl.tile.write(local_stats, [0, 0], ms)
                pl.tile.write(local_stats, [0, 1], mt)
                pl.tile.write(local_stats, [0, 8], zs)
                pl.tile.write(local_stats, [0, 9], zt)
                pl.store(local_stats, [row, 0], stats)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_kl_ready", deps=[published]) as sent:
            for peer in pl.range(ranks):
                pld.system.notify(ready, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_kl_wait", deps=[sent], allow_early_resolve=False) as available:
            for peer in pl.range(ranks):
                pld.system.wait(ready, offsets=[peer, 0], expected=2 * epoch - 1, cmp=pld.WaitCmp.Ge)
        with pl.spmd(rows, name_hint="glm_kl_gradient", deps=[available]) as consumed:
            output_row = pl.tile.get_block_idx()
            if output_row < valid:
                first_peer = pl.cast(0, pl.INT32)
                maxima = pld.tile.remote_load(stats, peer=first_peer, offsets=[output_row, 0], shape=[1, 8])
                denom = pld.tile.remote_load(stats, peer=first_peer, offsets=[output_row, 8], shape=[1, 8])
                peer_floor = pl.tile.full([1, 8], dtype=pl.FP32, value=-80.0)
                for peer in pl.range(1, ranks):
                    peer_max = pld.tile.remote_load(stats, peer=peer, offsets=[output_row, 0], shape=[1, 8])
                    peer_sum = pld.tile.remote_load(stats, peer=peer, offsets=[output_row, 8], shape=[1, 8])
                    next_max = pl.tile.maximum(maxima, peer_max)
                    previous = pl.tile.exp(pl.tile.maximum(pl.tile.sub(maxima, next_max), peer_floor))
                    incoming = pl.tile.exp(pl.tile.maximum(pl.tile.sub(peer_max, next_max), peer_floor))
                    denom = pl.tile.add(pl.tile.mul(denom, previous), pl.tile.mul(peer_sum, incoming))
                    maxima = next_max
                ms = pl.tile.read(maxima, [0, 0])
                mt = pl.tile.read(maxima, [0, 1])
                denominator_floor = pl.tile.full([1, 8], dtype=pl.FP32, value=1.0)
                log_denom = pl.tile.log(pl.tile.maximum(denom, denominator_floor))
                ls = pl.tile.read(log_denom, [0, 0])
                lt = pl.tile.read(log_denom, [0, 1])
                loss_lanes = pl.tile.full([1, tile], dtype=pl.FP32, value=0.0)
                ramp = pl.tile.arange(0, [1, tile], dtype=pl.INT32)
                lo = pl.tile.full([1, tile], dtype=pl.INT32, value=0)
                hi = pl.tile.full([1, tile], dtype=pl.INT32, value=1)
                zero_exp = pl.tile.full([1, tile], dtype=pl.FP32, value=0.0)
                lower_exp = pl.tile.full([1, tile], dtype=pl.FP32, value=-80.0)
                normalizer = pl.cast(valid, pl.FP32)
                for offset in pl.range(0, padded_vocab, tile):
                    pos = pl.tile.adds(ramp, pl.cast(offset + 1 - local_vocab, pl.INT32))
                    invalid = pl.tile.minimum(pl.tile.maximum(pos, lo), hi)
                    present = pl.tile.cast(pl.tile.sub(hi, invalid), pl.FP32)
                    s = pl.load(student, [output_row, offset], [1, tile])
                    t = pl.load(teacher, [output_row, offset], [1, tile])
                    log_s = pl.tile.subs(pl.tile.subs(s, ms), ls)
                    log_t = pl.tile.subs(pl.tile.subs(t, mt), lt)
                    p = pl.tile.exp(pl.tile.minimum(pl.tile.maximum(log_s, lower_exp), zero_exp))
                    q = pl.tile.exp(pl.tile.minimum(pl.tile.maximum(log_t, lower_exp), zero_exp))
                    dlogits = pl.tile.divs(pl.tile.mul(pl.tile.sub(p, q), present), normalizer)
                    gradient = pl.store(dlogits, [output_row, offset], gradient)
                    terms = pl.tile.mul(pl.tile.mul(q, present), pl.tile.sub(log_t, log_s))
                    loss_lanes = pl.tile.add(loss_lanes, terms)
                local_loss = pl.cast(0.0, pl.FP32)
                for lane in pl.range(tile):
                    local_loss = local_loss + pl.tile.read(loss_lanes, [0, lane])
                row_loss = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                pl.tile.write(row_loss, [0, 0], local_loss / normalizer)
                loss = pl.store(row_loss, [output_row, 0], loss)
            else:
                loss = pl.store(pl.tile.full([1, 8], dtype=pl.FP32, value=0.0), [output_row, 0], loss)
                for offset in pl.range(0, padded_vocab, tile):
                    zero = pl.tile.full([1, tile], dtype=pl.FP32, value=0.0)
                    gradient = pl.store(zero, [output_row, offset], gradient)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_kl_retire", deps=[consumed]) as done:
            for peer in pl.range(ranks):
                pld.system.notify(ready, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="glm_kl_done", deps=[done], allow_early_resolve=False):
            for peer in pl.range(ranks):
                pld.system.wait(ready, offsets=[peer, 0], expected=2 * epoch, cmp=pld.WaitCmp.Ge)
        return gradient, loss

    @pl.jit.host
    def entry(student: pl.Tensor[[ranks, rows, padded_vocab], pl.FP32],
              teacher: pl.Tensor[[ranks, rows, padded_vocab], pl.FP32],
              gradient: pl.Out[pl.Tensor[[ranks, rows, padded_vocab], pl.FP32]],
              loss: pl.Out[pl.Tensor[[ranks, rows, 8], pl.FP32]],
              valid: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32]):
        stats_handle = pld.alloc_window_buffer([rows, 32], dtype=pl.FP32)
        ready_handle = pld.alloc_window_buffer([ranks, 1], dtype=pl.INT32)
        for chip in pl.range(pld.world_size()):
            stats = pld.window(stats_handle, [rows, 32], dtype=pl.FP32)
            ready = pld.window(ready_handle, [ranks, 1], dtype=pl.INT32)
            kl(student[chip], teacher[chip], gradient[chip], loss[chip], stats, ready, valid, chip, epoch, device=chip)

    return kl if device_entry else entry
