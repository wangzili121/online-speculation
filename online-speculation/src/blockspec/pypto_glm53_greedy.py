"""Keep the draft and verifier logits on the device during greedy decoding."""


PROGRAM = "orthrus_greedy"


def top1_buffer_rows(width):
    """Align each rank's compact INT32 output to a 32-byte row."""
    return (width + 7) // 8 * 8


def prepare_greedy(api, compiler):
    return compiler.compile(PROGRAM, make_sharded_top1(top1_buffer_rows(api.VERIFY_QUERY_WIDTH)))


def make_sharded_top1(rows):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.config import LOCAL_VOCAB, TP_SIZE
    from models.glm5_3_flash.lm_head import HEAD_VOCAB_PAD, TP_CANDIDATE_TILE, lm_head_tp_reduce

    pad = HEAD_VOCAB_PAD
    tile = 256
    groups = 8
    stripe = ((LOCAL_VOCAB + groups * tile - 1) // (groups * tile)) * tile
    if rows < 1:
        raise ValueError("the device top-1 requires at least one query row")

    @pl.jit(auto_scope=False)
    def top1(
        logits: pl.Tensor[[rows, pad], pl.FP32],
        ids: pl.Out[pl.Tensor[[rows], pl.INT32]],
        values: pld.DistributedTensor[[rows, TP_CANDIDATE_TILE], pl.FP32],
        indices: pld.DistributedTensor[[rows, TP_CANDIDATE_TILE], pl.INT32],
        arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32], active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        local_scores = pl.create_tensor([rows], dtype=pl.FP32)
        local_ids = pl.create_tensor([rows], dtype=pl.INT32)
        global_scores = pl.create_tensor([rows], dtype=pl.FP32)
        global_ids = pl.create_tensor([rows], dtype=pl.INT32)
        partials = pl.create_tensor([rows * groups, 16], dtype=pl.FP32)
        with pl.spmd(rows * groups, name_hint="glm_orthrus_top1_scan") as scanned:
            task = pl.tile.get_block_idx()
            row = task // groups
            first = task % groups * stripe
            sentinel = pl.cast(1073741824, pl.INT32)
            ramp = pl.arange(0, [1, tile], dtype=pl.INT32)
            lanes = pl.col_expand(pl.full([8, tile], dtype=pl.INT32, value=0), ramp)
            lane_values = pl.cast(lanes, pl.FP32)
            ones = pl.full([8, tile], dtype=pl.INT32, value=1)
            best_score = pl.cast(-3.4028234663852886e38, pl.FP32)
            best_id = pl.cast(-1, pl.INT32)
            is_finite = pl.cast(1, pl.INT32)
            if row < active:
                for column in pl.range(first, pl.min(first + stripe, LOCAL_VOCAB), tile):
                    width = pl.min(tile, LOCAL_VOCAB - column)
                    source = pl.slice(logits, [8, tile], [row, column], valid_shape=[1, width])
                    scores = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.min), 8, tile)
                    # Only real vocabulary columns participate in the finite check.
                    valid_width = pl.cast(pl.cast(width, pl.INT32), pl.FP32)
                    keep = pl.cast(pl.cmp(lane_values, valid_width, cmp_type=2), pl.INT32)
                    flags = pl.cast(pl.cmp(pl.abs(scores), 3.4028234663852886e38, cmp_type=3), pl.INT32)
                    valid = pl.maximum(flags, pl.sub(ones, keep))
                    is_finite = pl.min(is_finite, pl.read(pl.row_min(valid), [0, 0]))
                    value = pl.read(pl.row_max(scores), [0, 0])
                    if best_id < 0 or value > best_score:
                        matches = pl.cast(pl.cmp(scores, value, cmp_type=0), pl.INT32)
                        candidates = pl.add(pl.mul(pl.mul(matches, keep), pl.sub(lanes, sentinel)), sentinel)
                        index = pl.read(pl.row_min(candidates), [0, 0])
                        if index < sentinel:
                            best_score = value
                            best_id = pl.cast(column, pl.INT32) + index
            # Every worker writes its own 64-byte row. GLM token IDs fit exactly in FP32.
            partial = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            pl.write(partial, [0, 0], best_score)
            pl.write(partial, [0, 1], pl.cast(best_id, pl.FP32))
            pl.write(partial, [0, 2], pl.cast(is_finite, pl.FP32))
            partials[task:task + 1, :] = partial

        with pl.spmd(1, name_hint="glm_orthrus_top1_merge", deps=[scanned]):
            writer = pl.tile.get_block_idx()
            inf_bits = pl.full([1, 8], dtype=pl.INT32, value=2139095040)
            positive_inf = pl.read(pl.reinterpret_view(inf_bits, pl.FP32), [0, 0])
            for row in pl.range(writer, rows):
                best_score = pl.cast(-3.4028234663852886e38, pl.FP32)
                best_id = pl.cast(-1, pl.INT32)
                is_finite = pl.cast(1, pl.INT32)
                for group in pl.range(groups):
                    offset = row * groups + group
                    token = pl.cast(pl.read(partials, [offset, 1]), pl.INT32)
                    value = pl.read(partials, [offset, 0])
                    ok = pl.cast(pl.read(partials, [offset, 2]), pl.INT32)
                    is_finite = pl.min(is_finite, ok)
                    if token >= 0 and (best_id < 0 or value > best_score):
                        best_id = token
                        best_score = value
                # Nonfinite input wins the TP comparison with a reserved score.
                # This carries the fault to every rank without another collective.
                if is_finite == 0:
                    best_score = positive_inf
                    best_id = pl.cast(0, pl.INT32)
                pl.write(local_scores, [row], best_score)
                pl.write(local_ids, [row], best_id)
        lm_head_tp_reduce(local_scores, local_ids, values, indices, arrived, global_scores, global_ids, rank, 0, epoch)
        for output_writer in pl.spmd(1, name_hint="glm_orthrus_top1_finish"):
            for output_row in pl.range(output_writer, rows):
                if pl.read(global_scores, [output_row]) > 3.4028234663852886e38:
                    pl.write(ids, [output_row], pl.cast(-1, pl.INT32))
                else:
                    pl.write(ids, [output_row], pl.read(global_ids, [output_row]))
        return ids

    @pl.jit.host
    def entry(
        logits: pl.Tensor[[TP_SIZE, rows, pad], pl.FP32],
        ids: pl.Out[pl.Tensor[[TP_SIZE, rows], pl.INT32]],
        active: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    ):
        scores_buffer = pld.alloc_window_buffer([rows, TP_CANDIDATE_TILE], dtype=pl.FP32)
        ids_buffer = pld.alloc_window_buffer([rows, TP_CANDIDATE_TILE], dtype=pl.INT32)
        ready_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            scores = pld.window(scores_buffer, [rows, TP_CANDIDATE_TILE], dtype=pl.FP32)
            indices = pld.window(ids_buffer, [rows, TP_CANDIDATE_TILE], dtype=pl.INT32)
            ready = pld.window(ready_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            top1(logits[rank], ids[rank], scores, indices, ready, rank, active, epoch, device=rank)

    return entry


def top1_predictions(runner, inputs, phase, rows):
    """Read compact IDs from a resident forward, or use the full-logit path."""
    program = runner.auxiliary_programs.get(PROGRAM)
    api = runner._api
    if program is None or rows > api.VERIFY_QUERY_WIDTH:
        pending = runner._dispatch_step(inputs, phase=phase, buffer_slot=0)
        pending.logit_count = rows
        logits = runner._reclaim_step(pending)
        runner._pending_forwards.pop(pending.buffer_slot)
        return logits.argmax(dim=1).tolist()

    pending = runner._dispatch_step(inputs, phase=phase, buffer_slot=0, task_args=runner._orthrus_greedy_args[phase])
    pending.dispatch.wait()
    ids = runner.stacked["orthrus_greedy_ids"]
    runner._run_l3(program, runner._orthrus_greedy_logit_view, ids, rows, runner._orthrus_greedy_epoch,
                   config=runner.config)
    runner._orthrus_greedy_epoch += 1
    host_ids = runner._orthrus_greedy_result
    runner.runtime.copy_from(host_ids.data_ptr(), ids.shards[0].data_ptr, host_ids.nbytes, worker_id=0)
    active_ids = host_ids[:rows]
    if bool((active_ids < 0).any()):
        runner.guard_events.append(f"nonfinite logits at step {pending.epoch}")
        raise FloatingPointError(f"invalid logits in GLM {phase} at step {pending.epoch}")
    runner._pending_forwards.pop(pending.buffer_slot)
    runner.calls += 1
    return active_ids.tolist()
