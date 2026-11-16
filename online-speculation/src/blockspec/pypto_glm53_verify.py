"""Target verification and cache commit for the PyPTO GLM runner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GreedyVerification:
    """Emitted tokens exclude the input anchor and may end in a new anchor."""

    emitted: tuple[int, ...]
    committed: int


def accept_greedy(block, predictions, stop_ids=()):
    """Match draft tokens with the causal prediction from the preceding row."""
    if not block or len(block) != len(predictions):
        raise ValueError("verification needs one target prediction per block row")
    accepted = []
    for index, guess in enumerate(block[1:]):
        target = predictions[index]
        if guess != target:
            return GreedyVerification((*accepted, target), 1 + len(accepted))
        accepted.append(guess)
        if guess in stop_ids:
            return GreedyVerification(tuple(accepted), 1 + len(accepted))
    return GreedyVerification((*accepted, predictions[-1]), len(block))


def bind_verify_snapshots(api, inputs, slots, counts):
    """Redirect verification writes into request-private KDA scratch rows."""
    rows = inputs["kda_state_rows"]
    for slot, count in zip(slots, counts, strict=True):
        first = api.BATCH + slot * api.VERIFY_QUERY_WIDTH
        if first + count > api.KDA_CHECKPOINT_BASE:
            raise ValueError("verification scratch rows overlap persistent checkpoints")
        rows[slot, 1:] = -1
        for index in range(count):
            rows[slot, 2 + index] = first + index
        rows[slot, 1] = first + count - 1


def verify_greedy(runner, token_sequences, request_slots, *, stop_ids=(), cache_groups=None):
    """Verify a batch of externally supplied proposals in the resident target.

    Each block starts at the next uncached token. In Serving, the grouped
    scheduler leases the lookahead pages. Standalone runs use the local pool.
    """
    api = runner._api
    try:
        commit_program = runner.auxiliary_programs["orthrus_commit"]
    except KeyError as error:
        raise RuntimeError("open a GLM runner with Orthrus verification enabled") from error
    with runner._state_lock:
        runner._drain_forwards()
        counts = tuple(map(len, token_sequences))
        if any(not 1 <= count <= api.VERIFY_QUERY_WIDTH for count in counts):
            raise ValueError("verification needs an anchor and up to five draft tokens")
        inputs, slots = runner._prepare_chunk_inputs(
            token_sequences, request_slots, phase="verify", cache_groups=cache_groups,
        )
        bind_verify_snapshots(api, inputs, slots, counts)
        total = sum(counts)
        try:
            pending = runner._dispatch_step(inputs, phase="verify", buffer_slot=0)
            pending.logit_count = total
            predictions = runner._reclaim_step(pending).argmax(dim=1).tolist()
            results = []
            metadata = runner._orthrus_commit_metadata
            feedback, source, destination = metadata
            for table in metadata:
                table.fill_(-1)
            snapshot_rows = inputs["kda_state_rows"][:, 2:].tolist()
            boundary = []
            for position, (block, slot, count) in enumerate(zip(token_sequences, slots, counts, strict=True)):
                offset = int(inputs["query_start_loc"][slot])
                result = accept_greedy(block, predictions[offset:offset + count], stop_ids)
                results.append(result)
                relative = result.committed - 1
                feedback[:, slot] = offset + relative
                source[:, slot] = snapshot_rows[slot][relative]
                destination[:, slot] = slot
                endpoint = int(runner._seq_lens[slot]) + result.committed
                if endpoint % api.BLOCK_SIZE == 0:
                    page = endpoint // api.BLOCK_SIZE - 1
                    groups = (runner._local_cache.ensure_group_blocks(str(slot), endpoint)
                              if cache_groups is None else cache_groups[position])
                    boundary.append((slot, snapshot_rows[slot][relative],
                                     api.KDA_CHECKPOINT_BASE + groups["kda_state"][page % api.KDA_STATE_BLOCKS]))

            def commit():
                runner._run_l3(
                    commit_program, runner.stacked["target_feedback"], runner.stacked["target_feedback_last"],
                    feedback, runner.stacked["kda_conv_state"], runner.stacked["kda_recurrent_state"],
                    source, destination, config=runner.config,
                )

            commit()
            if boundary:
                for table in metadata:
                    table.fill_(-1)
                for slot, snapshot, checkpoint in boundary:
                    source[:, slot] = snapshot
                    destination[:, slot] = checkpoint
                commit()
        except BaseException as error:
            runner._invalid_slots.update(slots)
            runner._poison_forwards(error)
            raise
        runner._pending_forwards.pop(pending.buffer_slot)
        for slot, result in zip(slots, results, strict=True):
            runner._seq_lens[slot] += result.committed
        runner._prefix_seed_slots.difference_update(slots)
        return tuple(results)
