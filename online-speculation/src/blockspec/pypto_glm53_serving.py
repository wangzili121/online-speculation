"""Stage a single Orthrus block and return GLM Serving's accepted tokens."""

from .pypto_glm53_verify import bind_verify_snapshots, verify_greedy


def propose_greedy(runner, slot, prefix_length, anchor, mask_id, *, cache_groups=None):
    """Execute the hybrid draft with the resident target's prefix leases."""
    api = runner._api
    width = api.VERIFY_QUERY_WIDTH
    block = [anchor] + [mask_id] * (width - 1)
    inputs, slots = runner._prepare_chunk_inputs([block], [slot], [prefix_length], phase="verify",
                                                 cache_groups=cache_groups)
    pools, tail = divmod(prefix_length, api.INDEX_KPOOL)
    inputs["execution_mode"].fill_(3)
    inputs["pool_count_tok"][:width].fill_(pools)
    inputs["tail_start"][:width].fill_(prefix_length - tail)
    inputs["tail_count"][:width].fill_(tail)
    inputs["kv_len_tok"][:width].fill_(prefix_length)
    bind_verify_snapshots(api, inputs, slots, [width])
    try:
        pending = runner._dispatch_step(inputs, phase="orthrus", buffer_slot=0)
        pending.logit_count = width - 1
        guesses = runner._reclaim_step(pending).argmax(1).tolist()
    except BaseException as error:
        runner._invalid_slots.add(slot)
        runner._poison_forwards(error)
        raise
    runner._pending_forwards.pop(pending.buffer_slot)
    return guesses


def serve_orthrus(runner, batch, prepared):
    """Return a normal target decode result when this request cannot speculate."""
    from pypto_serving.config.types import DecodeResult

    if "orthrus" not in runner.programs or len(batch.sampling_params) != 1:
        return None
    policy = batch.sampling_params[0]
    if policy.temperature > 0 or policy.forbidden_token_ids or batch.constraint_states or batch.stop_token_masks:
        return None
    if len(prepared.request_ids) != 1:
        return None
    request_id, = prepared.request_ids
    with runner._state_lock:
        runner._drain_forwards()
        state = runner._requests.get(request_id)
        if state is None or state.committed_length < state.prompt_length:
            raise ValueError("GLM Orthrus needs a fully prefetched request")
        prefix = state.committed_length
        if int(prepared.seq_lens[0]) != prefix + 1:
            raise ValueError("GLM Orthrus anchor does not continue committed cache")
        width = runner._api.VERIFY_QUERY_WIDTH
        if prefix + width > runner._max_seq_len:
            return None
        anchor = int(batch.token_ids.reshape(-1)[0])
        guesses = propose_greedy(runner, state.slot, prefix, anchor, runner._orthrus_mask_id,
                                 cache_groups=prepared.cache_groups)
        result, = verify_greedy(runner, [[anchor, *guesses]], [state.slot], cache_groups=prepared.cache_groups)
        state.committed_length += result.committed
    return DecodeResult(hidden_states=None, logits=None, accepted_token_ids=[list(result.emitted)],
                        num_draft_tokens=[len(guesses)])
