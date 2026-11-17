"""Stage a single Orthrus block and return GLM Serving's accepted tokens."""

from contextlib import contextmanager

from .pypto_glm53_greedy import top1_predictions
from .pypto_glm53_verify import bind_verify_snapshots, verify_greedy


def _draft_inputs(runner, slot, prefix_length, anchor, mask_id, *, cache_groups=None):
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
    return inputs


def draft_logits(runner, slot, prefix_length, anchor, mask_id, *, cache_groups=None):
    """Execute the hybrid draft with the resident target's prefix leases."""
    inputs = _draft_inputs(runner, slot, prefix_length, anchor, mask_id, cache_groups=cache_groups)
    try:
        pending = runner._dispatch_step(inputs, phase="orthrus", buffer_slot=0)
        pending.logit_count = runner._api.VERIFY_QUERY_WIDTH - 1
        logits = runner._reclaim_step(pending)
    except BaseException as error:
        runner._invalid_slots.add(slot)
        runner._poison_forwards(error)
        raise
    runner._pending_forwards.pop(pending.buffer_slot)
    return logits


def propose_greedy(runner, slot, prefix_length, anchor, mask_id, *, cache_groups=None):
    inputs = _draft_inputs(runner, slot, prefix_length, anchor, mask_id, cache_groups=cache_groups)
    try:
        return top1_predictions(runner, inputs, "orthrus", runner._api.VERIFY_QUERY_WIDTH - 1)
    except BaseException as error:
        runner._invalid_slots.add(slot)
        runner._poison_forwards(error)
        raise


@contextmanager
def _clean_replay(runner, tokens, anchor, slot):
    """Keep replay cache writes in an unused slot while holding the runner lock."""
    api = runner._api
    width = api.VERIFY_QUERY_WIDTH
    if anchor < 0 or anchor + width > len(tokens):
        raise ValueError("online replay needs an anchor and five known continuation tokens")
    if len(tokens) > api.MAX_SEQ_LEN or not 0 <= slot < api.BATCH:
        raise ValueError("online replay exceeds the compiled GLM context or slot geometry")
    if "orthrus" not in runner.programs or "verify" not in runner.programs:
        raise RuntimeError("online replay needs the PyPTO draft and verifier programs")
    with runner._state_lock:
        if any(state.slot == slot for state in runner._requests.values()) or int(runner._seq_lens[slot]):
            raise ValueError("online replay needs a private, unused slot")
        runner._drain_forwards()
        runner.reset_requests([slot])
        try:
            for offset in range(0, anchor, api.CAPACITY):
                prefix_chunk = list(tokens[offset:min(offset + api.CAPACITY, anchor)])
                runner.forward_chunks([prefix_chunk], [slot])
            yield
        finally:
            if runner._dispatch_error is None:
                runner.reset_requests([slot])


def _teacher_inputs(runner, tokens, anchor, slot):
    width = runner._api.VERIFY_QUERY_WIDTH
    block = list(tokens[anchor:anchor + width])
    inputs, slots = runner._prepare_chunk_inputs([block], [slot], [anchor], phase="verify")
    bind_verify_snapshots(runner._api, inputs, slots, [width])
    return inputs


def collect_clean_logits(runner, tokens, anchor, slot, mask_id):
    """Replay a clean, fixed-width block in a private slot for online distillation."""
    with _clean_replay(runner, tokens, anchor, slot):
        student = draft_logits(runner, slot, anchor, tokens[anchor], mask_id)
        inputs = _teacher_inputs(runner, tokens, anchor, slot)
        pending = runner._dispatch_step(inputs, phase="verify", buffer_slot=0)
        pending.logit_count = runner._api.VERIFY_QUERY_WIDTH - 1
        teacher = runner._reclaim_step(pending)
        runner._pending_forwards.pop(pending.buffer_slot)
    return student, teacher


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
