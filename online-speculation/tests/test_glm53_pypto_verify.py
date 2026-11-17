from types import SimpleNamespace
from threading import RLock

import pytest
import torch

from blockspec.pypto_glm53_greedy import PROGRAM, top1_predictions
from blockspec.pypto_glm53_serving import collect_clean_logits
from blockspec.pypto_glm53_verify import GreedyVerification, accept_greedy, bind_verify_snapshots


def test_greedy_prefix_and_correction_are_aligned_to_the_previous_row():
    block = [10, 11, 12, 13, 14, 15]
    golden = [11, 12, 13, 14, 15, 16]
    for matched in range(len(block) - 1):
        predictions = golden.copy()
        predictions[matched] = 20
        assert accept_greedy(block, predictions) == GreedyVerification((*golden[:matched], 20), matched + 1)
    assert accept_greedy(block, golden) == GreedyVerification(tuple(golden), len(block))
    assert accept_greedy([10], [11]) == GreedyVerification((11,), 1)
    assert accept_greedy(block, golden, {12}) == GreedyVerification((11, 12), 3)


def test_bind_uses_private_kda_scratch_and_preserves_prefix_seed():
    api = SimpleNamespace(BATCH=32, VERIFY_QUERY_WIDTH=6, KDA_CHECKPOINT_BASE=224)
    rows = torch.full((32, 8), -1, dtype=torch.int32)
    rows[:, 0] = torch.arange(32)
    rows[29, 0] = 300
    baseline = rows.clone()
    bind_verify_snapshots(api, {"kda_state_rows": rows}, [29, 31], [1, 6])
    assert torch.equal(rows[:, 0], baseline[:, 0])
    assert rows[29].tolist() == [300, 206, 206, -1, -1, -1, -1, -1]
    assert rows[31].tolist() == [31, 223, 218, 219, 220, 221, 222, 223]
    assert torch.equal(rows[:29], baseline[:29])


def test_teacher_replay_rejects_a_live_request_slot():
    api = SimpleNamespace(BATCH=32, VERIFY_QUERY_WIDTH=6, MAX_SEQ_LEN=1280)
    runner = SimpleNamespace(_api=api, _state_lock=RLock(), programs={"orthrus", "verify"},
                             _seq_lens=[0] * api.BATCH, _requests={"live": SimpleNamespace(slot=1)})
    with pytest.raises(ValueError, match="private, unused slot"):
        collect_clean_logits(runner, range(12), 1, 1, 0)
    runner._requests.clear()
    runner._seq_lens[1] = 2
    with pytest.raises(ValueError, match="private, unused slot"):
        collect_clean_logits(runner, range(12), 1, 1, 0)


def test_resident_greedy_dispatch_returns_ids_and_uses_a_separate_epoch():
    api = SimpleNamespace(VERIFY_QUERY_WIDTH=6)
    binding = object()
    logit_view = object()
    ids = torch.full((8,), -1, dtype=torch.int32)
    device_ids = SimpleNamespace(shards=[SimpleNamespace(data_ptr=42)])
    config_token = object()
    dispatched = []

    def dispatch(inputs, *, phase, buffer_slot, task_args):
        assert (inputs, phase, buffer_slot, task_args) == ({}, "verify", 0, binding)
        return SimpleNamespace(dispatch=SimpleNamespace(wait=lambda: dispatched.append("forward")),
                               buffer_slot=0, epoch=7)

    def run(program, logits, out_ids, rows, epoch, *, config):
        assert (program, logits, out_ids, rows, config) == (PROGRAM, logit_view, device_ids, 3, config_token)
        dispatched.append(epoch)

    def copy_from(destination, source, size, *, worker_id):
        assert (destination, source, size, worker_id) == (ids.data_ptr(), 42, ids.nbytes, 0)
        ids[:3] = torch.tensor([17, 53 if dispatched[-1] == 1 else -1, 4])

    runner = SimpleNamespace(
        auxiliary_programs={PROGRAM: PROGRAM}, _api=api, _dispatch_step=dispatch, _run_l3=run,
        _orthrus_greedy_args={"verify": binding}, _orthrus_greedy_logit_view=logit_view,
        _orthrus_greedy_result=ids, _orthrus_greedy_epoch=1, config=config_token,
        runtime=SimpleNamespace(copy_from=copy_from), stacked={"orthrus_greedy_ids": device_ids},
        calls=0, guard_events=[], _pending_forwards={0: object()},
    )
    assert top1_predictions(runner, {}, "verify", 3) == [17, 53, 4]
    assert not runner._pending_forwards
    assert runner.calls == 1
    runner._pending_forwards[0] = object()
    with pytest.raises(FloatingPointError, match="step 7"):
        top1_predictions(runner, {}, "verify", 3)
    assert dispatched == ["forward", 1, "forward", 2]
    assert runner.calls == 1
