"""EOS clips output, feedback and rollback at the same verified boundary."""

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("serving.runtime")
from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner, check_sampled_chunk, served_prefix
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online


@dataclass
class Slot:
    request: object
    phase: str
    position: int
    token_ids: tuple


@pytest.mark.parametrize("rows", [0, 3, 8, True])
def test_verification_chunks_stay_within_native_snapshot_capacity(rows):
    config = SimpleNamespace(speculative_tokens=0, prefix_cache=False, max_model_len=512,
                             long_prefill_token_threshold=4, max_num_batched_tokens=128, max_num_seqs=2)
    with pytest.raises(ValueError, match="verification chunk"):
        OrthrusTP2Runner(config, "unused", verify_chunk_size=rows)


@pytest.mark.parametrize("stops,previous,emitted,expected,finished", [
    ((12,), 0, [11, 12, 55, 66], [11, 12], True),
    ((), 0, [11, 12, 55, 66], [11, 12, 55, 66], False),
    ((), 3, [11, 12, 55, 66], [11, 12], True),
    ((12,), 0, [11], [11], False),
])
def test_only_delivered_rows_survive_eos_or_budget(stops, previous, emitted, expected, finished):
    request = SimpleNamespace(max_tokens=6, generated=[1], stop_token_ids=frozenset(stops))
    assert served_prefix(request, previous, emitted) == (expected, finished)


def test_reached_feedback_includes_eos_and_excludes_every_row_after_it():
    trainer = TP2Online(None, device=0, last_layers=1)
    trainer.feedback = dict(complete=False, valid=0, candidates=31)
    calls = []

    def capture(batch, offset, count, *, final):
        calls.append((batch, offset, count, final))
        trainer.feedback["complete"] = final

    trainer.capture_teacher = capture
    request = SimpleNamespace(max_tokens=128, generated=[], stop_token_ids=frozenset([12]))
    slot = SimpleNamespace(request=request)
    trainer.observe_teacher(slot, 0, [11, 12, 55, 66], [], 0, 4, final=False, selected=True)
    trainer.observe_teacher(slot, 0, [77], [11, 12], 4, 1, final=True, selected=True)
    assert calls == [(0, 0, 2, True)]


def test_sampling_does_not_read_target_rows_after_an_accepted_eos():
    q = torch.nn.functional.one_hot(torch.tensor(12), 128).float()
    reads = []

    def target(row):
        reads.append(row)
        return q

    result = check_sampled_chunk([99, 12, 12, 12], 0, 4, target, [q] * 3,
                                 torch.Generator().manual_seed(743), stop_tokens={12})
    assert result == ([12], 1, 1, True)
    assert reads == [0]


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
@pytest.mark.parametrize("guesses,selected,accepted,rejected", [
    ([11, 12, 55, 66], [11, 12, 55, 66], 2, 0),  # EOS is accepted; no later chunk is needed.
    ([11, 12, 55, 66], [11, 12, 99, 66], 2, 0),  # A rejection after EOS is not a real rejection.
    ([11, 55, 55, 66], [11, 12, 55, 66], 1, 1),  # Rejection correction itself emits EOS.
])
def test_real_runner_stops_verification_and_restores_the_eos_prefix(
        stochastic, chunk_size, guesses, selected, accepted, rejected):
    runner = object.__new__(OrthrusTP2Runner)
    runner.online = None
    runner.stochastic, runner.lazy_sampling = stochastic, False
    runner.output_candidates, runner.draft_batch_size = 31, 1
    runner.verify_chunk_size = chunk_size
    runner.spec_stats = Counter()
    runner.control_ids = torch.empty((2, 32), dtype=torch.int32)
    runner.control_values = torch.empty_like(runner.control_ids)
    runner.restore, runner.runconfig, runner.resident = object(), None, [None] * 6
    request = SimpleNamespace(request_id="r", max_tokens=128, generated=[],
                              sampling=SimpleNamespace(temperature=int(stochastic)),
                              generator=torch.Generator().manual_seed(743), stop_token_ids=frozenset([12]))
    slot = Slot(request=request, phase="decode", position=10, token_ids=(99,))
    pages = object()
    runner.requests = {"r": dict(slot=3, pages=pages)}
    committed, restored, executions = [], [], []
    runner.cache = SimpleNamespace(commit=lambda page, end: committed.append((page, end)))
    runner._reserve = lambda *_: None
    cursor = [0]
    def execute(*args, **kwargs):
        executions.append(args[2])
        cursor[0] = args[3][0] - 10
        return 1, None
    runner._execute = execute

    def proposals(items, capture=None, saved=None, **kwargs):
        result = [(guesses + [77] * 31)[:budget] for _, budget in items]
        if saved is not None:
            saved["r"] = [torch.nn.functional.one_hot(torch.tensor(token), 128).float()
                          for token in result[0]]
        return result

    runner.propose_many = proposals
    runner.propose = lambda item, budget: proposals([(item, budget)])[0]
    runner._distribution = lambda batch, index, row, req: (
        torch.nn.functional.one_hot(torch.tensor(selected[cursor[0] + row]), 128).float()
        if stochastic else selected[cursor[0] + row])
    runner._worker_call = lambda *_a, **_kw: restored.append(runner.control_values[:, 0].tolist())

    assert runner.run_batch([slot]) == [[11, 12]]
    assert len(executions) == (2 if chunk_size == 1 else 1)
    assert restored == ([[2, 2]] if chunk_size == 4 else [])
    assert committed == ([(pages, 11), (pages, 12)] if chunk_size == 1 else [(pages, 12)])
    assert runner.spec_stats["accepted"] == accepted
    assert runner.spec_stats["rejected_rounds"] == rejected
