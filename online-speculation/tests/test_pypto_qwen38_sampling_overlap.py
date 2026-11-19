"""Colocated replay owns the GPU only between prefetched CPU sampling and the next device call."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('serving.runtime')
from blockspec.sampling import SamplingConfig
from blockspec.pypto_qwen38.lazy_proposals import LazyProposals
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online
from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner, TensorParallelRunner


def test_prefetch_preserves_rows_and_rng_until_sampling():
    source = torch.tensor([[2., -1.], [-3., 1.], [.2, .1], [1., 1.]])
    original = source.clone()
    calls = []
    generator = torch.Generator().manual_seed(743)
    state = generator.get_state().clone()
    def read(first, count):
        calls.append((first, count))
        return source[first:first+count]  # 模拟会被下一次读取覆盖的暂存区。
    lazy = LazyProposals(4, read, SamplingConfig(1), generator)
    lazy.prefetch(4)
    assert not lazy.tokens and torch.equal(state, generator.get_state())
    source.zero_()
    lazy.ensure(2)
    lazy.ensure(4)
    assert calls == [(0, 4)]
    expected = original.softmax(-1)
    expected.div_(expected.sum(-1, keepdim=True))
    assert torch.equal(torch.stack(lazy.distributions), expected)


@pytest.mark.parametrize('options', [dict(background=True), dict(last_layers=2), dict(device=2)])
def test_overlap_is_explicit_and_uses_no_third_card(options):
    kw = dict(device=0, last_layers=1, overlap_sampling=True) | options
    with pytest.raises(ValueError, match='overlap'):
        learner = TP2Online(None, **kw)
        learner.bind_devices((0, 1))


@pytest.mark.parametrize('last_layers', [1, 4])
def test_queue_launches_on_sampling_window_and_commits_on_device_owner(monkeypatch, last_layers):
    learner = TP2Online(None, device=0, last_layers=last_layers, overlap_sampling=True)
    learner.bind_devices((0, 1))
    learner.runner = SimpleNamespace(spec_stats=Counter(), suffix='initial')
    learner.initial_step, learner.stride = 0, 2
    learner.updated, learner.packed = [], []
    learner.feedback = dict(valid=1, candidates=1, complete=True)
    entered, release = Event(), Event()
    owner, commits = get_ident(), []
    def train(replay, round_number):
        assert get_ident() != owner
        assert replay[0]['valid'] == 1
        entered.set()
        assert release.wait(5)
        learner.version += 1
        return dict(version=learner.version, round=round_number, update_ns=1,
                    publish_read_ns=0, publish_pack_ns=0)
    original_commit = learner.commit_update
    def commit(result):
        commits.append(get_ident())
        original_commit(result)
    monkeypatch.setattr(learner, 'train', train)
    monkeypatch.setattr(learner, 'commit_update', commit)
    with ThreadPoolExecutor(max_workers=1) as learner.executor:
        learner.finish_batch(2)
        assert learner.queued_update and not entered.is_set()
        assert learner.feedback_index(2) is None
        learner.launch_sampling_update()
        try:
            assert entered.wait(5) and not commits
            assert learner.feedback_index(2) is None
            release.set()
            learner.before_device()
        finally:
            release.set()
    assert commits == [owner]
    assert learner.completed_version == learner.published_version == 1
    assert learner.queued_update is learner.pending_update is None


def test_last_four_overlap_keeps_the_whole_suffix_on_serving_rank_zero():
    learner = TP2Online(None, device=0, last_layers=4, overlap_sampling=True)
    learner.bind_devices((0, 1))
    assert learner.colocated and learner.device == 0 and not learner.background
    assert tuple(learner.layers) == (60, 61, 62, 63)
    assert learner.publish_banks == frozenset(range(15))


@pytest.mark.parametrize('view', ['initial', 'current'])
def test_frozen_interleaving_cannot_drain_queued_work_for_free(view):
    learner = TP2Online(None, device=0, last_layers=1, overlap_sampling=True)
    learner.queued_update = ([], 16, 16)
    learner.pending_evaluation = view
    with pytest.raises(RuntimeError, match='synchronous'):
        learner.prepare_batch()
    assert learner.queued_update is not None


def test_failed_future_prevents_any_next_device_work():
    learner = TP2Online(None, device=0, last_layers=1, overlap_sampling=True)
    learner.runner = SimpleNamespace(_terminal_device_error=None)
    future = Future()
    future.set_exception(FloatingPointError('bad gradient'))
    learner.pending_update = future
    with pytest.raises(FloatingPointError, match='bad gradient'):
        learner.before_device()
    assert learner.runner._terminal_device_error and learner.pending_update is None


@dataclass
class Slot:
    request: object
    phase: str
    position: int
    token_ids: tuple


@pytest.mark.parametrize('queued', [False, True])
def test_runner_prefetches_both_requests_before_learning_and_drains_before_completion(monkeypatch, queued):
    events, busy = [], [False]
    runner = object.__new__(OrthrusTP2Runner)
    def start_update():
        events.append('start')
        busy[0] = True
        online.queued_update = None
    def before_device():
        events.append('wait')
        busy[0] = False
    online = runner.online = SimpleNamespace(
        overlap_sampling=True, evaluation_paused=False, prepare_batch=lambda: None,
        queued_update=object() if queued else None,
        feedback_index=lambda _: None, launch_sampling_update=start_update,
        before_device=before_device, observe_teacher=lambda *a, **k: None,
        finish_batch=lambda _: events.append('finish'), flush_updates=lambda: events.append('drain'))
    original_prefetch = LazyProposals.prefetch
    def prefetch(self, stop):
        assert queued, 'ordinary sampling must not clone a prefetched logits buffer'
        return original_prefetch(self, stop)
    monkeypatch.setattr(LazyProposals, 'prefetch', prefetch)
    runner.stochastic, runner.lazy_sampling = True, True
    runner.output_candidates, runner.draft_batch_size, runner.verify_chunk_size = 31, 2, 4
    runner.spec_stats = Counter()
    runner.control_ids = torch.empty((2, 32), dtype=torch.int32)
    runner.control_values = torch.empty_like(runner.control_ids)
    runner._reserve = lambda *_: None
    runner.cache = SimpleNamespace(commit=lambda *_: None)
    slots = []
    for index in range(2):
        request = SimpleNamespace(request_id=str(index), max_tokens=2, generated=[], stop_token_ids={12},
            sampling=SimpleNamespace(temperature=1), generator=torch.Generator().manual_seed(743+index))
        slots.append(Slot(request, 'decode', 10, (99,)))
    runner.requests = {str(i):dict(slot=i, pages=object()) for i in range(2)}
    def propose(items, capture, proposals, **kwargs):
        for slot, budget in items:
            def read(first, count, identity=slot.request.request_id):
                assert not busy[0], 'device read overlapped a running learner'
                events.append('read'+identity)
                value = torch.full((count, 128), -1000.)
                value[:, 10] = 1000.
                return value
            proposals[slot.request.request_id] = LazyProposals(budget, read, SamplingConfig(1), slot.request.generator)
        return [[] for _ in items]
    runner.propose_many = propose
    def device_call(self, *args, **kwargs):
        assert not busy[0]
        events.append('gpu')
    monkeypatch.setattr(TensorParallelRunner, '_worker_call', device_call)
    def execute(*args, **kwargs):
        runner._worker_call('run', None)
        return 2, None
    runner._execute = execute
    runner._distribution = lambda batch, index, row, request: torch.nn.functional.one_hot(
        torch.tensor(10 if row == 0 else 12), 128).float()
    assert runner.run_batch(slots) == [[10, 12], [10, 12]]
    assert events[:2] == ['read0', 'read1']
    if queued:
        assert events[2] == 'start'
    else:
        assert 'start' not in events
    assert events.index('wait') < events.index('gpu')
    assert events[-2:] == ['finish', 'drain']


def test_native_fallback_flushes_and_frozen_native_does_not(monkeypatch):
    runner = object.__new__(OrthrusTP2Runner)
    events = []
    runner.online = SimpleNamespace(overlap_sampling=True, evaluation_paused=False,
        observe_native=lambda *_: events.append('observe'), flush_updates=lambda: events.append('drain'))
    monkeypatch.setattr(TensorParallelRunner, 'run_batch', lambda *_: [[12]])
    assert runner._native_batch([]) == [[12]]
    assert events == ['observe', 'drain']
    runner.online.evaluation_paused = True
    events.clear()
    runner._native_batch([])
    assert not events
