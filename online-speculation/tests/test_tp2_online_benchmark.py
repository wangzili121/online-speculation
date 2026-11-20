"""Keep capped learning separate from continuous full-stream measurements."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1] / "scripts/benchmark_qwen38_oneserving_tp2_online.py"
spec = importlib.util.spec_from_file_location("tp2_online_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_continuous_measurement_rejects_a_step_cap():
    assert benchmark.update_policy(None, require_continuous=True) == "continuous_periodic"
    assert benchmark.update_policy(64) == "capped_then_frozen"
    with pytest.raises(ValueError, match="cannot have a step cap"):
        benchmark.update_policy(64, require_continuous=True)
    assert benchmark.update_policy(None, require_continuous=True, warm_updates=64) == "continuous_two_stage"
    with pytest.raises(ValueError, match="cannot have a step cap"):
        benchmark.update_policy(64, require_continuous=True, warm_updates=64)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_update_limits(limit):
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.update_policy(limit)


def test_full_stream_and_active_window_are_reconstructible():
    rows = [dict(seconds=2.0, tokens=[[1, 2], [3]],
                 stats=dict(rounds=2, updates=1, publications=1),
                 learner=dict(rounds=2, optimizer_version=161, published_version=161)),
            dict(seconds=1.0, tokens=[[4], [5, 6, 7]],
                 stats=dict(rounds=2, updates=0, publications=0),
                 learner=dict(rounds=2, optimizer_version=161, published_version=161))]
    timeline = benchmark.request_timeline(rows)
    assert [item["requests"] for item in timeline] == [[1, 2], [3, 4]]
    assert sum(sum(item["output_tokens"]) for item in timeline) == 7
    assert sum(item["seconds"] for item in timeline) == 3.0
    assert timeline[0]["stats"]["updates"] == 1
    assert timeline[1]["learner"]["rounds"] == 2


def test_startup_error_keeps_the_underlying_device_failure():
    cause = PermissionError("halMemCtl failed: 13")
    error = RuntimeError("Batch runner failed to start")
    error.__cause__ = cause
    cause.__context__ = error
    assert benchmark.exception_chain(error) == [
        dict(type="RuntimeError", message="Batch runner failed to start"),
        dict(type="PermissionError", message="halMemCtl failed: 13")]


@pytest.mark.parametrize("primary", [False, True])
def test_cleanup_failure_is_saved_without_overwriting_the_primary_error(primary):
    def stop():
        raise RuntimeError("cleanup failed")
    engine = SimpleNamespace(scheduler=SimpleNamespace(thread=SimpleNamespace(ident=1)), stop=stop)
    summary = {"completed": True, **({"error": "primary failure"} if primary else {})}
    saved = []
    if primary:
        benchmark.finish_engine(engine, summary, lambda: saved.append(dict(summary)))
    else:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            benchmark.finish_engine(engine, summary, lambda: saved.append(dict(summary)))
    assert saved[0]["completed"] is False
    assert saved[0]["cleanup_error"] == "RuntimeError('cleanup failed')"
    assert saved[0]["error"] == ("primary failure" if primary else "RuntimeError('cleanup failed')")


def test_normal_shutdown_is_saved():
    engine = SimpleNamespace(scheduler=SimpleNamespace(thread=SimpleNamespace(ident=1)), stop=lambda: None)
    summary, saved = {"completed": True}, []
    benchmark.finish_engine(engine, summary, lambda: saved.append(dict(summary)))
    assert saved == [{"completed": True, "clean_shutdown": True}]


def test_interleaving_preserves_online_order_and_charges_each_group_once():
    trainer = SimpleNamespace(background=False, pending_update=None, rounds=0, version=0,
                              completed_version=0, published_version=0, update_limit_reached=lambda: False)
    calls = []

    def issue(pair, index):
        online = trainer.pending_evaluation == "current"
        if online:
            trainer.rounds += 2
            trainer.version += 1
            trainer.completed_version = trainer.published_version = trainer.version
        calls.append((index, trainer.pending_evaluation, trainer.version))
        return dict(seconds=5.0 if online else 10.0, tokens=[[index], [index]],
                    metrics=[dict(tpot_ms=1)],
                    stats=dict(updates=int(online), publications=int(online), rounds=2, accepted=1))

    frozen, online = benchmark.measure_interleaved([("a", "b"), ("c", "d"), ("e", "f")], issue, trainer)
    assert [(i, view) for i, view, _ in calls] == [
        (0, "initial"), (0, "current"), (1, "current"), (1, "initial"),
        (2, "initial"), (2, "current")]
    assert [row["learner"]["optimizer_version"] for row in online] == [1, 2, 3]
    assert sum(row["stats"]["updates"] for row in frozen) == 0
    assert sum(row["stats"]["updates"] for row in online) == 3
    assert benchmark.summarize(frozen)["seconds"] == 30
    assert benchmark.summarize(online)["seconds"] == 15


@pytest.mark.parametrize("background,pending", [(True, None), (False, object())])
def test_interleaving_cannot_borrow_free_learning_time_from_the_frozen_group(background, pending):
    trainer = SimpleNamespace(background=background, pending_update=pending)
    with pytest.raises(ValueError, match="synchronous"):
        benchmark.measure_interleaved([("a", "b")], lambda *_: pytest.fail("must not generate"), trainer)


def test_ar_triplets_balance_all_six_orders_without_free_updates():
    trainer = SimpleNamespace(background=False, pending_update=None, queued_update=None,
                              rounds=0, version=0, completed_version=0, published_version=0,
                              update_limit_reached=lambda: False)
    calls = []
    def issue(pair, index, ar=False):
        mode = 'ar' if ar else 'online' if trainer.pending_evaluation == 'current' else 'frozen'
        if mode == 'online':
            trainer.rounds += 2
            trainer.version += 1
            trainer.completed_version = trainer.published_version = trainer.version
        else:
            assert trainer.pending_evaluation == 'initial'
        calls.append((index, mode))
        return dict(seconds={'ar': 2., 'frozen': 3., 'online': 1.}[mode], tokens=[[index], [index]],
                    metrics=[dict(tpot_ms=1)],
                    stats=dict(rounds=2, accepted=1, updates=int(mode == 'online'),
                               publications=int(mode == 'online')))
    frozen, online, ar = benchmark.measure_interleaved(
        [('a', 'b')] * 6, issue, trainer, ar_issue=lambda p, i: issue(p, i, ar=True))
    orders = [tuple(mode for _, mode in calls[3*i:3*i+3]) for i in range(6)]
    assert len(set(orders)) == 6
    for mode in ('ar', 'frozen', 'online'):
        assert [sum(order[position] == mode for order in orders) for position in range(3)] == [2, 2, 2]
    assert [r['learner']['optimizer_version'] for r in online] == list(range(1, 7))
    assert benchmark.summarize(ar)['seconds'] == 12.
    assert benchmark.summarize(frozen)['seconds'] == 18.
    assert benchmark.summarize(online)['seconds'] == 6.
    assert sum(r['stats']['updates'] for r in ar + frozen) == 0
    assert trainer.rounds == 12 and trainer.version == 6


@pytest.mark.parametrize('fault', ['hidden_update', 'feedback_cost', 'pending_work'])
def test_ar_control_cannot_advance_or_hide_learning(fault):
    trainer = SimpleNamespace(background=False, pending_update=None, queued_update=None,
                              rounds=0, version=0, completed_version=0, published_version=0,
                              update_limit_reached=lambda: False)
    def ar_issue(pair, index):
        stats = dict(updates=0, publications=0)
        if fault == 'hidden_update': trainer.rounds += 1
        elif fault == 'feedback_cost': stats['feedback_ns'] = 1
        else: trainer.queued_update = object()
        return dict(stats=stats)
    with pytest.raises(RuntimeError, match='learning'):
        benchmark.measure_interleaved([('a', 'b')], lambda *_: pytest.fail('AR should fail first'),
                                      trainer, ar_issue=ar_issue)
