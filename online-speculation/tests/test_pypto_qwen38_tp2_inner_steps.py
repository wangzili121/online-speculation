"""Verified replay may take several real Adam steps before one serving publication."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("serving.runtime")

from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online


@pytest.mark.parametrize("steps,publish,limit", [(0, 1, None), (True, 1, None),
                                                (2, 1, None), (3, 4, None), (2, 4, 3)])
def test_groups_end_at_an_unambiguous_publication_boundary(steps, publish, limit):
    with pytest.raises(ValueError, match="inner_steps"):
        TP2Online(None, device=0, last_layers=1, inner_steps=steps,
                  publish_every=publish, max_updates=limit)


@pytest.mark.parametrize("background", [False, True])
def test_each_inner_step_recomputes_at_new_parameters_then_publishes_once(monkeypatch, background):
    trainer = TP2Online(None, device=2 if background else 0, last_layers=1,
                        background=background, inner_steps=2, publish_every=2,
                        warmup_updates=4, learning_rate=1e-4,
                        warm_update_stride=2, warm_update_steps=2)
    copies, calls = [], []
    trainer.runner = SimpleNamespace(spec_stats=Counter(), suffix="initial",
                                     _worker_call=lambda *a, **kw: copies.append((a, kw)))
    trainer.original, trainer.updated, trainer.packed = "initial", [], []
    trainer.initial_step = 0
    trainer.stride = 8
    trainer.optimizer = SimpleNamespace(steps=0)
    trainer.feedback = dict(valid=2, candidates=2, complete=True, served_logits=object(), inputs={})

    def train(replay, round_number):
        rate = trainer.step_learning_rate()
        calls.append((trainer.version, rate, "served_logits" in replay[0]))
        trainer.optimizer.steps += 1
        trainer.version += 1
        return dict(version=trainer.version, round=round_number, update_ns=10,
                    publish_read_ns=0, publish_pack_ns=0)

    monkeypatch.setattr(trainer, "train", train)
    if background:
        trainer.executor = ThreadPoolExecutor(max_workers=1)
    try:
        trainer.finish_batch(2)
        trainer.collect_update(wait=True)
    finally:
        if trainer.executor:
            trainer.executor.shutdown(wait=True)
    assert calls == [(0, 2.5e-5, True), (1, 5e-5, False)]
    assert trainer.scheduled_updates == 1
    assert trainer.version == trainer.completed_version == trainer.published_version == 2
    assert trainer.runner.spec_stats["updates"] == 2
    assert trainer.runner.spec_stats["publications"] == 1
    assert trainer.runner.spec_stats["update_ns"] == 20
    assert trainer.update_interval() == 8
    assert [(r["inner_step"], r["inner_steps"], r["interval"]) for r in trainer.records] == [
        (1, 2, 2), (2, 2, 2)]
    assert [r["published_at_round"] for r in trainer.records] == [None, 2]


def test_failed_group_never_promotes_partial_weights(monkeypatch):
    trainer = TP2Online(None, device=0, last_layers=1, inner_steps=2, publish_every=2)
    trainer.initial_step, trainer.stride = 0, 2
    trainer.runner = SimpleNamespace(spec_stats=Counter(), suffix="initial")
    trainer.feedback = dict(valid=1, candidates=1, complete=True)
    calls = []
    def train(*_):
        calls.append(1)
        if len(calls) == 2:
            raise FloatingPointError("bad second gradient")
        trainer.version += 1
        return dict(version=trainer.version)
    monkeypatch.setattr(trainer, "train", train)
    with pytest.raises(FloatingPointError, match="second gradient"):
        trainer.finish_batch(2)
    assert trainer.completed_version == trainer.published_version == 0
    assert not trainer.records and trainer.runner.suffix == "initial"


@pytest.mark.parametrize("versions", [[1], [1, 3], [2, 3], []])
def test_invalid_group_is_rejected_before_any_commit(monkeypatch, versions):
    trainer = TP2Online(None, device=0, last_layers=1, inner_steps=2, publish_every=2)
    monkeypatch.setattr(trainer, "commit_update", lambda *_: pytest.fail("partial commit"))
    with pytest.raises(RuntimeError, match="update group"):
        trainer.commit_scheduled([dict(version=v) for v in versions])
