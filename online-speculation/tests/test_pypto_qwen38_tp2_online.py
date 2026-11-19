"""Exercise the serving and learner handoff without allocating an NPU."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("serving.runtime")

from blockspec.pypto_qwen38.oneserving_tp2 import check_sampled_chunk
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online


def test_colocated_background_is_rejected_before_worker_start():
    trainer = TP2Online(None, device=0, last_layers=1, background=True)
    with pytest.raises(ValueError, match="colocated training requires synchronous updates"):
        trainer.bind_devices((0, 1))


def test_serving_can_advance_while_one_update_is_pending(monkeypatch):
    trainer = TP2Online(None, device=1, last_layers=1, background=True)
    trainer.runner = SimpleNamespace(spec_stats=Counter())
    trainer.stride = 2
    trainer.feedback = {"valid": 1, "candidates": 1, "complete": True}
    started, release = Event(), Event()
    scheduler = get_ident()
    published = []

    def train(replay, round_number):
        assert get_ident() != scheduler
        assert [state["valid"] for state in replay] == [1]
        started.set()
        assert release.wait(10)
        return round_number

    def publish(round_number):
        assert get_ident() == scheduler
        published.append(round_number)

    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr(trainer, "commit_update", publish)
    with ThreadPoolExecutor(max_workers=1) as trainer.executor:
        try:
            assert trainer.feedback_index(2) == 1
            trainer.finish_batch(2)
            assert started.wait(10)
            trainer.prepare_batch()
            assert published == []
            assert trainer.feedback_index(2) is None
            trainer.finish_batch(2)
            assert trainer.runner.spec_stats["busy_updates"] == 1
        finally:
            release.set()
        trainer.pending_drain = True
        trainer.prepare_batch()
        assert published == [2]
        assert trainer.pending_update is None
        assert trainer.feedback_index(2) == 1


def test_separate_replay_and_publication_cadence(monkeypatch):
    trainer = TP2Online(None, device=1, last_layers=1, replay_blocks=4, publish_every=2)
    trainer.runner = SimpleNamespace(spec_stats=Counter(), suffix="start")
    trainer.initial_step = 0
    trainer.stride = 8
    trainer.updated = ["updated"]
    trainer.packed = [None]
    trained = []

    def train(replay, round_number):
        trained.append((round_number, [block["valid"] for block in replay]))
        return dict(version=len(trained), update_ns=1, publish_read_ns=0, publish_pack_ns=0)

    monkeypatch.setattr(trainer, "train", train)
    for window in range(2):
        for i in range(4):
            assert trainer.feedback_index(2) == 1
            trainer.feedback = dict(valid=i + 1, candidates=4, complete=True)
            trainer.finish_batch(2)
        assert len(trained) == window + 1
        assert trainer.completed_version == window + 1
        assert trainer.published_version == (0 if window == 0 else 2)
    assert trained == [(8, [1, 2, 3, 4]), (16, [1, 2, 3, 4])]
    assert trainer.runner.spec_stats["publications"] == 1
    assert trainer.runner.suffix is trainer.updated


def test_sampled_verification_commits_only_reached_rows():
    first, second = torch.tensor([1., 0.]), torch.tensor([0., 1.])
    sequence = [1, 0, 0, 0]
    q = [first] * 3
    rng = torch.Generator().manual_seed(42)
    assert check_sampled_chunk(sequence, 0, 2, lambda row: first, q, rng) == ([0, 0], 2, 2, False)

    def rejected(row):
        assert row == 0
        return second * 1.00001  # Simulate roundoff in a full-vocabulary softmax.

    assert check_sampled_chunk(sequence, 2, 2, rejected, q, rng) == ([1], 1, 0, True)
    assert check_sampled_chunk(sequence, 2, 2, lambda row: (first, second)[row], q, rng) == ([0, 1], 2, 1, True)


def test_update_budget_counts_from_the_checkpoint_and_stops_feedback():
    trainer = TP2Online(None, device=0, last_layers=1, max_updates=2)
    trainer.initial_step = 160
    trainer.completed_version = 161
    trainer.stride = 2
    assert trainer.feedback_index(2) == 1
    trainer.completed_version = 162
    assert trainer.feedback_index(2) is None
    trainer.finish_batch(2)
    assert trainer.rounds == 0


def test_unlimited_learning_continues_past_64_updates(monkeypatch):
    trainer = TP2Online(None, device=0, last_layers=1)
    trainer.runner = SimpleNamespace(spec_stats=Counter(), suffix="start")
    trainer.initial_step = trainer.version = trainer.completed_version = trainer.published_version = 160
    trainer.stride = 2
    trainer.updated, trainer.packed = [], []

    def train(replay, round_number):
        trainer.version += 1
        return dict(version=trainer.version, round=round_number, update_ns=1,
                    publish_read_ns=0, publish_pack_ns=0)

    monkeypatch.setattr(trainer, "train", train)
    for _ in range(80):
        assert trainer.feedback_index(2) == 1
        trainer.feedback = dict(valid=1, candidates=1, complete=True)
        trainer.finish_batch(2)
    assert trainer.rounds == 160
    assert trainer.completed_version == trainer.published_version == 240
    assert trainer.runner.spec_stats["updates"] == 80
    assert not trainer.update_limit_reached()
