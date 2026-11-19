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


@pytest.mark.parametrize("weight,objective", [(0, "tv"), (True, "tv"), (2, "kl")])
def test_focus_requires_a_positive_tv_weight(weight, objective):
    with pytest.raises(ValueError, match="TV focus"):
        TP2Online(None, device=0, last_layers=1, objective=objective, focus_weight=weight)


def test_training_teacher_temperature_does_not_mutate_verifier_logits():
    raw = torch.tensor([[-2., 2.]], dtype=torch.bfloat16)
    original = raw.clone()
    trainer = TP2Online(None, device=0, last_layers=1, teacher_temperature=.25)
    assert torch.equal(trainer.training_teacher(raw), raw * 4)
    assert torch.equal(raw, original)
    ordinary = TP2Online(None, device=0, last_layers=1)
    assert ordinary.training_teacher(raw) is raw


def test_warmup_uses_actual_optimizer_steps_including_checkpoint_resume():
    trainer = TP2Online(None, device=0, last_layers=1, learning_rate=1e-4, warmup_updates=4)
    trainer.optimizer = SimpleNamespace(steps=0)
    assert trainer.step_learning_rate() == 2.5e-5
    trainer.optimizer.steps = 3
    assert trainer.step_learning_rate() == 1e-4
    trainer.optimizer.steps = 160
    assert trainer.step_learning_rate() == 1e-4


def test_inverse_sqrt_decays_after_warmup_without_stopping_updates():
    trainer = TP2Online(None, device=0, last_layers=1, learning_rate=1e-4,
                        warmup_updates=4, lr_schedule="inverse_sqrt")
    trainer.optimizer = SimpleNamespace(steps=3)
    assert trainer.step_learning_rate() == 1e-4
    trainer.optimizer.steps = 15
    assert trainer.step_learning_rate() == 5e-5
    trainer.optimizer.steps = 160
    assert 0 < trainer.step_learning_rate() < 5e-5
    assert trainer.max_updates is None


def test_teacher_shards_are_interleaved_by_row_in_one_upload(monkeypatch):
    from blockspec.pypto_qwen38 import oneserving_tp2_online as module
    monkeypatch.setattr(module, "TP_VOCAB", 4)
    trainer = TP2Online(None, device=0, last_layers=1)
    trainer.feedback = dict(candidates=7, valid=2, complete=False)
    trainer.runner = SimpleNamespace(outputs={"target": (None, "logits")},
                                    _last_layout={"output_key": "target"}, spec_stats=Counter())
    reads, uploads = [], []
    def read(rank, source, destination, offset):
        reads.append((rank, offset))
        destination.copy_(torch.arange(12).reshape(3, 4) + rank * 100)
    monkeypatch.setattr(trainer, "read", read)
    monkeypatch.setattr(trainer, "upload", lambda key, data, offset: uploads.append((key, data.clone(), offset)))
    trainer.capture_teacher(1, 2, 3, final=True, source_row=1)
    assert reads == [(0, 40), (1, 40)]
    assert len(uploads) == 1 and uploads[0][2] == 32
    assert uploads[0][1].tolist() == [[0,1,2,3,100,101,102,103],
                                      [4,5,6,7,104,105,106,107],
                                      [8,9,10,11,108,109,110,111]]
    assert trainer.feedback["valid"] == 5 and trainer.feedback["complete"]


@pytest.mark.parametrize("temperature", [0., -1., float("inf"), float("nan")])
def test_invalid_training_teacher_temperatures_are_rejected(temperature):
    with pytest.raises(ValueError, match="teacher temperature"):
        TP2Online(None, device=0, last_layers=1, teacher_temperature=temperature)


def test_focused_replay_uses_weighted_not_raw_position_count():
    trainer = TP2Online(None, device=0, last_layers=1, focus_weight=4)
    calls = []
    trainer.shared, trainer.pack_stats = {}, {}
    trainer.optimizer = SimpleNamespace(zero_grad=lambda: None, tensors=lambda: {}, gradients={},
                                        accumulate=lambda: None, step=lambda *args: {})
    trainer.workspace = SimpleNamespace(run=lambda *args, **kw: calls.append(kw), values={"loss": None})
    trainer.graph = SimpleNamespace(losses=SimpleNamespace(name="loss"))
    trainer.worker = SimpleNamespace(run=lambda *args, **kw: None, copy_from=lambda *args: None)
    trainer.loss_host = torch.zeros((1, 8))
    trainer.loss, trainer.loss_program, trainer.config = SimpleNamespace(data_ptr=0), None, None
    result = trainer.train([dict(valid=n, position=0, width=32, inputs={}) for n in (1, 3, 5)], 64)
    assert [call["normalizer"] for call in calls] == [18, 18, 18]
    assert result["supervised"] == 9 and result["supervision_weight"] == 18


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
