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


def test_resident_evaluation_views_preserve_learning_state_and_never_publish_uninitialized_banks():
    trainer = TP2Online(None, device=0, last_layers=1)
    trainer.initial_step = 0
    trainer.version = trainer.completed_version = trainer.published_version = 4
    trainer.rounds = 128
    trainer.original, trainer.updated = object(), object()
    trainer.runner = SimpleNamespace(suffix=trainer.updated)
    trainer.replay = [object()]
    feedback, replay = trainer.feedback, list(trainer.replay)
    trainer.collect_update = lambda **_: None
    trainer.pending_evaluation = "initial"
    trainer.prepare_batch()
    assert trainer.evaluation_paused and trainer.runner.suffix is trainer.original
    assert (trainer.rounds, trainer.version, trainer.published_version) == (128, 4, 4)
    assert trainer.replay == replay and trainer.feedback is feedback
    trainer.pending_evaluation = "current"
    trainer.prepare_batch()
    assert not trainer.evaluation_paused and trainer.runner.suffix is trainer.updated
    trainer.published_version = trainer.initial_step
    trainer.pending_evaluation = "current"
    trainer.prepare_batch()
    assert trainer.runner.suffix is trainer.original


@pytest.mark.parametrize("background,pending", [(True, None), (False, object())])
def test_view_switch_rejects_background_work_before_any_collection(background, pending):
    trainer = TP2Online(None, device=0, last_layers=1, background=background)
    trainer.pending_update, trainer.pending_evaluation = pending, "initial"
    trainer.collect_update = lambda **_: pytest.fail("background work must not run during frozen evaluation")
    with pytest.raises(RuntimeError, match="synchronous"):
        trainer.prepare_batch()


def test_frozen_view_does_not_send_native_rows_to_the_learner(monkeypatch):
    from blockspec.pypto_qwen38.oneserving_tp2 import OrthrusTP2Runner, TensorParallelRunner
    runner = object.__new__(OrthrusTP2Runner)
    runner.online = SimpleNamespace(evaluation_paused=True,
                                   observe_native=lambda *_: pytest.fail("frozen feedback"))
    expected = object()
    monkeypatch.setattr(TensorParallelRunner, "run_batch", lambda *_: expected)
    assert runner._native_batch([]) is expected


@pytest.mark.parametrize("layers,indices", [(1, {12, 13}), (4, {8, 12, 13})])
def test_norm_scope_only_packs_changed_norm_banks(layers, indices):
    from math import prod
    from blockspec.pypto_qwen38.oneserving_tp2 import SHAPES
    from blockspec.pypto_qwen38.oneserving_tp2_online import pack_execution
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
    from blockspec.pypto_qwen38.training_flat import FlatLayout
    trainer = TP2Online(None, device=0, last_layers=layers, parameter_scope="norms")
    assert trainer.publish_banks == indices
    graph = SuffixGraph(layers, parameter_scope="norms")
    layout = FlatLayout(graph.parameters)
    master = torch.arange(layout.elements, dtype=torch.float32).reshape(layout.shape) / 1024
    ordinary = master.bfloat16()
    special = torch.zeros(layout.special_shape, dtype=torch.float32)
    for entry in layout.entries:
        if entry.special is not None:
            size = prod(entry.parameter.shape)
            special[0, entry.special:entry.special + size] = ordinary[0, entry.offset:entry.offset + size].float() + 1
    packed = [torch.empty((2, *shape), dtype=dtype) if i in indices else None
              for i, (shape, dtype) in enumerate(SHAPES)]
    pack_execution(layout, ordinary, special, packed)
    for bank in indices:
        assert torch.equal(packed[bank][0], packed[bank][1])
        assert torch.isfinite(packed[bank]).all()
    assert sum(x.nbytes for x in packed if x is not None) < 8192


@pytest.mark.parametrize("layers,banks", [(1, {12, 13, 14}), (4, {2, 8, 12, 13, 14})])
def test_output_norm_scope_publishes_only_selected_banks(layers, banks):
    trainer = TP2Online(None, device=0, last_layers=layers, parameter_scope="output_norms")
    assert trainer.publish_banks == banks
    with pytest.raises(ValueError, match="fresh Adam"):
        TP2Online("optimizer-checkpoint", device=0, last_layers=layers, parameter_scope="output_norms")


def test_colocated_background_is_rejected_before_worker_start():
    trainer = TP2Online(None, device=0, last_layers=1, background=True)
    with pytest.raises(ValueError, match="colocated training requires synchronous updates"):
        trainer.bind_devices((0, 1))


@pytest.mark.parametrize("layers,scope,device", [(1, "full", True), (1, "output_norms", True),
                                                (1, "norms", False), (4, "full", False)])
def test_publication_auto_uses_the_measured_path(layers, scope, device):
    assert TP2Online(None, device=0, last_layers=layers, parameter_scope=scope).device_publication is device
    assert not TP2Online(None, device=0, last_layers=layers, publication_backend="cpu").device_publication
    with pytest.raises(ValueError, match="publication"):
        TP2Online(None, device=0, last_layers=layers, publication_backend="unknown")


def test_device_publication_copies_owned_bases_with_explicit_offsets(monkeypatch):
    from blockspec.pypto_qwen38 import oneserving_tp2_online as module
    trainer = TP2Online(None, device=0, last_layers=1, publication_backend="device")
    trainer.initial_step = 0
    copied, runs = [], []
    trainer.worker = SimpleNamespace(copy_from=lambda *a, **kw: copied.append((a, kw)))
    trainer.publication = SimpleNamespace(
        run=lambda *args: runs.append(args),
        sources=lambda *args: {9: (4096, 128), 14: (8192, 0)})
    trainer.publication_buffers = {14: object()}
    trainer.packed = [None] * 15
    trainer.packed[9] = SimpleNamespace(data_ptr=lambda: 10000, nbytes=64)
    trainer.packed[14] = SimpleNamespace(data_ptr=lambda: 20000, nbytes=32)
    monkeypatch.setattr(module, "pack_execution", lambda *args: pytest.fail("CPU repacking was called"))
    layout, buffers = object(), (object(), object())
    trainer.publish_flat(layout, buffers, expected_version=0)
    assert trainer.version == 1 and len(runs) == 1
    assert copied == [((10000, 4096, 64), {"src_offset": 128}),
                      ((20000, 8192, 32), {"src_offset": 0})]
    assert all(value >= 0 for value in trainer.pack_stats.values())


@pytest.mark.parametrize("backend", ["cpu", "device"])
def test_dedicated_learner_inherits_its_dma_destinations(backend):
    trainer = TP2Online(None, device=2, last_layers=1, publication_backend=backend)
    trainer.ordinary, trainer.special = object(), object()
    one, two = object(), object()
    trainer.packed = [None, one, None, two]
    assert trainer.learner_host_tensors() == [trainer.ordinary, trainer.special, *(
        [one, two] if backend == "device" else [])]


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
        return dict(round=round_number)

    def publish(result):
        assert get_ident() == scheduler
        published.append(result["round"])

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


def test_two_stage_cadence_keeps_learning_after_the_switch(monkeypatch):
    trainer = TP2Online(None, device=0, last_layers=1, warm_update_stride=2, warm_update_steps=4)
    trainer.runner = SimpleNamespace(spec_stats=Counter(), suffix="start")
    trainer.initial_step = 160
    trainer.version = trainer.completed_version = trainer.published_version = 160
    trainer.stride = 8
    trainer.updated, trainer.packed = [], []
    observed = []

    def train(replay, round_number):
        observed.append((round_number, trainer.update_interval()))
        trainer.version += 1
        return dict(version=trainer.version, round=round_number, update_ns=1,
                    publish_read_ns=0, publish_pack_ns=0)

    monkeypatch.setattr(trainer, "train", train)
    for _ in range(16):
        if trainer.feedback_index(2) is not None:
            trainer.feedback = dict(valid=1, candidates=1, complete=True)
        trainer.finish_batch(2)
    assert observed == [(2, 2), (4, 2), (6, 2), (8, 2), (16, 8), (24, 8), (32, 8)]
    assert trainer.scheduled_updates == trainer.runner.spec_stats["updates"] == 7
    assert trainer.completed_version == trainer.published_version == 167
    assert not trainer.update_limit_reached()


def test_two_stage_async_switch_waits_for_commit_and_records_submitted_interval(monkeypatch):
    trainer = TP2Online(None, device=2, last_layers=1, warm_update_stride=2, warm_update_steps=1)
    trainer.runner = SimpleNamespace(spec_stats=Counter(), suffix="start")
    trainer.initial_step = 0
    trainer.stride, trainer.updated, trainer.packed = 8, [], []
    entered, release = Event(), Event()

    def train(replay, round_number):
        entered.set()
        assert release.wait(5)
        trainer.version += 1
        return dict(version=trainer.version, round=round_number, update_ns=1,
                    publish_read_ns=0, publish_pack_ns=0)

    monkeypatch.setattr(trainer, "train", train)
    with ThreadPoolExecutor(max_workers=1) as executor:
        trainer.executor = executor
        trainer.feedback = dict(valid=1, candidates=1, complete=True)
        trainer.finish_batch(2)
        assert entered.wait(5)
        assert trainer.update_interval() == 2
        trainer.finish_batch(2)
        assert trainer.runner.spec_stats["busy_updates"] == 1
        trainer.stride = None  # 模拟流结束，异步更新仍需完成并记录原来的间隔。
        release.set()
        trainer.collect_update(wait=True)
    trainer.stride = 8
    assert trainer.update_interval() == 8
    assert trainer.records[0]["interval"] == 2
    assert trainer.scheduled_updates == 2


@pytest.mark.parametrize("stride,steps", [(None, 1), (2, 0), (1, 4), (True, 4), (2, -1), (2, True)])
def test_two_stage_cadence_requires_an_explicit_valid_pair(stride, steps):
    with pytest.raises(ValueError, match="two-stage cadence"):
        TP2Online(None, device=0, warm_update_stride=stride, warm_update_steps=steps)
