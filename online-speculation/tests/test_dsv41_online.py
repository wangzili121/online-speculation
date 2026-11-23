import json
from types import SimpleNamespace

import pytest
import torch

from blockspec.dsv41_online import (
    DSV41DelayedTeacherBlock,
    DSV41OnlineConfig,
    DSV41OnlineController,
    DSV41PreparedUpdate,
)
from blockspec.pypto_dsv41_online_programs import loss_vocabulary_tile


def test_v41_full_vocabulary_uses_complete_1280_element_tiles():
    assert loss_vocabulary_tile(129280) == 1280
    assert loss_vocabulary_tile(248320) == 2560
    with pytest.raises(ValueError):
        loss_vocabulary_tile(129281)


def test_dsv41_online_policy_has_two_stage_cadence_and_warmup():
    policy = DSV41OnlineConfig()
    assert policy.update_interval(0) == 16
    assert policy.update_interval(255) == 16
    assert policy.update_interval(256) == 128
    assert policy.should_capture(16, 0)
    assert not policy.should_capture(15, 0)
    assert policy.should_capture(128, 256)
    assert policy.learning_rate_for(1) == pytest.approx(1.25e-6)
    assert policy.learning_rate_for(8) == pytest.approx(1e-5)
    assert policy.learning_rate_for(80) == pytest.approx(1e-5)
    assert policy.should_publish(4) and not policy.should_publish(3)


def test_dsv41_online_policy_loads_the_versioned_study_contract(tmp_path):
    candidate = DSV41OnlineConfig(last_layers=4, objective="forward_kl").contract()
    study = tmp_path / "study.json"
    study.write_text(json.dumps({
        "format": "orthrus-dsv41-gsm8k-study-v1",
        "candidate_default": candidate,
    }))

    loaded = DSV41OnlineConfig.from_study(
        study, overrides={"last_layers": 2, "objective": "tv"},
    )
    assert loaded.contract() == {
        **candidate, "last_layers": 2, "objective": "tv",
    }

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "format": "orthrus-dsv41-gsm8k-study-v1",
        "candidate_default": {**candidate, "typo": 1},
    }))
    with pytest.raises(ValueError, match="complete online policy"):
        DSV41OnlineConfig.from_study(bad)
    with pytest.raises(ValueError, match="unknown DSV4.1 online overrides"):
        DSV41OnlineConfig.from_study(study, overrides={"typo": 1})


def test_controller_prepares_before_device_update_and_serializes_snapshots():
    class Learner:
        last_layers = 1

        def __init__(self):
            self.calls = []

        def update_rows(self, replay, teacher, **options):
            self.calls.append((replay, teacher.clone(), options))
            return .1, .2, len(self.calls)

    learner = Learner()
    policy = DSV41OnlineConfig(
        teacher_rows=1, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=2,
    )
    controller = DSV41OnlineController(learner, policy)
    assert controller.begin_verifier_round()
    assert controller.begin_round("first", (3,), "replay")
    assert not controller.begin_round("second", (4,), "stale")
    prepared = controller.prepare(
        "first", (3,), (3,), (5,), torch.randn(1, 7),
    )
    assert isinstance(prepared, DSV41PreparedUpdate)
    assert learner.calls == [] and controller.completed_updates == 0
    record = controller.apply(prepared)
    assert record["update"] == 1 and len(learner.calls) == 1
    with pytest.raises(ValueError, match="stale"):
        controller.apply(prepared)


@pytest.mark.parametrize("last_layers", [1, 2, 4, 8, 40])
def test_dsv41_online_policy_accepts_a_configurable_suffix(last_layers):
    assert DSV41OnlineConfig(last_layers=last_layers).last_layers == last_layers


def test_delayed_teacher_follows_corrected_prefix_and_stops_at_width():
    replay = SimpleNamespace(marker="snapshot")
    block = DSV41DelayedTeacherBlock("request", [10, 11], replay, width=4)
    first = torch.arange(10, dtype=torch.float32).view(2, 5)
    assert block.append_verified("request", [10, 11], [11, 12], [12, 13], first) == 2
    assert block.prefix == (10, 11, 12, 13) and not block.ready
    second = torch.arange(15, dtype=torch.float32).view(3, 5)
    assert block.append_verified("request", block.prefix, [13, 14, 15], [14, 15, 16], second) == 1
    assert block.full and block.ready and block.teacher().shape == (3, 5)
    assert block.replay is replay


def test_delayed_teacher_rejects_wrong_branch_and_keeps_real_short_block():
    block = DSV41DelayedTeacherBlock("request", [20], object(), width=8)
    rows = torch.randn(2, 7)
    with pytest.raises(ValueError, match="wrong-branch"):
        block.append_verified("request", [20], [99, 21], [21, 22], rows)
    block.append_verified("request", [20], [20, 21], [21, 22], rows)
    assert block.finish() and block.teacher().shape == (2, 7)


def test_request_finish_prepares_real_short_block_without_padding():
    class Learner:
        last_layers = 1

        def update_rows(self, replay, teacher, **options):
            return .1, .2, 1

    controller = DSV41OnlineController(
        Learner(),
        DSV41OnlineConfig(
            teacher_rows=7, warm_update_stride=1, warm_update_steps=1,
            steady_update_stride=1, publish_every=2,
        ),
    )
    assert controller.begin_verifier_round()
    assert controller.begin_round("short", (4,), "snapshot")
    assert controller.prepare(
        "short", (4,), (4, 5), (5, 6), torch.randn(2, 13),
    ) is None
    prepared = controller.finish_request("short")
    assert prepared.teacher.shape == (2, 13)
    assert prepared.replay == "snapshot" and prepared.update_number == 1


def test_controller_updates_only_after_corrected_rows_and_publishes_atomically():
    class Learner:
        def __init__(self):
            self.calls = []

        def update_rows(self, replay, teacher, **options):
            self.calls.append((replay, teacher.clone(), options))
            return 0.25, 0.5, len(self.calls)

    learner, published = Learner(), []
    config = DSV41OnlineConfig(
        teacher_rows=3, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=1,
    )
    controller = DSV41OnlineController(
        learner, config, publisher=lambda owner, version: published.append((owner, version)),
    )
    replay = object()
    assert controller.begin_verifier_round()
    assert controller.begin_round("r", [7], replay)
    first = controller.observe("r", [7], [7, 8], [8, 9], torch.randn(2, 11))
    assert first is None and not learner.calls
    record = controller.observe("r", [7, 8, 9], [9], [10], torch.randn(1, 11))
    assert record["teacher_rows"] == 3 and record["published"]
    assert learner.calls[0][0] is replay and learner.calls[0][1].shape == (3, 11)
    assert learner.calls[0][2]["kind"] == "tv"
    assert published == [(learner, 1)]
    assert controller.summary() == {
        "verifier_rounds": 1,
        "updates": 1,
        "publications": 1,
        "update_seconds": record["update_seconds"],
        "publication_seconds": record["publication_seconds"],
        "active_feedback_blocks": 0,
        "prepared_update": None,
        "last_update": record,
    }


@pytest.mark.parametrize("kwargs", [
    {"last_layers": 41}, {"teacher_rows": 32}, {"learning_rate": 0},
    {"objective": "cross_entropy"},
])
def test_dsv41_online_policy_rejects_unverified_settings(kwargs):
    with pytest.raises(ValueError):
        DSV41OnlineConfig(**kwargs)
