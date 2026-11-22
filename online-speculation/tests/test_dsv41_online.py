from types import SimpleNamespace

import pytest
import torch

from blockspec.dsv41_online import (
    DSV41DelayedTeacherBlock,
    DSV41OnlineConfig,
    DSV41OnlineController,
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
        "active_feedback_blocks": 0,
        "last_update": record,
    }


@pytest.mark.parametrize("kwargs", [
    {"last_layers": 4}, {"teacher_rows": 32}, {"learning_rate": 0},
    {"objective": "cross_entropy"},
])
def test_dsv41_online_policy_rejects_unverified_settings(kwargs):
    with pytest.raises(ValueError):
        DSV41OnlineConfig(**kwargs)
