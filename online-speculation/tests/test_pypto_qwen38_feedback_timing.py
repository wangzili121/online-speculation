"""靠近更新点采集完整的真实教师窗口；不改变更新周期或监督长度。"""

import pytest

pytest.importorskip("serving.runtime")
from blockspec.pypto_qwen38.oneserving_tp2_delayed import TP2DelayedOnline


def learner(**kwargs):
    obj = TP2DelayedOnline(None, device=0, last_layers=1, teacher_tokens=7, **kwargs)
    obj.stride, obj.initial_step = 64, 0
    return obj


def test_default_keeps_the_existing_early_collection():
    obj = learner()
    assert obj.feedback_index(2) == 0
    assert obj.feedback_index(2) == 1


def test_near_update_waits_without_consuming_a_capture_or_changing_the_period():
    obj = learner(feedback_timing="near_update")
    for rounds in [0, 2, 30, 46]:
        obj.rounds = rounds
        assert obj.feedback_index(2) is None
    assert obj.captures == 0 and obj.stride == 64 and obj.teacher_tokens == 7
    obj.rounds = 48
    assert obj.feedback_index(2) == 0 and obj.captures == 1
    obj.rounds = 64
    assert obj.feedback_index(2) is None


def test_each_missing_replay_block_reserves_its_own_verification_budget():
    obj = learner(feedback_timing="near_update", replay_blocks=2)
    obj.rounds = 32
    assert obj.feedback_index(2) is None
    obj.rounds = 34
    assert obj.feedback_index(2) == 0
    obj.replay = [object()]
    obj.rounds = 40
    assert obj.feedback_index(2) is None
    obj.rounds = 48
    assert obj.feedback_index(2) == 1
    obj.replay.append(object())
    assert obj.feedback_index(2) is None


def test_full31_teacher_rows_are_kept_even_when_their_lead_spans_the_period():
    obj = learner(feedback_timing="near_update")
    obj.teacher_tokens = 31
    assert obj.feedback_index(2) == 0
    assert obj.teacher_tokens == 31


def test_an_incomplete_window_is_never_replaced_at_the_update_boundary():
    obj = learner(feedback_timing="near_update")
    obj.rounds, obj.window = 62, object()
    window = obj.window
    assert obj.feedback_index(2) is None and obj.window is window


def test_cadence_change_uses_the_current_completed_update_count():
    obj = learner(feedback_timing="near_update", warm_update_stride=8, warm_update_steps=4)
    assert obj.feedback_index(2) == 0  # 短周期仍预留完整监督的时间。
    obj.completed_version, obj.rounds = 4, 32
    assert obj.update_interval() == 64 and obj.feedback_index(2) is None
    obj.rounds = 48
    assert obj.feedback_index(2) == 1


@pytest.mark.parametrize("mode", ["late", "unknown"])
def test_unknown_collection_modes_fail_before_runtime_initialization(mode):
    with pytest.raises(ValueError, match="feedback timing"):
        learner(feedback_timing=mode)
