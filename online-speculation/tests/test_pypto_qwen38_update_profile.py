"""计时必须不改变调用行为，且父子窗口不能重复累加。"""

import json

import pytest

from blockspec.pypto_qwen38.update_profile import UpdateProfile


def test_nested_windows_have_correct_exclusive_cost_and_exception(tmp_path):
    clock = iter([0, 10, 30, 50, 60, 70])
    profile = UpdateProfile(tmp_path / "profile.json", clock=lambda: next(clock))
    with profile.span("train"):
        with profile.span("child"):
            pass
    with pytest.raises(ValueError):
        with profile.span("failure"):
            raise ValueError("original failure")
    parent, child, failed = profile.events
    assert (parent["inclusive_ns"], parent["exclusive_ns"], parent["child_ns"]) == (50, 30, 20)
    assert child["parent"] == parent["id"] and child["exclusive_ns"] == 20
    assert failed["parent"] is None and "original failure" in failed["error"]
    assert profile.stack() == []
    profile.write()
    assert json.loads(profile.output.read_text())["completed"]


def test_wrapping_preserves_args_results_errors_and_ignores_initialization(tmp_path):
    sentinel = object()
    class Target:
        def __init__(self):
            self.inputs = []
        def child(self, value, *, flag):
            self.inputs.append((value, flag))
            if flag == "fail":
                raise RuntimeError("child failed")
            return sentinel
        def run(self, value, *, flag):
            return self.child(value, flag=flag)
    original = Target.child
    p = UpdateProfile(tmp_path / "profile.json")
    p.wrap(Target, "child", "child")
    p.wrap(Target, "run", "run", root=True)
    target = Target()
    try:
        assert target.child(1, flag="initialization") is sentinel and not p.events
        assert target.run(2, flag="online") is sentinel
        assert target.inputs == [(1, "initialization"), (2, "online")]
        assert [row["label"] for row in p.events] == ["run", "child"]
        with pytest.raises(RuntimeError, match="child failed"):
            target.run(3, flag="fail")
        assert p.stack() == []
        assert sum(row["exclusive_ns"] for row in p.events[:2]) == p.events[0]["inclusive_ns"]
    finally:
        p.restore()
    assert Target.child is original


def test_train_and_commit_keep_keyword_arguments_and_scalar_metadata(tmp_path):
    class Target:
        def train(self, replay, round_number):
            return dict(version=1, round=round_number, supervised=2, loss=.25)

        def commit(self, result):
            result["publish_upload_ns"] = 42

    profile = UpdateProfile(tmp_path / "profile.json")
    profile.wrap(Target, "train", "train", root=True)
    profile.wrap(Target, "commit", "commit", root=True)
    try:
        target = Target()
        result = target.train(replay=[{"served_logits": object()}], round_number=3)
        assert target.commit(result=result) is None
        assert profile.events[0]["metadata"] == dict(
            feedback_blocks=1, replay_audit=True, version=1, round=3, supervised=2, loss=.25)
        assert profile.events[1]["metadata"]["publish_upload_ns"] == 42
        assert result["publish_upload_ns"] == 42
    finally:
        profile.restore()
