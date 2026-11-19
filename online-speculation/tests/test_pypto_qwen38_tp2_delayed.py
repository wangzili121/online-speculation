"""Exercise corrected teacher collection and scheduling without an NPU."""

from collections import Counter
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

import pytest

pytest.importorskip("serving.runtime")
from blockspec.pypto_qwen38.oneserving_tp2_delayed import TP2DelayedOnline
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online


@dataclass
class Slot:
    request: object
    position: int
    token_ids: tuple
    phase: str = "decode"


@pytest.fixture
def learner(monkeypatch):
    trainer = TP2DelayedOnline(None, device=0, last_layers=1)
    trainer.stride = 16
    trainer.initial_step = 0
    trainer.runner = SimpleNamespace(spec_stats=Counter())
    trainer.transfers = []

    def capture(self, slot, offset, count, *, final, source_row=0):
        assert offset == self.feedback["valid"]
        self.transfers.append((slot, offset, count, source_row))
        self.feedback["valid"] += min(count, self.feedback["candidates"] - offset)
        self.feedback["complete"] = final

    monkeypatch.setattr(TP2Online, "capture_teacher", capture)
    return trainer


def begin(trainer, *, width=32, maximum=64, prompt=(10,), stops=()):
    request = SimpleNamespace(request_id="r", prompt_ids=list(prompt), generated=[], max_tokens=maximum,
                              stop_strings=(), stop_token_ids=frozenset(stops),
                              finish_reason=None, cancelled=Event())
    trainer.feedback = dict(width=width, candidates=7, valid=0, complete=False)
    trainer.capture_request(Slot(request, len(prompt) - 1, (prompt[-1],)))
    return request


def observe(trainer, request, inputs, emitted, previous=()):
    position = len(request.prompt_ids) + len(request.generated) + len(previous) - 1
    trainer.observe_teacher(Slot(request, position, tuple(inputs)), 0, emitted, previous, 0,
                            len(emitted), final=True, selected=False)


def test_a_rejection_does_not_end_the_delayed_block(learner):
    request = begin(learner, width=6)
    observe(learner, request, [10, 11, 12], [11, 12, 99])
    assert learner.take_feedback() is None
    assert learner.feedback["valid"] == 3
    request.generated += [11, 12, 99]
    observe(learner, request, [99, 13], [13, 14])
    state = learner.take_feedback()
    assert state["valid"] == 5 and state["full_feedback"]
    assert learner.window is None and learner.ready_for_update() is False


def test_wrong_candidate_branch_never_reaches_the_teacher_buffer(learner):
    request = begin(learner)
    observe(learner, request, [10], [99])
    request.generated += [99]
    with pytest.raises(ValueError, match="wrong-branch"):
        observe(learner, request, [11], [12])
    assert len(learner.transfers) == 1 and learner.feedback["valid"] == 1


@pytest.mark.parametrize("stops,maximum", [((99,), 64), ((), 2)])
def test_eos_and_output_budget_trim_unserved_rows(learner, stops, maximum):
    request = begin(learner, stops=stops, maximum=maximum)
    observe(learner, request, [10, 11, 99], [11, 99, 12])
    state = learner.take_feedback()
    assert state["valid"] == 2 and not state["full_feedback"]
    assert learner.transfers == [(0, 0, 2, 0)]


def test_native_prefill_last_row_can_finish_a_window(learner):
    request = begin(learner, width=2, prompt=(1, 2, 3, 4))
    learner.observe_native([Slot(request, 0, (1, 2, 3, 4), "prefill")], [[5]])
    assert learner.transfers == [(0, 0, 1, 3)]
    assert len(learner.replay) == 1 and learner.replay[0]["valid"] == 1
    assert learner.rounds == 0


def test_waiting_for_future_tokens_is_counted_and_cannot_overlap_training(learner):
    learner.stride = 2
    request = begin(learner)
    observe(learner, request, [10], [11])
    learner.finish_batch(2)
    assert learner.rounds == 2 and not learner.ready_for_update()
    assert learner.runner.spec_stats["feedback_skips"] == 1
    assert learner.feedback_index(2) is None


def test_other_requests_and_second_chunks_remain_separate(learner):
    request = begin(learner, width=6)
    observe(learner, request, [10, 11, 12], [11, 12, 13])
    # Another verifier tile of the same run_batch; generated has not advanced yet.
    observe(learner, request, [13, 14], [14, 15], previous=[11, 12, 13])
    assert learner.feedback["valid"] == 5
    assert learner.transfers == [(0, 0, 3, 0), (0, 3, 2, 0)]


def test_full_k32_supervision_does_not_change_seven_token_proposals(learner):
    request = begin(learner)
    assert learner.feedback["candidates"] == 31
    for i in range(31):
        prefix_token = 10 if i == 0 else request.generated[-1]
        observe(learner, request, [prefix_token], [i + 20])
        request.generated.append(i + 20)
    assert learner.take_feedback()["valid"] == 31


def test_teacher_horizon_does_not_change_the_k32_forward(learner):
    learner.teacher_tokens = 7
    request = begin(learner)
    assert learner.feedback["width"] == 32 and learner.feedback["candidates"] == 7
    for i in range(7):
        anchor = 10 if i == 0 else request.generated[-1]
        observe(learner, request, [anchor], [i + 20])
        request.generated.append(i + 20)
    assert learner.take_feedback()["valid"] == 7


@pytest.mark.parametrize("count", [0, 32, True])
def test_invalid_teacher_horizons_are_rejected_before_loading(count):
    with pytest.raises(ValueError, match="teacher_tokens"):
        TP2DelayedOnline(None, device=0, last_layers=1, teacher_tokens=count)
