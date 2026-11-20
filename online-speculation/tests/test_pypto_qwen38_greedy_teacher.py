"""硬监督仅使用已提交的 AR greedy token；不读取未验证的候选或教师分片。"""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("serving.runtime")
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online
from blockspec.pypto_qwen38.oneserving_tp2_delayed import TP2DelayedOnline


def learner(monkeypatch, *, delayed=False):
    from blockspec.pypto_qwen38 import oneserving_tp2_online as module
    monkeypatch.setattr(module, "TP_VOCAB", 4)
    cls = TP2DelayedOnline if delayed else TP2Online
    obj = cls(None, device=0, last_layers=1, objective="kl", teacher_targets="greedy")
    obj.feedback_teacher = torch.empty((4, 8), dtype=torch.bfloat16).share_memory_()
    obj.feedback = dict(candidates=7, valid=0, complete=False, width=32)
    # 故意不提供 outputs/read：该路径应完全不读取目标 logits。
    obj.runner = SimpleNamespace(spec_stats=Counter())
    obj.saved = torch.full((32, 8), -7., dtype=torch.bfloat16)
    obj.upload = lambda name, data, offset: obj.saved[offset // 16:offset // 16 + len(data)].copy_(data)
    return obj


@pytest.mark.parametrize("offset,count,labels", [(0, 4, [0, 3, 4, 7]), (4, 3, [7, 0, 5, 2])])
def test_private_rows_and_vocabulary_shard_boundaries(monkeypatch, offset, count, labels):
    obj = learner(monkeypatch)
    obj.feedback["valid"] = offset
    obj.capture_teacher(1, offset, count, final=True, source_row=2, target_ids=labels)
    probs = obj.saved[offset:offset + count].float().softmax(-1)
    assert torch.equal(probs, torch.nn.functional.one_hot(torch.tensor(labels[:count]), 8).float())
    assert (obj.saved[:offset] == -7).all() and (obj.saved[offset + count:] == -7).all()
    assert obj.feedback["valid"] == offset + count and obj.feedback["complete"]


@pytest.mark.parametrize("labels", [None, [], [-1], [8], [True], [1.0]])
def test_invalid_labels_never_upload_or_advance_feedback(monkeypatch, labels):
    obj = learner(monkeypatch)
    with pytest.raises(ValueError, match="verified vocabulary ID"):
        obj.capture_teacher(0, 0, 1, final=True, target_ids=labels)
    assert (obj.saved == -7).all() and obj.feedback["valid"] == 0


@pytest.mark.parametrize("options", [dict(objective="tv"), dict(teacher_temperature=.25),
                                   dict(teacher_scaling="device"), dict(teacher_targets="unknown")])
def test_greedy_objective_is_explicit_and_cannot_silently_change_sampling(options):
    config = dict(device=0, last_layers=1, objective="kl", teacher_targets="greedy")
    config.update(options)
    with pytest.raises(ValueError, match="greedy teacher targets"):
        TP2Online(None, **config)


def test_stochastic_targets_are_rejected_before_capture(monkeypatch):
    obj = learner(monkeypatch)
    with pytest.raises(ValueError, match="deterministic"):
        obj.capture_request(SimpleNamespace(request=SimpleNamespace(sampling=SimpleNamespace(temperature=1))))


def test_delayed_targets_follow_corrected_prefix_and_stop_at_eos(monkeypatch):
    obj = learner(monkeypatch, delayed=True)
    obj.stride = 16
    request = SimpleNamespace(request_id="r", prompt_ids=[1], generated=[], stop_strings=(),
                              stop_token_ids={7}, max_tokens=20, sampling=SimpleNamespace(temperature=0))
    slot = SimpleNamespace(request=request, position=0, token_ids=(1, 2, 7))
    obj.capture_request(slot)
    # 实际输出为 2、7(EOS)；其后的候选 3 不得进入监督。
    obj.observe_teacher(slot, 1, [2, 7, 3], [], 0, 3, final=True, selected=True)
    assert obj.feedback["valid"] == 2 and obj.feedback["complete"]
    assert obj.saved[:2].argmax(-1).tolist() == [2, 7]
    assert (obj.saved[2:] == -7).all()


def test_one_hot_kl_matches_independent_label_cross_entropy_gradient():
    torch.manual_seed(147)
    logits = torch.randn(4, 64, requires_grad=True)
    labels = torch.tensor([0, 31, 32, 63])
    targets = torch.full_like(logits, -10000., dtype=torch.bfloat16)
    targets[torch.arange(4), labels] = 0.
    log_p, log_q = targets.float().log_softmax(-1), logits.log_softmax(-1)
    loss = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    expected = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, logits)[0])
