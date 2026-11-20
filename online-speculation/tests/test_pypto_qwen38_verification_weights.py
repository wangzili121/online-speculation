"""按首个拒绝点加权的数学与反馈归属；参考梯度由 Torch 自动微分产生。"""

from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_qwen38.verification_weights import rejection_weights
from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
from blockspec.pypto_qwen38.training_autograd import AutogradReference
from blockspec.pypto_qwen38.training_reference import evaluate
from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online
from test_pypto_qwen38_training import sample_inputs
from test_pypto_qwen38_suffix_training import sample


@pytest.mark.parametrize("valid,rejection,prefix", [
    (7, 2, [16, 16, 16, 8, 4, 2, 1]),
    (7, 0, [16, 8, 4, 2, 1, 1, 1]),
    (7, 6, [16] * 7), (7, None, [16] * 7), (3, 7, [16] * 3),
])
def test_weighting_keeps_rejection_and_masks_unverified_rows(valid, rejection, prefix):
    weights, total = rejection_weights(valid, rejection)
    assert weights == prefix + [0] * (32 - valid)
    assert total == sum(prefix)


@pytest.mark.parametrize("valid,rejection", [(0, 0), (32, 1), (3, -1), (3, 31), (3, True)])
def test_weighting_rejects_invalid_boundaries(valid, rejection):
    with pytest.raises(ValueError):
        rejection_weights(valid, rejection)


def test_mixed_replay_preserves_common_weight_scale():
    # 一个无拒绝块与一个早拒绝块共用分母，等价于标准的 1/.5/... 权重。
    left, left_total = rejection_weights(7, None)
    right, right_total = rejection_weights(7, 2)
    actual = torch.tensor(left[:7] + right[:7], dtype=torch.float64) / (left_total + right_total)
    expected = torch.tensor([1.] * 7 + [1., 1., 1., .5, .25, .125, .0625], dtype=torch.float64)
    torch.testing.assert_close(actual, expected / expected.sum(), atol=0, rtol=0)


@pytest.mark.parametrize("valid,rejection,prefix", [
    (7, 2, [1, 1, 16, 1, 1, 1, 1]),
    (3, 0, [16, 1, 1]), (3, 7, [1, 1, 1]), (3, None, [1, 1, 1]),
])
def test_first_rejection_focus_uses_only_an_observed_valid_position(valid, rejection, prefix):
    weights, total = rejection_weights(valid, rejection, mode="first_rejection")
    assert weights == prefix + [0] * (32 - valid)
    assert total == sum(prefix)


def test_unknown_weighting_mode_is_rejected():
    with pytest.raises(ValueError):
        rejection_weights(7, 2, mode="unknown")


def test_first_rejection_mixed_replay_keeps_common_unit_weight():
    left, lt = rejection_weights(7, None, mode="first_rejection")
    right, rt = rejection_weights(7, 2, mode="first_rejection")
    actual = torch.tensor(left[:7] + right[:7], dtype=torch.float64) / (lt + rt)
    expected = torch.ones(14, dtype=torch.float64)
    expected[9] = 16
    torch.testing.assert_close(actual, expected / expected.sum(), atol=0, rtol=0)


@pytest.mark.parametrize("objective", ["tv", "kl"])
@pytest.mark.parametrize("last_layers", [1, 4])
@pytest.mark.parametrize("weighting", ["rejection_decay", "first_rejection"])
def test_weighted_manual_chain_matches_independent_autograd(objective, last_layers, weighting):
    shape = SuffixShape(hidden=128, intermediate=256, vocabulary=128, q_heads=4, kv_heads=2,
                        dimension=64, context=64, rotary=32)
    graph = (LastLayerGraph(shape, objective=objective, position_weighted=True) if last_layers == 1 else
             SuffixGraph(4, shape, GDNShape(heads=8, query_heads=4, dimension=64),
                         objective=objective, position_weighted=True))
    inputs, masters = sample_inputs(graph, fp32=True) if last_layers == 1 else sample(graph, True)
    weights, normalizer = rejection_weights(7, 2, mode=weighting)
    inputs["position_weights"] = torch.tensor(weights, dtype=torch.float32)[:, None].repeat(1, 8)
    reference = AutogradReference(graph, {k: v for k, v in inputs.items() if k not in masters},
                                  masters, quantize=False)
    logits = reference.forward(inputs, past=7, width=8)[:7]
    log_q, log_p = logits.log_softmax(-1), inputs["teacher"][:7].log_softmax(-1)
    per_row = ((log_q.exp() - log_p.exp()).abs().sum(-1) * .5 if objective == "tv" else
               (log_p.exp() * (log_p - log_q)).sum(-1))
    expected = (per_row * torch.tensor(weights[:7])).sum() / (normalizer * 3)
    actual = evaluate(graph, inputs, past=7, width=8, valid=7, normalizer=normalizer * 3, fp32=True)
    torch.testing.assert_close(actual[graph.losses.name][:, 0].sum(), expected)
    gradients = torch.autograd.grad(expected, tuple(masters.values()))
    for name, gradient in zip(masters, gradients, strict=True):
        torch.testing.assert_close(actual[graph.gradients[name].name], gradient, atol=3e-8, rtol=.002)
    loss_node = next(n for n in graph.nodes if n.kind == objective)
    assert loss_node.inputs[2].name == "position_weights"


def test_capture_ignores_other_drafts_and_keeps_first_original_rejection():
    trainer = object.__new__(TP2Online)
    trainer.feedback = dict(position=10, first_rejection=None)
    trainer.observe_verification(SimpleNamespace(position=12), 2, 1, True, selected=False)
    trainer.observe_verification(SimpleNamespace(position=12), 2, 2, False, selected=True)
    assert trainer.feedback["first_rejection"] is None
    trainer.observe_verification(SimpleNamespace(position=12), 2, 1, True, selected=True)
    assert trainer.feedback["first_rejection"] == 3
    with pytest.raises(RuntimeError):
        trainer.observe_verification(SimpleNamespace(position=12), 2, 1, True, selected=True)


@pytest.mark.parametrize("weighting", ["rejection_decay", "first_rejection"])
def test_feedback_owns_weights_and_counts_their_transfer(weighting):
    trainer = object.__new__(TP2Online)
    trainer.position_weighting = weighting
    trainer.runner = SimpleNamespace(spec_stats={"feedback_ns": 0})
    trainer.feedback = dict(complete=True, valid=7, candidates=31, first_rejection=2, inputs={})
    uploaded = {}
    trainer.upload = lambda key, value: uploaded.update({key: value.clone()})
    feedback = trainer.take_feedback()
    weights, total = rejection_weights(7, 2, mode=weighting)
    assert feedback["supervision_weight"] == total
    assert torch.equal(uploaded["position_weights"][:, 0], torch.tensor(weights))
    assert trainer.feedback is None and trainer.runner.spec_stats["feedback_ns"] > 0


@pytest.mark.parametrize("mode", ["rejection_decay", "first_rejection"])
def test_dynamic_and_static_position_weights_cannot_be_mixed(mode):
    with pytest.raises(ValueError):
        LastLayerGraph(position_weighted=True, focus_weight=4)
    with pytest.raises(ValueError):
        TP2Online(None, device=0, last_layers=1, position_weighting=mode, focus_weight=4)


def test_power_of_two_teacher_temperature_preserves_bf16_greedy_labels():
    trainer = object.__new__(TP2Online)
    trainer.teacher_temperature = .0625
    logits = torch.tensor([[12.8125, 12.875, 12.75], [-31.0, -30.875, -31.25]], dtype=torch.bfloat16)
    sharpened = trainer.training_teacher(logits)
    assert torch.equal(sharpened.float(), logits.float() * 16)
    assert torch.equal(sharpened.argmax(-1), logits.argmax(-1))
