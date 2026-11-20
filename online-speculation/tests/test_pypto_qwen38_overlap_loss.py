"""接受重合概率损失：独立 autograd 核对整段手写反向及特殊边界。"""

import pytest
import torch

from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online
from blockspec.pypto_qwen38.training_autograd import AutogradReference
from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
from blockspec.pypto_qwen38.training_reference import evaluate
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from test_pypto_qwen38_training import sample_inputs
from test_pypto_qwen38_suffix_training import sample


@pytest.mark.parametrize("layers", [1, 4])
@pytest.mark.parametrize("weighted", [False, True])
def test_overlap_manual_suffix_matches_independent_autograd(layers, weighted):
    torch.set_num_threads(4)
    shape = SuffixShape(hidden=128, intermediate=256, vocabulary=128,
                        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32)
    options = dict(objective="log_overlap", position_weighted=weighted)
    graph = (LastLayerGraph(shape, **options) if layers == 1 else
             SuffixGraph(4, shape, GDNShape(heads=8, query_heads=4, dimension=64), **options))
    inputs, masters = sample_inputs(graph, fp32=True) if layers == 1 else sample(graph, True)
    weights = torch.tensor([16, 16, 16, 8, 4, 2, 1] if weighted else [1] * 7)
    normalizer = int(weights.sum()) * 3
    if weighted:
        inputs["position_weights"] = torch.cat([weights, torch.zeros(25)])[:, None].repeat(1, 8)
    reference = AutogradReference(graph, {k: v for k, v in inputs.items() if k not in masters},
                                  masters, quantize=False)
    logits = reference.forward(inputs, past=7, width=8)[:7]
    overlap = torch.minimum(logits.softmax(-1), inputs["teacher"][:7].softmax(-1)).sum(-1)
    expected = (-overlap.clamp_min(1e-8).log() * weights).sum() / normalizer
    actual = evaluate(graph, inputs, past=7, width=8, valid=7, normalizer=normalizer, fp32=True)
    torch.testing.assert_close(actual[graph.losses.name][:, 0].sum(), expected)
    gradients = torch.autograd.grad(expected, tuple(masters.values()))
    for name, gradient in zip(masters, gradients, strict=True):
        torch.testing.assert_close(actual[graph.gradients[name].name], gradient, atol=3e-8, rtol=.002)
    loss = next(node for node in graph.nodes if node.kind == "tv")
    assert loss.options["log_overlap"] is True
    delta = actual[loss.outputs[0].name]
    assert torch.count_nonzero(delta[7:]) == 0


@pytest.mark.parametrize("case", ["tied", "disjoint", "partial_ties"])
def test_overlap_gradient_handles_ties_and_clamped_zero_mass(case):
    shape = SuffixShape(hidden=128, intermediate=256, vocabulary=128,
                        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32)
    graph = LastLayerGraph(shape, objective="log_overlap")
    inputs, _ = sample_inputs(graph, fp32=True)
    student = torch.zeros(32, 128)
    teacher = student.clone()
    if case == "disjoint":
        student.fill_(-10000.)
        teacher.fill_(-10000.)
        student[:, 0], teacher[:, 1] = 0., 0.
    elif case == "partial_ties":
        student[:, :2] = torch.tensor([1., -1.])
        teacher[:, :2] = torch.tensor([-1., 1.])
    student.requires_grad_()
    inputs["teacher"] = teacher
    actual = evaluate(graph, inputs, past=7, width=8, valid=7, fp32=True,
                      observed={graph.logits.name: student})
    expected = -torch.minimum(student[:7].softmax(-1), teacher[:7].softmax(-1)).sum(-1).clamp_min(1e-8).log().mean()
    gradient, = torch.autograd.grad(expected, student)
    loss = next(node for node in graph.nodes if node.kind == "tv")
    torch.testing.assert_close(actual[graph.losses.name][:, 0].sum(), expected)
    torch.testing.assert_close(actual[loss.outputs[0].name], gradient, atol=1e-9, rtol=1e-5)
    if case in {"tied", "disjoint"}:
        assert torch.count_nonzero(actual[loss.outputs[0].name]) == 0


def test_overlap_is_explicit_and_does_not_change_existing_tv_or_teacher_contract():
    graph = LastLayerGraph()
    assert "log_overlap" not in next(node for node in graph.nodes if node.kind == "tv").options
    trainer = TP2Online(None, device=0, last_layers=1, objective="log_overlap")
    assert trainer.objective == "log_overlap" and trainer.teacher_targets == "distribution"
    with pytest.raises(ValueError, match="greedy"):
        TP2Online(None, device=0, last_layers=1, objective="log_overlap", teacher_targets="greedy")
