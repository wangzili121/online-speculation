"""独立 autograd 核对手工链式反向，而非用手写梯度生成参考。"""

import pytest
import torch

from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
from blockspec.pypto_qwen38.training_reference import evaluate


def small_graph():
    return LastLayerGraph(SuffixShape(hidden=128, intermediate=256, vocabulary=128,
        q_heads=4, kv_heads=2, dimension=64, context=64, rotary=32))


def sample_inputs(graph, *, fp32=False):
    torch.manual_seed(91)
    inputs, masters = {}, {}
    trainable = set(graph.gradients)
    for name, value in graph.inputs.items():
        x = torch.randn(value.shape) * (.03 if "weight" in name or "mlp" in name or name == "lm_head" else .2)
        if name in trainable:
            raw = torch.nn.Parameter(x)
            masters[name] = raw
            execution = raw if fp32 else raw.bfloat16()
            inputs[name] = execution.float() + 1 if "gamma" in name else execution
        elif "gamma" in name:
            inputs[name] = x + 1
        else:
            inputs[name] = x if fp32 else x.bfloat16()
    return inputs, masters


@pytest.mark.parametrize("past,width,valid", ((0, 32, 31), (7, 17, 4), (32, 32, 1)))
@pytest.mark.parametrize("fp32", (True, False))
def test_manual_last_layer_vjp_matches_autograd(past, width, valid, fp32):
    torch.set_num_threads(4)
    graph = small_graph()
    inputs, masters = sample_inputs(graph, fp32=fp32)
    values = evaluate(graph, inputs, past=past, width=width, valid=valid, fp32=fp32)
    loss = values[graph.losses.name][:, 0].sum()
    grads = torch.autograd.grad(loss, tuple(masters.values()))
    for (name, _), expected in zip(masters.items(), grads, strict=True):
        actual = values[graph.gradients[name].name]
        if fp32:
            torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-4, msg=lambda msg, name=name: name + ": " + msg)
        else:
            # BF16 在共享节点归并梯度时可有不同舍入顺序；同时限制整体误差和最大绝对误差。
            error = (actual - expected).norm() / expected.norm().clamp_min(1e-20)
            assert error < .02, (name, error)
            assert (actual - expected).abs().max() < 2e-5, name
        assert torch.count_nonzero(actual)
    assert len(graph.gradients) == 6
    assert all(not x.requires_grad for name, x in inputs.items() if name not in masters)


def test_prefix_and_history_are_explicit_constants():
    graph = small_graph()
    assert set(graph.gradients) == {"q_weight", "k_weight", "v_weight", "o_weight", "q_gamma", "k_gamma"}
    assert not any(node.kind == "norm_dw" and node.inputs[0].shape[1] == graph.shape.hidden for node in graph.nodes)
    inputs, _ = sample_inputs(graph)
    for args in ((-1, 32, 1), (40, 32, 1), (0, 1, 1), (0, 32, 32)):
        with pytest.raises(ValueError):
            evaluate(graph, inputs, past=args[0], width=args[1], valid=args[2])


def test_replay_uses_one_global_position_normalizer():
    graph = small_graph()
    inputs, masters = sample_inputs(graph, fp32=True)
    values = evaluate(graph, inputs, past=5, width=32, valid=4, normalizer=17, fp32=True)
    reference = evaluate(graph, inputs, past=5, width=32, valid=4, fp32=True)
    for name, gradient in graph.gradients.items():
        torch.testing.assert_close(values[gradient.name], reference[gradient.name]*4/17, atol=2e-8, rtol=2e-4)
    loss = values[graph.losses.name][:, 0].sum()
    gradients = torch.autograd.grad(loss, tuple(masters.values()))
    for (name, _), expected in zip(masters.items(), gradients, strict=True):
        torch.testing.assert_close(values[graph.gradients[name].name], expected, atol=2e-8, rtol=2e-4)


def test_full_attention_returns_input_gradient_for_preceding_draft_layer():
    graph = LastLayerGraph(small_graph().shape, backprop_input=True)
    inputs, _ = sample_inputs(graph, fp32=True)
    inputs["hidden"].requires_grad_()
    values = evaluate(graph, inputs, past=7, width=17, valid=4, fp32=True)
    expected, = torch.autograd.grad(values[graph.losses.name][:, 0].sum(), inputs["hidden"])
    torch.testing.assert_close(values[graph.input_gradient.name], expected, atol=2e-8, rtol=2e-4)
