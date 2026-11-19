"""冻结部分 draft 参数后，前向及保留参数的梯度必须与完整图一致。"""

import pytest
import torch

from blockspec.pypto_qwen38.training_graph import SuffixShape
from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
from blockspec.pypto_qwen38.training_reference import evaluate
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph


def graphs(n, objective, scope="norms"):
    shape = SuffixShape(hidden=128, intermediate=256, vocabulary=128, q_heads=4,
                        kv_heads=2, dimension=64, context=64, rotary=32)
    gdn = GDNShape(heads=8, query_heads=4, dimension=64)
    return [SuffixGraph(n, shape, gdn, objective=objective, parameter_scope=option)
            for option in ("full", scope)]


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("objective", ["tv", "kl"])
@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("scope", ["norms", "output_norms"])
def test_pruned_forward_and_selected_vjp_match_full_graph(n, objective, fp32, scope):
    torch.set_num_threads(4)
    full, reduced = graphs(n, objective, scope)
    torch.manual_seed(71)
    selected = {p.name: p for p in reduced.parameters}
    inputs, masters = {}, {}
    for name, value in full.inputs.items():
        raw = torch.randn(value.shape) * .03
        if name in selected:
            raw = torch.nn.Parameter(raw)
            masters[name] = raw
            x = raw if fp32 else raw.bfloat16()
            inputs[name] = x.float() + 1 if selected[name].offset_norm else x
        elif "gamma" in name:
            inputs[name] = raw + 1
        else:
            inputs[name] = raw if fp32 or value.dtype == "fp32" else raw.bfloat16()
    expected = evaluate(full, inputs, past=7, width=17, valid=4, normalizer=7, fp32=fp32)
    actual = evaluate(reduced, inputs, past=7, width=17, valid=4, normalizer=7, fp32=fp32)
    assert torch.equal(actual[reduced.logits.name], expected[full.logits.name])
    assert torch.equal(actual[reduced.losses.name], expected[full.losses.name])
    independent = torch.autograd.grad(actual[reduced.losses.name][:, 0].sum(), tuple(masters.values()))
    for (name, _), gradient in zip(masters.items(), independent, strict=True):
        got = actual[reduced.gradients[name].name]
        assert torch.equal(got, expected[full.gradients[name].name])
        torch.testing.assert_close(got, gradient, atol=1e-5 if fp32 else .001, rtol=.04)
    assert len(reduced.nodes) < len(full.nodes)
    assert sum(bool(node.options.get("gradient_weight")) for node in reduced.nodes) == (0 if scope == "norms" else n)
    assert len(reduced.frozen_attention) + len(reduced.parameters) == len(full.parameters)


@pytest.mark.parametrize("scope,n,expected", [("norms", 1, 512), ("norms", 4, 896),
                                            ("output_norms", 1, 31457792), ("output_norms", 4, 125830016)])
def test_real_model_partial_update_parameter_count(scope, n, expected):
    graph = SuffixGraph(n, parameter_scope=scope)
    assert sum(torch.tensor(p.shape).prod().item() for p in graph.parameters) == expected
    assert set(graph.gradients) == {p.name for p in graph.parameters}


def test_invalid_scope_does_not_build_a_graph():
    with pytest.raises(ValueError, match="parameter scope"):
        SuffixGraph(1, parameter_scope="other")


def test_memory_estimate_keeps_frozen_draft_constants():
    from blockspec.pypto_qwen38.training_flat import FlatLayout
    from blockspec.pypto_qwen38.training_memory import suffix_memory
    graph = graphs(1, "tv")[1]
    layout = FlatLayout(graph.parameters)
    actual = suffix_memory(graph, layout, 1)["learner"]
    constants = sum(torch.tensor(p.shape).prod().item() *
                    (4 if p.execution_dtype == "fp32" else 2) for p in graph.frozen_attention)
    graph.frozen_attention = []
    assert actual - suffix_memory(graph, layout, 1)["learner"] == constants
