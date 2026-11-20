"""低秩链式反向对照独立 autograd；既有 dense/last-N 路径保持独立。"""

from math import prod

import pytest
import torch

from blockspec.pypto_qwen38.training_flat import FlatLayout
from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
from blockspec.pypto_qwen38.training_graph import SuffixShape
from blockspec.pypto_qwen38.training_lora_graph import LoRAMergeGraph, LoRASuffixGraph
from blockspec.pypto_qwen38.training_reference import evaluate
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph


def graph(n, cls=LoRASuffixGraph, **kwargs):
    return cls(n, SuffixShape(hidden=128, intermediate=256, vocabulary=128, q_heads=4, kv_heads=2,
                              dimension=64, context=64, rotary=32),
               GDNShape(heads=8, query_heads=4, dimension=64), objective="kl", **kwargs)


def inputs(g, *, fp32, zero_b):
    torch.manual_seed(154)
    values, masters = {}, {}
    trainable = {p.name: p for p in g.parameters}
    cast = (lambda x: x) if fp32 else (lambda x: x.bfloat16())
    for name, value in g.inputs.items():
        x = torch.randn(value.shape) * (.03 if "weight" in name or "mlp" in name or name == "lm_head" else .15)
        if name in trainable:
            if name.endswith("_lora_b") and zero_b:
                x.zero_()
            raw = torch.nn.Parameter(x)
            masters[name] = raw
            parameter = trainable[name]
            execution = raw if fp32 or parameter.raw_dtype == "fp32" else raw.bfloat16()
            values[name] = execution.float() + 1 if parameter.offset_norm else execution
        else:
            values[name] = x + 1 if "gamma" in name else x if fp32 or value.dtype == "fp32" else x.bfloat16()
    bases = {}
    for name, adapter in g.adapters.items():
        bases[name] = values[name]
        delta = cast(values[adapter.b.name] @ values[adapter.a.name])
        values[name] = cast(bases[name].float() + delta.float())
    assert all(not v.requires_grad for n, v in values.items()
               if n not in masters and n not in g.adapters)
    return values, masters, bases


@pytest.mark.parametrize("layers", [1, 4])
@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("zero_b", [True, False])
def test_adapter_vjp_matches_independent_autograd(layers, fp32, zero_b):
    torch.set_num_threads(4)
    g = graph(layers)
    values, masters, _ = inputs(g, fp32=fp32, zero_b=zero_b)
    result = evaluate(g, values, past=7, width=17, valid=4, normalizer=7, fp32=fp32)
    expected = torch.autograd.grad(result[g.losses.name][:, 0].sum(), tuple(masters.values()))
    for (name, _), reference in zip(masters.items(), expected, strict=True):
        actual = result[g.gradients[name].name]
        if fp32:
            torch.testing.assert_close(actual, reference, atol=4e-8, rtol=.001,
                                       msg=lambda message, name=name: name + ": " + message)
        elif zero_b and name.endswith("_lora_a"):
            assert torch.count_nonzero(actual) == 0 and torch.count_nonzero(reference) == 0
        else:
            relative = (actual - reference).norm() / reference.norm().clamp_min(1e-20)
            assert relative < .05, (name, float(relative))
            assert (actual - reference).abs().max() < .001, name


@pytest.mark.parametrize("layers", [1, 4])
def test_zero_b_keeps_the_complete_initial_forward_bitwise(layers):
    g = graph(layers)
    values, _, bases = inputs(g, fp32=False, zero_b=True)
    assert all(torch.equal(values[name], base) for name, base in bases.items())
    dense = graph(layers, SuffixGraph)
    original = {name: bases.get(name, values[name]) for name in dense.inputs}
    a = evaluate(g, values, past=7, width=17, valid=4)
    b = evaluate(dense, original, past=7, width=17, valid=4)
    assert torch.equal(a[g.logits.name], b[dense.logits.name])
    assert torch.equal(a[g.losses.name], b[dense.losses.name])


def test_real_model_parameter_reduction_and_special_layout_compatibility():
    for layers in [1, 4]:
        g = LoRASuffixGraph(layers)
        assert g.trainable_parameter_count < g.dense_parameter_count / 30
        full, low = FlatLayout(g.original_parameters), FlatLayout(g.parameters)
        specials = lambda layout: [(e.parameter.name, e.special, e.parameter.shape)
                                   for e in layout.entries if e.special is not None]
        assert specials(full) == specials(low)
    assert LoRASuffixGraph(1).trainable_parameter_count == 2621952


def test_initialization_preserves_nonprojection_parameters_and_has_nonzero_a_zero_b():
    g = graph(1)
    layout = FlatLayout(g.parameters)
    read = lambda p: torch.full(p.shape, .125, dtype=torch.float32 if p.raw_dtype == "fp32" else torch.bfloat16)
    one = g.initial_master(layout, read, seed=71)
    assert torch.equal(one, g.initial_master(layout, read, seed=71))
    for entry in layout.entries:
        p = entry.parameter
        value = one[0, entry.offset:entry.offset + prod(p.shape)]
        if p.name.endswith("_lora_a"):
            assert torch.count_nonzero(value) and value.abs().max() <= p.shape[1] ** -.5
        elif p.name.endswith("_lora_b"):
            assert torch.count_nonzero(value) == 0
        else:
            assert (value == .125).all()


@pytest.mark.parametrize("rank", [0, 32, True, 256])
def test_invalid_or_oversized_rank_is_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        graph(1, rank=rank)


def test_low_rank_gradient_scheduling_keeps_enough_independent_tiles():
    g = LoRASuffixGraph(4)
    for node in g.nodes:
        if "task_group" not in node.options:
            continue
        rows, columns = node.outputs[0].shape
        tasks = rows // (64 if rows % 64 == 0 else 32) * (columns // 64)
        group = node.options["task_group"]
        assert 1 <= group <= 16 and (tasks + group - 1) // group >= min(32, tasks)
    assert not any("task_group" in n.options for n in SuffixGraph(1).nodes)


def test_weight_cast_and_merge_use_bounded_task_grids():
    g = LoRASuffixGraph(4)
    merge = LoRAMergeGraph(g)
    values = [n for n in [*g.nodes, *merge.nodes] if n.kind == "element" and "tile" in n.options]
    assert len(values) == 2 * len(g.adapters)
    for node in values:
        elements = prod(node.outputs[0].shape)
        tile, group = node.options["tile"], node.options["task_group"]
        assert tile == 4096 and 1 <= group <= 16
        assert (elements // tile + group - 1) // group <= 4096
        assert (elements // tile + group - 1) // group >= min(32, elements // tile)


def test_fp32_weight_intermediates_keep_explicit_input_precision():
    g = LoRASuffixGraph(4)
    merge = LoRAMergeGraph(g)
    for current in (g, merge):
        for node in current.nodes:
            if node.kind == "element" and any(v.dtype == "fp32" for v in node.inputs):
                assert node.options["input_fp32"] == tuple(v.dtype == "fp32" for v in node.inputs)
    assert all("input_fp32" not in n.options for n in SuffixGraph(1).nodes)


@pytest.mark.parametrize("layers", [1, 4])
@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("zero_b", [True, False])
@pytest.mark.parametrize("fused", [False, True])
def test_merge_graph_matches_independent_weight_formula_and_preserves_base(layers, fp32, zero_b, fused):
    g = graph(layers)
    tensors, _, bases = inputs(g, fp32=fp32, zero_b=zero_b)
    merge = LoRAMergeGraph(g, fused=fused)
    original = {name: value.clone() for name, value in bases.items()}
    bound = {alias: bases[name] for alias, name in merge.base_inputs.items()}
    bound.update({alias: tensors[name] for alias, name in merge.optimizer_inputs.items()})
    result = evaluate(merge, bound, past=0, width=32, valid=7, fp32=fp32)
    for name, value in merge.targets.items():
        if name in g.adapters:
            adapter = g.adapters[name]
            delta = tensors[adapter.b.name] @ tensors[adapter.a.name]
            expected = bases[name] + delta if fp32 else (bases[name].float() + delta.float()).bfloat16()
        else:
            expected = tensors[name]
        assert torch.equal(result[value.name], expected), name
    assert all(torch.equal(original[name], bases[name]) for name in bases)
    assert set(merge.scratch) == (set() if fused else set(g.adapters))
    assert all(value.shape == g.dense_gradients[name].shape and value.dtype == "fp32"
               for name, value in merge.scratch.items())


def test_fused_merge_keeps_the_same_external_contract_without_dense_delta_outputs():
    g = LoRASuffixGraph(4)
    old, new = LoRAMergeGraph(g), LoRAMergeGraph(g, fused=True)
    assert old.inputs == new.inputs and old.layout.schema() == new.layout.schema()
    assert set(old.targets) == set(new.targets)
    assert sum(n.kind == "lora_merge" for n in new.nodes) == len(g.adapters)
    assert not new.scratch and all(v.dtype == "bf16" for n in new.nodes for v in n.outputs)


@pytest.mark.parametrize("shape", [(128, 128, 32), (0, 128, 64), (True, 128, 64), (128, 129, 64)])
def test_fused_merge_rejects_unaligned_or_invalid_dimensions_before_compilation(shape):
    from blockspec.pypto_qwen38.training_lora_programs import make_weight_merge
    with pytest.raises(ValueError, match="64-aligned"):
        make_weight_merge(*shape)


@pytest.mark.parametrize("fp32", [False, True])
def test_fused_merge_rank128_keeps_the_two_rounding_boundaries(fp32):
    g = graph(1, rank=128)
    values, _, bases = inputs(g, fp32=fp32, zero_b=False)
    results = []
    for fused in (False, True):
        merge = LoRAMergeGraph(g, fused=fused)
        bound = {alias: bases[name] for alias, name in merge.base_inputs.items()}
        bound.update({alias: values[name] for alias, name in merge.optimizer_inputs.items()})
        result = evaluate(merge, bound, past=0, width=32, valid=7, fp32=fp32)
        results.append({name: result[value.name] for name, value in merge.targets.items()})
    assert results[0].keys() == results[1].keys()
    assert all(torch.equal(results[0][name], results[1][name]) for name in results[0])
