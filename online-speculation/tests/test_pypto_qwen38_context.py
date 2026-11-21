"""上下文扩展须同时覆盖 draft、教师历史与训练，不能截断到前 512 个位置。"""

from dataclasses import replace

import pytest
import torch

from blockspec.pypto_qwen38.attention import AttentionShape, attention_golden
from blockspec.pypto_qwen38.paged_attention_programs import capacity_for, make_paged_attention
from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
from blockspec.pypto_qwen38.training_reference import evaluate


@pytest.mark.parametrize("total,capacity", ((1, 128), (128, 128), (129, 256), (256, 256),
                                          (257, 512), (512, 512), (513, 1024), (1024, 1024)))
def test_capacity_covers_history_and_temporary_block(total, capacity):
    assert capacity_for(total) == capacity


@pytest.mark.parametrize("total", (0, -1, 1025, True, 512.0))
def test_invalid_length_does_not_silently_dispatch_to_smaller_kernel(total):
    with pytest.raises(ValueError):
        capacity_for(total)


@pytest.mark.parametrize("shape,capacity", ((AttentionShape(context=512), 1024),
    (AttentionShape(context=2048), None), (AttentionShape(context=1024, query_tile=32), None)))
def test_unsupported_geometry_rejected_before_pypto_import(shape, capacity):
    with pytest.raises(ValueError):
        make_paged_attention(shape, capacity=capacity)


@pytest.mark.parametrize("page_size", (16, 128))
def test_long_history_is_visible_and_target_pools_remain_readonly(page_size):
    shape = AttentionShape(block_size=16, q_heads=2, kv_heads=1, head_dim=64,
                           context=1024, page_size=page_size)
    past, width = 1008, 16
    q = torch.zeros((16, 128), dtype=torch.bfloat16)
    block = torch.zeros((16, 64), dtype=torch.bfloat16)
    pool = torch.zeros((1, 1024 // page_size + 3, page_size, 64), dtype=torch.bfloat16)
    table = torch.arange(1024 // page_size - 1, -1, -1, dtype=torch.int32)
    values = pool.clone()
    # 修改 512 之后、anchor 之前的一个真实历史值，均匀 Attention 必须读到它。
    logical = 800
    values[0, table[logical // page_size], logical % page_size] = 1024.
    original = values.clone()
    result = attention_golden(shape, q, block, block, q, pool, values, table, past, width)
    torch.testing.assert_close(result, torch.full_like(result, .5), rtol=0, atol=0)
    assert torch.equal(values, original)


@pytest.mark.parametrize("objective", ("tv", "kl"))
def test_larger_training_history_preserves_short_prefix_forward_and_gradients(objective):
    torch.set_num_threads(4)
    small = SuffixShape(hidden=128, intermediate=256, vocabulary=128, q_heads=4,
                        kv_heads=2, dimension=64, context=512, rotary=32, block=16)
    original = LastLayerGraph(small, objective=objective)
    extended = LastLayerGraph(replace(small, context=1024), objective=objective)
    generator = torch.Generator().manual_seed(273)
    inputs, longer = {}, {}
    for name, value in original.inputs.items():
        data = torch.randn(value.shape, generator=generator) * .03
        if "gamma" in name:
            data += 1
        inputs[name] = data if value.dtype == "fp32" else data.bfloat16()
        if name.startswith("history_"):
            data = inputs[name].reshape(2, 512, 64)
            longer[name] = torch.cat((data, torch.full_like(data, float("nan"))), dim=1).reshape(2048, 64)
        else:
            longer[name] = inputs[name].clone()
    before = evaluate(original, inputs, past=127, width=8, valid=7)
    after = evaluate(extended, longer, past=127, width=8, valid=7)
    torch.testing.assert_close(after[extended.logits.name][:8], before[original.logits.name][:8], atol=0, rtol=0)
    for name in original.gradients:
        torch.testing.assert_close(after[extended.gradients[name].name], before[original.gradients[name].name],
                                   atol=1e-8, rtol=1e-5)


def test_serving_and_learner_allocate_the_same_extended_context():
    pytest.importorskip("serving.runtime")
    from blockspec.pypto_qwen38.oneserving_tp2 import draft_inputs
    from blockspec.pypto_qwen38.oneserving_tp2_online import TP2Online

    sizes = {name: shape for name, shape, _ in draft_inputs(2, context=1024)}
    assert sizes["table"] == (2, 8) and sizes["bias"] == (2, 1024)
    learner = TP2Online(None, device=0, last_layers=1, training_rows=16, context=1024)
    graph = learner.build_training_graph()
    assert graph.shape.context == 1024
    assert graph.inputs["l63_history_key"].shape == (4096, 256)
    assert graph.inputs["hidden"].shape == (16, 5120)
