"""全块 KL 的梯度须独立匹配 autograd；TV 默认行为保持原样。"""

import pytest
import torch

from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
from blockspec.pypto_qwen38.training_reference import evaluate
from blockspec.pypto_qwen38.online import OnlineConfig, OnlineLearner, FeedbackBlock
from test_pypto_qwen38_training import sample_inputs


@pytest.mark.parametrize("past,width,valid", ((0,8,7), (13,32,31)))
def test_kl_manual_chain_matches_independent_autograd(past, width, valid):
    graph = LastLayerGraph(SuffixShape(hidden=128,intermediate=256,vocabulary=128,
        q_heads=4,kv_heads=2,dimension=64,context=64,rotary=32), objective="kl")
    inputs, masters = sample_inputs(graph, fp32=True)
    values = evaluate(graph, inputs, past=past, width=width, valid=valid, normalizer=valid*3, fp32=True)
    logits = values[graph.logits.name]
    teacher = inputs["teacher"].float().softmax(-1)
    expected = torch.nn.functional.kl_div(logits[:valid].log_softmax(-1), teacher[:valid], reduction="sum")/(valid*3)
    torch.testing.assert_close(values[graph.losses.name][:,0].sum(),expected,atol=2e-7,rtol=1e-4)
    gradients = torch.autograd.grad(expected,tuple(masters.values()))
    for (name,_),gradient in zip(masters.items(),gradients,strict=True):
        torch.testing.assert_close(values[graph.gradients[name].name],gradient,atol=3e-8,rtol=3e-4)


def test_full_teacher_block_supervises_all_rows_without_acceptance_input():
    learner = OnlineLearner(None,OnlineConfig(objective="kl",stride=2,replay_blocks=2))
    block = FeedbackBlock(learner,{})
    block.rows = block.width = 8
    learner.current = block
    learner.observe_clean_block()
    assert block.valid == 7 and learner.rounds == 1
    learner.current = None
    with pytest.raises(RuntimeError,match="complete clean teacher"):
        learner.observe_clean_block()
    assert OnlineConfig().objective == "tv"


def test_kl_propagates_through_three_gdn_layers():
    from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
    from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
    from test_pypto_qwen38_suffix_training import sample
    graph=SuffixGraph(4,SuffixShape(hidden=128,intermediate=256,vocabulary=128,q_heads=4,kv_heads=2,
        dimension=64,context=64,rotary=32),GDNShape(heads=8,query_heads=4,dimension=64),objective="kl")
    inputs,masters=sample(graph,True)
    values=evaluate(graph,inputs,past=7,width=8,valid=7,fp32=True)
    logits=values[graph.logits.name]
    loss=torch.nn.functional.kl_div(logits[:7].log_softmax(-1),inputs["teacher"][:7].softmax(-1),reduction="sum")/7
    expected=torch.autograd.grad(loss,tuple(masters.values()))
    for (name,_),gradient in zip(masters.items(),expected,strict=True):
        torch.testing.assert_close(values[graph.gradients[name].name],gradient,atol=3e-8,rtol=.0005)
    assert sum(node.kind=="kl" for node in graph.nodes)==1
    assert not any(node.kind=="tv" for node in graph.nodes)
