"""活动行裁剪不改变块语义；参考梯度不调用任何手写 backward。"""

import pytest
import torch

from blockspec.pypto_qwen38.training_autograd import AutogradReference
from blockspec.pypto_qwen38.training_graph import LastLayerGraph, SuffixShape
from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph
from blockspec.pypto_qwen38.training_reference import evaluate
from test_pypto_qwen38_suffix_training import sample
from test_pypto_qwen38_training import sample_inputs


@pytest.mark.parametrize("last_layers",(1,4))
def test_active_forward_and_autograd_match_unpadded_fp32_graph(last_layers):
    shape=SuffixShape(hidden=128,intermediate=256,vocabulary=128,q_heads=4,kv_heads=2,
        dimension=64,context=64,rotary=32)
    graph=(LastLayerGraph(shape,objective="kl") if last_layers==1 else
           SuffixGraph(4,shape,GDNShape(heads=8,query_heads=4,dimension=64),objective="kl"))
    inputs,masters=(sample_inputs(graph,fp32=True) if last_layers==1 else sample(graph,True))
    shared={name:value for name,value in inputs.items() if name not in masters}
    reference=AutogradReference(graph,shared,masters,quantize=False)
    actual,logits=reference.loss(inputs,past=7,width=8,valid=7,normalizer=56)
    values=evaluate(graph,inputs,past=7,width=8,valid=7,normalizer=56,fp32=True)
    torch.testing.assert_close(logits,values[graph.logits.name][:8],atol=2e-6,rtol=1e-4)
    torch.testing.assert_close(actual,values[graph.losses.name][:,0].sum(),atol=1e-7,rtol=1e-4)
    gradients=torch.autograd.grad(actual,tuple(masters.values()))
    for (name,_),gradient in zip(masters.items(),gradients,strict=True):
        torch.testing.assert_close(gradient,values[graph.gradients[name].name],atol=3e-8,rtol=.001)
    assert not any(node.kind.endswith(("_dx","_dw")) for node in reference.nodes)


def test_execution_norm_rounds_raw_gamma_before_adding_one():
    graph=LastLayerGraph(SuffixShape(hidden=128,intermediate=256,vocabulary=128,q_heads=4,kv_heads=2,
        dimension=64,context=64,rotary=32),objective="kl")
    inputs,masters=sample_inputs(graph,fp32=True)
    reference=AutogradReference(graph,{},masters)
    execution=reference.execution()
    for name,p in masters.items():
        expected=p.bfloat16().float()+(1 if "gamma" in name else 0)
        torch.testing.assert_close(execution[name],expected,atol=0,rtol=0)
    execution["q_gamma"].sum().backward()
    assert torch.equal(masters["q_gamma"].grad,torch.ones_like(masters["q_gamma"]))


def test_eight_block_gradients_use_the_same_global_normalizer():
    graph=LastLayerGraph(SuffixShape(hidden=128,intermediate=256,vocabulary=128,q_heads=4,kv_heads=2,
        dimension=64,context=64,rotary=32),objective="kl")
    inputs,masters=sample_inputs(graph,fp32=True)
    shared={name:value for name,value in inputs.items() if name not in masters}
    reference=AutogradReference(graph,shared,masters,quantize=False)
    expected={name:torch.zeros_like(p) for name,p in masters.items()}
    for i in range(8):
        block={**inputs,"teacher":inputs["teacher"].roll(i,1)}
        loss,_=reference.loss(block,past=7,width=8,valid=7,normalizer=56)
        loss.backward()
        values=evaluate(graph,block,past=7,width=8,valid=7,normalizer=56,fp32=True)
        for name,value in graph.gradients.items():expected[name]+=values[value.name].detach()
    for name,p in masters.items():torch.testing.assert_close(p.grad,expected[name],atol=3e-8,rtol=.001)
