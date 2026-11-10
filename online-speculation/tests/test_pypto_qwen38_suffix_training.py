"""跨 GDN/Full Attention 的完整后段蒸馏梯度。"""

import pytest
import torch

from blockspec.pypto_qwen38.training_gdn_reference import GDNShape
from blockspec.pypto_qwen38.training_graph import SuffixShape
from blockspec.pypto_qwen38.training_reference import evaluate
from blockspec.pypto_qwen38.training_suffix_graph import SuffixGraph


def small_suffix(n):
    return SuffixGraph(n,SuffixShape(hidden=128,intermediate=256,vocabulary=128,q_heads=4,kv_heads=2,
        dimension=64,context=64,rotary=32),GDNShape(heads=8,query_heads=4,dimension=64))


def sample(graph,fp32):
    torch.manual_seed(71)
    parameters = {p.name:p for p in graph.parameters}
    inputs,masters = {},{}
    for name,value in graph.inputs.items():
        x = torch.randn(value.shape)*(.03 if "weight" in name or "mlp" in name or name == "lm_head" else .15)
        if name in parameters:
            param = parameters[name]
            raw = torch.nn.Parameter(x)
            masters[name] = raw
            execution = raw if fp32 or param.raw_dtype == "fp32" else raw.bfloat16()
            inputs[name] = execution.float()+1 if param.offset_norm else execution
        elif "gamma" in name:
            inputs[name] = x+1
        else:
            inputs[name] = x if fp32 or value.dtype == "fp32" else x.bfloat16()
    return inputs,masters


@pytest.mark.parametrize("n", (2,3,4))
@pytest.mark.parametrize("fp32", (True,False))
def test_suffix_vjp_matches_independent_autograd(n,fp32):
    torch.set_num_threads(4)
    graph = small_suffix(n)
    inputs,masters = sample(graph,fp32)
    values = evaluate(graph,inputs,past=7,width=17,valid=4,normalizer=7,fp32=fp32)
    expected = torch.autograd.grad(values[graph.losses.name][:,0].sum(),tuple(masters.values()))
    for (name,_),gradient in zip(masters.items(),expected,strict=True):
        actual = values[graph.gradients[name].name]
        if fp32:
            torch.testing.assert_close(actual,gradient,atol=3e-8,rtol=.0005,msg=lambda msg,name=name:name+": "+msg)
        else:
            error = float(((actual-gradient).norm()/gradient.norm().clamp_min(1e-20)).detach())
            assert error < .04,(name,error)
            assert float((actual-gradient).abs().max().detach()) < .001,name
        assert torch.count_nonzero(actual),name
    assert len(graph.gradients) == 9*(n-1)+6
    assert len(graph.frozen) == n*5+2
    assert graph.capture_layer == 64-n
    assert all(not value.requires_grad for name,value in inputs.items() if name not in masters)


def test_suffix_rejects_layers_outside_validated_group():
    for n in (0,1,5,64):
        with pytest.raises(ValueError):
            small_suffix(n)
