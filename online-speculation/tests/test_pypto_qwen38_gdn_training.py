"""GDN 显式反向以 autograd 为独立参照，并覆盖非零历史和部分块。"""

import pytest
import torch

from blockspec.pypto_qwen38.training_gdn_reference import (GDNShape, convolution_backward,
    convolution_forward, gates_backward, gates_forward, recurrence_backward, recurrence_forward, tail_backward, tail_forward)


@pytest.mark.parametrize("width", (1, 2, 17, 32))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_recurrence_vjp(width, dtype):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    s = GDNShape(heads=4, query_heads=2, dimension=64)
    packed = (torch.randn(32, s.channels)*.2).to(dtype).requires_grad_()
    g = (-torch.rand(32, s.heads)*.3).requires_grad_()
    b = torch.rand(32, s.heads).bfloat16().float().requires_grad_()
    history = torch.randn(s.heads, s.dimension, s.dimension)*.03
    old = history.clone()
    output, states, _ = recurrence_forward(packed, g, b, history, width, s)
    dy = torch.randn_like(output)*.1
    expected = torch.autograd.grad((output*dy).sum(), (packed, g, b))
    actual = recurrence_backward(packed.detach(), g.detach(), b.detach(), states.detach(), dy, width, s)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.float(), e.float(), atol=2e-6 if dtype == torch.float32 else .0002,
                                   rtol=.0002 if dtype == torch.float32 else .02)
    assert torch.equal(old, history)
    assert not actual[0][width:].count_nonzero()


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_conv_vjp(dtype):
    torch.manual_seed(19)
    x = torch.randn(32, 128).to(dtype).requires_grad_()
    w = (torch.randn(4, 128)*.1).to(dtype).requires_grad_()
    history = torch.randn(4, 128).to(dtype)
    y, pre = convolution_forward(x, w, history, 17)
    dy = torch.randn_like(y)*.2
    expected = torch.autograd.grad((y*dy).sum(), (x, w))
    actual = convolution_backward(x.detach(), w.detach(), history, pre.detach(), dy, 17)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.float(), e.float(), atol=2e-7 if dtype == torch.float32 else .0001, rtol=.0002)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_gate_vjp(dtype):
    torch.manual_seed(29)
    a = torch.randn(32, 64).to(dtype).requires_grad_()
    b = torch.randn_like(a).requires_grad_()
    al = torch.randn(1, 64, requires_grad=True)
    bias = (torch.randn(1, 64)*.1).to(dtype).requires_grad_()
    g, beta = gates_forward(a, b, al, bias, 48)
    dg, db = torch.randn_like(g), torch.randn_like(beta)
    expected = torch.autograd.grad((g*dg+beta*db).sum(), (a,b,al,bias))
    actual = gates_backward(a.detach(),b.detach(),al.detach(),bias.detach(),dg,db,48)
    for x,y in zip(actual,expected,strict=True):
        torch.testing.assert_close(x.float(),y.float(),atol=3e-5,rtol=2e-5)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_gated_rms_vjp(dtype):
    torch.manual_seed(31)
    core = (torch.randn(32,4,64)*.1).to(dtype).requires_grad_()
    z = torch.randn(32,256).to(dtype).requires_grad_()
    weight = torch.randn(1,64).to(dtype).requires_grad_()
    y,normalized,weighted,inverse = tail_forward(core,z,weight)
    dy = torch.randn_like(y)*.1
    expected = torch.autograd.grad((y*dy).sum(),(core,z,weight))
    actual = tail_backward(core.detach(),z.detach(),weight.detach(),normalized.detach(),weighted.detach(),inverse.detach(),dy)
    for x,y in zip(actual,expected,strict=True):
        torch.testing.assert_close(x.float(),y.float(),atol=2e-6 if dtype == torch.float32 else .003,
                                   rtol=.0002 if dtype == torch.float32 else .02)
