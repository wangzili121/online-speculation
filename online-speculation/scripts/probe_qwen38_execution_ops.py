#!/usr/bin/env python3
"""NPU forward/backward checks for the production Qwen3.8 execution adapters."""

import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch import nn
from torch.nn import functional as F

from blockspec.vllm_ascend.qwen38_ops import GatedRMSNorm, causal_conv, delta_rule, l2norm, swiglu


def error(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    delta = actual - expected
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-12)))


def main():
    from vllm_ascend.utils import enable_custom_op
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    if not enable_custom_op():
        raise RuntimeError("installed NPU GDN operators required")
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope  # noqa: F401
    init_device_properties_triton()
    torch.manual_seed(71)
    device, report = "npu:0", {"convolution": [], "gdn": [], "activations": [], "attention": [], "gqa": []}
    for length in (1, 4, 17, 32, 36, 65):
        x = torch.randn(1, 128, length, device=device, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(128, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
        with torch.no_grad():
            actual = causal_conv(x, weight)
        expected = causal_conv(x, weight)
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.0001)
        expected.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() for p in (x, weight))
        report["convolution"].append(dict(length=length, forward=error(actual, expected), backward=True))
    for length in (1, 8, 17, 32):
        q, k = [torch.randn(1, length, 8, 128, device=device, dtype=torch.bfloat16).repeat_interleave(3, 2)
                .requires_grad_() for _ in range(2)]
        value = torch.randn(1, length, 24, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        g = (-torch.rand(1, length, 24, device=device)).requires_grad_()
        beta = torch.rand(1, length, 24, device=device, dtype=torch.bfloat16, requires_grad=True)
        initial = torch.randn(1, 24, 128, 128, device=device) * .01
        before = initial.clone()
        kwargs = dict(fallback=torch_chunk_gated_delta_rule, groups=3, g=g, beta=beta,
                      initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True)
        with torch.no_grad():
            actual, final = delta_rule(q, k, value, **kwargs)
        expected, state = delta_rule(q, k, value, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=.025, atol=.001)
        torch.testing.assert_close(final, state, rtol=.025, atol=.001)
        assert torch.equal(initial, before)
        expected.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
                   for p in (q, k, value, g, beta))
        report["gdn"].append(dict(length=length, forward=error(actual, expected),
                                  state=error(final, state), history_unchanged=True, backward=True))
    for length in (1, 16, 32):
        packed = torch.randn(length, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
        reference = packed.detach().float().requires_grad_()
        gate, up = reference.chunk(2, -1)
        actual, expected = swiglu(packed), (F.silu(gate) * up).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.0001)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        torch.testing.assert_close(packed.grad.float(), reference.grad, rtol=.025, atol=.0001)
        x, gate = [torch.randn(length, 128, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
        norm = GatedRMSNorm(nn.Parameter(torch.randn(128, device=device, dtype=torch.bfloat16)), 1e-6)
        with torch.no_grad():
            forward = norm(x, gate)
            norm_l2 = l2norm(x)
        differentiable = norm(x, gate)
        torch.testing.assert_close(forward, differentiable, rtol=.01, atol=.001)
        torch.testing.assert_close(norm_l2, l2norm(x), rtol=.01, atol=.0001)
        differentiable.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
                   for p in (x, gate, norm.weight))
        report["activations"].append(dict(length=length, swiglu=error(actual, expected),
                                          norm=error(forward, differentiable), backward=True))
    # 原生 GQA 与显式复制 KV 的数学含义一致；反向允许 BF16 归并顺序带来的舍入差异。
    for past in (0, 11, 491, 4096):
        for causal in (False, True):
            q = torch.randn(1, 12, 32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
            k, v = [torch.randn(1, 2, past + 32, 256, device=device, dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(2)]
            blocked = torch.zeros(1, 1, 32, past + 32, device=device, dtype=torch.bool)
            if causal:
                blocked |= torch.arange(past + 32, device=device)[None] > torch.arange(past, past + 32,
                                                                                     device=device)[:, None]
            def attention(key, value, query=q, mask=blocked):
                return torch_npu.npu_fusion_attention(query, key, value, 12, input_layout="BNSD",
                    atten_mask=mask, scale=256 ** -.5, keep_prob=1., sparse_mode=0)[0]
            direct = attention(k, v)
            expanded = attention(k.repeat_interleave(6, dim=1), v.repeat_interleave(6, dim=1))
            forward = error(direct, expanded)
            if forward["relative_l2"] > .001:
                raise AssertionError(f"GQA forward differs: {forward}")
            direct_grad = torch.autograd.grad(direct.float().square().mean(), (q, k, v))
            expanded_grad = torch.autograd.grad(expanded.float().square().mean(), (q, k, v))
            gradients = [error(a, b) for a, b in zip(direct_grad, expanded_grad, strict=True)]
            if any(not torch.isfinite(g).all() for g in direct_grad) or any(
                    g["relative_l2"] > .01 for g in gradients):
                raise AssertionError(f"GQA backward differs: {gradients}")
            report["gqa"].append(dict(past=past, causal=causal, forward=forward, gradients=gradients))
    from blockspec.vllm_ascend.qwen38_weights import SharedRMSNorm
    x, residual = [torch.randn(32, 5120, device=device, dtype=torch.bfloat16, requires_grad=True)
                   for _ in range(2)]
    weight = nn.Parameter(torch.randn(5120, device=device, dtype=torch.bfloat16) * .1, requires_grad=False)
    norm = SharedRMSNorm(weight, 1e-6, native_residual=True)
    with torch.no_grad():
        expected, _, saved = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, 1 + weight, None, 1e-6)
        actual = norm(x + residual)
        full = x.float() + residual.float()
        unrounded = torch_npu.npu_rms_norm(full, torch.ones_like(weight, dtype=torch.float32), 1e-6)[0]
        unrounded = unrounded.bfloat16() * (1 + weight)
        metrics = error(actual, expected)
        if metrics["relative_l2"] > .001:
            raise AssertionError(f"residual norm differs from native: {metrics}")
        assert torch.equal(saved, x + residual)
    norm(x + residual).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() for p in (x, residual))
    report["residual_norm"] = dict(width=5120, rounded=metrics, unrounded=error(unrounded, expected),
                                   residual_equal=True, backward=True)

    import copy
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    from blockspec.parallel.qwen38 import _ScratchCache
    from blockspec.vllm_ascend.qwen38_ops import configure_attention

    config = Qwen3_5TextConfig(hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=256)
    config._attn_implementation = "sdpa"
    mixer = Qwen3_5Attention(config, 0).bfloat16().npu().eval()
    with torch.no_grad():
        mixer.q_norm.weight.normal_(std=.1)
        mixer.k_norm.weight.normal_(std=.1)
    reference = copy.deepcopy(mixer).float().cpu()

    class Rotary:
        def __init__(self):
            angles = torch.randn(64, 32, device=device)
            self.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
            self.mrope_section = [11, 11, 10]
            self.mrope_interleaved = True
            self.rotary_dim = 64

    native_rope = Rotary()
    configure_attention(mixer, native_rope)
    for causal in (False, True):
        mixer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        x = torch.randn(1, 32, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        ref_x = x.detach().cpu().float().requires_grad_()
        key, value = [torch.randn(1, 2, 11, 256, device=device, dtype=torch.bfloat16) for _ in range(2)]
        before = key.clone(), value.clone()
        rotary = tuple(v.squeeze(1) for v in mixer._native_rope.coefficients(
            torch.arange(11, 43, device=device), torch.bfloat16))
        mask = torch.zeros(1, 1, 32, 43, device=device, dtype=torch.bfloat16)
        if causal:
            mask.masked_fill_(torch.arange(43, device=device)[None] > torch.arange(11, 43, device=device)[:, None],
                              float("-inf"))
        with torch.no_grad():
            inference, _ = mixer(x, position_embeddings=rotary, attention_mask=mask,
                past_key_values=_ScratchCache(((key, value),), ("full_attention",)))
        cache = _ScratchCache(((key, value),), ("full_attention",))
        actual, _ = mixer(x, position_embeddings=rotary, attention_mask=mask, past_key_values=cache)
        inference_error = error(inference, actual)
        if inference_error["relative_l2"] > .01:
            raise AssertionError(f"inference and differentiable attention differ: {inference_error}")
        ref_cache = _ScratchCache(((key.cpu().float(), value.cpu().float()),), ("full_attention",))
        expected, _ = reference(ref_x, tuple(v.cpu().float() for v in rotary), mask.cpu().float(), ref_cache)
        metrics = error(actual.cpu(), expected)
        if metrics["relative_l2"] > .025:
            raise AssertionError(f"fused attention differs from FP32 reference: {metrics}")
        actual.float().square().mean().backward()
        expected.bfloat16().float().square().mean().backward()
        gradients = {}
        for name, p in mixer.named_parameters():
            expected_grad = dict(reference.named_parameters())[name].grad
            if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.count_nonzero():
                raise AssertionError(f"invalid attention gradient: {name}")
            gradients[name] = error(p.grad.cpu(), expected_grad)
            if gradients[name]["relative_l2"] > .05:
                raise AssertionError(f"attention gradient differs from reference: {name} {gradients[name]}")
        assert torch.equal(key, before[0]) and torch.equal(value, before[1])
        report["attention"].append(dict(causal=causal, forward=metrics, gradients=gradients,
                                        native_vs_differentiable=inference_error,
                                        history_unchanged=True, backward=True))
    report["passed"] = True
    Path(os.environ["ORTHRUS_EXECUTION_OPS_OUTPUT"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
