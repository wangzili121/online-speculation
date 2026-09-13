"""Read-only numerical audits of equivalent causal execution layouts."""

from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .distillation import paired_batch


def difference(actual, expected):
    actual, expected = actual.float(), expected.float()
    error = actual - expected
    return {"max_abs": float(error.abs().max()), "rms_abs": float(error.square().mean().sqrt())}


@torch.no_grad()
def audit_paired_teacher(model, clean, *, block_size=4, attention="default", reduced_bf16=True):
    """Compare ordinary and paired teacher rows, recording the earliest drift.

    Attention overrides are diagnostic oracles, not silently enabled model modes.
    All global flags/hooks are restored even if a model call fails.
    """
    if attention not in ("default", "math", "fp32"):
        raise ValueError("unknown attention diagnostic mode")
    length = clean.shape[1]
    rng = torch.Generator(device=clean.device).manual_seed(42017)
    noisy = torch.randint(model.config.vocab_size, clean.shape, device=clean.device, generator=rng)
    paired = paired_batch(clean, block_size, noisy=noisy)
    names = ["model.layers.0.input_layernorm", "model.layers.0.self_attn.q_proj",
             "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj",
             "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj"]
    names += [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    names += ["model.norm", "lm_head"]
    trace = {}
    handles = []
    for name in names:
        def hook(module, inputs, result, key=name):
            value = result[0] if isinstance(result, tuple) else result
            trace[key] = value[:, :length].detach().clone()
        handles.append(model.get_submodule(name).register_forward_hook(hook))
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    def full_precision_attention(q, k, v, **kwargs):
        return original_sdpa(q.float(), k.float(), v.float(), **kwargs).to(q.dtype)
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    context = sdpa_kernel(SDPBackend.MATH) if attention == "math" else nullcontext()
    replacement = patch("torch.nn.functional.scaled_dot_product_attention", full_precision_attention)
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_bf16
        with context, replacement if attention == "fp32" else nullcontext():
            normal_logits = model(clean)
            expected = dict(trace)
            packed_logits = model(paired.tokens, positions=paired.positions, allowed=paired.allowed,
                                  adapter_mask=paired.adapter_mask)[:, :length]
        layers = [{"name": name, **difference(trace[name], expected[name])} for name in names]
        p, q = normal_logits.float().softmax(-1), packed_logits.float().softmax(-1)
        return {"attention": attention, "reduced_bf16": reduced_bf16,
                "dtype": str(next(model.parameters()).dtype),
                "logits": difference(packed_logits, normal_logits),
                "mean_tv": float((p - q).abs().sum(-1).mean() / 2),
                "argmax_agreement": float((p.argmax(-1) == q.argmax(-1)).float().mean()),
                "first_drift": next((x for x in layers if x["max_abs"] > 0), None),
                "layers": layers}
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        for handle in handles:
            handle.remove()
