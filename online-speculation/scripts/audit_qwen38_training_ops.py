#!/usr/bin/env python3
"""Compare installed training operators on the actual Qwen TP shard shapes."""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import warnings

import torch
import torch_npu
from torch.nn import functional as F

from blockspec.vllm_ascend.qwen38_ops import GatedRMSNorm, causal_conv


def error(actual, expected):
    delta = actual.detach().float() - expected.detach().float()
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / expected.detach().float().norm().clamp_min(1e-12)))


def measure(function, inputs, repeats):
    def run():
        output = function(*inputs)
        gradients = torch.autograd.grad(output.float().square().mean(), inputs)
        if any(not torch.isfinite(g).all() or not g.count_nonzero() for g in gradients):
            raise AssertionError("missing, zero or nonfinite training gradient")
        return output, gradients

    output, gradients = run()
    for _ in range(3):
        torch.autograd.grad(function(*inputs).float().square().mean(), inputs)
    torch.npu.synchronize()
    times = []
    torch.npu.reset_peak_memory_stats()
    before = torch.npu.memory_allocated()
    for _ in range(repeats):
        start = time.perf_counter()
        torch.autograd.grad(function(*inputs).float().square().mean(), inputs)
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=statistics.median(times), samples_ms=times,
                peak_extra_bytes=torch.npu.max_memory_allocated() - before), output, gradients


def elapsed(operation, repeats):
    for _ in range(3):
        operation()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fla-path", type=Path, help="optional isolated fla-core/FLA installation; never installed globally")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True).text_config
    torch.manual_seed(731)
    report = dict(torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                  device=torch.npu.get_device_name(0), convolution=[], norm=[], gdn=[])
    for length in (1, 16, 32):
        channels = (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                    + cfg.linear_num_value_heads * cfg.linear_value_head_dim) // 2
        x = torch.randn(1, channels, length, device="npu", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(channels, cfg.linear_conv_kernel_dim, device="npu",
                        dtype=torch.bfloat16, requires_grad=True)
        metrics, _, _ = measure(causal_conv, (x, w), args.repeats)
        report["convolution"].append(dict(length=length, channels=channels,
            kernel=cfg.linear_conv_kernel_dim, current=metrics))
        # GDN norm is per value head, not over the packed projection dimension.
        shape = (length * cfg.linear_num_value_heads // 2, cfg.linear_value_head_dim)
        x, z = [torch.randn(shape, device="npu", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
        weight = torch.randn(shape[-1], device="npu", dtype=torch.bfloat16, requires_grad=True)

        def reference(x, z, weight):
            full = x.float()
            normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + 1e-6)
            return (normalized * weight.float() * F.silu(z.float())).to(x.dtype)

        def rms_fused(x, z, weight):
            normalized = torch_npu.npu_rms_norm(x.float(), weight.float(), 1e-6)[0]
            return (normalized * F.silu(z.float())).to(x.dtype)

        original, out, gradients = measure(reference, (x, z, weight), args.repeats)
        fused, actual, grads = measure(rms_fused, (x, z, weight), args.repeats)
        differences = dict(forward=error(actual, out),
                           gradients=[error(a, b) for a, b in zip(grads, gradients, strict=True)])
        if differences["forward"]["relative_l2"] > .001 or any(
                g["relative_l2"] > .01 for g in differences["gradients"]):
            raise AssertionError(differences)
        report["norm"].append(dict(length=length, shape=shape, current=original,
                                    fused_rms=fused, differences=differences))
        norm = GatedRMSNorm(torch.nn.Parameter(weight.detach()), 1e-6)
        torch.testing.assert_close(norm(x, z), reference(x, z, weight), rtol=.01, atol=.001)

    fla_cases = []
    # An inference interface is not a training interface merely because it returns tensors.
    for length in (1, 8, 17, 32):
        nk, nv, dim = cfg.linear_num_key_heads // 2, cfg.linear_num_value_heads // 2, 128
        q, k = [F.normalize(torch.randn(length, nk, dim, device="npu", dtype=torch.bfloat16), dim=-1)
                .detach().requires_grad_() for _ in range(2)]
        v = torch.randn(length, nv, dim, device="npu", dtype=torch.bfloat16, requires_grad=True)
        g = (-torch.rand(length, nv, device="npu")).requires_grad_()
        beta = torch.rand(length, nv, device="npu", dtype=torch.bfloat16, requires_grad=True)
        initial = torch.randn(1, nv, dim, dim, device="npu") * .01

        def reference(q, k, v, g, beta, groups=nv // nk, state=initial, chunk_size=64):
            return torch_chunk_gated_delta_rule(q[None].repeat_interleave(groups, 2),
                k[None].repeat_interleave(groups, 2), v[None], g=g[None], beta=beta[None],
                initial_state=state, output_final_state=False, chunk_size=chunk_size)[0]

        current, out, grads = measure(reference, (q, k, v, g, beta), args.repeats)
        size = 1 << (length - 1).bit_length()
        smaller, actual, actual_grads = measure(partial(reference, chunk_size=size),
                                               (q, k, v, g, beta), args.repeats)
        differences = dict(forward=error(actual, out),
            gradients=[error(a, b) for a, b in zip(actual_grads, grads, strict=True)])
        if differences["forward"]["relative_l2"] > .001 or any(
                item["relative_l2"] > .01 for item in differences["gradients"]):
            raise AssertionError(differences)
        row = dict(length=length, current=current, short_chunk=smaller,
                   chunk_size=size, differences=differences)
        # Check the adapter itself, including final state and immutable AR history.
        from blockspec.vllm_ascend.qwen38_ops import delta_rule
        before = initial.clone()
        repeated = q[None].repeat_interleave(nv // nk, 2), k[None].repeat_interleave(nv // nk, 2)
        expected, state = torch_chunk_gated_delta_rule(*repeated, v[None], g=g[None], beta=beta[None],
            initial_state=initial, output_final_state=True)
        adapted, adapted_state = delta_rule(*repeated, v[None], fallback=torch_chunk_gated_delta_rule,
            groups=nv // nk, g=g[None], beta=beta[None], initial_state=initial, output_final_state=True)
        row["adapter"] = dict(forward=error(adapted, expected), final_state=error(adapted_state, state),
                              history_unchanged=torch.equal(initial, before))
        torch.testing.assert_close(adapted, expected, rtol=.001, atol=.00001)
        torch.testing.assert_close(adapted_state, state, rtol=.001, atol=.00001)
        assert row["adapter"]["history_unchanged"]
        if length == 8:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    out, _ = torch_npu.npu_chunk_gated_delta_rule(q, k, v, beta=beta, g=g,
                        initial_state=initial.transpose(-1, -2).bfloat16().contiguous(),
                        actual_seq_lengths=torch.tensor([length], device="npu", dtype=torch.int32), scale=dim ** -.5)
                    gradients = torch.autograd.grad(out.float().sum(), (q, k, v, g, beta), allow_unused=True)
                    row["fused_gradient_present"] = [gradient is not None for gradient in gradients]
                except RuntimeError as exc:
                    row["fused_error"] = str(exc)
                row["fused_warnings"] = [str(w.message) for w in caught]
        report["gdn"].append(row)
        fla_cases.append(dict(length=length, inputs=(q, k, v, g, beta), initial=initial,
                              expected=expected.detach(), final_state=state.detach(), gradients=grads))
    # Representative TP2 GDN packed-projection weight, not an extrapolated full-step timer.
    master = torch.nn.Parameter(torch.zeros(channels, cfg.hidden_size))
    gradient = torch.randn(master.shape, device="npu", dtype=torch.bfloat16)
    resident = torch.empty_like(gradient)
    report["transfer_probe"] = dict(shape=list(master.shape), host_dtype=str(master.dtype),
        device_dtype=str(gradient.dtype), host_pinned=master.is_pinned(),
        h2d_with_cast=elapsed(lambda: master.to("npu", dtype=torch.bfloat16), args.repeats),
        d2h_with_cast=elapsed(lambda: gradient.to("cpu", dtype=torch.float32), args.repeats))
    with torch.no_grad():
        report["transfer_probe"]["publish"] = elapsed(lambda: resident.copy_(master), args.repeats)
    master.grad = gradient.float().cpu()
    optimizer = torch.optim.AdamW([master], lr=1e-4, weight_decay=0)
    report["transfer_probe"]["optimizer_cpu"] = elapsed(optimizer.step, args.repeats)
    if args.fla_path:
        sys.path.insert(0, str(args.fla_path))
        external = report["external_fla"] = dict(path=str(args.fla_path), cases=[])
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            for sample in fla_cases:
                initial = sample["initial"]
                before = initial.clone()

                def fused(q, k, v, g, beta, history=initial, groups=nv // nk, final=False):
                    result = delta_rule(q[None].repeat_interleave(groups, 2),
                        k[None].repeat_interleave(groups, 2), v[None], g=g[None], beta=beta[None],
                        initial_state=history, output_final_state=final, groups=groups,
                        fallback=torch_chunk_gated_delta_rule, training_fallback=chunk_gated_delta_rule)
                    return result if final else result[0]

                metrics, actual, actual_grads = measure(fused, sample["inputs"], args.repeats)
                _, final = fused(*sample["inputs"], final=True)
                row = dict(length=sample["length"], fused=metrics, forward=error(actual, sample["expected"]),
                    final_state=error(final, sample["final_state"]), history_unchanged=torch.equal(initial, before),
                    gradients=[error(a, b) for a, b in zip(actual_grads, sample["gradients"], strict=True)])
                row["numerical_pass"] = (row["forward"]["relative_l2"] < .01
                    and row["final_state"]["relative_l2"] < .02 and row["history_unchanged"]
                    and all(g["relative_l2"] < .03 for g in row["gradients"]))
                external["cases"].append(row)
            external["numerical_pass"] = all(row["numerical_pass"] for row in external["cases"])
        except Exception:
            external["error"] = traceback.format_exc()
    Path("/results/training-ops.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
