#!/usr/bin/env python3
"""Check device KL and logit gradients at the published GLM vocabulary width."""

import argparse
import json
from pathlib import Path

import torch

from blockspec.pypto_glm53_online_loss import make_kl_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mla-output", action="store_true", help="validate the draft output projection's VJP")
    modes.add_argument("--optimizer", action="store_true", help="validate a complete AdamW parameter update")
    modes.add_argument("--vocab-backward", action="store_true", help="validate a frozen TP8 vocabulary shard VJP")
    modes.add_argument("--norm-backward", action="store_true", help="validate final RMSNorm and mHC mean backward")
    modes.add_argument("--kda-backward", action="store_true", help="validate the frozen KDA recurrence adjoint")
    modes.add_argument("--mhc-backward", action="store_true", help="validate the mHC residual combination VJP")
    modes.add_argument("--sinkhorn-backward", action="store_true", help="validate mHC residual normalization VJP")
    modes.add_argument("--swiglu-backward", action="store_true", help="validate the frozen expert activation VJP")
    modes.add_argument("--joint-backward", action="store_true", help="validate joint history/draft softmax VJP")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path")

    from models.glm5_3_flash.config import D, VOCAB
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig

    torch.set_num_threads(4)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    generator = torch.Generator().manual_seed(142)
    if args.joint_backward:
        from blockspec.pypto_glm53_online_attention import make_joint_softmax_backward

        rows, hist, block = 48, 2176, 128
        shape = (rows, hist + block)
        raw = torch.randn(shape, generator=generator).double()
        raw[0, :hist] = -torch.inf
        raw[:, 2051:hist] = -torch.inf
        raw[:, hist + 6:] = -torch.inf
        raw.requires_grad_()
        prob = raw.softmax(-1)
        upstream = torch.randn(shape, generator=generator)
        expected, = torch.autograd.grad((prob * upstream).sum(), (raw,))
        output = torch.empty(shape, dtype=torch.float32)
        execute(make_joint_softmax_backward(rows, hist, block), (prob.detach().float(), upstream, output),
                2, (), config, {"program": "GLM joint softmax VJP", "passed": False}, args.output,
                (expected,), ({"atol": 1e-7, "rtol": 1e-5},))
        return
    if args.swiglu_backward:
        from models.glm5_3_flash.config import FLASH
        from blockspec.pypto_glm53_online_moe import make_swiglu_backward

        shape = (16, FLASH.moe_intermediate_size)
        gate, up = ((torch.randn(shape, generator=generator) * 12).bfloat16() for _ in range(2))
        upstream = torch.randn(shape, generator=generator) * 0.1
        limit = FLASH.swiglu_limit
        g, u = gate.double().requires_grad_(), up.double().requires_grad_()
        bounded_gate, bounded_up = (g.clamp(max=limit), u.clamp(-limit, limit)) if limit else (g, u)
        ref = torch.nn.functional.silu(bounded_gate) * bounded_up
        expected = torch.autograd.grad((ref * upstream).sum(), (g, u))
        outputs = tuple(torch.empty(shape, dtype=torch.float32) for _ in range(2))
        execute(make_swiglu_backward(*shape, limit), (gate, up, upstream, *outputs), 3, (), config,
                {"program": "GLM expert SwiGLU VJP", "passed": False}, args.output, expected,
                ({"atol": 2e-6, "rtol": 2e-5},) * 2)
        return
    if args.sinkhorn_backward:
        from models.glm5_3_flash.mhc import HC_EPS, HC_SINKHORN_ITER
        from blockspec.pypto_glm53_online_mhc import make_sinkhorn_backward

        tokens = 6
        logits = torch.randn((tokens, 4, 4), generator=generator) * 2
        logits[0] *= 5
        upstream = torch.randn(logits.shape, generator=generator)
        z = logits.double().requires_grad_()
        y = z.softmax(-1) + HC_EPS
        for step in range(2 * HC_SINKHORN_ITER - 1):
            y = y / (y.sum(-2 if step % 2 == 0 else -1, keepdim=True) + HC_EPS)
        expected, = torch.autograd.grad((y * upstream).sum(), (z,))

        def pack(tensor):
            return tensor.repeat(1, 2, 2).reshape(-1, 8)

        output = torch.empty((tokens * 8, 8), dtype=torch.float32)
        execute(make_sinkhorn_backward(tokens, HC_SINKHORN_ITER, HC_EPS), (pack(logits), pack(upstream), output),
                2, (), config, {"program": "GLM mHC Sinkhorn VJP", "passed": False}, args.output,
                (pack(expected),), ({"atol": 1e-6, "rtol": 1e-5},))
        return
    if args.mhc_backward:
        from blockspec.pypto_glm53_online_mhc import make_mhc_post_backward

        tokens, streams = 6, 4
        layout = (tokens, streams, D)
        sublayer = torch.randn((tokens, D), generator=generator).bfloat16()
        residual = torch.randn(layout, generator=generator).bfloat16()
        post = torch.rand((tokens, streams), generator=generator).bfloat16()
        mix = torch.rand((tokens, streams, streams), generator=generator).bfloat16()
        upstream = torch.randn(layout, generator=generator)
        y, x, p, r = (value.double().requires_grad_() for value in (sublayer, residual, post, mix))
        output = y[:, None, :] * p[:, :, None] + torch.einsum("tij,tid->tjd", r, x)
        dy, dx, dp, dr = torch.autograd.grad((output * upstream).sum(), (y, x, p, r))
        pad = torch.nn.functional.pad
        inputs = (sublayer, residual.flatten(0, 1), pad(post.float(), (0, 4)), pad(mix.float().flatten(0, 1), (0, 4)),
                  upstream.flatten(0, 1))
        refs = (dy, dx.flatten(0, 1), pad(dp, (0, 4)), pad(dr.flatten(0, 1), (0, 4)))
        outputs = tuple(torch.empty(ref.shape, dtype=torch.float32) for ref in refs)
        execute(make_mhc_post_backward(D, tokens), inputs + outputs, len(inputs), (), config,
                {"program": "GLM mHC post VJP", "passed": False}, args.output,
                refs, ({"atol": 4e-5, "rtol": 3e-5},) * len(refs))
        return
    if args.kda_backward:
        from blockspec.pypto_glm53_online_kda import make_kda_recurrence_backward

        tokens, heads, width = 6, 8, 128
        shape = (tokens, heads, width)
        q, k, v, decay = (torch.randn(shape, generator=generator) for _ in range(4))
        q = torch.nn.functional.normalize(q, dim=-1) * width**-0.5
        k = torch.nn.functional.normalize(k, dim=-1)
        beta = torch.rand((tokens, heads), generator=generator)
        initial = torch.randn((heads, width, width), generator=generator) * 0.05
        replay = [initial]
        ref_q, ref_k, ref_v, ref_decay, ref_beta = (
            x.double().requires_grad_() for x in (q, k, v, -decay.abs(), beta)
        )
        grad = torch.randn(shape, generator=generator) * 0.1
        state, loss = initial.double(), 0
        for step in range(tokens):
            state = state * ref_decay[step, :, None, :].exp()
            memory = (state * ref_k[step, :, None, :]).sum(-1)
            delta = (ref_v[step] - memory) * ref_beta[step, :, None]
            state = state + delta[:, :, None] * ref_k[step, :, None, :]
            replay.append(state.detach().float())
            out = (state * ref_q[step, :, None, :]).sum(-1)
            loss = loss + (out * grad[step]).sum()
        refs = torch.autograd.grad(loss, (ref_q, ref_k, ref_v, ref_decay, ref_beta))
        args_kda = tuple(x.detach().float().reshape(-1, width) for x in (ref_q, ref_k, ref_v, ref_decay))
        args_kda += (beta.reshape(-1, 1).expand(-1, 8).contiguous(), torch.cat(replay).reshape(-1, width),
                     grad.reshape(-1, width))
        refs = tuple(x.reshape(-1, width) for x in refs[:-1]) + (refs[-1].reshape(-1, 1).expand(-1, 8),)
        outputs = tuple(torch.empty(ref.shape, dtype=torch.float32) for ref in refs)
        execute(make_kda_recurrence_backward(tokens, heads), args_kda + outputs, len(args_kda), (), config,
                {"tokens": tokens, "heads": heads, "program": "GLM KDA adjoint", "passed": False}, args.output,
                refs, ({"atol": 1.5e-7, "rtol": 3e-5},) * len(refs))
        return
    if args.norm_backward:
        from models.glm5_3_flash.rmsnorm import EPS
        from blockspec.pypto_glm53_online_head import make_final_norm_backward

        x = torch.randn((16, D), generator=generator).bfloat16()
        x[1].mul_(0.001)
        gamma = (torch.rand(D, generator=generator) + 0.5).bfloat16()
        upstream = torch.randn((16, D), generator=generator) * 0.01
        ref_x = x.double().requires_grad_()
        projected = ref_x * (ref_x.square().mean(-1, keepdim=True) + EPS).rsqrt() * gamma.double()
        dx, = torch.autograd.grad((projected * upstream).sum(), (ref_x,))
        expected = dx[:, None, :].expand(-1, 4, -1).reshape(-1, D) / 4
        output = torch.empty(expected.shape, dtype=torch.float32)
        execute(make_final_norm_backward(D, EPS), (x, gamma, upstream, output), 3, (), config,
                {"program": "GLM final normalization VJP", "passed": False}, args.output,
                (expected,), ({"atol": 2e-6, "rtol": 3e-5},))
        return
    if args.vocab_backward:
        from blockspec.pypto_glm53_online_head import make_vocab_backward

        vocab = VOCAB // 8
        padded = (vocab + 63) // 64 * 64
        gradients = torch.randn((16, padded), generator=generator) * 0.01
        gradients[:, vocab:].zero_()
        weights = (torch.randn((padded, D), generator=generator) * 0.02).bfloat16()
        output = torch.empty((16, D), dtype=torch.float32)
        execute(make_vocab_backward(vocab, D), (gradients, weights, output), 2, (), config,
                {"local_vocab": vocab, "program": "GLM vocabulary VJP", "passed": False}, args.output,
                (gradients @ weights.float(),), ({"atol": 1.5e-6, "rtol": 3e-5},))
        return
    if args.optimizer:
        from blockspec.pypto_glm53_mla import HEADS, VALUE
        from blockspec.pypto_glm53_online_optimizer import make_adamw_update

        shape = (D, HEADS * VALUE)
        gradient, master, moment = (torch.randn(shape, generator=generator) * scale for scale in (0.01, 0.02, 0.01))
        variance = torch.rand(shape, generator=generator) * 1e-4
        serving = torch.empty(shape, dtype=torch.bfloat16)
        scalars = (1e-4, 0.01, 1 / (1 - 0.9**3), 1 / (1 - 0.999**3))
        lr, decay, bias_one, bias_two = scalars
        m = moment * 0.9 + gradient * 0.1
        v = variance * 0.999 + gradient.square() * 0.001
        after = master - lr * (m * bias_one / ((v * bias_two).sqrt() + 1e-8) + master * decay)
        tolerances = ({"atol": 1e-8, "rtol": 2e-6},) * 3 + ({"atol": 0, "rtol": 0.008},)
        execute(make_adamw_update(*shape), (gradient, master, moment, variance, serving), 1, scalars, config,
                {"shape": shape, "program": "MLA AdamW", "passed": False}, args.output,
                (after, m, v, after.bfloat16()), tolerances)
        return
    if args.mla_output:
        from blockspec.pypto_glm53_mla import HEADS, VALUE
        from blockspec.pypto_glm53_online_attention import make_output_backward

        rows, width = 16, HEADS * VALUE
        context = torch.randn((rows, width), generator=generator).bfloat16()
        weight = (torch.randn((D, width), generator=generator) * 0.02).bfloat16()
        upstream = torch.randn((rows, D), generator=generator).bfloat16()
        upstream[5:].zero_()
        weight_grad = torch.empty(weight.shape, dtype=torch.float32)
        context_grad = torch.empty(context.shape, dtype=torch.float32)
        tensors = (context, weight, upstream, weight_grad, context_grad)
        expected = (upstream.float().T @ context.float(), upstream.float() @ weight.float())
        report = {"rows": rows, "program": "MLA output VJP", "passed": False}
        execute(make_output_backward(), tensors, 3, (), config, report, args.output, expected,
                ({"atol": 2e-5, "rtol": 2e-5}, {"atol": 2e-5, "rtol": 2e-5}))
        return

    rows, active = 6, 5
    student = torch.randn((rows, VOCAB), generator=generator)
    teacher = torch.randn((rows, VOCAB), generator=generator)
    student[1, -1] = 60
    teacher[2, 0] = 65
    student[3].fill_(0)
    teacher[3].fill_(0)
    gradient, loss = torch.empty_like(student), torch.empty((rows, 8), dtype=torch.float32)
    tensors = (student, teacher, gradient, loss)
    expected_p = teacher.double().log_softmax(-1)
    expected_q = student.double().log_softmax(-1)
    expected_loss = (expected_p.exp() * (expected_p - expected_q)).sum(1) / active
    expected_grad = (expected_q.exp() - expected_p.exp()) / active
    expected_loss[active:] = 0
    expected_grad[active:] = 0

    report = {"vocabulary": VOCAB, "rows": rows, "active_rows": active, "passed": False}
    execute(make_kl_program(rows, VOCAB), tensors, 2, (active, active), config, report, args.output,
            (expected_grad, expected_loss[:, None].expand(rows, 8)),
            ({"atol": 2e-7, "rtol": 2e-4}, {"atol": 3e-5, "rtol": 2e-5}))


def execute(entry, tensors, source_count, scalars, config, report, output, expected, tolerances):
    from pypto.runtime import DistributedWorker

    worker, handles = None, []
    try:
        tensors = tuple(value.share_memory_() for value in tensors)
        program = entry.compile(*tensors, *scalars, config=config)
        worker = DistributedWorker([program], config=config, inherited_host_tensors=tensors)
        for tensor in tensors:
            handles.append(worker.alloc_tensor(tuple(tensor.shape), tensor.dtype, init=tensor))
        worker.run(program, *handles, *scalars, config=config)
        errors = []
        for tensor, handle, reference, tolerance in zip(
            tensors[source_count:], handles[source_count:], expected, tolerances, strict=True,
        ):
            worker.copy_from(tensor.data_ptr(), handle.data_ptr, tensor.nbytes)
            torch.testing.assert_close(tensor, reference.to(tensor.dtype), **tolerance)
            errors.append(float((tensor - reference).abs().max()))
        report.update(passed=True, output_max_abs=errors)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if worker is not None:
                try:
                    for handle in reversed(handles):
                        worker.free_tensor(handle)
                finally:
                    worker.close()
        finally:
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
