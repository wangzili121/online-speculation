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
    parser.add_argument("--peer-device", type=int, help="second device for the packed MLA projection adjoint")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mla-output", action="store_true", help="validate the draft output projection's VJP")
    modes.add_argument("--mla-projection-fp32", action="store_true", help="validate MLA projection with FP32 upstream gradients")
    modes.add_argument("--optimizer", action="store_true", help="validate a complete AdamW parameter update")
    modes.add_argument("--mla-update", action="store_true", help="validate device MLA output VJP plus AdamW")
    modes.add_argument("--vocab-backward", action="store_true", help="validate a frozen TP8 vocabulary shard VJP")
    modes.add_argument("--norm-backward", action="store_true", help="validate final RMSNorm and mHC mean backward")
    modes.add_argument("--kda-backward", action="store_true", help="validate the frozen KDA recurrence adjoint")
    modes.add_argument("--kda-conv-backward", action="store_true", help="validate the frozen KDA convolution and SiLU VJP")
    modes.add_argument("--kda-qk-backward", action="store_true", help="validate the KDA Q/K L2 normalization VJP")
    modes.add_argument("--kda-output-backward", action="store_true", help="validate the KDA output projection and gated RMSNorm")
    modes.add_argument("--kda-projection-backward", action="store_true", help="validate the five KDA input projection branches")
    modes.add_argument("--mhc-backward", action="store_true", help="validate the mHC residual combination VJP")
    modes.add_argument("--mhc-pre-backward", action="store_true", help="validate the mHC stream collapse VJP")
    modes.add_argument("--mhc-mix-backward", action="store_true", help="validate the mHC coefficient projection VJP")
    modes.add_argument("--sinkhorn-backward", action="store_true", help="validate mHC residual normalization VJP")
    modes.add_argument("--swiglu-backward", action="store_true", help="validate the frozen expert activation VJP")
    modes.add_argument("--joint-backward", action="store_true", help="validate joint history/draft softmax VJP")
    modes.add_argument("--mla-score-backward", action="store_true", help="validate the latent MLA score VJP")
    modes.add_argument("--mla-value-backward", action="store_true", help="validate the two latent MLA value branches")
    modes.add_argument("--mla-mixture-backward", action="store_true", help="validate the shared probabilities and draft value VJP")
    modes.add_argument("--mla-latent-backward", action="store_true", help="validate the complete latent attention adjoint")
    modes.add_argument("--mla-replay-tape", action="store_true", help="validate MLA training replay from its forward inputs")
    modes.add_argument("--mla-absorb-backward", action="store_true", help="validate the two MLA query bases")
    modes.add_argument("--mla-norm-backward", choices=("q", "kv"), help="validate a trainable MLA RMSNorm")
    modes.add_argument("--mla-input-projections", action="store_true", help="validate both MLA input projection adjoints")
    modes.add_argument("--mla-local-backward", action="store_true", help="validate the local MLA output-to-input adjoint")
    modes.add_argument("--int8-backward", action="store_true", help="validate frozen W8A8 input gradients")
    modes.add_argument("--expert-backward", action="store_true", help="validate the complete frozen W8A8 expert adjoint")
    modes.add_argument("--shared-expert-backward", action="store_true", help="validate the frozen NZ shared expert adjoint")
    modes.add_argument("--router-backward", action="store_true", help="validate the selected GLM router weight adjoint")
    modes.add_argument("--router-replay", action="store_true", help="validate native routing inputs for the expert adjoint")
    parser.add_argument("--history-capacity", type=int, default=256, help="padded history length for the MLA replay tests")
    parser.add_argument("--checkpoint", type=Path, help="use real checkpoint weights for the selected adjoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output path")
    if args.peer_device is not None and (not args.mla_input_projections or args.peer_device == args.device):
        parser.error("--peer-device requires --mla-input-projections and a second device")
    if args.shared_expert_backward and args.checkpoint is None:
        parser.error("--shared-expert-backward requires the GLM checkpoint")
    if args.router_replay and args.checkpoint is None:
        parser.error("--router-replay requires the GLM checkpoint")
    if args.mla_local_backward and (args.checkpoint is None or args.history_capacity % 128 or args.history_capacity < 256):
        parser.error("--mla-local-backward requires the checkpoint and aligned history capacity of at least 256")

    from models.glm5_3_flash.config import D, VOCAB
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig

    torch.set_num_threads(4)
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0))
    generator = torch.Generator().manual_seed(142)
    if args.mla_input_projections:
        projection_shards(args, generator)
        return
    if args.shared_expert_backward:
        shared_expert(args, generator, config)
        return
    if args.mla_local_backward:
        local_attention(args, generator, config)
        return
    if args.router_replay:
        router_replay(args, generator, config)
        return
    if args.mla_absorb_backward:
        from blockspec.pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, VALUE
        from blockspec.pypto_glm53_online_attention import make_mla_absorb_backward

        width = HEADS * VALUE
        if args.checkpoint is None:
            target_basis = (torch.randn(width, LATENT, generator=generator) * 0.04).bfloat16()
        else:
            from models.glm5_3_flash.utils import Checkpoint

            attn = Checkpoint(args.checkpoint).attention(43, tp_rank=0, tp=8, mla_mode="native")
            target_basis = attn["w_k"].reshape(width, LATENT).contiguous()
        draft_basis = target_basis.flip(1).contiguous()
        q = (torch.randn(16, width, generator=generator) * 0.3).bfloat16()
        d_h, d_b = (torch.randn(QUERY_ROWS, HEAD_TILE, LATENT, generator=generator) * 0.025 for _ in range(2))
        projected_q = q.double().requires_grad_()
        basis = draft_basis.double().requires_grad_()
        live = projected_q[:QUERY_ROWS].view(QUERY_ROWS, HEADS, VALUE)
        history = torch.einsum("thd,hdl->thl", live, target_basis.double().view(HEADS, VALUE, LATENT))
        block = torch.einsum("thd,hdl->thl", live, basis.view(HEADS, VALUE, LATENT))
        expected = torch.autograd.grad((history, block), (projected_q, basis),
                                       (d_h[:, :HEADS].double(), d_b[:, :HEADS].double()))
        inputs = (q, target_basis, draft_basis, d_h.flatten(0, 1), d_b.flatten(0, 1))
        outputs = (torch.empty(q.shape), torch.empty(draft_basis.shape))
        execute(make_mla_absorb_backward(), (*inputs, *outputs), len(inputs), (), config,
                {"program": "GLM MLA absorbed history/draft query adjoint", "passed": False}, args.output,
                expected, ({"atol": 3e-6, "rtol": 4e-5},) * 2)
        return
    if args.mla_norm_backward:
        from blockspec.pypto_glm53_online_attention import make_mla_norm_backward
        from models.glm5_3_flash.config import FLASH

        eps = FLASH.rms_norm_eps
        width = FLASH.q_lora_rank if args.mla_norm_backward == "q" else FLASH.kv_lora_rank
        if args.checkpoint is None:
            weight = torch.rand(width, generator=generator).bfloat16()
        else:
            from models.glm5_3_flash.utils import Checkpoint

            attn = Checkpoint(args.checkpoint).attention(43, tp_rank=0, tp=8, mla_mode="native")
            name = "q_a_norm_weight" if args.mla_norm_backward == "q" else "kv_a_norm_weight"
            weight = attn[name].contiguous()
        x = torch.randn(16, width, generator=generator).bfloat16()
        x[0].mul_(0.001)
        x[1].zero_()
        upstream = torch.randn(16, width, generator=generator) * 0.02
        upstream[6:] = 0
        x_ref, w_ref = x.double().requires_grad_(), weight.double().requires_grad_()
        y = x_ref * (x_ref.square().mean(-1, keepdim=True) + eps).rsqrt() * w_ref
        dx, dw = torch.autograd.grad(y, (x_ref, w_ref), upstream.double())
        inputs = (x, weight, upstream)
        outputs = (torch.empty_like(upstream), torch.empty(1, width))
        execute(make_mla_norm_backward(width, eps), (*inputs, *outputs), len(inputs), (), config,
                {"program": "GLM trainable MLA RMSNorm adjoint", "kind": args.mla_norm_backward, "passed": False},
                args.output, (dx, dw.unsqueeze(0)), ({"atol": 4e-6, "rtol": 6e-5},) * 2)
        return
    if args.kda_projection_backward:
        from blockspec.pypto_glm53_online_kda import make_kda_projection_backward
        from models.glm5_3_flash.config import FLASH, KDA_DIM, KDA_H

        tokens, heads, dim = 16, KDA_H // 8, KDA_DIM
        local, lower = heads * dim, FLASH.linear_lower_bound
        names = ("w_q", "w_k", "w_v", "w_f_a", "w_f_b", "w_b", "w_g_a", "w_g_b")
        if args.checkpoint is None:
            shapes = ((local, D), (local, D), (local, D), (dim, D), (local, dim), (heads, D), (dim, D), (local, dim))
            weight = {name: (torch.randn(shape, generator=generator) * 0.01).bfloat16()
                      for name, shape in zip(names, shapes, strict=True)}
            weight["a_log"] = torch.linspace(-0.4, 1.0, heads)
            weight["dt_bias"] = torch.linspace(-1.0, 1.0, local)
        else:
            from models.glm5_3_flash.utils import Checkpoint

            weight = Checkpoint(args.checkpoint).attention(44, tp_rank=0, tp=8)
        x = (torch.randn(tokens, D, generator=generator) * 0.3).bfloat16().double().requires_grad_()

        def project(value, name):
            out = value @ weight[name].double().T
            return out + (out.bfloat16().double() - out).detach()

        qkv = torch.cat(tuple(project(x, name) for name in names[:3]), dim=-1)
        forget = project(project(x, "w_f_a"), "w_f_b") + weight["dt_bias"]
        rates = weight["a_log"].double().exp().repeat_interleave(dim)
        decay = (forget * rates).sigmoid() * lower
        beta = project(x, "w_b").sigmoid()
        output_gate = project(project(x, "w_g_a"), "w_g_b")
        upstream = tuple(torch.randn(value.shape, generator=generator) * 0.02
                         for value in (qkv, decay, beta, output_gate))
        for tensor in upstream:
            tensor[6:] = 0
        (expected,) = torch.autograd.grad((qkv, decay, beta, output_gate), x,
                                         tuple(tensor.double() for tensor in upstream))
        d_qkv, d_decay, d_beta, d_gate = upstream
        result = torch.empty(tokens, D)
        inputs = (d_qkv, decay.detach().float(), weight["a_log"], d_decay, beta.detach().float(), d_beta, d_gate,
                  *(weight[name].contiguous() for name in names))
        execute(make_kda_projection_backward(tokens, heads, D, lower), (*inputs, result), len(inputs), (), config,
                {"program": "GLM frozen KDA QKV and gate projection adjoint", "passed": False,
                 "checkpoint_weights": args.checkpoint is not None},
                args.output, (expected,), ({"atol": 1e-5, "rtol": 9e-5},))
        return
    if args.kda_output_backward:
        from blockspec.pypto_glm53_online_kda import make_kda_output_backward
        from models.glm5_3_flash.config import FLASH, KDA_DIM, KDA_H

        tokens, heads = 16, KDA_H // 8
        shape = (tokens * heads, KDA_DIM)
        if args.checkpoint is None:
            projection = (torch.randn(D, heads * KDA_DIM, generator=generator) * 0.02).bfloat16()
            norm = torch.rand(KDA_DIM, generator=generator).bfloat16()
        else:
            from models.glm5_3_flash.utils import Checkpoint

            weights = Checkpoint(args.checkpoint).attention(44, tp_rank=0, tp=8)
            projection, norm = weights["w_o"].contiguous(), weights["norm_weight"]
        core = torch.randn(shape, generator=generator).bfloat16()
        core[0].mul_(0.002)
        gate = (torch.randn(shape, generator=generator) * 4).bfloat16()
        upstream = torch.randn(tokens, D, generator=generator) * 0.02
        upstream[6:] = 0
        x, logit = core.double().requires_grad_(), gate.double().requires_grad_()
        inv_rms = (x.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps).rsqrt()
        projected = (x * inv_rms * norm.double() * logit.sigmoid()).reshape(tokens, -1) @ projection.double().T
        expected = torch.autograd.grad(projected, (x, logit), upstream.double())
        outputs = (torch.empty(shape), torch.empty(shape))
        inputs = (upstream, core, gate, projection, norm)
        execute(make_kda_output_backward(tokens, heads, D, FLASH.rms_norm_eps), (*inputs, *outputs),
                len(inputs), (), config,
                {"program": "GLM KDA output projection and gated RMSNorm adjoint", "passed": False,
                 "checkpoint_weights": args.checkpoint is not None},
                args.output, expected, ({"atol": 6e-6, "rtol": 5e-5},) * 2)
        return
    if args.router_backward:
        from blockspec.pypto_glm53_online_moe import make_router_backward
        from models.glm5_3_flash.config import FLASH

        rows, active = 16, 6
        if args.checkpoint is None:
            weight = (torch.randn(FLASH.n_routed_experts, D, generator=generator) * 0.01).bfloat16()
        else:
            from models.glm5_3_flash.utils import Checkpoint

            weight = Checkpoint(args.checkpoint).read("model.language_model.layers.44.mlp.gate.weight").bfloat16()
        x = torch.randn(rows, D, generator=generator).bfloat16().double().requires_grad_()
        scores = (x @ weight.double().T).sigmoid()
        bias = torch.randn(FLASH.n_routed_experts, generator=generator) * 0.2
        ids = (scores.detach() + bias).topk(8, dim=1).indices
        selected_scores = scores.gather(1, ids)
        differentiable = selected_scores * FLASH.routed_scaling_factor / (selected_scores.sum(1, keepdim=True) + 1e-20)
        gradient = torch.randn(rows, 8, generator=generator) * 0.2
        (expected,) = torch.autograd.grad(differentiable[:active], x, gradient[:active].double())
        ids[active:] = -1
        output = torch.empty(rows, D)
        inputs = (selected_scores.detach().float(), ids.int(), gradient, weight)
        execute(make_router_backward(rows, D, FLASH.n_routed_experts, FLASH.routed_scaling_factor),
                (*inputs, output), len(inputs), (active,), config,
                {"program": "GLM selected sigmoid routing adjoint", "passed": False,
                 "checkpoint_weights": args.checkpoint is not None}, args.output, (expected,),
                ({"atol": 2e-6, "rtol": 3e-5},))
        return
    if args.expert_backward:
        from blockspec.pypto_glm53_online_moe import make_expert_backward
        from models.glm5_3_flash.config import FLASH
        from models.glm5_3_flash.quantization import quantize_per_token_int8

        rows, hidden = 16, FLASH.moe_intermediate_size
        limit = FLASH.swiglu_limit
        if args.checkpoint is None:
            weights = (
                torch.randint(-127, 128, (2 * hidden, D), dtype=torch.int8, generator=generator),
                torch.rand(2 * hidden, generator=generator) * 0.001,
                torch.randint(-127, 128, (D, hidden), dtype=torch.int8, generator=generator),
                torch.rand(D, generator=generator) * 0.0005,
            )
        else:
            from models.glm5_3_flash.utils import Checkpoint

            checkpoint = Checkpoint(args.checkpoint)
            root = "model.language_model.layers.44.mlp.experts.0."

            def parameters(name):
                return checkpoint.read(root + name + ".weight"), checkpoint.read(root + name + ".weight_scale").flatten()

            gate, sg = parameters("gate_proj")
            up, su = parameters("up_proj")
            down, sd = parameters("down_proj")
            weights = (torch.cat((gate, up)), torch.cat((sg, su)).float(), down, sd.float())
        gu, gu_scale, down, down_scale = weights
        x = torch.randn(rows, D, generator=generator)
        x[1:4].mul_(2.5)
        x_q, x_scale = quantize_per_token_int8(x)
        raw = (x_q.int() @ gu.int().T).float() * x_scale * gu_scale.bfloat16().float()
        g, u = raw.chunk(2, dim=-1)
        activation = torch.nn.functional.silu(g.clamp_max(limit)) * u.clamp(-limit, limit)
        h_q, h_scale = quantize_per_token_int8(activation)
        y = ((h_q.int() @ down.int().T).float() * h_scale * down_scale.bfloat16().float()).bfloat16()
        route = torch.zeros(rows, 8)
        route[:6, 0] = torch.linspace(0.15, 1.0, 6)
        grad = torch.randn(rows, D, generator=generator) * 0.01
        grad[6:] = 0

        x_ref = x.double().requires_grad_()
        r_ref = route.double().requires_grad_()
        effective_gu = gu.double() * gu_scale.bfloat16().double().unsqueeze(1)
        effective_down = down.double() * down_scale.bfloat16().double().unsqueeze(1)
        # Evaluate derivatives at the deployed forward values and use an identity VJP for activation Q/DQ.
        g, u = (raw.double() + (x_ref - x_ref.detach()) @ effective_gu.T).chunk(2, dim=-1)
        h = torch.nn.functional.silu(g.clamp_max(limit)) * u.clamp(-limit, limit)
        out = y.double() + (h - h.detach()) @ effective_down.T
        expected = torch.autograd.grad(out * r_ref[:, :1], (x_ref, r_ref), grad.double())
        outputs = (torch.empty_like(x), torch.empty_like(route))
        inputs = (grad, x_q, torch.nn.functional.pad(x_scale, (0, 7)), route, *weights)
        execute(make_expert_backward(rows, D, hidden, limit), (*inputs, *outputs), len(inputs), (), config,
                {"program": "GLM frozen W8A8 expert input and route adjoint", "passed": False,
                 "checkpoint_weights": args.checkpoint is not None}, args.output, expected,
                ({"atol": 4e-5, "rtol": 9e-5}, {"atol": 4e-5, "rtol": 9e-5}))
        return
    if args.mla_latent_backward or args.mla_replay_tape:
        from blockspec.pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, TILE, VALUE
        from blockspec.pypto_glm53_online_attention import make_mla_latent_backward, make_mla_replay_tape

        aligned = args.history_capacity
        if aligned < 2 * TILE or aligned % TILE:
            parser.error("--history-capacity must be a multiple of 128 and at least 256")

        def normal_bf16(*shape):
            return (torch.randn(shape, generator=generator) * 0.1).bfloat16()

        history = normal_bf16(QUERY_ROWS, aligned, LATENT)
        block = normal_bf16(TILE, LATENT)
        qh = normal_bf16(QUERY_ROWS, HEADS, LATENT)
        qb = normal_bf16(QUERY_ROWS, HEADS, LATENT)
        wh = normal_bf16(HEADS, VALUE, LATENT)
        wb = normal_bf16(HEADS, VALUE, LATENT)
        qh_ref, qb_ref, block_ref, wb_ref = (t.double().requires_grad_() for t in (qh, qb, block, wb))
        hist_scores = qh_ref @ history.double().transpose(1, 2)
        block_scores = qb_ref @ block_ref.t()
        logits = torch.cat((hist_scores, block_scores), -1) * VALUE ** -0.5
        lengths = torch.tensor((0, 1, 16, 127, 129, aligned - 15))
        hist_mask = torch.arange(aligned) >= lengths[:, None]
        block_mask = (torch.arange(TILE) >= QUERY_ROWS).expand(QUERY_ROWS, -1)
        prob = logits.masked_fill(torch.cat((hist_mask, block_mask), dim=-1)[:, None], -torch.inf).softmax(-1)
        mixed_hist = prob[..., :aligned] @ history.double()
        mixed_block = prob[..., aligned:] @ block_ref

        def padded_heads(tensor):
            return torch.nn.functional.pad(tensor.detach(), (0, 0, 0, HEAD_TILE - HEADS)).flatten(0, 1)

        history_bias = torch.zeros(hist_mask.shape).masked_fill(hist_mask, -1e30)
        block_bias = torch.zeros((1, TILE)).masked_fill(block_mask[:1], -1e30)
        replay_inputs = (padded_heads(qh), padded_heads(qb), history.flatten(0, 1), block, history_bias, block_bias)
        if args.mla_replay_tape:
            expected = (padded_heads(prob), padded_heads(mixed_block))
            outputs = tuple(torch.empty(t.shape, dtype=torch.float32) for t in expected)
            execute(make_mla_replay_tape(aligned), (*replay_inputs, *outputs), len(replay_inputs), (), config,
                    {"program": "GLM MLA joint normalization replay", "passed": False}, args.output,
                    expected, ({"atol": 3e-6, "rtol": 5e-5},) * 2)
            return

        context = torch.einsum("thl,hvl->thv", mixed_hist, wh.double())
        context = context + torch.einsum("thl,hvl->thv", mixed_block, wb_ref)
        ctx_grad = torch.randn(context.shape, generator=generator) * 0.05
        dqh, dqb, dw, dblock = torch.autograd.grad(context, (qh_ref, qb_ref, wb_ref, block_ref), ctx_grad.double())

        expected = (padded_heads(dqh), padded_heads(dqb), dw.flatten(0, 1), dblock)
        inputs = (padded_heads(ctx_grad), wh.flatten(0, 1), wb.flatten(0, 1), *replay_inputs)
        outputs = tuple(torch.empty(t.shape) for t in expected)
        execute(make_mla_latent_backward(aligned), (*inputs, *outputs), len(inputs), (), config,
                {"program": "GLM shared-history and bidirectional draft latent attention adjoint", "passed": False},
                args.output, expected, ({"atol": 3e-6, "rtol": 5e-5},) * 4)
        return
    if args.mla_value_backward:
        from blockspec.pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, VALUE
        from blockspec.pypto_glm53_online_attention import make_mla_value_backward

        ctx = torch.randn(QUERY_ROWS, HEAD_TILE, VALUE, generator=generator) * 0.1
        mixture = torch.randn(QUERY_ROWS, HEAD_TILE, LATENT, generator=generator)
        shape = (HEADS, VALUE, LATENT)
        w_hist = (torch.randn(shape, generator=generator) * 0.05).bfloat16()
        w_draft = (torch.randn(shape, generator=generator) * 0.08).bfloat16()
        context = ctx[:, :HEADS].double()
        expected_hist = torch.einsum("thv,hvl->thl", context, w_hist.double())
        expected_draft = torch.einsum("thv,hvl->thl", context, w_draft.double())
        expected_w = torch.einsum("thv,thl->hvl", context, mixture[:, :HEADS].double())
        head_padding = (0, 0, 0, HEAD_TILE - HEADS)
        reference = (torch.nn.functional.pad(expected_hist, head_padding).flatten(0, 1),
                     torch.nn.functional.pad(expected_draft, head_padding).flatten(0, 1),
                     expected_w.flatten(0, 1))
        inputs = tuple(tensor.flatten(0, 1) for tensor in (ctx, mixture, w_hist, w_draft))
        outputs = tuple(torch.empty(tensor.shape) for tensor in reference)
        execute(make_mla_value_backward(), (*inputs, *outputs), 4, (), config,
                {"program": "GLM history and draft latent value bases VJP", "passed": False},
                args.output, reference, ({"atol": 5e-6, "rtol": 3e-5},) * 3)
        return
    if args.mla_score_backward or args.mla_mixture_backward:
        from blockspec.pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, TILE, VALUE
        from blockspec.pypto_glm53_online_attention import make_mla_mixture_backward, make_mla_score_backward

        aligned = 2 * TILE
        logits_grad = torch.randn(QUERY_ROWS, HEAD_TILE, aligned + TILE, generator=generator) * 0.1
        logits_grad[:, HEADS:] = 0
        logits_grad[:, :, aligned + QUERY_ROWS:] = 0
        for row, length in enumerate((0, 1, 16, 127, 129, 241)):
            logits_grad[row, :, length:aligned] = 0
        hist = torch.randn(QUERY_ROWS, aligned, LATENT, generator=generator).bfloat16()
        block = torch.randn(TILE, LATENT, generator=generator).bfloat16()
        if args.mla_mixture_backward:
            mass = logits_grad.exp() * logits_grad.ne(0)
            probability = mass / mass.sum(-1, keepdim=True).clamp_min(1.0)
            dh = torch.randn(QUERY_ROWS, HEAD_TILE, LATENT, generator=generator) * 0.1
            db = torch.randn(QUERY_ROWS, HEAD_TILE, LATENT, generator=generator) * 0.1
            expected_prob = torch.cat((dh.double() @ hist.double().transpose(1, 2),
                                       db.double() @ block.double().t()), dim=-1).flatten(0, 1)
            expected_block = (probability[..., aligned:].double().transpose(1, 2) @ db.double()).sum(0)
            outputs = (torch.empty(expected_prob.shape), torch.empty(expected_block.shape))
            inputs = (probability.flatten(0, 1), hist.flatten(0, 1), block, dh.flatten(0, 1), db.flatten(0, 1))
            execute(make_mla_mixture_backward(aligned), (*inputs, *outputs), 5, (), config,
                    {"program": "GLM latent probability and draft value VJP", "passed": False}, args.output,
                    (expected_prob, expected_block), ({"atol": 2e-5, "rtol": 3e-5},) * 2)
            return
        q_block = torch.randn(QUERY_ROWS, HEAD_TILE, LATENT, generator=generator).bfloat16()
        hist_grad, block_grad = logits_grad.double().split((aligned, TILE), dim=-1)
        scale = VALUE ** -0.5
        expected_hist = (hist_grad @ hist.double()).reshape(-1, LATENT) * scale
        expected_draft = (block_grad @ block.double()).reshape(-1, LATENT) * scale
        expected_k = (block_grad.transpose(1, 2) @ q_block.double()).sum(0) * scale
        outputs = (torch.empty((QUERY_ROWS * HEAD_TILE, LATENT)),
                   torch.empty((QUERY_ROWS * HEAD_TILE, LATENT)), torch.empty((TILE, LATENT)))
        inputs = (logits_grad.reshape(-1, aligned + TILE), hist.flatten(0, 1), block, q_block.flatten(0, 1))
        execute(make_mla_score_backward(aligned), (*inputs, *outputs), 4, (), config,
                {"program": "GLM joint causal-history and bidirectional-block latent scores VJP", "passed": False},
                args.output, (expected_hist, expected_draft, expected_k), ({"atol": 4e-6, "rtol": 4e-5},) * 3)
        return
    if args.mhc_mix_backward:
        from blockspec.pypto_glm53_online_mhc import make_mhc_mix_backward
        from models.glm5_3_flash.config import FLASH

        tokens, width, pad = 6, 4 * D, 8
        x = torch.randn(tokens, width, generator=generator).bfloat16()
        x[0].mul_(0.001)
        weights = (torch.randn(24, width, generator=generator) * 0.01).bfloat16()
        scale = torch.tensor([0.2, -0.3, 0.4])
        bias = torch.randn(24, generator=generator)
        dp = torch.randn(tokens, pad, generator=generator) * 0.1
        dq = torch.randn(tokens, pad, generator=generator) * 0.1
        dr = torch.randn(tokens, pad, pad, generator=generator) * 0.1
        xd = x.double().requires_grad_()
        normalized = xd * (xd.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps).rsqrt()
        raw = normalized @ weights.double().t()
        p = (raw[:, :4] * scale[0] + bias[:4]).sigmoid() + FLASH.hc_eps
        q = (raw[:, 4:8] * scale[1] + bias[4:8]).sigmoid() * 2.0
        r = (raw[:, 8:] * scale[2] + bias[8:]).reshape(tokens, 4, 4)
        (expected,) = torch.autograd.grad((p, q, r), xd, (dp[:, :4].double(), dq[:, :4].double(), dr[:, :4, :4].double()))
        p = torch.nn.functional.pad(p.detach().float(), (0, 4))
        q = torch.nn.functional.pad(q.detach().float(), (0, 4))
        output = torch.empty(x.shape)
        execute(make_mhc_mix_backward(tokens, D, FLASH.rms_norm_eps, FLASH.hc_eps),
                (x, weights, scale, p, q, dp, dq, dr.reshape(tokens * pad, pad), output), 8, (), config,
                {"program": "GLM mHC coefficient projection VJP", "passed": False}, args.output,
                (expected,), ({"atol": 3e-6, "rtol": 5e-5},))
        return
    if args.mhc_pre_backward:
        from blockspec.pypto_glm53_online_mhc import make_mhc_pre_backward

        tokens, streams, pad = 6, 4, 8
        x = torch.randn(tokens * streams, D, generator=generator).bfloat16()
        weights = torch.rand(tokens, pad, generator=generator)
        grad = torch.randn(tokens, D, generator=generator) * 0.01
        x_ref = x.double().requires_grad_()
        w_ref = weights.double().requires_grad_()
        collapsed = (x_ref.reshape(tokens, streams, D) * w_ref[:, :streams, None]).sum(1)
        expected = torch.autograd.grad(collapsed, (x_ref, w_ref), grad.double())
        outputs = (torch.empty(x.shape), torch.empty_like(weights))
        execute(make_mhc_pre_backward(D, tokens), (x, weights, grad, *outputs), 3, (), config,
                {"program": "GLM mHC stream collapse VJP", "passed": False}, args.output, expected,
                ({"atol": 2e-6, "rtol": 3e-5},) * 2)
        return
    if args.kda_qk_backward:
        from blockspec.pypto_glm53_online_kda import make_kda_qk_backward
        from models.glm5_3_flash.config import KDA_DIM, KDA_H

        tokens, heads = 6, KDA_H // 8
        shape = (tokens * heads, KDA_DIM)
        query = torch.randn(shape, generator=generator).bfloat16()
        key = torch.randn(shape, generator=generator).bfloat16()
        query[0].zero_()
        key[1].mul_(0.0001)
        grad_q = torch.randn(shape, generator=generator)
        grad_k = torch.randn(shape, generator=generator)
        q, k = query.double().requires_grad_(), key.double().requires_grad_()
        norm = lambda x: x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        expected = torch.autograd.grad((norm(q) * KDA_DIM ** -0.5, norm(k)), (q, k),
                                       (grad_q.double(), grad_k.double()))
        outputs = (torch.empty_like(grad_q), torch.empty_like(grad_k))
        execute(make_kda_qk_backward(tokens, heads), (query, key, grad_q, grad_k, *outputs),
                4, (), config, {"program": "GLM KDA Q/K L2 normalization VJP", "passed": False},
                args.output, expected, ({"atol": 2e-6, "rtol": 2e-5},) * 2)
        return
    if args.kda_conv_backward:
        from blockspec.pypto_glm53_online_kda import make_kda_convolution_backward
        from models.glm5_3_flash.config import KDA_CONV_K, KDA_DIM, KDA_H

        tokens = 6
        channels = 3 * (KDA_H // 8) * KDA_DIM
        mixed = torch.randn(tokens, channels, generator=generator).bfloat16()
        history = torch.randn(KDA_CONV_K - 1, channels, generator=generator).bfloat16()
        weight = (torch.randn(KDA_CONV_K, channels, generator=generator) * 0.25).bfloat16()
        gradient = torch.randn(tokens, channels, generator=generator) * 0.05
        x = mixed.double().requires_grad_()
        joined = torch.cat((history.double(), x))
        convolution = (joined.unfold(0, KDA_CONV_K, 1) * weight.double().t()).sum(-1)
        (expected,) = torch.autograd.grad(torch.nn.functional.silu(convolution), x, gradient.double())
        output = torch.empty_like(gradient)
        execute(make_kda_convolution_backward(tokens, channels), (mixed, history, weight, gradient, output),
                4, (), config, {"program": "GLM causal KDA convolution and SiLU VJP", "passed": False},
                args.output, (expected,), ({"atol": 4e-7, "rtol": 2e-5},))
        return
    if args.int8_backward:
        from models.glm5_3_flash.config import FLASH
        from blockspec.pypto_glm53_online_moe import make_frozen_int8_linear_backward

        rows, width = 16, FLASH.moe_intermediate_size
        gradient = torch.randn((rows, D), generator=generator) * 0.01
        weights = torch.randint(-128, 128, (D, width), generator=generator, dtype=torch.int8)
        scale = torch.rand(D, generator=generator) * 0.0005
        output = torch.empty((rows, width), dtype=torch.float32)
        expected = gradient.double() @ (weights.double() * scale.double().unsqueeze(1))
        execute(make_frozen_int8_linear_backward(rows, width, D), (gradient, weights, scale, output),
                3, (), config, {"program": "GLM W8A8 frozen down projection VJP", "passed": False}, args.output,
                (expected,), ({"atol": 1e-6, "rtol": 3e-5},))
        return
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
        post = torch.rand((tokens, streams), generator=generator)
        mix = torch.rand((tokens, streams, streams), generator=generator)
        upstream = torch.randn(layout, generator=generator)
        y, x, p, r = (value.double().requires_grad_() for value in (sublayer, residual, post, mix))

        def round_ste(tensor):
            return tensor + (tensor.bfloat16().double() - tensor).detach()

        update = round_ste(y[:, None, :] * round_ste(p)[:, :, None])
        mixed = round_ste(torch.einsum("tij,tid->tjd", round_ste(r), x))
        output = round_ste(update + mixed)
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
    if args.optimizer or args.mla_update:
        from blockspec.pypto_glm53_mla import HEADS, VALUE
        from blockspec.pypto_glm53_online_optimizer import make_adamw_update, make_mla_output_step

        shape = (D, HEADS * VALUE)
        gradient, master, moment = (torch.randn(shape, generator=generator) * scale for scale in (0.01, 0.02, 0.01))
        variance = torch.rand(shape, generator=generator) * 1e-4
        serving = master.bfloat16()
        if args.mla_update:
            context = (torch.randn((16, shape[1]), generator=generator) * 0.2).bfloat16()
            upstream = (torch.randn((16, D), generator=generator) * 0.1).bfloat16()
            upstream[5:].zero_()
            gradient = upstream.float().T @ context.float()
            d_context = upstream.float() @ serving.float()
        scalars = (1e-4, 0.01, 1 / (1 - 0.9**3), 1 / (1 - 0.999**3))
        lr, decay, bias_one, bias_two = scalars
        m = moment * 0.9 + gradient * 0.1
        v = variance * 0.999 + gradient.square() * 0.001
        after = master - lr * (m * bias_one / ((v * bias_two).sqrt() + 1e-8) + master * decay)
        tolerances = ({"atol": 1e-8, "rtol": 2e-6},) * 3 + ({"atol": 0, "rtol": 0.008},)
        reports = {"shape": shape, "passed": False}
        tensors = (master, moment, variance, serving)
        expected = (after, m, v, after.bfloat16())
        if args.mla_update:
            kernel = make_mla_output_step()
            reports["program"] = "MLA output device update"
            source = (context, upstream)
            tensors += (torch.empty_like(gradient), torch.empty_like(d_context))
            expected += (gradient, d_context)
            tolerances += ({"atol": 2e-5, "rtol": 2e-5},) * 2
        else:
            kernel = make_adamw_update(*shape)
            reports["program"] = "MLA AdamW"
            source = (gradient,)
        execute(kernel, source + tensors, len(source), scalars, config, reports, args.output, expected, tolerances)
        return
    if args.mla_output or args.mla_projection_fp32:
        from blockspec.pypto_glm53_mla import HEADS, VALUE
        from blockspec.pypto_glm53_online_attention import make_output_backward, make_projection_backward

        rows, width = 16, HEADS * VALUE
        context = torch.randn((rows, width), generator=generator).bfloat16()
        weight = (torch.randn((D, width), generator=generator) * 0.02).bfloat16()
        upstream = torch.randn((rows, D), generator=generator)
        if not args.mla_projection_fp32:
            upstream = upstream.bfloat16()
        upstream[5:].zero_()
        weight_grad = torch.empty(weight.shape, dtype=torch.float32)
        context_grad = torch.empty(context.shape, dtype=torch.float32)
        tensors = (context, weight, upstream, weight_grad, context_grad)
        expected = (upstream.float().T @ context.float(), upstream.float() @ weight.float())
        report = {"rows": rows, "program": "MLA output VJP", "passed": False}
        program = make_projection_backward(width, D) if args.mla_projection_fp32 else make_output_backward()
        tolerance = {"atol": 2e-4, "rtol": 3e-5} if args.mla_projection_fp32 else {"atol": 2e-5, "rtol": 2e-5}
        execute(program, tensors, 3, (), config, report, args.output, expected, (tolerance, tolerance))
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


def router_replay(args, generator, config):
    from models.glm5_3_flash.config import D, TOPK
    from models.glm5_3_flash.gate import gate
    from models.glm5_3_flash.rmsnorm import golden_rmsnorm_quant
    from models.glm5_3_flash.utils import Checkpoint, layer_router

    from blockspec.pypto_glm53_online_moe import make_router_replay

    rows, active = 16, 6
    checkpoint = Checkpoint(args.checkpoint)
    router = layer_router(checkpoint, 44)
    norm = checkpoint.read(checkpoint.layer_key(44, "post_attention_layernorm.weight")).bfloat16()
    x = torch.randn(rows, D, generator=generator).bfloat16()
    normalized, quantized, scale = golden_rmsnorm_quant(x[:active], norm)
    coefficients, selected = gate(normalized, router.weight, router.correction_bias)
    picked = (normalized.double() @ router.weight.double().T).sigmoid().gather(1, selected.long())
    expected = tuple(torch.nn.functional.pad(value, (0, 0, 0, rows - active))
                     for value in (selected, coefficients, picked, quantized, scale))
    outputs = (torch.empty((rows, TOPK), dtype=torch.int32), torch.empty(rows, TOPK),
               torch.empty(rows, TOPK), torch.empty_like(x, dtype=torch.int8), torch.empty(rows, 1))
    inputs = (x, norm, router.weight, router.correction_bias)
    execute(make_router_replay(), (*inputs, *outputs), len(inputs), (active,), config,
            {"program": "GLM native router replay", "passed": False, "layer": 44}, args.output, expected,
            ({"atol": 0, "rtol": 0}, {"atol": 2e-6, "rtol": 3e-5}, {"atol": 2e-6, "rtol": 3e-5},
             {"atol": 1, "rtol": 0}, {"atol": 2e-7, "rtol": 2e-5}))


def local_attention(args, generator, config):
    from models.glm5_3_flash.utils import Checkpoint

    from blockspec.pypto_glm53_head import head_layout, initialize_rank_head
    from blockspec.pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, TILE, VALUE
    from blockspec.pypto_glm53_online_attention import make_mla_local_backward

    history_capacity = args.history_capacity
    width, rows = HEADS * VALUE, 16
    layout, _ = head_layout()
    packed = initialize_rank_head(Checkpoint(args.checkpoint), 43, 0)

    def weight(name):
        offset, count, shape = layout[name]
        return packed[offset:offset + count].reshape(shape)

    def sample(*shape):
        return (torch.randn(shape, generator=generator) * 0.12).bfloat16()

    def round_ste(tensor):
        return tensor + (tensor.bfloat16().double() - tensor).detach()

    def head_padding(tensor):
        return torch.nn.functional.pad(tensor, (0, 0, 0, HEAD_TILE - HEADS)).flatten(0, 1)

    q = sample(rows, width)
    old = sample(QUERY_ROWS, history_capacity, LATENT)
    block = sample(TILE, LATENT)
    ar_key, ar_value, out_weight = (weight(name) for name in ("key", "value", "out"))
    draft_key = ar_key.flip(1).contiguous()
    draft_value = ar_value.flip(1).contiguous()
    q_ref, block_ref, out_ref, key_ref, value_ref = (
        t.double().requires_grad_() for t in (q, block, out_weight, draft_key, draft_value)
    )
    active_q = q_ref[:QUERY_ROWS].reshape(QUERY_ROWS, HEADS, VALUE)
    q_history, q_block = (
        round_ste(torch.einsum("thv,hvl->thl", active_q, w.reshape(HEADS, VALUE, LATENT)))
        for w in (ar_key.double(), key_ref)
    )
    score_h = q_history @ old.double().transpose(-1, -2)
    score_b = q_block @ block_ref.T
    lengths = torch.tensor([0, 1, 127, 128, history_capacity - 1, history_capacity])
    history_mask = torch.arange(history_capacity) >= lengths[:, None]
    block_mask = torch.arange(TILE)[None, :] >= QUERY_ROWS
    combined_mask = torch.cat((history_mask, block_mask.expand(QUERY_ROWS, -1)), dim=1)
    scores = (torch.cat((score_h, score_b), dim=-1) * VALUE ** -0.5).masked_fill(combined_mask[:, None], -torch.inf)
    p_old, p_block = scores.softmax(-1).split((history_capacity, TILE), dim=-1)
    hidden_h = p_old @ old.double()
    hidden_b = p_block @ block_ref
    context = torch.einsum("thl,hvl->thv", hidden_h, ar_value.double().reshape(HEADS, VALUE, LATENT))
    context = context + torch.einsum("thl,hvl->thv", hidden_b, value_ref.reshape(HEADS, VALUE, LATENT))
    context = round_ste(torch.nn.functional.pad(context.flatten(1), (0, 0, 0, rows - QUERY_ROWS)))
    output = context @ out_ref.T
    upstream = torch.randn(output.shape, generator=generator) * 0.02
    upstream[QUERY_ROWS:] = 0
    expected = torch.autograd.grad(output, (out_ref, key_ref, value_ref, q_ref, block_ref), upstream.double())
    outputs = tuple(torch.empty(t.shape, dtype=torch.float32) for t in expected)
    mask_old = torch.zeros(history_mask.shape).masked_fill(history_mask, -3.4e38)
    mask_block = torch.zeros(block_mask.shape).masked_fill(block_mask, -3.4e38)
    inputs = (
        upstream, context.detach().bfloat16(), out_weight, q, ar_key, draft_key, ar_value, draft_value,
        head_padding(q_history.detach().bfloat16()), head_padding(q_block.detach().bfloat16()), old.flatten(0, 1),
        block, mask_old, mask_block,
    )
    execute(make_mla_local_backward(history_capacity), (*inputs, *outputs), len(inputs), (), config,
            {"program": "GLM MLA output through absorbed attention adjoint", "passed": False,
             "checkpoint_weights": True, "history_capacity": history_capacity,
             "outputs": ["out", "key", "value", "query", "block_latent"]}, args.output, expected,
            ({"atol": 1e-5, "rtol": 2e-4},) * len(expected))


def shared_expert(args, generator, config):
    from models.glm5_3_flash.config import D, FLASH
    from models.glm5_3_flash.quantization import quantize_per_token_int8
    from models.glm5_3_flash.utils import Checkpoint, load_shared_expert, pack_int8_nz

    from blockspec.pypto_glm53_online_moe import make_shared_expert_backward

    rows, hidden = 16, FLASH.moe_intermediate_size // 8
    limit = FLASH.swiglu_limit
    shard = load_shared_expert(Checkpoint(args.checkpoint), 44, tp_rank=0, tp_size=8)
    gu, gu_scale = shard.gate_up, shard.gate_up_scale.flatten().float()
    down, down_scale = shard.down, shard.down_scale.flatten().float()
    x = torch.randn(rows, D, generator=generator)
    x[:3].mul_(3.0)
    x_q, scale = quantize_per_token_int8(x)
    upstream = torch.randn(rows, D, generator=generator) * 0.02
    upstream[6:] = 0

    x_ref = x.double().requires_grad_()
    dequantized_gu = gu.double() * gu_scale.bfloat16().double()[:, None]
    dequantized_down = down.double() * down_scale.bfloat16().double()[:, None]
    gu_forward = ((x_q.int() @ gu.int().T).float() * scale * gu_scale.bfloat16().float()).bfloat16()
    gate, up = (gu_forward.double() + (x_ref - x_ref.detach()) @ dequantized_gu.T).chunk(2, dim=-1)
    activation = torch.nn.functional.silu(gate.clamp_max(limit)) * up.clamp(-limit, limit)
    (expected,) = torch.autograd.grad(activation @ dequantized_down.T, x_ref, upstream.double())
    inputs = (upstream, x_q, torch.nn.functional.pad(scale, (0, 7)),
              pack_int8_nz(gu), gu_scale, pack_int8_nz(down), down_scale)
    output = torch.empty_like(x)
    execute(make_shared_expert_backward(rows, D, hidden, limit), (*inputs, output), len(inputs), (), config,
            {"program": "GLM shared NZ expert input adjoint", "passed": False, "layer": 44, "tp_rank": 0},
            args.output, (expected,), ({"atol": 4e-6, "rtol": 9e-5},))


def projection_shards(args, generator):
    from models.glm5_3_flash.config import D, FLASH
    from models.glm5_3_flash.utils import Checkpoint
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig

    from blockspec.pypto_glm53_head import head_layout, initialize_rank_head
    from blockspec.pypto_glm53_online_attention import make_mla_input_projection_backward

    devices = [args.device] if args.peer_device is None else [args.device, args.peer_device]
    ranks = len(devices)
    fields = ("q_a", "q_norm", "q_b", "kv_a", "kv_norm")
    layout, _ = head_layout()
    if args.checkpoint is None:
        weights = [(torch.randn(layout[name][2], generator=generator) * 0.04).bfloat16()
                   for name in fields]
        local_query = torch.stack([weights[2] for _ in devices])
    else:
        checkpoint = Checkpoint(args.checkpoint)
        packed = [initialize_rank_head(checkpoint, 43, rank) for rank in range(ranks)]

        def read_field(bank, name):
            start, length, shape = layout[name]
            return bank[start:start + length].reshape(shape)

        weights = [read_field(packed[0], name) for name in fields]
        for bank in packed[1:]:
            for name in fields:
                if name != "q_b" and not torch.equal(read_field(bank, name), read_field(packed[0], name)):
                    raise AssertionError(f"{name} is not replicated in the checkpoint")
        local_query = torch.stack([read_field(bank, "q_b") for bank in packed])
    weights[2] = local_query
    x = (torch.randn(16, D, generator=generator) * 0.1).bfloat16()
    x_ref = x.double().requires_grad_()
    qa, qn, qb, ka, kn = (w.double().requires_grad_() for w in weights)

    def rounded(value):
        return value + (value.bfloat16().double() - value).detach()

    def rms(value, weight):
        inv = (value.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps).rsqrt()
        return rounded(value * inv * weight)

    q_raw, kv_raw = (rounded(x_ref @ weight.T) for weight in (qa, ka))
    q_normalized = rms(q_raw, qn)
    projected_query = rounded(q_normalized @ qb.transpose(-1, -2))
    projected_kv = rms(kv_raw, kn)
    dy = torch.randn(projected_query.shape, generator=generator) * 0.025
    dz = torch.randn((ranks, *projected_kv.shape), generator=generator) * 0.025
    dy[:, 6:] = 0
    dz[:, 6:] = 0
    gradients = torch.autograd.grad((projected_query, projected_kv), (qa, qn, qb, ka, kn, x_ref),
                                    (dy.double(), dz.double().sum(0)))

    def replicate(tensor):
        return tensor.detach().unsqueeze(0).expand(ranks, *tensor.shape).contiguous()

    expected = tuple(
        t if name == "q_b" else replicate(t.unsqueeze(0) if t.ndim == 1 else t)
        for name, t in zip((*fields, "x"), gradients, strict=True)
    )
    outputs = tuple(torch.empty(t.shape, dtype=torch.float32) for t in expected)
    inputs = tuple(replicate(t.bfloat16()) for t in (x, q_raw, q_normalized, kv_raw))
    inputs += (dy, dz, *(w if name == "q_b" else replicate(w)
                         for name, w in zip(fields, weights, strict=True)))
    scratch = tuple(torch.empty_like(v, dtype=torch.float32) for v in (
        inputs[1], inputs[1], inputs[3], inputs[1], inputs[3], inputs[0], inputs[0],
    ))
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    report = {"ranks": ranks, "passed": False, "checkpoint_weights": args.checkpoint is not None}
    try:
        program = make_mla_input_projection_backward(ranks).compile(*inputs, *outputs, *scratch, 1, config=config)
        with DistributedWorker([program], config=config, persistent=True) as worker:
            handles = [worker.alloc_stacked_tensor(t.share_memory_()) for t in (*inputs, *outputs, *scratch)]
            worker.run(program, *handles, 1, config=config)
            metrics = {}
            for name, actual, expected_value, handle in zip(
                (*fields, "x"), outputs, expected, handles[len(inputs):len(inputs) + len(outputs)], strict=True,
            ):
                for rank, shard in enumerate(handle.shards):
                    local = actual[rank]
                    worker.copy_from(local.data_ptr(), shard.data_ptr, local.nbytes, worker_id=rank)
                metrics[name] = float((actual - expected_value).abs().max())
                torch.testing.assert_close(actual, expected_value.float(), atol=8e-6, rtol=9e-5)
                if name != "q_b" and any(not torch.equal(actual[0], t) for t in actual[1:]):
                    raise AssertionError(f"{name} gradients differ between replicas")
            report.update(passed=True, max_abs=metrics)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


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
