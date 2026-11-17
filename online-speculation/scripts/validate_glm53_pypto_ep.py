#!/usr/bin/env python3
"""Compare the distributed routed-expert adjoint with full-precision references."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch

from blockspec.pypto_glm53_online_ep import ROUTES, ROUTE_STRIDE, ROWS, make_frozen_moe_backward, make_routed_ep_backward


def reference(x_q, scale, upstream, ids, weights, parameters, active, limit):
    from models.glm5_3_flash.quantization import quantize_per_token_int8

    width = x_q.shape[1]
    output = torch.zeros((ROWS, width + ROUTE_STRIDE), dtype=torch.float64)
    gate_up, gate_up_scale, down, down_scale = (t.flatten(0, 1) for t in parameters)
    for expert in range(gate_up.shape[0]):
        token, route = torch.where(ids[:active] == expert)
        if not token.numel():
            continue
        gu, ds = gate_up_scale[expert].bfloat16(), down_scale[expert].bfloat16()
        raw = (x_q[token].int() @ gate_up[expert].int().T).float() * scale[token] * gu.float()
        gate, up = raw.double().chunk(2, -1)
        g, u = gate.clamp_max(limit), up.clamp(-limit, limit)
        activation = torch.nn.functional.silu(g) * u
        h_q, h_scale = quantize_per_token_int8(activation.float())
        value = ((h_q.int() @ down[expert].int().T).float() * h_scale * ds.float()).bfloat16()
        dy = upstream[token].double()
        output[token, width + route] = (dy * value.double()).sum(-1)

        # Frozen W8A8 weights and identity derivative through activation Q/DQ.
        d_h = ((dy * weights[token, route, None].double()) * ds.double()) @ down[expert].double()
        p = g.sigmoid()
        d_gate = d_h * u * p * (1 + g * (1 - p)) * (gate <= limit)
        d_up = d_h * g * p * ((up >= -limit) & (up <= limit))
        d_raw = torch.cat((d_gate, d_up), dim=-1)
        output[token, :width] += (d_raw * gu.double()) @ gate_up[expert].double()
    return output


def full_check(args, programs, config, report, checkpoint, parameters):
    from models.glm5_3_flash.config import D, FLASH
    from models.glm5_3_flash.gate import gate
    from models.glm5_3_flash.rmsnorm import golden_rmsnorm_quant
    from models.glm5_3_flash.utils import layer_router, load_shared_expert, pack_int8_nz
    from pypto.runtime import DistributedWorker

    ranks = len(args.devices)
    rng = torch.Generator().manual_seed(53)
    router = layer_router(checkpoint, args.layer)
    norm = checkpoint.read(checkpoint.layer_key(args.layer, "post_attention_layernorm.weight")).bfloat16()
    bias = torch.full_like(router.correction_bias, -50)
    bias[:ranks * parameters[0].shape[1]].fill_(50)
    shared = [load_shared_expert(checkpoint, args.layer, tp_rank=rank, tp_size=ranks) for rank in range(ranks)]
    packed = (torch.stack([pack_int8_nz(s.gate_up) for s in shared]),
              torch.stack([s.gate_up_scale for s in shared]),
              torch.stack([pack_int8_nz(s.down) for s in shared]),
              torch.stack([s.down_scale for s in shared]))

    def replicate(value):
        return value.unsqueeze(0).repeat((ranks,) + (1,) * value.ndim).share_memory_()

    active = 6
    x = (torch.randn(ROWS, D, generator=rng) * 0.4).bfloat16()
    upstream = torch.randn(ROWS, D, generator=rng) * 0.01
    route_shape = (ranks, ROWS, ROUTES)
    selected = torch.empty(route_shape, dtype=torch.int32)
    weights, scores = (torch.empty(route_shape) for _ in range(2))
    tokens = torch.empty((ranks, ROWS, D), dtype=torch.int8)
    scales = torch.empty((ranks, ROWS, 1))
    partial, combined = (torch.empty((ranks, ROWS, D + ROUTE_STRIDE)) for _ in range(2))
    shared_partial, shared_sum, norm_gradient = (torch.empty((ranks, ROWS, D)) for _ in range(3))
    model = (replicate(x), replicate(upstream), replicate(norm), replicate(router.weight))
    router_args = (model[0], model[2], model[3], replicate(bias), selected, weights, scores, tokens, scales)
    routed_args = (tokens, scales, model[1], selected, weights, *parameters, partial, combined)
    result = torch.empty((ranks, ROWS, D)).share_memory_()
    finish_args = (*model, *packed, selected, scores, tokens, scales, combined, shared_partial, shared_sum,
                   torch.empty(route_shape), norm_gradient, result)
    with DistributedWorker(programs, config=config, persistent=True, reset_persistent_windows=False) as worker:
        handles = {}
        for arg in (*router_args, *routed_args, *finish_args):
            if id(arg) not in handles:
                handles[id(arg)] = worker.alloc_stacked_tensor(arg.share_memory_())
        arguments = tuple(tuple(handles[id(arg)] for arg in stage_args)
                          for stage_args in (router_args, routed_args, finish_args))
        samples = {stage: [] for stage in ("route", "routed_vjp", "shared_and_norm_vjp")}
        for epoch in range(1, args.profile_steps + 2):
            for index, (stage, program, bindings) in enumerate(zip(samples, programs, arguments, strict=True)):
                scalars = (active,) if index == 0 else (active, epoch)
                start = perf_counter_ns()
                worker.run(program, *bindings, *scalars, config=config)
                if epoch > 1:
                    samples[stage].append((perf_counter_ns() - start) / 1e6)
        if args.profile_steps:
            report["profile"] = {"measured_steps": args.profile_steps,
                                 "dispatch_ms_median": {key: median(ms) for key, ms in samples.items()},
                                 "dispatch_ms_samples": samples}
        for rank, shard in enumerate(handles[id(result)].shards):
            worker.copy_from(result[rank].data_ptr(), shard.data_ptr, result[rank].nbytes, worker_id=rank)

    normed, x_q, scale = golden_rmsnorm_quant(x, norm)
    route_weights, ids = gate(normed, router.weight, bias)
    data = reference(x_q, scale, upstream, ids, route_weights, parameters, active, FLASH.swiglu_limit)
    d_normed, d_route = data[:, :D], data[:, D:D + ROUTES]
    normed_ref = normed[:active].double().requires_grad_()
    selected_gate = router.weight.double()[ids[:active].long()]
    score = torch.einsum("td,trd->tr", normed_ref, selected_gate).sigmoid()
    routing = score / score.sum(-1, keepdim=True) * FLASH.routed_scaling_factor
    d_router, = torch.autograd.grad(routing, normed_ref, d_route[:active])
    d_normed[:active].add_(d_router)
    limit = FLASH.swiglu_limit
    for shard in shared:
        gu_scale = shard.gate_up_scale.bfloat16().float()
        d_scale = shard.down_scale.bfloat16().double()
        raw = (x_q[:active].int() @ shard.gate_up.int().T).float() * scale[:active] * gu_scale
        gate_raw, up_raw = raw.bfloat16().double().requires_grad_().chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate_raw.clamp_max(limit)) * up_raw.clamp(-limit, limit)
        d_h = (upstream[:active].double() * d_scale) @ shard.down.double()
        d_gate, d_up = torch.autograd.grad(h, (gate_raw, up_raw), d_h)
        d_normed[:active].add_((torch.cat((d_gate, d_up), dim=-1) * gu_scale.double()) @ shard.gate_up.double())
    x_ref = x.double().requires_grad_()
    normed_ref = x_ref * (x_ref.square().mean(-1, keepdim=True) + FLASH.rms_norm_eps).rsqrt() * norm.double()
    expected, = torch.autograd.grad(normed_ref, x_ref, d_normed)
    expected = expected.float().expand_as(result)
    torch.testing.assert_close(result, expected, atol=8e-6, rtol=4e-4)
    if result[:, active:].count_nonzero():
        raise AssertionError("inactive rows received a full MoE gradient")
    report["cases"].append({"rows": active, "input_max_abs": float((result - expected).abs().max())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, choices=(43, 44), default=44)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--full-moe", action="store_true", help="include native routing, shared experts and RMSNorm")
    parser.add_argument("--profile-steps", type=int, default=0, help="time extra full MoE steps after warmup")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = len(args.devices)
    if args.profile_steps < 0 or (args.profile_steps and (not args.full_moe or args.compile_only)):
        parser.error("--profile-steps needs device execution with --full-moe")
    if ranks not in (1, 2, 4, 8) or len(set(args.devices)) != ranks or args.output.exists():
        parser.error("use distinct devices and a fresh output path")
    if args.checkpoint is None and not args.compile_only:
        parser.error("device execution needs a GLM W8A8 checkpoint")

    from models.glm5_3_flash.config import D, FLASH
    from models.glm5_3_flash.quantization import quantize_per_token_int8
    from models.glm5_3_flash.utils import Checkpoint, shard_routed_experts
    from pypto.ir import DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig

    torch.set_num_threads(4)
    local_experts = FLASH.n_routed_experts // ranks if args.compile_only else (
        max(4, ROUTES // ranks) if args.full_moe else 4
    )
    config = RunConfig(platform="a2a3", distributed_config=DistributedConfig(
        device_ids=args.devices, num_sub_workers=0,
    ))
    report = {"passed": False, "ranks": ranks, "layer": args.layer, "local_experts": local_experts, "cases": []}
    try:
        report["full_moe"] = args.full_moe
        entries = (make_frozen_moe_backward(ranks, local_experts) if args.full_moe
                   else (make_routed_ep_backward(ranks, local_experts),))
        programs = []
        report["compiled_stages"] = []
        for entry in entries:
            programs.append(entry.compile(config=config))
            report["compiled_stages"].append(entry._func.__name__)
        report["compiled"] = True
        if args.compile_only:
            report["passed"] = True
            return
        checkpoint = Checkpoint(args.checkpoint)
        shards = [shard_routed_experts(checkpoint, args.layer, rank, ranks,
                                       local_experts=local_experts, global_experts=ranks * local_experts)
                  for rank in range(ranks)]
        parameters = tuple(torch.stack([getattr(shard, name) for shard in shards]).share_memory_()
                           for name in ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale"))
        if args.full_moe:
            full_check(args, programs, config, report, checkpoint, parameters)
            report["passed"] = True
            return
        replicated = (torch.empty(ranks, ROWS, D, dtype=torch.int8), torch.empty(ranks, ROWS, 1),
                      torch.empty(ranks, ROWS, D), torch.empty(ranks, ROWS, ROUTES, dtype=torch.int32),
                      torch.empty(ranks, ROWS, ROUTES))
        partial = torch.zeros((ranks, ROWS, D + ROUTE_STRIDE)).share_memory_()
        combined = torch.empty_like(partial).share_memory_()
        rng = torch.Generator().manual_seed(45)
        program, = programs
        with DistributedWorker([program], config=config, persistent=True, reset_persistent_windows=False) as worker:
            inputs = (*replicated, *parameters, partial, combined)
            handles = [worker.alloc_stacked_tensor(t.share_memory_()) for t in inputs]
            for epoch, active in enumerate((6, 3), start=1):
                x = torch.randn(ROWS, D, generator=rng) * 0.4
                x_q, scale = quantize_per_token_int8(x)
                upstream = torch.randn(ROWS, D, generator=rng) * 0.01
                ids = torch.full((ROWS, ROUTES), -1, dtype=torch.int32)
                weights = torch.rand((ROWS, ROUTES), generator=rng)
                for row in range(active):
                    selected = (row, 1, ranks * local_experts - 1) if epoch == 1 else (1,)
                    selected = list(dict.fromkeys(e % (ranks * local_experts) for e in selected))
                    ids[row, :len(selected)] = torch.tensor(selected, dtype=torch.int32)
                weights.masked_fill_(ids < 0, 0.0)
                for dst, source, handle in zip(replicated, (x_q, scale, upstream, ids, weights), handles[:5], strict=True):
                    dst.copy_(source.unsqueeze(0).expand_as(dst))
                    for rank, shard in enumerate(handle.shards):
                        worker.copy_to(shard.data_ptr, dst[rank].data_ptr(), dst[rank].nbytes, worker_id=rank)
                worker.run(program, *handles, active, epoch, config=config)
                for rank, shard in enumerate(handles[-1].shards):
                    worker.copy_from(combined[rank].data_ptr(), shard.data_ptr, combined[rank].nbytes, worker_id=rank)
                expected = reference(x_q, scale, upstream, ids, weights, parameters, active, FLASH.swiglu_limit)
                expected = expected.float().expand_as(combined)
                torch.testing.assert_close(combined[..., :D], expected[..., :D], atol=3e-6, rtol=1e-4)
                torch.testing.assert_close(combined[..., D:D + ROUTES], expected[..., D:D + ROUTES],
                                           atol=5e-4, rtol=1e-3)
                if combined[:, active:].count_nonzero() or combined[..., D + ROUTES:].count_nonzero():
                    raise AssertionError("inactive rows or routing padding received a gradient")
                for replica in combined[1:]:
                    if not torch.equal(replica, combined[0]):
                        raise AssertionError("EP gradient replicas differ")
                error = (combined - expected).abs()
                report["cases"].append({"epoch": epoch, "rows": active,
                                        "input_max_abs": float(error[..., :D].max()),
                                        "route_max_abs": float(error[..., D:D + ROUTES].max())})
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
