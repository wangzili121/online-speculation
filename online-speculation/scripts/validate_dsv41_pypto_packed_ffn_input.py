#!/usr/bin/env python3
"""Validate the complete TP lane-local FFN input backward after routed expert return."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, choices=(2, 4), default=2)
    parser.add_argument("--block", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--routed", action="store_true", help="Use the actual W8A8 experts selected by the gate")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    block, tp, world = args.block, args.tp, len(devices)
    if len(set(devices)) != world or world % tp:
        parser.error("Use distinct devices and complete TP groups")
    if not args.compile_only and args.checkpoint is None:
        parser.error("Pass the authentic W8A8 checkpoint for a numerical check")

    import torch
    import torch.nn.functional as F
    from blockspec.dsv41_packed_ep_dispatch import PackedExpertDispatch, make_packed_ep_dispatch
    from blockspec.dsv41_packed_ep_reverse import (
        PackedExpertGradients,
        PackedExpertReverse,
        make_packed_ep_reverse,
    )
    from blockspec.dsv41_packed_expert_vjp import packed_expert_vjp_reference
    from blockspec.dsv41_packed_ffn_input_vjp import PackedFFNInputVJP, make_packed_ffn_input_vjp
    from blockspec.dsv41_packed_ffn_tape import PackedFFNPostGradient, packed_ffn_tape_shapes
    from models.deepseek_v4_1_flash_a3.config import FLASH as model, SHARED_NZ_PANEL
    from models.deepseek_v4_1_flash_a3.gate import golden_gate_reference
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights, pack_projection_int8_nz
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    dp, local, last_n = world // tp, block // tp, 4
    ranks, owners = tuple(range(world)), tuple(range(0, world, tp))
    hidden, topk = model.hidden_size, model.num_experts_per_tok
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = make_packed_ffn_input_vjp(block, last_n, model, dp, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, tp=tp, dp=dp, ep=world, last_n=last_n)))
        return
    dispatch_graph = make_packed_ep_dispatch(block, last_n, hidden, model.hc_mult, topk,
                                             model.n_routed_experts, dp, tp).compile(config=config)

    checkpoint = Checkpoint(args.checkpoint)
    weights = ResidentWeights(checkpoint)
    source_model_layer, resident_layer, tape_slot = 39, 1, 3
    prefix = f"layers.{source_model_layer}.ffn.shared_experts"
    w1, s1 = weights.projection(prefix + ".w1", output_major=True)
    w3, s3 = weights.projection(prefix + ".w3", output_major=True)
    w2, s2 = weights.projection(prefix + ".w2", output_major=True)
    paired = torch.stack((w1.reshape(-1, SHARED_NZ_PANEL, hidden),
                          w3.reshape(-1, SHARED_NZ_PANEL, hidden)), dim=1).reshape(-1, hidden).T
    ff_gate = checkpoint.tensor(f"layers.{source_model_layer}.ffn.gate.weight").float()
    bias = checkpoint.tensor(f"layers.{source_model_layer}.ffn.gate.bias")
    gamma = checkpoint.tensor(f"layers.{source_model_layer}.ffn_norm.weight")
    rank_bank = {
        "shared_w13": pack_projection_int8_nz(paired), "shared_w1_scale": s1, "shared_w3_scale": s3,
        "shared_w2": pack_projection_int8_nz(w2.T), "shared_w2_scale": s2, "gate_weight": ff_gate,
        "ffn_norm_weight": gamma,
    }
    generator = torch.Generator().manual_seed(483806)
    hidden_state = torch.randn((dp, block, hidden), generator=generator, dtype=torch.bfloat16)
    normed = (hidden_state.float() * torch.rsqrt(hidden_state.float().square().mean(-1, keepdim=True)
                                                + model.rms_norm_eps) * gamma.float()).bfloat16()
    dy = torch.randn((dp, block, hidden), generator=generator) * .1
    route_factors, route_ids = golden_gate_reference(normed.reshape(-1, hidden), ff_gate, bias)
    route_factors = route_factors.reshape(dp, block, topk)
    route_ids = route_ids.reshape(dp, block, topk)
    routed_dx = torch.randn((dp, block, hidden), generator=generator) * .003
    droute = torch.randn((dp, block, 8), generator=generator) * .1
    shapes = packed_ffn_tape_shapes(block, last_n, tp, hidden, model.hc_mult)
    tape_names = ("ffn_tape_vectors", "ffn_tape_route_ids", "ffn_tape_route_weights")
    fixtures = ((owners[0], 1), (owners[-1], min(block, 7)), (owners[0], 3))
    programs = [dispatch_graph, graph]
    routed_weights = {}
    if args.routed:
        programs.append(make_packed_ep_reverse(block, hidden, model.moe_intermediate_size, model.swiglu_limit,
                                               topk, model.n_routed_experts, dp, tp).compile(config=config))
        active_experts = {value for owner, active in fixtures
                          for value in route_ids[owner // tp, :active].flatten().tolist()}
        for expert in sorted(active_experts):
            stem = f"layers.{source_model_layer}.ffn.experts.{expert}"
            routed_weights[expert] = tuple(item for name in ("w1", "w3", "w2")
                                           for item in weights.projection(f"{stem}.{name}", output_major=True))
    cases = []
    with DistributedWorker(programs, config, persistent=True, reset_persistent_windows=False) as runtime:
        def upload(tensor, rank):
            return runtime.alloc_tensor(tensor.shape, tensor.dtype, init=tensor, worker_id=rank)

        def stacked(tensor):
            return StackedDeviceTensor([upload(tensor[rank // tp, rank % tp::tp].contiguous(), rank)
                                        for rank in ranks], (world, local, tensor.shape[-1]), ranks)

        def read(device, rank):
            result = torch.empty(device.shape, dtype=device.dtype)
            runtime.copy_from(result.data_ptr(), device.data_ptr, result.nbytes, worker_id=rank)
            return result

        ranks_tape, banks = [], []
        vector_offset = tape_slot * (2 * block + local)
        routes_offset = tape_slot * local
        for rank in ranks:
            replica, lane = divmod(rank, tp)
            vectors, ids, coefficients = [torch.zeros(shapes[key][0], dtype=getattr(torch, shapes[key][1]))
                                           for key in tape_names]
            vectors[vector_offset:vector_offset + block] = hidden_state[replica]
            vectors[vector_offset + 2 * block:vector_offset + 2 * block + local] = normed[replica, lane::tp]
            ids[routes_offset:routes_offset + local, :topk] = route_ids[replica, lane::tp]
            coefficients[routes_offset:routes_offset + local, :topk] = route_factors[replica, lane::tp]
            ranks_tape.append({key: upload(value, rank)
                               for key, value in zip(tape_names, (vectors, ids, coefficients), strict=True)})
            banks.append({key: upload(torch.stack((torch.zeros_like(value), value)), rank)
                          for key, value in rank_bank.items()})
        owner_grad = StackedDeviceTensor([upload(dy[index], rank) for index, rank in enumerate(owners)],
                                         dy.shape, owners)
        proposer = SimpleNamespace(runtime=runtime, config=config, ranks=ranks_tape, world=world, tp=tp,
                                   block=block, last_layers=last_n)
        first_layer = resident_layer - tape_slot
        post = SimpleNamespace(owners=owners, first_layer=first_layer, outputs=(owner_grad,))
        dispatch = PackedExpertDispatch(proposer, post, dispatch_graph, topk, hidden)
        actor_model = SimpleNamespace(num_hidden_layers=first_layer + last_n, hidden_size=hidden,
                                      moe_intermediate_size=model.moe_intermediate_size,
                                      n_routed_experts=model.n_routed_experts)
        complete = PackedFFNInputVJP(proposer, graph, banks, actor_model)
        if args.routed:
            local_experts = model.n_routed_experts // world
            bank_keys = ("routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale")
            sample = next(iter(routed_weights.values()))
            for rank in ranks:
                for key, tensor in zip(bank_keys, sample, strict=True):
                    matrix_shape = tensor.shape[-1:] if key.endswith("_scale") else tensor.shape
                    shape = ((resident_layer + 1) * local_experts, *matrix_shape)
                    banks[rank][key] = runtime.alloc_tensor(shape, tensor.dtype, worker_id=rank)
            for expert, matrices in routed_weights.items():
                rank, index = divmod(expert, local_experts)
                for key, tensor in zip(bank_keys, matrices, strict=True):
                    offset = (resident_layer * local_experts + index) * tensor.nbytes
                    runtime.copy_to(banks[rank][key].data_ptr, tensor.data_ptr(), tensor.nbytes,
                                    dst_offset=offset, worker_id=rank)
            reverse = PackedExpertReverse(runtime, config, programs[-1], banks, block, hidden,
                                          model.moe_intermediate_size, topk, local_experts, tp)
        else:
            routed = stacked(routed_dx)
            droutes = stacked(droute)
        for owner, active in fixtures:
            replica = owner // tp
            student = SimpleNamespace(owner=owner, ffn_tape={rank: ranks_tape[rank]
                                                           for rank in range(owner, owner + tp)})
            boundary = PackedFFNPostGradient(owner, resident_layer, active, owner_grad.shards[replica], None, None, 0.)
            packets = dispatch(student, boundary)
            for rank in range(owner, owner + tp):
                positions = torch.arange(local) * tp + rank - owner
                expected = torch.where((positions < active)[:, None], dy[replica, positions], 0.)
                torch.testing.assert_close(read(packets.local_gradient.shards[rank], rank), expected, atol=0, rtol=0)
            returned = (reverse(packets) if args.routed else
                        PackedExpertGradients(owner, resident_layer, active, routed, droutes, 0.))
            steps = [complete(student, packets, returned) for _ in range(6)]
            result = read(steps[-1].collapsed, owner)

            x = normed[replica, :active]
            _, shared, _ = packed_expert_vjp_reference(x, dy[replica, :active], torch.ones(active, 1), w1, s1,
                                                       w3, s3, w2, s2, model.swiglu_limit)
            gate_input = x.double().requires_grad_(True)
            logits = ((gate_input @ ff_gate.double().T) / model.gate_temperature).gather(
                1, route_ids[replica, :active].long())
            score = F.softplus(logits).sqrt()
            weight = model.routed_scaling_factor * score / (score.sum(-1, keepdim=True) + 1e-20)
            if args.routed:
                gate_grad = torch.zeros(active, topk)
                expert_grad = torch.zeros(active, topk, hidden)
                for expert, matrices in routed_weights.items():
                    token, slot = (route_ids[replica, :active] == expert).nonzero(as_tuple=True)
                    if len(token):
                        _, dx, dw = packed_expert_vjp_reference(
                            x[token], dy[replica, token], route_factors[replica, token, slot, None],
                            *matrices, limit=model.swiglu_limit)
                        expert_grad[token, slot] = dx
                        gate_grad[token, slot] = dw[:, 0]
                expert_input = expert_grad.double().sum(dim=1)
            else:
                gate_grad = droute[replica, :active, :topk]
                expert_input = routed_dx[replica, :active].double()
            router, = torch.autograd.grad(weight, gate_input, gate_grad.double())
            collapsed = hidden_state[replica, :active].double().requires_grad_(True)
            norm_fp64 = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + model.rms_norm_eps)
            expected, = torch.autograd.grad(norm_fp64 * gamma.double(), collapsed,
                                             shared.double() + router + expert_input)
            delta = result[:active].double() - expected
            error = float(delta.norm() / expected.norm())
            assert error < .006 and not torch.count_nonzero(result[active:]), (owner, active, error)
            cases.append(dict(owner=owner, active_rows=active, relative_l2=error, max_abs=float(delta.abs().max()),
                              warm_median_ms=1000 * median(step.seconds for step in steps[1:])))
    print(json.dumps(dict(passed=True, block=block, tp=tp, dp=dp, ep=world, checkpoint=str(args.checkpoint),
                          checkpoint_layer=source_model_layer, resident_layer=resident_layer,
                          tape_slot=tape_slot, routed_experts=len(routed_weights), cases=cases)), flush=True)


if __name__ == "__main__":
    main()
