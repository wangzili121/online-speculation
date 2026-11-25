#!/usr/bin/env python3
"""Validate resident routed MoE backward from saved TP lanes through the EP return."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=2, choices=(2, 4))
    parser.add_argument("--block", type=int, default=32, choices=(8, 16, 32))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if len(set(devices)) != len(devices) or len(devices) % args.tp or args.block % args.tp:
        parser.error("Use distinct devices and a whole number of TP groups")
    if not args.compile_only and args.checkpoint is None:
        parser.error("Pass the real W8A8 checkpoint for an expert reverse run")

    import torch
    from blockspec.dsv41_packed_ep_dispatch import PackedExpertDispatch, make_packed_ep_dispatch
    from blockspec.dsv41_packed_ep_reverse import PackedExpertReverse, make_packed_ep_reverse
    from blockspec.dsv41_packed_expert_vjp import packed_expert_vjp_reference
    from blockspec.dsv41_packed_ffn_tape import PackedFFNPostGradient, packed_ffn_tape_shapes
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    model = FLASH
    ranks = tuple(range(len(devices)))
    tp, block, topk = args.tp, args.block, model.num_experts_per_tok
    world, dp, local = len(ranks), len(ranks) // tp, block // tp
    local_experts = model.n_routed_experts // world if args.compile_only else 4
    experts = world * local_experts
    hidden, inter = model.hidden_size, model.moe_intermediate_size
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    reverse_graph = make_packed_ep_reverse(block, hidden, inter, model.swiglu_limit,
                                          topk, experts, dp, tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, block=block, tp=tp, dp=dp, ep=world,
                              experts_per_rank=local_experts)))
        return
    dispatch_graph = make_packed_ep_dispatch(block, 1, hidden, model.hc_mult, topk, experts,
                                             dp, tp).compile(config=config)

    checkpoint = Checkpoint(args.checkpoint)
    native_weights = ResidentWeights(checkpoint)
    generator = torch.Generator().manual_seed(63126)
    x = torch.randn(world, local, hidden, generator=generator, dtype=torch.bfloat16)
    dy = torch.randn(dp, block, hidden, generator=generator) * .1
    factor = torch.rand(world, local, topk, generator=generator)
    factor[0, 0, 0] = 0
    bank_names = ("routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale")
    real_ids = [lane * (model.n_routed_experts // world) + index
                for lane in ranks for index in range(local_experts)]
    weights = [tuple(item for name in ("w1", "w3", "w2")
                     for item in native_weights.projection(f"layers.39.ffn.experts.{expert}.{name}", output_major=True))
               for expert in real_ids]
    # Owner 0 sends every row to the same experts to exercise the 32-row branch.
    # The other owner spreads its packets across the selected real experts.
    ids = torch.arange(topk, dtype=torch.int32).expand(world, local, -1).clone()
    permutation = torch.rand(world - tp, local, experts, generator=generator)
    ids[tp:] = permutation.topk(topk).indices.to(torch.int32)
    owners = tuple(range(0, world, tp))
    shapes = packed_ffn_tape_shapes(block, 1, tp, hidden, model.hc_mult)
    tape_names = ("ffn_tape_vectors", "ffn_tape_route_ids", "ffn_tape_route_weights")
    cases = []

    with DistributedWorker([dispatch_graph, reverse_graph], config, persistent=True,
                           reset_persistent_windows=False) as runtime:
        def upload(value, rank):
            return runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

        def read(shards, rank):
            device = shards.shards[rank]
            host = torch.empty(device.shape, dtype=device.dtype)
            runtime.copy_from(host.data_ptr(), device.data_ptr, host.nbytes, worker_id=rank)
            return host

        banks, tapes = [], []
        for rank in ranks:
            tensors = [torch.zeros(shapes[name][0], dtype=getattr(torch, shapes[name][1])) for name in tape_names]
            tensors[0][-local:] = x[rank]
            tensors[1][-local:, :topk] = ids[rank]
            tensors[2][-local:, :topk] = factor[rank]
            tapes.append({name: upload(value, rank) for name, value in zip(tape_names, tensors, strict=True)})
            owned = weights[rank * local_experts:(rank + 1) * local_experts]
            bank = {}
            for name, parts in zip(bank_names, zip(*owned, strict=True), strict=True):
                layer = torch.cat(parts, dim=0) if name.endswith("_scale") else torch.stack(parts)
                if not name.endswith("_scale"):
                    # The resident handle describes [K, N], with bytes uploaded in [N, K] order.
                    layer = layer.view(local_experts, layer.shape[-1], layer.shape[-2])
                two_layers = torch.cat((torch.zeros_like(layer), layer))
                bank[name] = upload(two_layers, rank)
            banks.append(bank)
        gradient = StackedDeviceTensor([upload(dy[index], rank) for index, rank in enumerate(owners)],
                                       dy.shape, owners)
        proposer = SimpleNamespace(runtime=runtime, config=config, world=world, ranks=tapes, block=block)
        post = SimpleNamespace(owners=owners, first_layer=1, outputs=(gradient,))
        dispatch = PackedExpertDispatch(proposer, post, dispatch_graph, topk, hidden)
        reverse = PackedExpertReverse(runtime, config, reverse_graph, banks, block, hidden,
                                      inter, topk, local_experts, tp)
        for owner, active in ((owners[0], block), (owners[-1], 7), (owners[0], 1)):
            replica = owners.index(owner)
            taped = SimpleNamespace(owner=owner, ffn_tape={rank: tapes[rank] for rank in range(owner, owner + tp)})
            incoming = PackedFFNPostGradient(owner, 1, active, gradient.shards[replica], None, None, 0.)
            packet = dispatch(taped, incoming)
            elapsed = [reverse(packet) for _ in range(5)]
            result = elapsed[-1]
            expected_dx = torch.zeros(tp, local, topk, hidden)
            expected_route = torch.zeros(tp, local, (topk + 7) // 8 * 8)
            for expert, matrix in enumerate(weights):
                slots = [(token % tp, token // tp, token, choice)
                         for token in range(active) for choice in range(topk)
                         if ids[owner + token % tp, token // tp, choice] == expert]
                if not slots:
                    continue
                lane, row, activation, choice = (torch.tensor(indices) for indices in zip(*slots, strict=True))
                _, local_dx, droute = packed_expert_vjp_reference(
                    x[owner + lane, row], dy[replica, activation], factor[owner + lane, row, choice, None],
                    *matrix, limit=model.swiglu_limit)
                expected_dx[lane, row, choice] = local_dx
                expected_route[lane, row, choice] = droute[:, 0]
            summed = torch.zeros_like(expected_dx[:, :, 0])
            for choice in range(topk):
                summed += expected_dx[:, :, choice]
            actual_dx, actual_route = [torch.stack([read(values, rank) for rank in range(owner, owner + tp)])
                                       for values in (result.input, result.route)]
            rates = {}
            for name, got, expected in (("input", actual_dx, summed), ("route", actual_route, expected_route)):
                delta = (got - expected).double()
                rates[name] = dict(relative_l2=float(delta.norm() / expected.double().norm()),
                                   max_abs=float(delta.abs().max()))
                assert rates[name]["relative_l2"] < .006, (owner, active, name, rates[name])
            touched = torch.arange(local)[None, :] * tp + torch.arange(tp)[:, None] < active
            assert not torch.count_nonzero(actual_dx[~touched]) and not torch.count_nonzero(actual_route[~touched])
            cases.append(dict(owner=owner, active_rows=active, metrics=rates,
                              warm_median_ms=1000 * median(step.seconds for step in elapsed[1:])))
            print(json.dumps(cases[-1]), flush=True)
    print(json.dumps(dict(passed=True, block=block, dp=dp, tp=tp, ep=world, checkpoint=str(args.checkpoint),
                          checkpoint_layer=39, resident_bank_layer=1, actual_checkpoint_experts=real_ids, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
