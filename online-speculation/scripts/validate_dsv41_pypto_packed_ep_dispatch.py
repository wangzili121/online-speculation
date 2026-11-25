#!/usr/bin/env python3
"""Validate the resident TP-to-EP expert-gradient dispatch and epoch isolation."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--fixture-tp", type=int, choices=(2, 4), default=4)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--split-tp", action="store_true", help="Measure the separate TP scatter baseline")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    world, tp = len(devices), args.fixture_tp
    if len(set(devices)) != world or world % tp or 384 % world:
        parser.error("The fixture requires distinct devices, complete TP groups and even expert sharding")

    import torch
    from blockspec.dsv41_packed_ep_dispatch import PackedExpertDispatch, make_packed_ep_dispatch
    from blockspec.dsv41_packed_ffn_tape import PackedFFNPostGradient, packed_ffn_tape_shapes
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    hidden, copies, topk, experts = FLASH.hidden_size, FLASH.hc_mult, FLASH.num_experts_per_tok, FLASH.n_routed_experts
    block, last_layers = 32, 4
    local, dp = block // tp, world // tp
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    graph = make_packed_ep_dispatch(block, last_layers, hidden, copies, topk, experts, dp, tp,
                                    fuse_tp=not args.split_tp).compile(config=config)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, ep=world, tp=tp, dp=dp, fused_tp=not args.split_tp)))
        return

    shapes = packed_ffn_tape_shapes(block, last_layers, tp, hidden, copies)
    names = ("ffn_tape_vectors", "ffn_tape_route_ids", "ffn_tape_route_weights")
    generator = torch.Generator().manual_seed(48123)
    inputs = torch.randn(world, local, hidden, generator=generator, dtype=torch.bfloat16)
    if args.checkpoint:
        from models.deepseek_v4_1_flash_a3.gate import golden_gate_reference
        from models.deepseek_v4_1_flash_a3.weights import Checkpoint

        checkpoint = Checkpoint(args.checkpoint)
        gate = checkpoint.tensor("layers.39.ffn.gate.weight")
        bias = checkpoint.tensor("layers.39.ffn.gate.bias")
        weights, ids = golden_gate_reference(inputs.flatten(0, 1), gate, bias)
        weights, ids = weights.reshape(world, local, topk), ids.reshape(world, local, topk)
    else:
        ids = torch.randint(0, experts, (world, local, topk), dtype=torch.int32, generator=generator)
        weights = torch.rand(world, local, topk, generator=generator)
    dvalue = torch.randn(dp, block, hidden, generator=generator)
    ranks, owners = tuple(range(world)), tuple(range(0, world, tp))
    cases = []
    with DistributedWorker([graph], config, persistent=True, reset_persistent_windows=False) as runtime:
        tapes = []
        for rank in ranks:
            host = [torch.zeros(shapes[name][0], dtype=getattr(torch, shapes[name][1])) for name in names]
            host[0][-local:] = inputs[rank]
            host[1][-local:, :topk] = ids[rank]
            host[2][-local:, :topk] = weights[rank]
            tapes.append({name: runtime.alloc_tensor(tensor.shape, tensor.dtype, init=tensor, worker_id=rank)
                          for name, tensor in zip(names, host, strict=True)})
        gradient = StackedDeviceTensor([
            runtime.alloc_tensor(part.shape, part.dtype, init=part, worker_id=rank)
            for part, rank in zip(dvalue, owners, strict=True)
        ], dvalue.shape, owners)
        proposer = SimpleNamespace(runtime=runtime, config=config, world=world, ranks=tapes, block=block)
        post = SimpleNamespace(owners=owners, first_layer=40 - last_layers, outputs=(gradient,))
        dispatch = PackedExpertDispatch(proposer, post, graph, topk, hidden)

        def read(shards, rank):
            device = shards.shards[rank]
            cpu = torch.empty(device.shape, dtype=device.dtype)
            runtime.copy_from(cpu.data_ptr(), device.data_ptr, cpu.nbytes, worker_id=rank)
            return cpu

        for owner, active in ((owners[0], block), (owners[-1], 7), (owners[0], 1)):
            tape = {rank: tapes[rank] for rank in range(owner, owner + tp)}
            student = SimpleNamespace(owner=owner, ffn_tape=tape)
            replica = owners.index(owner)
            incoming = PackedFFNPostGradient(owner, 39, active, gradient.shards[replica], None, None, 0.)
            samples = [dispatch(student, incoming) for _ in range(10)]
            result = samples[-1]
            packet_counts = []
            local_experts = experts // world
            for rank in ranks:
                meta, route_weight, x, dy = [read(output, rank) for output in
                                              (result.metadata, result.weights, result.input, result.gradient)]
                expected = torch.zeros_like(meta)
                for token in range(active):
                    src, position = owner + token % tp, token // tp
                    for route in range(topk):
                        expert = int(ids[src, position, route])
                        if expert // local_experts != rank:
                            continue
                        packet = token * topk + route
                        expected[packet, :5] = torch.tensor([result.epoch, expert % local_experts, src, position, route])
                        assert route_weight[packet, 0] == weights[src, position, route]
                        assert not torch.count_nonzero(route_weight[packet, 1:])
                        torch.testing.assert_close(x[packet], inputs[src, position], atol=0, rtol=0)
                        torch.testing.assert_close(dy[packet], dvalue[replica, token], atol=0, rtol=0)
                torch.testing.assert_close(meta, expected, atol=0, rtol=0)
                packet_counts.append(int(torch.count_nonzero(meta[:, 0])))
            assert sum(packet_counts) == active * topk
            cases.append(dict(owner=owner, active_rows=active, packets=packet_counts,
                              warm_median_ms=1000 * median(sample.seconds for sample in samples[1:])))
    print(json.dumps(dict(passed=True, byte_exact=True, ep=world, dp=dp, tp=tp, fused_tp=not args.split_tp,
                          checkpoint=str(args.checkpoint) if args.checkpoint else None, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
