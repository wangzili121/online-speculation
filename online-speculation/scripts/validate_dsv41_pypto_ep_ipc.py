#!/usr/bin/env python3
"""Exercise graph-owned A3 buffers through a same-host external HCCL mapping."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace


def compile_programs(config, world, columns):
    import pypto.language as pl
    import pypto.language.distributed as pld

    @pl.jit
    def write_rank(output: pl.Tensor[[world, columns], pl.INT32],
                   rank: pl.Scalar[pl.INT32], version: pl.Scalar[pl.INT32]):
        for peer in pl.spmd(world):
            lanes = pl.tile.arange(0, [1, columns], dtype=pl.INT32)
            offset = pl.cast(version * 1000 + rank * 100 + peer * 10, pl.INT32)
            output = pl.store(pl.tile.add(lanes, offset), [peer, 0], output)
        return output

    @pl.jit
    def read_rank(received: pl.Tensor[[world, columns], pl.INT32],
                  output: pl.Tensor[[world, columns], pl.INT32]):
        for source in pl.spmd(world):
            row = pl.tile.load(received, [source, 0], [1, columns])
            output = pl.store(pl.tile.add(row, 7), [source, 0], output)
        return output

    @pl.jit.host
    def produce(
        output: pl.Out[pl.Tensor[[world, world, columns], pl.INT32]],
        version: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(pld.world_size()):
            write_rank(output[rank], rank, version, device=rank)

    @pl.jit.host
    def consume(
        received: pl.Tensor[[world, world, columns], pl.INT32],
        output: pl.Out[pl.Tensor[[world, world, columns], pl.INT32]],
    ):
        for rank in pl.range(pld.world_size()):
            read_rank(received[rank], output[rank], device=rank)

    return tuple(stage.compile(config=config) for stage in (produce, consume))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--columns", type=int, default=256)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--moe", action="store_true", help="Compile or exercise the EP16 MoE graph stages")
    parser.add_argument("--rows", type=int, default=64, choices=(16, 64, 80))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not 16 <= args.columns <= 1024 or args.columns % 16:
        parser.error("Choose between 16 and 1024 columns, divisible by 16")

    import torch
    from blockspec.dsv41_packed_ep_ipc import ResidentHcclMappings
    from blockspec.pypto_dsv41_cluster_transport import ClusterCollectives
    from models.deepseek_v4_1_flash_a3 import moe_transport
    from pypto.backend import BackendType
    from pypto.ir import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    devices = [int(value) for value in args.devices.split(",")]
    world, columns = len(devices), args.columns
    if devices != list(range(world)) or (not args.compile_only and world not in (1, 2)):
        parser.error("Use consecutive local device IDs and one or two devices for runtime validation")
    if args.moe and not args.compile_only and (world != 1 or args.checkpoint is None):
        parser.error("The real-weight MoE fixture uses --devices 0 and --checkpoint")
    ranks = tuple(range(world))
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    if args.moe:
        from models.deepseek_v4_1_flash_a3 import config as native_config
        from blockspec.dsv41_packed_ep_graph import make_hccl_ep_graphs

        tokens = 64 if args.rows == 80 or (args.compile_only and args.rows == 64) else 16
        native_config.set_moe_tokens(tokens)
        programs = {}
        for phase in ("prefill", "decode"):
            entries = make_hccl_ep_graphs(tokens, args.rows, world, prefill=phase == "prefill")
            programs[phase] = tuple(entry.compile(config=config) for entry in entries)
        report = dict(passed=True, local_ranks=world, ep=16, tokens=tokens, rows=args.rows,
                      compiled={phase: ["prepare", "experts", "restore"] for phase in programs})
        if not args.compile_only:
            report["real_weights"] = exercise_moe(config, programs, args.checkpoint, moe_transport, tokens, args.rows)
        print(json.dumps(report), flush=True)
        return
    produce, consume = compile_programs(config, world, columns)
    if args.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, ranks=world)), flush=True)
        return
    dimensions = (world, columns)
    with ExitStack() as stack:
        runtime = stack.enter_context(DistributedWorker([produce, consume], config, persistent=True))
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=world))

        def allocate(*, exportable):
            shards = [runtime.alloc_tensor(dimensions, torch.int32, worker_id=rank, exportable=exportable)
                      for rank in ranks]
            return StackedDeviceTensor(shards, (world, *dimensions), ranks)

        sent, received, observed = allocate(exportable=True), allocate(exportable=True), allocate(exportable=False)
        # The fixture uses one on-host group. Production EP uses the same resident transfer entry with two hosts.
        plan = SimpleNamespace(local_ranks=range(world), node=0, groups=lambda _: (ranks,))
        dispatcher = SimpleNamespace(devices=devices, map_ranks=lambda task: list(pool.map(task, ranks)))
        transport = stack.enter_context(ClusterCollectives(dispatcher, plan, SimpleNamespace(node=0),
                                                          native=moe_transport))
        mappings = transport.mappings = ResidentHcclMappings(runtime, transport.memory)
        for rank in ranks:
            mappings.register(rank, sent.shards[rank])
            mappings.register(rank, received.shards[rank])
        transfers = [((source, target, "packet"),)
                     for source, target in zip(sent.shards, received.shards, strict=True)]

        acl = transport.memory.library
        check = moe_transport._check
        stream = ctypes.c_void_p()
        if world == 1:
            check(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
            stack.callback(lambda: check(acl.aclrtDestroyStream(stream), "aclrtDestroyStream"))
        durations = []
        for iteration in range(6):
            runtime.run(produce, sent, iteration)
            started = perf_counter()
            if world == 2:
                transport.exchange_resident("ipc/round", transfers, group="tp")
            else:
                size = sent.shards[0].nbytes
                src = mappings.address(0, sent.shards[0])
                dst = mappings.address(0, received.shards[0])
                check(acl.aclrtMemcpyAsync(dst, size, src, size, 3, stream), "aclrtMemcpyAsync/D2D")
                check(acl.aclrtSynchronizeStream(stream), "aclrtSynchronizeStream/D2D")
            durations.append(perf_counter() - started)
            runtime.run(consume, received, observed)
            for rank, shard in enumerate(observed.shards):
                result = torch.empty(dimensions, dtype=torch.int32)
                runtime.copy_from(result.data_ptr(), shard.data_ptr, result.nbytes, worker_id=rank)
                expected = (torch.arange(world, dtype=torch.int32)[:, None] * 100
                            + torch.arange(columns, dtype=torch.int32)[None, :]
                            + iteration * 1000 + rank * 10 + 7)
                torch.testing.assert_close(result, expected, atol=0, rtol=0)

    print(json.dumps(dict(passed=True, physical_devices=len(devices), iterations=len(durations),
                          transport="hccl_all_to_all" if world == 2 else "acl_d2d",
                          host_staging=False, byte_exact=True, leases_retired=True,
                          warm_transfer_median_ms=1000 * median(durations[1:]))), flush=True)


def exercise_moe(config, programs, model_path, native, tokens, rows):
    """Run one owning lane with real weights and route its packets to itself."""
    import torch
    from blockspec.dsv41_packed_ep_graph import GraphResidentHcclMoE
    from models.deepseek_v4_1_flash_a3.config import FLASH, SHARED_NZ_PANEL
    from models.deepseek_v4_1_flash_a3.expert_routed import golden_expert_routed
    from models.deepseek_v4_1_flash_a3.expert_shared import golden_expert_shared
    from models.deepseek_v4_1_flash_a3.gate import golden_gate_reference
    from models.deepseek_v4_1_flash_a3.gemm import golden_activation_int8
    from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights, pack_projection_int8_nz
    from pypto.runtime import StackedDeviceTensor
    from pypto.runtime.distributed_runner import DistributedWorker
    from validate_dsv41_pypto_ep_hccl_resident import expert_bank

    torch.set_num_threads(4)
    checkpoint = Checkpoint(model_path)
    weights = ResidentWeights(checkpoint)
    d, inter, ep = FLASH.hidden_size, FLASH.moe_intermediate_size, 16
    owner, selected = 0, tuple(range(8))
    stem = "layers.39"
    norm = checkpoint.tensor(stem + ".ffn_norm.weight").bfloat16()
    gate = checkpoint.tensor(stem + ".ffn.gate.weight").float()
    bias = checkpoint.tensor(stem + ".ffn.gate.bias").float().clone()
    # The fixture activates only experts mapped to its real chip.
    bias[len(selected):] -= 1e4
    shared_name = stem + ".ffn.shared_experts."
    w1, s1 = weights.projection(shared_name + "w1", output_major=True)
    w3, s3 = weights.projection(shared_name + "w3", output_major=True)
    w2, s2 = weights.projection(shared_name + "w2", output_major=True)
    paired = torch.stack((w1.reshape(-1, SHARED_NZ_PANEL, d), w3.reshape(-1, SHARED_NZ_PANEL, d)),
                         dim=1).reshape(-1, d).T
    packed_w13 = pack_projection_int8_nz(paired)
    packed_w2 = pack_projection_int8_nz(w2.T)
    bank = expert_bank(FLASH.n_routed_experts // ep, d, inter, initialize=True)
    for index in selected:
        name = f"{stem}.ffn.experts.{index}."
        projection = [value for key in ("w1", "w3", "w2")
                      for value in weights.projection(name + key, output_major=True)]
        for destination, value in zip(bank, projection, strict=True):
            destination[index].copy_(value.reshape(destination[index].shape))
    generator = torch.Generator().manual_seed(34261)
    x = torch.randn(rows, d, dtype=torch.bfloat16, generator=generator)
    stages = [program for group in programs.values() for program in group]
    results = []
    with ExitStack() as stack:
        runtime = stack.enter_context(DistributedWorker(stages, config, persistent=True))
        memory = native.CANNMemory()
        acl, check = memory.library, native._check
        memory.select(owner)
        stream = ctypes.c_void_p()
        check(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
        transport = SimpleNamespace(plan=SimpleNamespace(counts=(8, 8), tp=4, local_ranks=(0,)),
                                    memory=memory, mappings=None)

        def close_transport():
            check(acl.aclrtDestroyStream(stream), "aclrtDestroyStream")
            if transport.mappings is not None:
                transport.mappings.close()

        stack.callback(close_transport)

        def local_exchange(_stage, batches):
            for source, target, _field in batches[0]:
                src = transport.mappings.address(owner, source)
                dst = transport.mappings.address(owner, target)
                check(acl.aclrtMemcpyAsync(dst, target.nbytes, src, source.nbytes, 3, stream), "aclrtMemcpyAsync/D2D")
            check(acl.aclrtSynchronizeStream(stream), "aclrtSynchronizeStream/D2D")

        transport.exchange_resident = local_exchange
        runner = GraphResidentHcclMoE(runtime, transport, programs, tokens=tokens, rows=rows)

        def upload(value):
            handle = runtime.alloc_tensor(value.shape, value.dtype, worker_id=owner, init=value)
            return StackedDeviceTensor([handle], (1, *value.shape), (owner,))

        def read(tensor):
            handle = tensor.shards[owner]
            value = torch.empty(handle.shape, dtype=handle.dtype)
            runtime.copy_from(value.data_ptr(), handle.data_ptr, value.nbytes, worker_id=owner)
            return value

        input_tensor = upload(x)
        shared = [upload(value) for value in (norm, gate, bias, packed_w13, s1, s3, packed_w2, s2)]
        routed = [upload(value) for value in bank]
        for phase in programs:
            for active in (1, min(rows, 17)):
                runner.bind_batch((active, 0, 0, 0))
                output = runner.run(f"{phase}/{active}", prefill=phase == "prefill", ffn_input=input_tensor,
                                    shared_weights=shared, routed_weights=routed)
                slots = torch.arange(0, active, 4)
                live = len(slots)
                normalized = read(runner.normed)[:live]
                factors, ids = golden_gate_reference(normalized, gate, bias)
                torch.testing.assert_close(read(runner.ids)[:live], ids, atol=0, rtol=0)
                qx = golden_activation_int8(normalized)
                scale = (normalized.float().abs().amax(-1) / 127).clamp_min(1e-30)
                combined = golden_expert_shared(qx, scale, w1.T, s1, w3.T, s3, w2.T, s2).float()
                for choice in range(FLASH.num_experts_per_tok):
                    for expert in selected:
                        matches = ids[:, choice] == expert
                        if matches.any():
                            parts = [item[expert].T if item.ndim == 3 else item[expert:expert + 1] for item in bank]
                            contribution = golden_expert_routed(qx[matches], scale[matches],
                                                               factors[matches, choice][None], *parts)
                            combined[matches] += contribution.float()
                target = torch.zeros_like(x)
                target[slots] = combined.bfloat16()
                observed = read(output)
                torch.testing.assert_close(observed, target, atol=0.003, rtol=0.03)
                results.append(dict(phase=phase, active=active, owner_rows=live,
                                    output_max_abs=(observed.float() - target.float()).abs().max().item()))
    return dict(layer=39, experts=list(selected), scope="single-owner EP16 device loopback", cases=results,
                dispatch_payload_bytes=runner.dispatch_send[1].shards[0].nbytes,
                return_payload_bytes=runner.return_send.shards[0].nbytes)


if __name__ == "__main__":
    main()
