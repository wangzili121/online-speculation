#!/usr/bin/env python3
"""Check 1122 W8A8 expert stages and resident EP exchange."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from blockspec.dsv41_packed_ep_hccl import make_ep_forward_experts, make_ep_forward_packets
from validate_dsv41_pypto_ep_hccl_packets import golden_buckets, golden_compact


def expert_bank(local, hidden, intermediate, *, initialize=False):
    create = torch.zeros if initialize else torch.empty
    return [create(shape, dtype=dtype) for shape, dtype in (
        ((local, intermediate, hidden), torch.int8), ((local, intermediate), torch.float32),
        ((local, intermediate, hidden), torch.int8), ((local, intermediate), torch.float32),
        ((local, hidden, intermediate), torch.int8), ((local, hidden), torch.float32),
    )]


def golden_experts(compact, bank, reference):
    counts, x, scale, weight, address = compact
    result = torch.zeros_like(x, dtype=torch.bfloat16)
    for index, (count, offset) in enumerate(counts.tolist()):
        if count:
            rows = slice(offset, offset + count)
            projections = [tensor[index].T if tensor.ndim == 3 else tensor[index:index + 1]
                           for tensor in bank]
            result[rows] = reference(x[rows], scale[0, rows], weight[:, rows], *projections)
    return result, address[0] >= 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--transport-source", type=Path,
                        help="moe_transport.py from the previous e32726b model revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=39)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--compile-only", action="store_true")
    options = parser.parse_args()
    devices = [int(device) for device in options.devices.split(",")]
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices):
        parser.error("Select one logical device for emulation or two for HCCL")
    use_hccl = len(devices) == 2 and not options.compile_only
    if options.iterations < 1:
        parser.error("Use at least one hot iteration")

    with ExitStack() as context:
        run(options, devices, use_hccl, context)


def run(options, devices, use_hccl, context):
    from models.deepseek_v4_1_flash_a3.config import FLASH
    from pypto.backend import BackendType
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    ranks = (0, 1)
    configs = [RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                         device_id=device) for device in devices]
    workers = [] if options.compile_only else [context.enter_context(ChipWorker(config)) for config in configs]
    tokens, world = (64, 16) if options.compile_only else (16, len(ranks))
    width, local, topk = FLASH.hidden_size, FLASH.n_routed_experts // 16, FLASH.num_experts_per_tok
    n = tokens * topk
    packets = world * n
    rows = packets + local * 16
    rng = torch.Generator().manual_seed(781)
    source = [[torch.randint(-127, 128, (tokens, width), dtype=torch.int8, generator=rng),
               torch.rand(1, tokens, generator=rng) * 0.015 + 0.015,
               torch.randint(local * world, (tokens, topk), dtype=torch.int32, generator=rng),
               torch.rand(tokens, topk, generator=rng)] for _ in ranks]
    selected = ([0, 3, 7, 16], [1, 5, 14, 23])
    if options.checkpoint:
        pool = torch.tensor([index + rank * local for rank in ranks for index in selected[rank]], dtype=torch.int32)
        for values in source:
            choices = torch.rand(tokens, len(pool), generator=rng).topk(topk).indices
            values[2] = pool[choices]
            values[3] /= values[3].sum(dim=1, keepdim=True)
    lives = (tokens, 5)
    expected_packets = [golden_buckets(*values, live, world, local) for values, live in zip(source, lives, strict=True)]
    outgoing = expected_packets[0]
    incoming = (torch.zeros(local, 2, dtype=torch.int32), torch.zeros(rows, width, dtype=torch.int8),
                torch.zeros(1, rows), torch.zeros(1, rows), torch.full((1, rows), -1, dtype=torch.int32))
    expert = torch.zeros(rows, width, dtype=torch.bfloat16)
    returning = torch.zeros(packets, width, dtype=torch.bfloat16)
    shared = torch.randn(tokens, width, dtype=torch.bfloat16, generator=rng)
    restored = torch.zeros_like(shared)
    args = [(*source[0], *outgoing, tokens), (*outgoing, *incoming),
            (expert, incoming[-1], incoming[0], returning), (returning, source[0][2], shared, restored, tokens)]
    programs = [entry.compile(*operands, config=configs[0])
                for entry, operands in zip(make_ep_forward_packets(tokens, width, world, local, topk), args, strict=True)]
    expert_programs = {}
    fused_programs = {}
    if options.checkpoint or options.compile_only:
        bank = expert_bank(local, width, FLASH.moe_intermediate_size)
        for phase in ("prefill", "decode"):
            factory = make_ep_forward_experts(tokens, world, local, prefill=phase == "prefill")
            expert_programs[phase] = factory.compile(*incoming, *bank, expert, config=configs[0])
            fused = make_ep_forward_experts(tokens, world, local, prefill=phase == "prefill", fuse_return=True)
            fused_programs[phase] = fused.compile(*incoming, *bank, returning, config=configs[0])
    if options.compile_only:
        print(json.dumps(dict(passed=True, compiled=True, world=world, tokens=tokens,
                              expert_phases=list(expert_programs), fused_return_phases=list(fused_programs))),
              flush=True)
        return

    weight_banks = []
    if options.checkpoint:
        from models.deepseek_v4_1_flash_a3.expert_routed import golden_expert_routed
        from models.deepseek_v4_1_flash_a3.weights import Checkpoint, ResidentWeights

        weights = ResidentWeights(Checkpoint(options.checkpoint))
        for rank in ranks:
            bank = expert_bank(local, width, FLASH.moe_intermediate_size, initialize=True)
            for index in selected[rank]:
                stem = f"layers.{options.layer}.ffn.experts.{rank * local + index}"
                projections = [value for name in ("w1", "w3", "w2")
                               for value in weights.projection(f"{stem}.{name}", output_major=True)]
                for destination, value in zip(bank, projections, strict=True):
                    destination[index].copy_(value.reshape(destination[index].shape))
            weight_banks.append(bank)

    channels = {}
    if use_hccl:
        if options.transport_source is None:
            from models.deepseek_v4_1_flash_a3 import moe_transport as native
        else:
            spec = importlib.util.spec_from_file_location("orthrus_validation.paired_transport",
                                                         options.transport_source)
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
        channels = native.HCCLBytes.create_group(native.CANNMemory(), devices)
        for channel in channels.values():
            context.callback(channel.close)
    else:
        configs *= world
        workers *= world

    def alloc(rank, values, upload=False):
        handles = [workers[rank].alloc_tensor(value.shape, value.dtype) for value in values]
        if upload:
            for value, handle in zip(values, handles, strict=True):
                workers[rank].copy_to(handle.data_ptr, value.data_ptr(), value.nbytes)
        return handles

    def read(rank, handles):
        out = [torch.empty(t.shape, dtype=t.dtype) for t in handles]
        for value, handle in zip(out, handles):
            workers[rank].copy_from(value.data_ptr(), handle.data_ptr, value.nbytes)
        return out

    def transposed(values, dest):
        return [torch.cat([row[index].chunk(world)[dest] for row in values]) for index in range(len(values[0]))]

    def exchange(send, recv, fields):
        if not use_hccl:
            sent = [read(rank, handles) for rank, handles in enumerate(send)]
            for rank in ranks:
                for value, handle in zip(transposed(sent, rank), recv[rank], strict=True):
                    workers[rank].copy_to(handle.data_ptr, value.data_ptr(), value.nbytes)
            return None

        def run(rank):
            transfers = list(zip(send[rank], recv[rank], fields, strict=True))
            channels[devices[rank]].exchange_device_batch(transfers)
        start = perf_counter()
        with ThreadPoolExecutor(max_workers=world) as pool:
            list(pool.map(run, ranks))
        return perf_counter() - start

    src = [alloc(rank, source[rank], upload=True) for rank in ranks]
    send = [alloc(rank, outgoing) for rank in ranks]
    recv = [alloc(rank, outgoing) for rank in ranks]
    for rank in ranks:
        workers[rank].run(programs[0], *src[rank], *send[rank], lives[rank], config=configs[rank])
    dispatch_seconds = exchange(send, recv, ("counts", "hidden", "auxiliary"))
    staged = [alloc(rank, incoming) for rank in ranks]
    outgoing_return = [alloc(rank, (returning,)) for rank in ranks]
    incoming_return = [alloc(rank, (returning,)) for rank in ranks]
    compact = []
    for rank in ranks:
        for value, target in zip(read(rank, recv[rank]), transposed(expected_packets, rank), strict=True):
            torch.testing.assert_close(value, target, atol=0, rtol=0)
        workers[rank].run(programs[1], *recv[rank], *staged[rank], config=configs[rank])
        used, expected = golden_compact(*transposed(expected_packets, rank), n, local, rows)
        actual = read(rank, staged[rank])
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1][:used], expected[1][:used], atol=0, rtol=0)
        for actual_row, expected_row in zip(actual[2:], expected[2:], strict=True):
            torch.testing.assert_close(actual_row[:, :used], expected_row[:, :used], atol=0, rtol=0)
        compact.append((used, expected))

    buffers = [alloc(rank, (expert,))[0] for rank in ranks]
    shared_buffers = [alloc(rank, (shared,), upload=True)[0] for rank in ranks]
    outputs = [alloc(rank, (restored,)) for rank in ranks]
    bank_buffers = [alloc(rank, bank, upload=True) for rank, bank in enumerate(weight_banks)]
    golden = []
    if weight_banks:
        golden = [golden_experts(expected, weight_banks[rank], golden_expert_routed)
                  for rank, (_, expected) in enumerate(compact)]
    phases = {}
    for phase in expert_programs or {"transport_only": None}:
        expected_return = []
        differences = []
        kernel_timings = []
        for rank in ranks:
            used, expected = compact[rank]
            buffer = buffers[rank]
            worker, config = workers[rank], configs[rank]
            if weight_banks:
                worker.run(expert_programs[phase], *staged[rank], *bank_buffers[rank], buffer, config=config)
                expert_result, = read(rank, (buffer,))
                reference, valid = golden[rank]
                output, target = expert_result[valid].float(), reference[valid].float()
                if not torch.isfinite(output).all():
                    raise AssertionError(f"{phase} rank {rank} produced nonfinite routed output")
                absolute = (output - target).abs()
                # Same tolerance as native l2_expert_routed validation.
                error_ratio = (absolute > 0.003 + target.abs() * 0.03).float().mean().item()
                differences.append(dict(rank=rank, valid_routes=int(valid.sum()),
                                        max_abs=absolute.max().item(), outlier_ratio=error_ratio,
                                        relative_l2=(absolute.norm() / target.norm()).item()))
                if error_ratio > 0.005:
                    raise AssertionError(f"{phase} checkpoint comparison: {differences[-1]}")
            else:
                expert_result = torch.zeros_like(expert)
                expert_result[:used].copy_(expected[1][:used].float() * expected[2][0, :used, None])
                worker.copy_to(buffer.data_ptr, expert_result.data_ptr(), expert_result.nbytes)
            packet_args = (buffer, staged[rank][-1], staged[rank][0], *outgoing_return[rank])
            worker.run(programs[2], *packet_args, config=config)
            expected_value = torch.zeros_like(returning)
            positions = expected[-1][0, :used].long()
            valid = positions >= 0
            expected_value[positions[valid]] = expert_result[:used][valid]
            expected_return.append((expected_value,))
            if weight_banks:
                expert_args = (*staged[rank], *bank_buffers[rank])
                worker.run(fused_programs[phase], *expert_args, *outgoing_return[rank], config=config)
                packet, = read(rank, outgoing_return[rank])
                torch.testing.assert_close(packet, expected_value, atol=0, rtol=0)
                separate, fused = [], []
                for _ in range(options.iterations):
                    start = perf_counter()
                    worker.run(expert_programs[phase], *expert_args, buffer, config=config)
                    worker.run(programs[2], *packet_args, config=config)
                    separate.append(perf_counter() - start)
                    start = perf_counter()
                    worker.run(fused_programs[phase], *expert_args, *outgoing_return[rank], config=config)
                    fused.append(perf_counter() - start)
                kernel_timings.append(dict(rank=rank, separate_median_ms=1000 * median(separate),
                                           fused_median_ms=1000 * median(fused)))
        return_seconds = exchange(outgoing_return, incoming_return, ("expert_return",))
        for rank in ranks:
            target, = transposed(expected_return, rank)
            received, = read(rank, incoming_return[rank])
            torch.testing.assert_close(received, target, atol=0, rtol=0)
            workers[rank].run(programs[3], *incoming_return[rank], src[rank][2], shared_buffers[rank],
                              *outputs[rank], lives[rank], config=configs[rank])
            expected = shared[:lives[rank]].float()
            for choice in range(topk):
                owner = source[rank][2][:lives[rank], choice].long() // local
                index = owner * n + torch.arange(lives[rank]) * topk + choice
                expected.add_(target[index].float())
            observed, = read(rank, outputs[rank])
            torch.testing.assert_close(observed[:lives[rank]], expected.bfloat16(), atol=0, rtol=0)
        phases[phase] = dict(exact_transport_and_combine=True, expert_differences=differences,
                             fused_return_byte_exact=bool(weight_banks), kernel_timings=kernel_timings,
                             return_seconds=return_seconds)
    report = dict(passed=True, world=world, tokens=tokens,
                  devices=devices, transport="hccl_device" if use_hccl else "host_rank_transpose",
                  checkpoint_layer=options.layer if weight_banks else None,
                  checkpoint_experts=pool.tolist() if weight_banks else [], phases=phases, iterations=options.iterations,
                  dispatch_seconds=dispatch_seconds,
                  transfers=[channel.transfers for channel in channels.values()])
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
