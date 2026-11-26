#!/usr/bin/env python3
"""Verify HCCL can exchange 1122 PyPTO-owned MoE buffers directly on two local NPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import torch

from blockspec.dsv41_packed_ep_hccl import make_ep_forward_packets
from validate_dsv41_pypto_ep_hccl_packets import golden_buckets, golden_compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--transport-source", type=Path, required=True,
                        help="moe_transport.py from the previous e32726b model revision")
    options = parser.parse_args()
    devices = [int(device) for device in options.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        parser.error("Select two distinct logical devices")

    from pypto.backend import BackendType
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("orthrus_validation.paired_transport", options.transport_source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    ranks = (0, 1)
    tokens, width, world, local, topk = 16, 5120, len(devices), 24, 6
    n = tokens * topk
    packets = world * n
    rows = packets + local * 16
    rng = torch.Generator().manual_seed(781)
    source = [[torch.randint(-128, 128, (tokens, width), dtype=torch.int8, generator=rng),
               torch.rand(1, tokens, generator=rng), torch.randint(local * world, (tokens, topk), dtype=torch.int32, generator=rng),
               torch.rand(tokens, topk, generator=rng)] for _ in ranks]
    lives = (tokens, 5)
    expected_packets = [golden_buckets(*values, live, world, local) for values, live in zip(source, lives, strict=True)]
    outgoing = expected_packets[0]
    incoming = (torch.zeros(local, 2, dtype=torch.int32), torch.zeros(rows, width, dtype=torch.int8),
                torch.zeros(1, rows), torch.zeros(1, rows), torch.zeros(1, rows, dtype=torch.int32))
    expert = torch.zeros(rows, width, dtype=torch.bfloat16)
    returning = torch.zeros(packets, width, dtype=torch.bfloat16)
    shared = torch.randn(tokens, width, dtype=torch.bfloat16, generator=rng)
    restored = torch.zeros_like(shared)
    args = [(*source[0], *outgoing, tokens), (*outgoing, *incoming),
            (expert, incoming[-1], incoming[0], returning), (returning, source[0][2], shared, restored, tokens)]
    cfg = [RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=i) for i in devices]
    programs = [entry.compile(*operands, config=cfg[0])
                for entry, operands in zip(make_ep_forward_packets(tokens, width, world, local, topk), args, strict=True)]
    with ExitStack() as context:
        workers = [context.enter_context(ChipWorker(config)) for config in cfg]
        memory = native.CANNMemory()
        channels = native.HCCLBytes.create_group(memory, devices)
        for channel in channels.values():
            context.callback(channel.close)

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

        def exchange(send, recv, fields):
            def run(rank):
                transfers = list(zip(send[rank], recv[rank], fields, strict=True))
                channels[devices[rank]].exchange_device_batch(transfers)
            start = perf_counter()
            with ThreadPoolExecutor(max_workers=world) as pool:
                list(pool.map(run, ranks))
            return perf_counter() - start

        def transposed(values, dest):
            return [torch.cat([row[index].chunk(world)[dest] for row in values]) for index in range(len(values[0]))]

        src = [alloc(rank, source[rank], upload=True) for rank in ranks]
        send = [alloc(rank, outgoing) for rank in ranks]
        recv = [alloc(rank, outgoing) for rank in ranks]
        for rank in ranks:
            workers[rank].run(programs[0], *src[rank], *send[rank], lives[rank], config=cfg[rank])
        dispatch_seconds = exchange(send, recv, ("counts", "hidden", "auxiliary"))
        staged = [alloc(rank, incoming) for rank in ranks]
        outgoing_return = [alloc(rank, (returning,)) for rank in ranks]
        incoming_return = [alloc(rank, (returning,)) for rank in ranks]
        expected_return = []
        for rank in ranks:
            for value, target in zip(read(rank, recv[rank]), transposed(expected_packets, rank), strict=True):
                torch.testing.assert_close(value, target, atol=0, rtol=0)
            workers[rank].run(programs[1], *recv[rank], *staged[rank], config=cfg[rank])
            used, expected = golden_compact(*transposed(expected_packets, rank), n, local, rows)
            actual = read(rank, staged[rank])
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
            torch.testing.assert_close(actual[1][:used], expected[1][:used], atol=0, rtol=0)
            for actual_row, expected_row in zip(actual[2:], expected[2:], strict=True):
                torch.testing.assert_close(actual_row[:, :used], expected_row[:, :used], atol=0, rtol=0)
            expert_result = torch.zeros_like(expert)
            expert_result[:used].copy_(expected[1][:used].float() * expected[2][0, :used, None])
            buffer, = alloc(rank, (expert_result,), upload=True)
            workers[rank].run(programs[2], buffer, staged[rank][-1], staged[rank][0], *outgoing_return[rank], config=cfg[rank])
            expected_value = torch.zeros_like(returning)
            positions = expected[-1][0, :used].long()
            valid = positions >= 0
            expected_value[positions[valid]] = expert_result[:used][valid]
            expected_return.append((expected_value,))
        return_seconds = exchange(outgoing_return, incoming_return, ("expert_return",))
        for rank in ranks:
            target, = transposed(expected_return, rank)
            shared_buf, = alloc(rank, (shared,), upload=True)
            out = alloc(rank, (restored,))
            workers[rank].run(programs[3], *incoming_return[rank], src[rank][2], shared_buf, *out, lives[rank], config=cfg[rank])
            expected = shared[:lives[rank]].float()
            for choice in range(topk):
                owner = source[rank][2][:lives[rank], choice].long() // local
                index = owner * n + torch.arange(lives[rank]) * topk + choice
                expected.add_(target[index].float())
            observed, = read(rank, out)
            torch.testing.assert_close(observed[:lives[rank]], expected.bfloat16(), atol=0, rtol=0)
        report = dict(passed=True, world=world, tokens=tokens, exact_output=True,
                      dispatch_seconds=dispatch_seconds, return_seconds=return_seconds,
                      transfers=[channel.transfers for channel in channels.values()])
        print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
