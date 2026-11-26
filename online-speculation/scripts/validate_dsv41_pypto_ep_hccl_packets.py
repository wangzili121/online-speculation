#!/usr/bin/env python3
"""Check the W8A8 EP AllToAll packet layout and native expert-row order."""

import argparse
import json
from time import perf_counter

import torch

from blockspec.dsv41_packed_ep_hccl import make_ep_forward_packets


def golden_buckets(x, scale, ids, weight, live, world, local):
    tokens, hidden = x.shape
    topk = ids.shape[1]
    routes = tokens * topk
    count_width = (local + 15) // 16 * 16
    counts = torch.zeros((world, count_width), dtype=torch.int32)
    payload = torch.zeros((world, routes, hidden), dtype=torch.int8)
    auxiliary = torch.zeros((world, routes, 8))
    for peer in range(world):
        cursor = 0
        for expert in range(local):
            locations = (ids[:live] == peer * local + expert).nonzero()
            size = len(locations)
            counts[peer, expert] = size
            token, slot = locations.unbind(dim=1)
            payload[peer, cursor:cursor + size] = x[token]
            auxiliary[peer, cursor:cursor + size, 0] = weight[token, slot]
            auxiliary[peer, cursor:cursor + size, 1] = scale[0, token]
            auxiliary[peer, cursor:cursor + size, 2] = token * topk + slot
            cursor += size
    return counts, payload.flatten(0, 1), auxiliary.flatten(0, 1)


def golden_compact(counts, payload, auxiliary, source_routes, local, rows):
    world = len(counts)
    offsets = torch.empty((local, 2), dtype=torch.int32)
    x = torch.zeros((rows, payload.shape[1]), dtype=torch.int8)
    scale = torch.zeros((1, rows))
    weights = torch.zeros((1, rows))
    address = torch.full((1, rows), -1, dtype=torch.int32)
    cursors = [source * source_routes for source in range(world)]
    start = 0
    for expert in range(local):
        position = start
        for source in range(world):
            size = int(counts[source, expert])
            first = cursors[source]
            selected = slice(first, first + size)
            incoming = slice(position, position + size)
            x[incoming] = payload[selected]
            weights[0, incoming] = auxiliary[selected, 0]
            scale[0, incoming] = auxiliary[selected, 1]
            address[0, incoming] = source * source_routes + auxiliary[selected, 2].to(torch.int32)
            position += size
            cursors[source] += size
        used = position - start
        offsets[expert] = torch.tensor((used, start))
        start += (used + 15) // 16 * 16
    return start, (offsets, x, scale, weights, address)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--world", type=int, choices=(2, 4, 8, 16), default=16)
    parser.add_argument("--tokens", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    from pypto.backend import BackendType
    from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
    from simpler_setup.log_config import configure_logging

    configure_logging("warn")
    torch.set_num_threads(4)
    world, tokens, hidden, local, topk = args.world, args.tokens, 5120, 24, 6
    source_routes = tokens * topk
    packets = world * source_routes
    rows = packets + local * 16
    rng = torch.Generator().manual_seed(315)
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=args.device)
    pack, unpack, pack_return, combine = make_ep_forward_packets(tokens, hidden, world, local, topk)
    source = (
        torch.randint(-128, 128, (tokens, hidden), dtype=torch.int8, generator=rng),
        torch.rand((1, tokens), generator=rng),
        torch.randint(world * local, (tokens, topk), dtype=torch.int32, generator=rng),
        torch.rand((tokens, topk), generator=rng),
    )
    outgoing = (torch.empty((world, 32), dtype=torch.int32), torch.empty((packets, hidden), dtype=torch.int8),
                torch.empty((packets, 8), dtype=torch.float32))
    incoming = (torch.empty((local, 2), dtype=torch.int32), torch.empty((rows, hidden), dtype=torch.int8),
                torch.empty((1, rows)), torch.empty((1, rows)), torch.empty((1, rows), dtype=torch.int32))
    expert_output = torch.empty((rows, hidden), dtype=torch.bfloat16)
    returned = torch.empty((packets, hidden), dtype=torch.bfloat16)
    shared = torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=rng)
    output = torch.empty_like(shared)
    pack = pack.compile(*source, *outgoing, tokens, config=config)
    unpack = unpack.compile(*outgoing, *incoming, config=config)
    pack_return = pack_return.compile(expert_output, incoming[-1], incoming[0], returned, config=config)
    combine = combine.compile(returned, source[2], shared, output, tokens, config=config)
    report = dict(passed=True, compiled=4, executed=False, world=world, tokens=tokens, width=hidden)
    if args.compile_only:
        print(json.dumps(report), flush=True)
        return

    with ChipWorker(config) as worker:
        def allocate(tensors):
            return tuple(worker.alloc_tensor(value.shape, value.dtype) for value in tensors)

        def move(tensors, handles, *, read):
            for value, device in zip(tensors, handles, strict=True):
                if read:
                    worker.copy_from(value.data_ptr(), device.data_ptr, value.nbytes)
                else:
                    worker.copy_to(device.data_ptr, value.data_ptr(), value.nbytes)

        source_buffers, packet_buffers, expert_buffers = map(allocate, (source, outgoing, incoming))
        result_buffers = allocate((expert_output, returned, shared, output))
        expert_result_buffer, return_buffer, shared_buffer, output_buffer = result_buffers
        move((shared,), (shared_buffer,), read=False)
        sent = []
        timings = []
        rank_inputs = []
        for rank in range(world):
            active = (0, tokens, tokens // 2, 1)[rank % 4]
            source[2].random_(world * local, generator=rng)
            if rank == 1:
                source[2][:, 0] = (world - 1) * local
            move(source, source_buffers, read=False)
            start = perf_counter()
            worker.run(pack, *source_buffers, *packet_buffers, active, config=config)
            timings.append(perf_counter() - start)
            move(outgoing, packet_buffers, read=True)
            expected = golden_buckets(*source, active, world, local)
            for observed, target in zip(outgoing, expected, strict=True):
                torch.testing.assert_close(observed, target, atol=0, rtol=0)
            sent.append(tuple(value.reshape(world, -1, value.shape[-1]).clone() for value in outgoing))
            rank_inputs.append((active, source[2].clone()))
        unpack_times = []
        returned_by_peer = []
        return_pack_times = []
        for peer in range(world):
            # HcclAlltoAll concatenates destination slices in ascending source-rank order.
            received = tuple(torch.cat([packet[index][peer] for packet in sent], dim=0) for index in range(3))
            move(received, packet_buffers, read=False)
            start = perf_counter()
            worker.run(unpack, *packet_buffers, *expert_buffers, config=config)
            unpack_times.append(perf_counter() - start)
            move(incoming, expert_buffers, read=True)
            used, expected = golden_compact(*received, source_routes, local, rows)
            valid = (incoming[0], incoming[1][:used], *(item[:, :used] for item in incoming[2:]))
            target = (expected[0], expected[1][:used], *(item[:, :used] for item in expected[2:]))
            for observed, result in zip(valid, target, strict=True):
                torch.testing.assert_close(observed, result, atol=0, rtol=0)

            # Give each valid expert row a reproducible BF16 output. Both sides of
            # this test leave the native quantized expert implementation unchanged.
            expert_output.zero_()
            expert_output[:used].copy_(
                (incoming[1][:used].float() + peer * .125) * incoming[2][0, :used, None] * incoming[3][0, :used, None]
            )
            move((expert_output,), (expert_result_buffer,), read=False)
            start = perf_counter()
            worker.run(pack_return, expert_result_buffer, expert_buffers[-1], expert_buffers[0],
                       return_buffer, config=config)
            return_pack_times.append(perf_counter() - start)
            move((returned,), (return_buffer,), read=True)
            expected_return = torch.zeros_like(returned)
            addresses = expected[-1][0, :used].long()
            valid = addresses >= 0
            expected_return[addresses[valid]] = expert_output[:used][valid]
            torch.testing.assert_close(returned, expected_return, atol=0, rtol=0)
            returned_by_peer.append(returned.view(world, source_routes, hidden).clone())

        combine_times = []
        for rank, (active, ids) in enumerate(rank_inputs):
            returned.copy_(torch.cat([value[rank] for value in returned_by_peer]))
            move((returned, ids), (return_buffer, source_buffers[2]), read=False)
            start = perf_counter()
            worker.run(combine, return_buffer, source_buffers[2], shared_buffer, output_buffer, active, config=config)
            combine_times.append(perf_counter() - start)
            move((output,), (output_buffer,), read=True)
            expected = shared[:active].float()
            for choice in range(topk):
                addresses = (ids[:active, choice] // local) * source_routes + torch.arange(active) * topk + choice
                expected.add_(returned[addresses].float())
            torch.testing.assert_close(output[:active], expected.bfloat16(), atol=0, rtol=0)
    report.update(executed=True, exact_source_packets=world, exact_destination_slabs=world,
                  exact_return_packets=world, exact_restored_ranks=world,
                  transfer="CPU emulation of AllToAll rank ordering", pack_seconds=timings, unpack_seconds=unpack_times,
                  return_pack_seconds=return_pack_times, combine_seconds=combine_times)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
