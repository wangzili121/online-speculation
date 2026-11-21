#!/usr/bin/env python3
"""Validate the native resident transport and cache rollback on one Ascend host."""

import argparse
import json
from pathlib import Path
import sys

import torch


def check_collectives(runtime, world, tp):
    # Exactly representable in BF16 across all supported test groups.
    rows = (torch.arange(world * 16).reshape(-1, 1) % 13).expand(-1, 16)
    payloads = {
        "values": [(rows + rank).bfloat16().contiguous() for rank in range(world)],
        "tags": [(rows + rank * world).to(torch.uint8).contiguous() for rank in range(world)],
    }
    tests = (("all_gather", "tp"), ("all_reduce", "tp"), ("reduce_scatter", "tp"),
             ("all_to_all", "ep"), ("all_gather", "dp"))
    passed = []
    for collective, group in tests:
        name = "tags" if collective == "all_to_all" else "values"
        values = payloads[name]
        packets = [{"draft": {name: value}} for value in values]
        result, receipts = runtime.exchange(f"validation/{group}/{collective}", packets, collective, group)
        for rank, receipt in enumerate(receipts):
            first = rank // tp * tp
            members = {
                "ep": list(range(world)),
                "tp": list(range(first, first + tp)),
                "dp": list(range(rank % tp, world, tp)),
            }[group]
            assert receipt["group_ranks"] == members
            peers = [values[peer] for peer in members]
            index, count = members.index(rank), len(members)
            if collective == "all_gather":
                expected = torch.cat(peers)
            elif collective == "all_to_all":
                expected = torch.cat([peer.chunk(count)[index] for peer in peers])
            else:
                expected = torch.stack(peers).sum(dim=0, dtype=torch.float32).bfloat16()
                if collective == "reduce_scatter":
                    expected = expected.chunk(count)[index]
            torch.testing.assert_close(runtime.read_output(result[rank]["draft"][name], rank), expected,
                                       atol=0, rtol=0)
            assert not receipt["activation_staging"]
        runtime.release_values()
        passed.append(f"{group}/{collective}")
    return passed


def check_rollback(runtime, world):
    class RejectedBlock(Exception):
        pass

    # Cross a physical page boundary and leave other bytes intact.
    intervals = ((1024 - 64, 128), (2048 + 32, 64))
    caches = [((torch.arange(3072) + rank) % 197).to(torch.uint8) for rank in range(world)]
    for rank, cache in enumerate(caches):
        runtime.pin(cache, (rank,), readonly=False, name=f"raw/target/rank{rank}")
    def restore(rank):
        cache = caches[rank]
        address = runtime.handle(cache, rank).data_ptr
        for reject in (False, True):
            try:
                with runtime.preserve_cache_ranges(rank, [(cache, intervals)]):
                    changed = cache.clone()
                    for offset, count in intervals:
                        runtime._device_fill(address + offset, count, 255, rank)
                        changed[offset:offset + count] = 255
                    assert torch.equal(runtime.read_output(cache, rank), changed)
                    if reject:
                        raise RejectedBlock()
            except RejectedBlock:
                assert reject
            assert torch.equal(runtime.read_output(cache, rank), cache)

    runtime.map_ranks(restore)
    return 2 * world * sum(count for _, count in intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--cards", type=int, choices=(2, 3, 4, 8), required=True)
    parser.add_argument("--tp", type=int, help="Default to one TP group over the selected cards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tp = args.cards if args.tp is None else args.tp
    if tp < 1 or args.cards % tp or args.output.exists():
        parser.error("TP must divide the number of cards and the output file must be new")
    sys.path.insert(0, str(args.lib.resolve()))
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    torch.set_num_threads(2)
    topology = dict(ep=args.cards, dp=args.cards // tp, tp=tp)
    with ResidentNetworkRuntime(list(range(args.cards)), topology) as runtime:
        collectives = check_collectives(runtime, args.cards, tp)
        restored_bytes = check_rollback(runtime, args.cards)
        assert runtime.stats["verify_backup_d2d_bytes"] == runtime.stats["verify_restore_d2d_bytes"] == restored_bytes
    report = dict(passed=True, topology=topology, collectives=collectives, cache_rollback_byte_exact=True,
                  restored_bytes=restored_bytes)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
