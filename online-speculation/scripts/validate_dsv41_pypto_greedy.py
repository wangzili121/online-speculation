#!/usr/bin/env python3
"""Compare the resident Orthrus device vocabulary reduction against Torch."""

import argparse
import json
from pathlib import Path
from statistics import median
from threading import RLock
from time import perf_counter

import torch

from blockspec.pypto_dsv41_greedy import PACKET_COLUMNS, make_local_greedy, merge_greedy_shards
from blockspec.pypto_dsv41_runtime import ResidentDraftAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tile", type=int, default=6464)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    from models.deepseek_v4_1_flash_a3 import network
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime
    from pypto.backend import BackendType
    from pypto.runtime import ExecutionMode, RunConfig

    torch.set_num_threads(2)
    torch.manual_seed(4)
    capacity, vocab, tp = 128, 129280, 4
    shard = vocab // tp
    logits = torch.randn((capacity, vocab), dtype=torch.bfloat16)
    logits[0].fill_(-5)
    logits[0, [123, 19000, shard + 50]] = 9
    logits[1].fill_(-25)
    logits[2, shard * 3 + shard - 1] = 17
    shards = [part.contiguous() for part in logits.split(shard, dim=-1)]
    invalid = shards[0].clone()
    invalid[0, 2] = float("nan")
    invalid[1, 97] = float("inf")
    invalid[2, 108] = float("-inf")
    expected = logits.argmax(dim=-1)
    mode = ExecutionMode.SIM if args.simulate else ExecutionMode.ONBOARD
    config = RunConfig(arch=BackendType.Ascend910, execution_mode=mode, device_id=args.device)
    if args.simulate:
        entry = make_local_greedy(capacity, shard, tile=args.tile)
        packets = [torch.empty((capacity, PACKET_COLUMNS), dtype=torch.float32) for _ in shards]
        for count in (32, 64, 96, 128):
            for source, packed in zip(shards, packets, strict=True):
                entry(source, packed, count, config=config)
            assert torch.equal(merge_greedy_shards([packed[:count] for packed in packets], shard), expected[:count])
        entry(invalid, packets[0], 3, config=config)
        assert packets[0][:3, 2].eq(0).all()
        report = dict(passed=True, simulator=True, tile=args.tile)
        args.output.write_text(json.dumps(report) + "\n")
        print(json.dumps(report))
        return
    results = {}
    with ResidentNetworkRuntime([args.device], dict(ep=1, dp=1, tp=1)) as runtime:
        for index, source in enumerate([*shards, invalid]):
            runtime.pin(source, [0], readonly=True, name=f"greedy-test/{index}")
        attention = ResidentDraftAttention.__new__(ResidentDraftAttention)
        attention.resident, attention.rank_ids = runtime, {0: 0}
        attention.compile_lock, attention.programs = RLock(), {}
        attention.output = args.output.parent / f"{args.output.stem}-program"
        attention.output.mkdir(exist_ok=False)
        attention.native = lambda name: network
        for count in (1, 7, 32, 64, 96, 128):
            for source in shards:
                attention.greedy_shard(0, source, count)
                runtime.release_values()
            samples = {"copy_and_argmax_ms": [], "device_packet_ms": []}
            for _ in range(args.repeats):
                old, new = [], []
                for source in shards:
                    t = perf_counter()
                    full = runtime.read_output(source[:count], 0)
                    old.append(full.argmax(-1))
                    samples["copy_and_argmax_ms"].append((perf_counter() - t) * 1000)
                    t = perf_counter()
                    new.append(attention.greedy_shard(0, source, count))
                    samples["device_packet_ms"].append((perf_counter() - t) * 1000)
                    assert torch.equal(old[-1], new[-1][:, 1].long())
                    assert torch.equal(full.max(-1).values.float(), new[-1][:, 0])
                    runtime.release_values()
                assert torch.equal(merge_greedy_shards(new, shard), expected[:count])
            results[count] = {key: median(values) for key, values in samples.items()}
            results[count]["bytes_per_shard"] = dict(full=count * shard * 2, compact=count * PACKET_COLUMNS * 4)
        status = attention.greedy_shard(0, invalid, 3)[:, 2]
        assert status.eq(0).all(), status
    report = dict(passed=True, active_rows=results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
