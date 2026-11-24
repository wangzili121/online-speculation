#!/usr/bin/env python3
"""Measure native TP reduction padding at 16 and 128 execution rows."""

import argparse
import json
from pathlib import Path
from statistics import median
from threading import RLock
from time import perf_counter

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.output.exists():
        parser.error("provide a positive iteration count and a new output path")

    from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
    from models.deepseek_v4_1_flash_a3.input_pack import (
        make_sp_reduce_narrow, make_sp_reduce_pack, make_sp_reduce_widen,
    )
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(41129)
    ranks, hidden = tuple(range(args.tp)), 5120
    contributions = [torch.randn((16, hidden), dtype=torch.bfloat16, generator=rng) for _ in ranks]
    summed = torch.stack(contributions).sum(dim=0, dtype=torch.float32).bfloat16()
    programs, reports = {}, []
    with ResidentNetworkRuntime(ranks, dict(ep=args.tp, dp=1, tp=args.tp)) as runtime:
        attention = ResidentDraftAttention.__new__(ResidentDraftAttention)
        attention.resident, attention.rank_ids = runtime, {rank: rank for rank in ranks}
        attention.compile_lock, attention.programs = RLock(), programs
        attention.native = native
        attention.output = args.output.parent / (args.output.stem + "-compiled")
        attention.output.mkdir(parents=True, exist_ok=False)
        inputs = {}
        for capacity in (16, 128):
            inputs[capacity] = [torch.zeros((capacity, hidden), dtype=torch.bfloat16) for _ in ranks]
            for rank, operand in enumerate(inputs[capacity]):
                operand[:16].copy_(contributions[rank])
                runtime.pin(operand, (rank,), readonly=True, name=f"fixture/{capacity}/{rank}")

        for live in (1, 8, 16):
            physical = (live + args.tp - 1) // args.tp

            def call(capacity, check=False):
                started = perf_counter()

                def pack_rank(rank):
                    packed_shape = (args.tp * capacity, hidden)
                    packed = runtime.workspace(packed_shape, torch.bfloat16, rank)
                    widened = runtime.workspace(packed_shape, torch.float32, rank)
                    attention.execute(
                        ("pack", capacity), rank,
                        lambda: make_sp_reduce_pack(capacity, hidden, args.tp),
                        dict(value=inputs[capacity][rank], packed=packed, active=live, physical=physical),
                    )
                    attention.execute(
                        ("widen", capacity), rank,
                        lambda: make_sp_reduce_widen(*packed_shape),
                        dict(value=packed, output=widened),
                    )
                    return {"wo-b": {"value": widened}}

                packed = runtime.map_ranks(pack_rank)
                result, _ = runtime.exchange("wo-b-fp32", packed, "reduce_scatter", "tp")

                def finish_rank(rank):
                    output = runtime.workspace((capacity, hidden), torch.bfloat16, rank)
                    attention.execute(
                        ("narrow", capacity), rank, lambda: make_sp_reduce_narrow(capacity, hidden),
                        dict(value=result[rank]["wo-b"]["value"], output=output),
                    )
                    return output

                outputs = runtime.map_ranks(finish_rank)
                elapsed = (perf_counter() - started) * 1e3
                if check:
                    for rank, output in enumerate(outputs):
                        count = min(physical, max(0, live - rank * physical))
                        expected = torch.zeros((capacity, hidden), dtype=torch.bfloat16)
                        expected[:count].copy_(summed[rank * physical:rank * physical + count])
                        assert torch.equal(runtime.read_output(output, rank), expected)
                runtime.release_values()
                return elapsed

            timing = {16: [], 128: []}
            for capacity in timing:
                call(capacity, check=True)
            for _ in range(args.iterations):
                for capacity in (128, 16):
                    timing[capacity].append(call(capacity))
            reports.append(dict(
                active_rows=live, iterations=args.iterations, exact=True,
                wide_ms=median(timing[128]), narrow_ms=median(timing[16]),
                wide_send_bytes_per_rank=args.tp * 128 * hidden * 4,
                narrow_send_bytes_per_rank=args.tp * 16 * hidden * 4,
            ))

    report = dict(passed=True, phase="native-tp-pack-widen-reduce-scatter-narrow",
                  capacity_rows=[16, 128], physical_devices=args.tp, tp=args.tp, hidden=hidden, cases=reports)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
