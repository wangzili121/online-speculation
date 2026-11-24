#!/usr/bin/env python3
"""Compare native TP reduction geometries or fused draft packet construction."""

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
    parser.add_argument("--draft-pack", action="store_true", help="Compare the fused draft pack at 128 rows")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.output.exists():
        parser.error("provide a positive iteration count and a new output path")

    from blockspec.pypto_dsv41_runtime import DraftRound, ResidentDraftAttention, native
    from models.deepseek_v4_1_flash_a3.input_pack import (
        make_sp_reduce_narrow, make_sp_reduce_pack, make_sp_reduce_widen,
    )
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(41129)
    ranks, hidden = tuple(range(args.tp)), 5120
    active_rows = (32, 64, 96, 128) if args.draft_pack else (1, 8, 16)
    variants = ((128, False), (128, True)) if args.draft_pack else ((128, False), (16, False))
    contributions = [torch.randn((max(active_rows), hidden), dtype=torch.bfloat16, generator=rng) for _ in ranks]
    summed = torch.stack(contributions).sum(dim=0, dtype=torch.float32).bfloat16()
    programs, reports = {}, []
    with ResidentNetworkRuntime(ranks, dict(ep=args.tp, dp=1, tp=args.tp)) as runtime:
        attention = ResidentDraftAttention.__new__(ResidentDraftAttention)
        attention.resident, attention.rank_ids = runtime, {rank: rank for rank in ranks}
        attention.compile_lock, attention.programs = RLock(), programs
        attention.native = native
        attention.topology = {"tp": args.tp}
        attention.output = args.output.parent / (args.output.stem + "-compiled")
        attention.output.mkdir(parents=True, exist_ok=False)
        draft = DraftRound.__new__(DraftRound)
        draft.attention = attention
        inputs = {}
        for capacity in sorted({capacity for capacity, _ in variants}):
            inputs[capacity] = [torch.zeros((capacity, hidden), dtype=torch.bfloat16) for _ in ranks]
            for rank, operand in enumerate(inputs[capacity]):
                operand[:max(active_rows)].copy_(contributions[rank])
                runtime.pin(operand, (rank,), readonly=True, name=f"fixture/{capacity}/{rank}")

        for live in active_rows:
            physical = (live + args.tp - 1) // args.tp

            def call(capacity, fused, check=False):
                started = perf_counter()

                def pack_rank(rank):
                    if fused:
                        widened = draft.pack_fp32(rank, inputs[capacity][rank], physical, live)
                        return {"wo-b": {"value": widened}}
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

            timing = {variant: [] for variant in variants}
            for variant in variants:
                call(*variant, check=True)
            for _ in range(args.iterations):
                for variant in variants:
                    timing[variant].append(call(*variant))
            before, after = (median(timing[variant]) for variant in variants)
            case = dict(active_rows=live, iterations=args.iterations, exact=True)
            if args.draft_pack:
                case.update(before_ms=before, after_ms=after,
                            saved_bf16_staging_bytes_per_rank=args.tp * 128 * hidden * 2)
            else:
                case.update(wide_ms=before, narrow_ms=after,
                            wide_send_bytes_per_rank=args.tp * 128 * hidden * 4,
                            narrow_send_bytes_per_rank=args.tp * 16 * hidden * 4)
            reports.append(case)

    phase = ("orthrus-draft-fp32-pack-with-tp-reduce-scatter" if args.draft_pack
             else "native-tp-pack-widen-reduce-scatter-narrow")
    report = dict(passed=True, phase=phase, capacity_rows=list(inputs),
                  physical_devices=args.tp, tp=args.tp, hidden=hidden, cases=reports)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
