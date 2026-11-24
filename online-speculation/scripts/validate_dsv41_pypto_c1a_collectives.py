#!/usr/bin/env python3
"""Compare the TP4 communication stages of C1A replay on four physical NPUs."""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from threading import RLock
from time import perf_counter
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.output.exists():
        parser.error("provide a positive iteration count and a new report path")

    from blockspec.dsv41_serving_online import DSV41SequenceSpan
    from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives
    from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    spec = importlib.util.spec_from_file_location("blockspec.c1a_transport_baseline", args.baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(41130)
    ranks = tuple(range(4))
    block, capacity, hidden = 32, 128, 5120
    reports = []
    with ResidentNetworkRuntime(ranks, dict(ep=4, dp=1, tp=4)) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids = runtime, {rank: rank for rank in ranks}
        draft.compile_lock, draft.programs = RLock(), {}
        draft.native = native
        draft.output = args.output.parent / (args.output.stem + "-compiled")
        draft.output.mkdir(parents=True, exist_ok=False)
        with_stages = {"baseline": baseline.DSV41ResidentPrimitives, "compact": DSV41ResidentPrimitives}
        scatter = {rank: torch.randn((block, hidden), generator=rng).to(torch.bfloat16) for rank in ranks}
        contributions = {rank: torch.randn((block, hidden), generator=rng).to(torch.bfloat16) for rank in ranks}
        for name, values in (("scatter", scatter), ("contribution", contributions)):
            for rank, value in values.items():
                runtime.pin(value, (rank,), readonly=True, name=f"fixture/{name}/{rank}")

        def primitive(cls):
            value = cls.__new__(cls)
            value.resident, value.draft, value.native, value.ranks = runtime, draft, SimpleNamespace(), ranks
            # Replace the compute on either side of the communication with
            # resident BF16 fixtures. The two collectives and their layout kernels
            # are the actual production methods and run on the four NPUs.
            value._reverse_attention_post = lambda _layer, rank, _capture, _dy: {"value": scatter[rank]}
            value._reverse_c1a_rank = lambda _layer, rank, _capture, grad: (contributions[rank], grad)
            value._reverse_attention_pre = lambda _layer, _rank, _cap, _shell, grad: grad
            return value

        arms = {name: primitive(cls) for name, cls in with_stages.items()}
        for active, begin in ((32, 0), (64, 32), (96, 32), (128, 96)):
            physical = active // len(ranks)
            spans = {rank: DSV41SequenceSpan.from_record(
                {"cp_token_range": (rank * physical, (rank + 1) * physical), "SP_valid_rows": physical},
                begin, begin + block,
            ) for rank in ranks}
            captured = {
                rank: dict(token_begin=begin, token_end=begin + block, sp_span=span,
                           attention_scattered=torch.empty((capacity, hidden), dtype=torch.bfloat16),
                           record=dict(SP_physical_rows=physical, positions=range(active)))
                for rank, span in spans.items()
            }
            gathered = torch.cat([scatter[rank][:spans[rank].rows] for rank in ranks])
            summed = torch.stack(list(contributions.values())).sum(dim=0, dtype=torch.float32).bfloat16()
            expected = {}
            for rank, span in spans.items():
                expected[rank] = torch.zeros((block, hidden), dtype=torch.bfloat16)
                expected[rank][:span.rows] = summed[span.block_start:span.block_start + span.rows]

            def run(arm, check=False):
                backend = arms[arm]
                original = backend._tp_collective
                bytes_sent = {}

                def record(values, operation, *, stage):
                    bytes_sent[operation] = next(iter(values.values())).nbytes
                    return original(values, operation, stage=stage)

                if check:
                    backend._tp_collective = record
                start = perf_counter()
                try:
                    outputs = backend.reverse_draft_attention_owner_batch(
                        layer=39, ranks=ranks, captured=captured, output_gradients=scatter,
                    )
                finally:
                    if check:
                        del backend._tp_collective
                elapsed = (perf_counter() - start) * 1e3
                if check:
                    for rank, (received, unpacked) in outputs.items():
                        assert torch.equal(runtime.read_output(received, rank), expected[rank])
                        assert torch.equal(runtime.read_output(unpacked, rank), gathered)
                runtime.release_values()
                return elapsed, bytes_sent

            traffic = {name: run(name, check=True)[1] for name in arms}
            samples = {name: [] for name in arms}
            for _ in range(args.iterations):
                for name in arms:
                    samples[name].append(run(name)[0])
            reports.append(dict(active_rows=active, request_first_row=begin, exact=True, iterations=args.iterations,
                                median_ms={name: median(values) for name, values in samples.items()},
                                send_bytes_per_rank=traffic))

    report = dict(passed=True, tp=4, capacity=capacity, request_rows=block, hidden=hidden,
                  scope="resident TP4 replay communication and layout kernels", cases=reports)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
