#!/usr/bin/env python3
"""Compare empty EP16 packet construction on real Ascend ranks."""

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
    parser.add_argument("--cards", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.iterations < 1:
        parser.error("provide a fresh report path and a positive iteration count")

    from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives
    from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    spec = importlib.util.spec_from_file_location("blockspec.ep_empty_baseline", args.baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)

    ranks = tuple(range(args.cards))
    torch.set_num_threads(2)
    results = {}
    with ResidentNetworkRuntime(ranks, dict(dp=args.cards, tp=1, ep=args.cards)) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids = runtime, {rank: rank for rank in ranks}
        draft.compile_lock, draft.programs, draft.native = RLock(), {}, native
        draft.output = args.output.parent / (args.output.stem + "-compiled")
        draft.output.mkdir(parents=True, exist_ok=False)

        def no_expert(*_args):
            raise AssertionError("empty incoming packets executed an expert")

        arms = {}
        for label, cls in (("before", baseline.DSV41ResidentPrimitives), ("after", DSV41ResidentPrimitives)):
            engine = cls.__new__(cls)
            engine.resident, engine.draft, engine.ranks = runtime, draft, ranks
            engine.native = SimpleNamespace(weights=SimpleNamespace(for_layer=lambda _: {"expert": no_expert}))
            arms[label] = engine

        for bucket in (32, 192):
            samples = {key: ([], []) for key in arms}

            def run(engine, check=False):
                started = perf_counter()
                sent, _ = engine._ep_dispatch_packets({}, bucket)
                dispatch_ms = (perf_counter() - started) * 1e3
                started = perf_counter()
                returned = engine._ep_owner_expert_packets(39, sent, bucket)
                return_ms = (perf_counter() - started) * 1e3
                if check:
                    for rank, (before_return, after_return) in enumerate(zip(sent, returned, strict=True)):
                        for tensor in (*before_return["frozen-tail"].values(), after_return["frozen-tail"]["dx"]):
                            assert not runtime.read_output(tensor, rank).view(torch.uint8).any()
                runtime.release_values()
                return dispatch_ms, return_ms

            for engine in arms.values():
                run(engine, check=True)
            for _ in range(args.iterations):
                for label, engine in arms.items():
                    dispatch, returned = run(engine)
                    samples[label][0].append(dispatch)
                    samples[label][1].append(returned)
            results[bucket] = {label: {"dispatch_ms": median(dispatch), "expert_side_ms": median(returned)}
                               for label, (dispatch, returned) in samples.items()}

    report = dict(passed=True, phase="ep16-empty-packet", physical_cards=args.cards, exact_all_zero=True,
                  iterations=args.iterations, scope="local pack and expert-side return without EP fabric",
                  buckets=results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
