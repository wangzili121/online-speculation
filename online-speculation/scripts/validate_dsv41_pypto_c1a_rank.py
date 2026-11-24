"""Compare C1A backward implementations using real DSV4.1 draft operands."""

import argparse
import json
import sys
from pathlib import Path
import runpy
from statistics import median
from time import perf_counter

import torch

from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives
from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
from validate_dsv41_pypto_resident_draft import prepare_round


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-layers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--baseline-file", type=Path, action="append", default=[],
                        help="Previous resident-primitives source to compare under the same TP4 scheduler")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(23)
    sys.path.insert(0, str(args.lib.resolve()))
    ranks = tuple(range(4))
    topology = dict(dp=1, tp=4, ep=4)
    with native("prefill_resident").ResidentNetworkRuntime(ranks, topology) as runtime:
        draft = ResidentDraftAttention(args.model, args.head, runtime, topology, args.output)
        plan = draft.enable_online(args.last_layers)
        sources = {native("cache").resolve_layer_role(layer)["index_source"] for layer in plan.layers}
        anchor = 257
        table, _pinned, _, record, width, hidden, cos, sin = prepare_round(
            runtime, draft, ranks, (anchor,), 1,
        )
        owner = 0, 0, 1
        draft.arm_online_capture({owner: ("benchmark", (11,) * anchor)}, plan)
        round_state = draft.start_round(table)

        def project(rank):
            output = runtime.workspace(hidden.shape, hidden.dtype, rank)
            round_state.project_layer(rank, current_layer, record, hidden, cos, sin, output)

        for current_layer in sorted(sources.union(plan.layers)):
            runtime.map_ranks(project)
            runtime.release_values(round_state.finish_layer(current_layer))

        snapshots = {(layer, rank): round_state.captured_attention[rank, layer, owner]
                     for layer in plan.layers for rank in ranks}
        upstream = {rank: (torch.randn(32, width) * 0.1).bfloat16() for rank in ranks}
        for rank, value in upstream.items():
            runtime.pin(value, (rank,), readonly=True, name=f"test/upstream/{rank}")
        def make_reverse(cls):
            engine = cls.__new__(cls)
            engine.resident, engine.draft, engine.ranks = runtime, draft, ranks

            def reverse(layer, parallel):
                def run(rank):
                    return engine._reverse_c1a_rank(layer, rank, snapshots[layer, rank], upstream[rank])

                if parallel:
                    return engine._map_owner_ranks(ranks, run)
                return {rank: run(rank) for rank in ranks}

            def run_layers(*, parallel=True):
                return {layer: reverse(layer, parallel) for layer in reversed(plan.layers)}

            return run_layers

        parallel = make_reverse(DSV41ResidentPrimitives)
        if not args.baseline_file:
            def serial():
                return parallel(parallel=False)

            variants = {"serial": serial, "parallel": parallel}
        else:
            def load_variant(path):
                module = runpy.run_path(str(path), run_name="blockspec.c1a_vjp_baseline")
                return make_reverse(module["DSV41ResidentPrimitives"])
            variants = {f"baseline_{index + 1}": load_variant(path)
                        for index, path in enumerate(args.baseline_file)}
            variants["candidate"] = parallel

        def read(outputs):
            copies = {}
            for layer, rows in outputs.items():
                for rank, (boundary, grads) in rows.items():
                    copies[layer, rank] = {
                        name: runtime.read_output(value, rank).clone()
                        for name, value in dict(boundary=boundary, **grads).items()
                    }
            return copies

        # Compile and warm every rank before measuring. Keep compilation out
        # of both arms, but include rank scheduling and kernel runtime.
        first_dispatch = {}

        def check(name, function):
            prior = len(draft.programs)
            started = perf_counter()
            outputs = function()
            first_dispatch[name] = {
                "milliseconds": (perf_counter() - started) * 1e3,
                "programs_compiled": len(draft.programs) - prior,
            }
            result = read(outputs)
            runtime.release_values()
            return result

        comparisons = iter(variants.items())
        baseline = check(*next(comparisons))
        for rank, tensors in baseline.items():
            assert len(tensors) == 9
            for name, value in tensors.items():
                assert value.isfinite().all() and value.count_nonzero(), (rank, name)
        for name, function in comparisons:
            candidate = check(name, function)
            for rank, tensors in baseline.items():
                for name, value in tensors.items():
                    assert torch.equal(value, candidate[rank][name]), (rank, name)
        samples, fills, evictions = {}, {}, {}
        for name, function in variants.items():
            # Measure one active variant at a time. Alternating the two
            # overfills the hardware callable cache with duplicate kernels.
            function()
            runtime.release_values()
            first_evictions = runtime.stats.get("program_evictions", 0)
            samples[name], fills[name] = [], []
            for _ in range(8):
                prior = runtime.stats["workspace_fill_bytes"]
                start = perf_counter()
                function()
                samples[name].append((perf_counter() - start) * 1e3)
                fills[name].append(runtime.stats["workspace_fill_bytes"] - prior)
                runtime.release_values()
            evictions[name] = runtime.stats.get("program_evictions", 0) - first_evictions
        print(json.dumps({
            "passed": True, "ranks": ranks, "layers": plan.layers, "index_sources": sorted(sources),
            "anchor": anchor, "block": 32, "tensors_per_layer_rank": 9, "byte_exact": True,
            "first_dispatch": first_dispatch,
            "repeat_count": 8, "wall_ms": {name: median(rows) for name, rows in samples.items()},
            "workspace_fill_bytes": {name: median(rows) for name, rows in fills.items()},
            "steady_callable_evictions": evictions,
        }), flush=True)
        runtime.release_replay_values()


if __name__ == "__main__":
    main()
