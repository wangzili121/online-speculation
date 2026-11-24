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
        plan = draft.enable_online(1)
        layer, = plan.layers
        source = native("cache").resolve_layer_role(layer)["index_source"]
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

        for current_layer in sorted({source, layer}):
            runtime.map_ranks(project)
            runtime.release_values(round_state.finish_layer(current_layer))

        snapshots = {rank: round_state.captured_attention[rank, layer, owner] for rank in ranks}
        upstream = {rank: (torch.randn(32, width) * 0.1).bfloat16() for rank in ranks}
        for rank, value in upstream.items():
            runtime.pin(value, (rank,), readonly=True, name=f"test/upstream/{rank}")
        def make_reverse(cls):
            engine = cls.__new__(cls)
            engine.resident, engine.draft, engine.ranks = runtime, draft, ranks

            def reverse(rank):
                return engine._reverse_c1a_rank(layer, rank, snapshots[rank], upstream[rank])

            return engine, reverse

        engine, reverse = make_reverse(DSV41ResidentPrimitives)

        def parallel():
            return engine._map_owner_ranks(ranks, reverse)

        if not args.baseline_file:
            def serial():
                return {rank: reverse(rank) for rank in ranks}

            variants = {"serial": serial, "parallel": parallel}
        else:
            def load_variant(path):
                module = runpy.run_path(str(path), run_name="blockspec.c1a_vjp_baseline")
                implementation, reverse_rank = make_reverse(module["DSV41ResidentPrimitives"])

                def run():
                    return implementation._map_owner_ranks(ranks, reverse_rank)

                return run

            variants = {f"baseline_{index + 1}": load_variant(path)
                        for index, path in enumerate(args.baseline_file)}
            variants["candidate"] = parallel

        def read(outputs):
            copies = {}
            for rank, (boundary, grads) in outputs.items():
                copies[rank] = {"boundary": runtime.read_output(boundary, rank).clone()}
                copies[rank].update({name: runtime.read_output(value, rank).clone()
                                     for name, value in grads.items()})
            return copies

        # Compile and warm every rank before measuring. Keep compilation out
        # of both arms, but include rank scheduling and kernel runtime.
        def check(function):
            result = read(function())
            runtime.release_values()
            return result

        comparisons = iter(variants.values())
        baseline = check(next(comparisons))
        for rank, tensors in baseline.items():
            assert len(tensors) == 9
            for name, value in tensors.items():
                assert value.isfinite().all() and value.count_nonzero(), (rank, name)
        for function in comparisons:
            candidate = check(function)
            for rank, tensors in baseline.items():
                for name, value in tensors.items():
                    assert torch.equal(value, candidate[rank][name]), (rank, name)
        samples = {name: [] for name in variants}
        fills = {name: [] for name in variants}
        for _ in range(8):
            for name, function in variants.items():
                prior = runtime.stats["workspace_fill_bytes"]
                start = perf_counter()
                function()
                samples[name].append((perf_counter() - start) * 1e3)
                fills[name].append(runtime.stats["workspace_fill_bytes"] - prior)
                runtime.release_values()
        print(json.dumps({
            "passed": True, "ranks": ranks, "layer": layer, "index_source": source,
            "anchor": anchor, "block": 32, "tensors_per_rank": 9, "byte_exact": True,
            "repeat_count": 8, "wall_ms": {name: median(rows) for name, rows in samples.items()},
            "workspace_fill_bytes": {name: median(rows) for name, rows in fills.items()},
        }), flush=True)
        runtime.release_replay_values()


if __name__ == "__main__":
    main()
