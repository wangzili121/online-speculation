"""Compare serial and parallel C1A reverse using real DSV4.1 draft operands."""

import argparse
import json
import sys
from pathlib import Path
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
        engine = DSV41ResidentPrimitives.__new__(DSV41ResidentPrimitives)
        engine.resident, engine.draft, engine.ranks = runtime, draft, ranks

        def reverse(rank):
            return engine._reverse_c1a_rank(
                layer, rank, snapshots[rank], upstream[rank],
            )

        def read(outputs):
            copies = {}
            for rank, (boundary, grads) in outputs.items():
                copies[rank] = {"boundary": runtime.read_output(boundary, rank).clone()}
                copies[rank].update({name: runtime.read_output(value, rank).clone()
                                     for name, value in grads.items()})
            return copies

        # Compile and warm every rank before measuring. Keep compilation out
        # of both arms, but include rank scheduling and kernel runtime.
        def serial():
            return {rank: reverse(rank) for rank in ranks}

        def parallel():
            return engine._map_owner_ranks(ranks, reverse)

        baseline = read(serial())
        runtime.release_values()
        candidate = read(parallel())
        runtime.release_values()
        nonzero = {}
        for rank in ranks:
            assert len(baseline[rank]) == 9
            nonzero[rank] = []
            for name, before in baseline[rank].items():
                assert torch.isfinite(before).all(), (rank, name)
                assert torch.equal(before, candidate[rank][name]), (rank, name)
                if torch.count_nonzero(before):
                    nonzero[rank].append(name)
            assert len(nonzero[rank]) == 9, (rank, nonzero[rank])
        samples = {"serial": [], "parallel": []}
        for _ in range(8):
            for name, function in (("serial", serial), ("parallel", parallel)):
                start = perf_counter()
                function()
                samples[name].append((perf_counter() - start) * 1e3)
                runtime.release_values()
        print(json.dumps({
            "passed": True, "ranks": ranks, "layer": layer, "index_source": source,
            "anchor": anchor, "block": 32, "tensors_per_rank": 9, "byte_exact": True,
            "repeat_count": 8, "wall_ms": {name: median(rows) for name, rows in samples.items()},
        }), flush=True)
        runtime.release_replay_values()


if __name__ == "__main__":
    main()
