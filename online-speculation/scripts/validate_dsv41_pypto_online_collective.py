"""Check write-only online collective receive buffers on real HCCL TP4."""

import argparse
import json
from pathlib import Path
import runpy
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import torch

from blockspec.pypto_dsv41_cluster_transport import ClusterCollectives
from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives
from blockspec.pypto_dsv41_runtime import native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    args = parser.parse_args()
    baseline = runpy.run_path(str(args.baseline_file), run_name="blockspec.baseline_collective")[
        "DSV41ResidentPrimitives"
    ]
    torch.manual_seed(41131)
    torch.set_num_threads(4)
    ranks = range(4)
    plan = SimpleNamespace(node=0, local_ranks=ranks, counts=(4, 4),
                           groups=lambda _: (tuple(ranks), (4, 5, 6, 7)))

    def peer(stage, payload):
        if stage.startswith("schedule/"):
            return dict(payload, packets=[dict(row, rank=row["rank"] + 4) for row in payload["packets"]])
        assert stage.startswith("ready/")
        return None

    cases = []
    with native("prefill_resident").ResidentNetworkRuntime(ranks, dict(dp=1, tp=4, ep=4)) as runtime:
        with ClusterCollectives(runtime, plan, SimpleNamespace(node=0, exchange=peer)) as transport:
            engines = {}
            for label, cls in (("before", baseline), ("after", DSV41ResidentPrimitives)):
                engine = cls.__new__(cls)
                engine.resident, engine.ranks = runtime, tuple(ranks)
                engine.draft = SimpleNamespace(rank_ids={rank: rank for rank in ranks})
                engine.native = SimpleNamespace(transport=transport)
                engines[label] = engine

            for collective, shape, dtype, stage in (
                ("all_gather", (32, 5120), torch.bfloat16, "attn-output-vjp"),
                ("reduce_scatter", (128, 5120), torch.float32, "attn-input-vjp"),
                ("all_reduce", (1280,), torch.float32, "q-norm-vjp"),
                ("all_reduce", (5120, 512), torch.float32, "wkv-vjp"),
                ("all_reduce", (5120, 1280), torch.float32, "wq-a-vjp"),
                ("all_reduce", (5120, 1280), torch.float32, "online-dp-receive"),
            ):
                inputs = {rank: torch.randint(-8, 8, shape, dtype=torch.int16).to(dtype) for rank in ranks}
                for rank, value in inputs.items():
                    runtime.pin(value, (rank,), readonly=True, name=f"{stage}/{rank}")

                def run(label, read=False):
                    prior = runtime.stats["workspace_fill_bytes"]
                    start = perf_counter()
                    if stage == "online-dp-receive":
                        packets = [{"gradient": {"value": inputs[rank]}} for rank in ranks]
                        results, _ = transport.exchange(
                            stage, packets, collective, "dp", write_only_receive=label == "after",
                        )
                        outputs = {rank: results[rank]["gradient"]["value"] for rank in ranks}
                    else:
                        outputs = engines[label]._tp_collective(inputs, collective, stage=stage)
                    elapsed = (perf_counter() - start) * 1e3
                    fills = runtime.stats["workspace_fill_bytes"] - prior
                    result = [runtime.read_output(outputs[rank], rank) for rank in ranks] if read else None
                    runtime.release_values()
                    return elapsed, fills, result

                _, _, expected = run("before", read=True)
                _, _, actual = run("after", read=True)
                assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
                if collective == "all_gather":
                    gold = torch.cat(list(inputs.values()))
                    assert all(torch.equal(gold, value) for value in actual)
                else:
                    gold = sum(inputs.values())
                    gold = gold.chunk(len(ranks)) if collective == "reduce_scatter" else (gold,) * len(ranks)
                    assert all(torch.equal(a, b) for a, b in zip(gold, actual, strict=True))
                times, fills = {key: [] for key in engines}, {}
                for _ in range(12):
                    for label in engines:
                        elapsed, filled, _ = run(label)
                        times[label].append(elapsed)
                        fills[label] = filled
                assert fills["before"] > 0 and fills["after"] == 0
                cases.append(dict(stage=stage, collective=collective, dtype=str(dtype), shape=shape,
                                  ms={label: median(samples) for label, samples in times.items()},
                                  workspace_fill_bytes=fills, byte_exact=True))
    print(json.dumps(dict(passed=True, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
