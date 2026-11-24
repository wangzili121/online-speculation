"""Validate resident zero-gradient send buffers with an actual local HCCL DP group."""

import argparse
import json
from pathlib import Path
import runpy
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import torch

from blockspec.dsv41_suffix import DSV41SuffixPlan
from blockspec.pypto_dsv41_cluster_transport import ClusterCollectives
from blockspec.pypto_dsv41_resident_primitives import DSV41ResidentPrimitives, _RankGradient
from blockspec.pypto_dsv41_runtime import native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    args = parser.parse_args()
    baseline = runpy.run_path(str(args.baseline_file), run_name="blockspec.baseline_dp_zero")
    torch.set_num_threads(4)
    torch.manual_seed(41131)
    ranks = tuple(range(4))
    plan = SimpleNamespace(node=0, local_ranks=range(4), counts=(4, 4),
                           groups=lambda _: (ranks, (4, 5, 6, 7)))
    suffix = DSV41SuffixPlan(1)
    names = suffix.trainable_names()
    shapes = ((5120, 1280), (1280,), (1280, 8192), (5120, 512),
              (512,), (16,), (2, 1024, 4096), (2048, 5120))

    def peer(stage, payload):
        if stage.startswith("schedule/"):
            return dict(payload, packets=[dict(row, rank=row["rank"] + 4)
                                          for row in payload["packets"]])
        assert stage.startswith("ready/")
        return None

    with native("prefill_resident").ResidentNetworkRuntime(
        ranks, dict(dp=4, tp=1, ep=4),
    ) as runtime, ClusterCollectives(runtime, plan, SimpleNamespace(node=0, exchange=peer)) as transport:
        variants = {}
        for label, cls, descriptor in (
            ("before", baseline["DSV41ResidentPrimitives"], baseline["_RankGradient"]),
            ("after", DSV41ResidentPrimitives, _RankGradient),
        ):
            engine = cls.__new__(cls)
            engine.resident, engine.ranks, engine.plan, engine._last_step = runtime, ranks, suffix, 0
            engine.draft = SimpleNamespace(rank_ids={rank: rank for rank in ranks})
            engine.native = SimpleNamespace(transport=transport)
            engine._adam = {
                (rank, 39, name.split(".attn.", 1)[1]):
                    SimpleNamespace(master=torch.empty(shape, device="meta"))
                for rank in ranks for name, shape in zip(names, shapes, strict=True)
            }
            engine._dp_zero_storage = None
            engine._dp_zero_ranks, engine._dp_zero_views = set(), {}
            variants[label] = engine, descriptor

        inputs = {name: torch.randint(-3, 4, shape, dtype=torch.int16).float()
                  for name, shape in zip(names, shapes, strict=True)}
        for name, value in inputs.items():
            runtime.pin(value, (0, 3), readonly=True, name=f"test/{name}")

        def run(label, owner, *, verify=False):
            engine, descriptor = variants[label]
            fill = runtime.stats["workspace_fill_bytes"]
            upload = runtime.stats["h2d_bytes"]
            start = perf_counter()
            if owner is None:
                gradients = engine.zero_gradients(suffix.layers, names)
            else:
                gradients = {name: descriptor(39, {owner: value}) for name, value in inputs.items()}
            result = engine.dp_all_reduce(gradients)
            elapsed = (perf_counter() - start) * 1e3
            if verify:
                for name, gradient in result.items():
                    for rank in ranks:
                        value = runtime.read_output(gradient.values[rank], rank)
                        if owner is None:
                            assert not value.any(), (name, rank)
                        else:
                            assert torch.equal(inputs[name], value), (name, rank)
            runtime.release_values()
            return elapsed, runtime.stats["workspace_fill_bytes"] - fill, runtime.stats["h2d_bytes"] - upload

        summaries = {}
        sealed_weight_uploads = runtime.stats["device_weight_upload_calls"]
        for owner in (0, 3, None):
            for label in variants:
                run(label, owner, verify=True)
            samples = {label: [run(label, owner) for _ in range(10)] for label in variants}
            assert all(row[1] > 0 for row in samples["before"])
            assert all(row[1:] == (0, 0) for row in samples["after"])
            summaries[str(owner)] = {
                label: {"wall_ms": median(row[0] for row in rows),
                        "fill_bytes": rows[0][1], "h2d_bytes": rows[0][2]}
                for label, rows in samples.items()
            }
        assert runtime.stats["device_weight_upload_calls"] == sealed_weight_uploads
        engine, _ = variants["after"]
        expected_bytes = max(torch.Size(shape).numel() for shape in shapes) * 4
        for rank in ranks:
            assert torch.count_nonzero(runtime.read_output(engine._dp_zero(rank, shapes[0]), rank)) == 0
            record = runtime.pinned[rank, engine._dp_zero_storage.untyped_storage()._cdata]
            assert record["readonly"] and record["nbytes"] == expected_bytes
    print(json.dumps(dict(passed=True, dp=4, local_fabric=True, byte_exact=True, static_weight_counter_unchanged=True,
                          repeats=10, zero_bytes_per_rank=expected_bytes, cases=summaries)), flush=True)


if __name__ == "__main__":
    main()
