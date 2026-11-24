"""Compare short-window TP4 verifier argmax and teacher-row feedback on NPU."""

import json
from pathlib import Path
from statistics import median
from threading import Lock
from time import perf_counter
from types import SimpleNamespace

import torch

from blockspec.pypto_dsv41_greedy import merge_greedy_shards
from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native


def main():
    torch.manual_seed(41131)
    torch.set_num_threads(4)
    world, shard = 4, 32320
    results = []
    boundary = native("network")._read_speculative_vocab
    with native("prefill_resident").ResidentNetworkRuntime(
        tuple(range(world)), dict(dp=1, tp=world, ep=world),
    ) as runtime:
        draft = ResidentDraftAttention.__new__(ResidentDraftAttention)
        draft.resident, draft.rank_ids, draft.native = runtime, {i: i for i in range(world)}, native
        draft.compile_lock, draft.programs = Lock(), {}
        draft.output = Path("short-greedy-compiled")
        draft.output.mkdir()

        for capacity in (16, 128):
            logits = torch.randn((capacity, world * shard), dtype=torch.bfloat16)
            logits[0, [33, shard + 99, 2 * shard + 20]] = 666
            shards = [row.contiguous() for row in logits.chunk(world, dim=1)]
            for rank, tensor in enumerate(shards):
                runtime.pin(tensor, (rank,), readonly=True, name=f"vocab{capacity}/rank{rank}")
            for active in (2, 4, 8, 16):
                for teachers in sorted({0, 1, active - 1}):
                    records = [{"positions": range(active)} for _ in range(world)]
                    sharded = [{"fresh": row} for row in shards]
                    executor = SimpleNamespace(
                        resident=runtime, rank_devices=dict(enumerate(range(world))),
                        map_ranks=runtime.map_ranks,
                    )

                    def old():
                        outputs, _, _ = boundary(sharded, records, "fresh", executor, world)
                        merged = outputs[0]
                        assert torch.isfinite(merged).all()
                        return merged.argmax(1), merged[:teachers]

                    def new():
                        packets = runtime.map_ranks(lambda rank: draft.greedy_shard(rank, shards[rank], active))
                        ids = merge_greedy_shards(packets, shard)
                        feedback = torch.cat(runtime.map_ranks(
                            lambda rank: runtime.read_output(shards[rank][:teachers], rank),
                        ), dim=1) if teachers else logits[:0]
                        return ids, feedback

                    def required_device(*args):
                        raise AssertionError("short window should select the host shard path")

                    def host_sharded():
                        _, ids, selected = boundary(
                            sharded, records, "fresh", executor, world,
                            greedy_shard=required_device,
                            logit_ranges=lambda _dp, _tokens: [(0, teachers)] if teachers else (),
                        )
                        return ids[0], selected.get((0, 0), logits[:0])

                    baseline, candidate = old(), new()
                    gold = logits[:active].argmax(1), logits[:teachers]
                    for arm in (baseline, candidate, host_sharded()):
                        assert all(torch.equal(a, b) for a, b in zip(arm, gold, strict=True))
                    runtime.release_values()
                    arms = {}
                    for name, call in (("full", old), ("packet", new), ("host_sharded", host_sharded)):
                        samples = []
                        copied = []
                        for _ in range(15):
                            before = runtime.stats["output_d2h_bytes"]
                            started = perf_counter()
                            call()
                            samples.append((perf_counter() - started) * 1e3)
                            copied.append(runtime.stats["output_d2h_bytes"] - before)
                            runtime.release_values()
                        arms[name] = dict(ms=median(samples), readback_bytes=median(copied))
                    results.append(dict(capacity=capacity, active=active, teachers=teachers, **arms))

        # The same short geometry must reject every nonfinite shard.
        contaminated = shards[0].clone()
        contaminated[0, 7] = float("nan")
        contaminated[1, 5] = float("inf")
        contaminated[2, 11] = float("-inf")
        runtime.pin(contaminated, (0,), readonly=True, name=f"vocab{capacity}/invalid")
        assert draft.greedy_shard(0, contaminated, 3)[:, 2].eq(0).all()
        invalid = [dict(fresh=contaminated), *sharded[1:]]
        records = [{"positions": range(3)} for _ in range(world)]
        try:
            boundary(invalid, records, "fresh", executor, world, greedy_shard=required_device)
        except RuntimeError as error:
            assert "Non-finite" in str(error)
        else:
            raise AssertionError("short verifier failed to reject nonfinite values")
        runtime.release_values()
    print(json.dumps(dict(passed=True, scopes=results, tp=world, vocabulary=world * shard)), flush=True)


if __name__ == "__main__":
    main()
