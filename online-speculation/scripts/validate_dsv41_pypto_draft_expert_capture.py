"""Compare native draft MoE combine with and without the online snapshot."""

import argparse
import json
from pathlib import Path
from statistics import median
from threading import Lock
from time import perf_counter

import torch
import pypto.language as pl

from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(41134)
    world, slab, hidden, topk, capacity = 16, 16, 5120, 6, 128
    routes = slab * topk
    program = native("prefill_moe").make_moe_combine(slab, hidden, topk, world, weight_dtype=pl.BF16)
    cases = []
    with native("prefill_resident").ResidentNetworkRuntime((0,), dict(ep=1, dp=1, tp=1)) as runtime:
        engine = ResidentDraftAttention.__new__(ResidentDraftAttention)
        engine.resident, engine.rank_ids, engine.native = runtime, {0: 0}, native
        engine.compile_lock, engine.programs, engine.output = Lock(), {}, args.output
        engine.output.mkdir()
        for active in (4, 8, 16):
            metadata = torch.full((world * routes, 4), -1, dtype=torch.int64)
            counts = torch.zeros((world, 8), dtype=torch.int32)
            for token in range(active):
                for slot in range(topk):
                    peer = (slot + token) % world
                    packet = peer * routes + int(counts[peer, 0])
                    metadata[packet] = torch.tensor([0, token, slot, peer * 24 + slot])
                    counts[peer, 0] += 1
            x = torch.randn((world * routes, hidden), dtype=torch.bfloat16, generator=rng)
            shared = torch.randn((slab, hidden), dtype=torch.bfloat16, generator=rng)
            weights = torch.rand((slab, topk), generator=rng)
            weights /= weights.sum(-1, keepdim=True)
            sources = dict(packet_routes=metadata, packet_hidden=x, packet_counts=counts,
                           route_weights=weights, shared=shared)
            for key, tensor in sources.items():
                runtime.pin(tensor, (0,), readonly=True, name=f"{active}/{key}")

            def run(capture, check=False):
                bytes_before = runtime.transfer_stats[0].get("d2d_bytes", 0)
                start = perf_counter()
                local = dict(
                    routed=runtime.workspace((slab, hidden), torch.bfloat16, 0),
                    output=runtime.workspace((slab, hidden), torch.bfloat16, 0),
                    returned_routes=runtime.workspace((routes, 4), torch.int64, 0, fill=-1),
                    expert_outputs=runtime.workspace((routes, hidden), torch.bfloat16, 0),
                )
                if capture:
                    snapshot = runtime.workspace((capacity * topk, hidden), torch.bfloat16, 0)
                engine.execute("moe-combine", 0, lambda: program, dict(sources, **local, active=active))
                if capture:
                    count = active * topk
                    runtime.copy_tensor(snapshot[:count], local["expert_outputs"][:count], 0)
                elapsed = (perf_counter() - start) * 1000
                moved = runtime.transfer_stats[0].get("d2d_bytes", 0) - bytes_before
                output = None
                if check:
                    output = runtime.read_output(local["output"][:active], 0)
                    if capture:
                        assert torch.equal(runtime.read_output(snapshot[:count], 0),
                                           runtime.read_output(local["expert_outputs"][:count], 0))
                runtime.release_values()
                return elapsed, moved, output

            _, before_bytes, before = run(True, check=True)
            _, after_bytes, after = run(False, check=True)
            assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
            assert before_bytes == active * topk * hidden * 2 and after_bytes == 0
            before_samples, after_samples = [], []
            for _ in range(12):
                before_samples.append(run(True)[0])
                after_samples.append(run(False)[0])
            cases.append(dict(active_rows=active, bytes_before=before_bytes, bytes_after=after_bytes,
                              bytes_snapshot=capacity * topk * hidden * 2, byte_exact=True,
                              before_ms=median(before_samples), after_ms=median(after_samples)))
    print(json.dumps(dict(passed=True, cases=cases)), flush=True)


if __name__ == "__main__":
    main()
