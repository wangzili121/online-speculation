#!/usr/bin/env python3
"""Compare native verifier cache work at 16 and 128 rows on one Ascend rank."""

import argparse
import json
from pathlib import Path
from statistics import median
from threading import RLock
from time import perf_counter

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.output.exists():
        parser.error("provide a positive iteration count and a new report path")

    from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
    from models.deepseek_v4_1_flash_a3.cache import make_network_cache_publish, make_network_sparse_read
    from models.deepseek_v4_1_flash_a3.prefill_resident import ResidentNetworkRuntime

    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(41128)
    dim, heads, blocks, page_rows = 512, 16, 4, 128
    offset = 64
    stride = page_rows * dim + offset
    plane = dict(blocks=blocks, stride=stride, offset=offset, width=dim)
    geometry = dict(own=plane, long=plane)
    config = dict(num_attention_heads=heads, head_dim=dim, sliding_window=page_rows)
    cache = torch.randn((blocks, stride), dtype=torch.bfloat16, generator=rng)
    compressed = torch.randn(cache.shape, dtype=cache.dtype, generator=rng)
    sink = torch.linspace(-4., 4., heads)
    query = torch.randn((16, heads * dim), dtype=torch.bfloat16, generator=rng)
    kv = torch.randn((16, dim), dtype=torch.bfloat16, generator=rng)

    # The anchor is the last row on a physical page. Subsequent tokens use
    # another, nonadjacent page. Each query sees only its causal SWA prefix.
    page_ids = (0, 3, 1)
    anchor = 255

    def address(position):
        page, row = divmod(position, page_rows)
        return page_ids[page], row

    def slot_id(position):
        page, row = address(position)
        return page * page_rows + row

    slots = torch.tensor([address(anchor + i) for i in range(16)])
    reports = []
    with ResidentNetworkRuntime((args.device,), dict(ep=1, dp=1, tp=1)) as runtime:
        attention = ResidentDraftAttention.__new__(ResidentDraftAttention)
        attention.resident, attention.rank_ids = runtime, {0: 0}
        attention.compile_lock, attention.programs = RLock(), {}
        attention.native = native
        attention.output = args.output.parent / (args.output.stem + "-compiled")
        attention.output.mkdir(parents=True, exist_ok=False)
        runtime.pin(cache, (0,), readonly=False, name="raw/short-validation/swa")
        runtime.pin(compressed, (0,), readonly=True, name="short-validation/compressed")
        runtime.pin(sink, (0,), readonly=True, name="short-validation/sink")

        for live in (1, 2, 8, 16):
            expected_cache = cache.clone()
            ranges_by_page = {}
            for token, (page, row) in enumerate(slots[:live].tolist()):
                begin = offset + row * dim
                expected_cache[page, begin:begin + dim].copy_(kv[token])
                ranges_by_page.setdefault(page, []).append((2 * (page * stride + begin), 2 * dim))
            ranges = sorted(interval for group in ranges_by_page.values() for interval in group)
            inputs = {}
            for capacity in (16, 128):
                ids = torch.full((capacity, 640), -1, dtype=torch.int32)
                for token in range(live):
                    position = anchor + token
                    ids[token, :128] = torch.tensor([slot_id(i) for i in range(position - 127, position + 1)])
                    ids[token, 128:160] = torch.arange(32, dtype=torch.int32)
                values, queries = torch.zeros((capacity, dim), dtype=kv.dtype), torch.zeros(
                    (capacity, heads * dim), dtype=query.dtype,
                )
                mapping = torch.full((capacity, 2), -1, dtype=torch.int64)
                values[:live], queries[:live], mapping[:live] = kv[:live], query[:live], slots[:live]
                inputs[capacity] = values, queries, mapping, ids
                for name, operand in zip(("kv", "query", "slots", "indices"), inputs[capacity], strict=True):
                    runtime.pin(operand, (0,), readonly=True, name=f"fixture/{live}/{capacity}/{name}")

            def call(capacity, check=False):
                values, queries, mapping, ids = inputs[capacity]
                output = runtime.workspace((capacity, heads * dim), torch.bfloat16, 0)
                start = perf_counter()
                with runtime.preserve_cache_ranges(0, [(cache, ranges)]):
                    attention.execute(
                        ("publish", capacity), 0, lambda: make_network_cache_publish(capacity, plane),
                        dict(value=values, slots=mapping, raw=cache, active=live),
                    )
                    if check:
                        assert torch.equal(runtime.read_output(cache, 0), expected_cache)
                    attention.execute(
                        ("read", capacity), 0, lambda: make_network_sparse_read(capacity, config, geometry),
                        dict(query=queries, indices=ids, sink=sink, long_raw=compressed,
                             own_raw=cache, output=output, active=live),
                    )
                    result = runtime.read_output(output, 0)[:live].clone() if check else None
                elapsed = (perf_counter() - start) * 1e3
                if check:
                    assert torch.equal(runtime.read_output(cache, 0), cache)
                runtime.release_values()
                return elapsed, result

            wide = call(128, check=True)[1]
            narrow = call(16, check=True)[1]
            assert torch.isfinite(wide).all()
            assert torch.equal(wide, narrow), (live, (wide.float() - narrow.float()).abs().max().item())
            timing = {16: [], 128: []}
            for _ in range(args.iterations):
                for capacity in (128, 16):
                    timing[capacity].append(call(capacity)[0])
            reports.append(dict(active_rows=live, exact_attention=True, exact_cache_write_and_restore=True,
                                restored_bytes=live * dim * 2, iterations=args.iterations,
                                wide_ms=median(timing[128]), narrow_ms=median(timing[16])))

    report = dict(passed=True, phase="native-cache-publish-attend-and-restore", tile_rows=[16, 128],
                  tp=4, rank_local_heads=heads, head_dim=dim, cases=reports)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
