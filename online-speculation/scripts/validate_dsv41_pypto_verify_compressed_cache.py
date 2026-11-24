"""Check native C1A/C2A writes and narrow verifier rollback on a real NPU."""

import argparse
import json
from pathlib import Path
import runpy
from statistics import median
from threading import Lock
from time import perf_counter
from types import SimpleNamespace

import torch

from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
from blockspec.pypto_dsv41_verify_cache import index_written_pages


def geometry_for(role, blocks):
    root = f"model.layers.{role['physical_layer']}.self_attn."
    ratio = role["ratio"]
    names = {key: root + suffix for key, suffix in (
        ("swa", "swa_cache"), ("long", "long_kv_cache"),
        ("index", "indexer.k_cache"), ("ring", "compressor.state_cache"),
    )}

    def plane(rows, width, dtype, offset=0, key="shared"):
        size = torch.empty((), dtype=dtype).element_size()
        return dict(storage_key=key, dtype=str(dtype), shape=[blocks, rows, 1, width],
                    stride=[131072 // size, width, width, 1], storage_offset_bytes=offset,
                    storage_bytes=131072 * blocks)

    index_key = "index" if ratio == 1 else "shared"
    index_offset = 0 if ratio == 1 else 65536
    encoded_rows = 128 // ratio
    views = {
        names["swa"]: plane(128, 512, torch.bfloat16),
        names["long"]: plane(encoded_rows, 512, torch.bfloat16),
        names["index"]: [
            plane(encoded_rows, 128, torch.int8, index_offset, index_key),
            plane(encoded_rows, 1, torch.float16, index_offset + encoded_rows * 128, index_key),
        ],
    }
    groups = [[names["swa"]], [names["long"], names["index"]]]
    if ratio == 2:
        views[names["ring"]] = plane(32, 1024, torch.float32)
        groups.append([names["ring"]])
    raw = {key: dict(storage_key=key, dtype="torch.int8", shape=[blocks * 131072],
                     stride=[1], storage_offset_bytes=0, storage_bytes=blocks * 131072)
           for key in {index_key, "shared"}}
    return names, groups, dict(views=views, raw=raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(41131)
    cache, compressor = native("cache"), native("compressor")
    old_cache = SimpleNamespace(**runpy.run_path(
        str(args.baseline_cache), run_name="models.deepseek_v4_1_flash_a3.baseline_cache",
    ))
    roles = {role["ratio"]: role for layer in reversed(range(40))
             if (role := cache.resolve_layer_role(layer))["mode"] == "full"}
    old_cache.make_network_cache_entries.__globals__["resolve_layer_role"] = cache.resolve_layer_role
    capacity, blocks = 128, 386
    results = []
    with native("prefill_resident").ResidentNetworkRuntime((0,), dict(ep=1, dp=1, tp=1)) as runtime:
        engine = ResidentDraftAttention.__new__(ResidentDraftAttention)
        engine.resident, engine.rank_ids, engine.native = runtime, {0: 0}, native
        engine.compile_lock, engine.programs = Lock(), {}
        engine.output = args.output
        engine.output.mkdir()

        for ratio in (1, 2):
            role = roles[ratio]
            names, groups, layout = geometry_for(role, blocks)
            geometry = cache.network_cache_geometry(layout, role, blocks)
            entries = cache.make_network_cache_entries(
                capacity, dict(num_attention_heads=64, head_dim=512), geometry, role,
            )
            old_entries = old_cache.make_network_cache_entries(
                capacity, dict(num_attention_heads=64, head_dim=512), geometry, role,
            )
            entries["index_publish"] = compressor.make_index_cache_publish_observed(capacity, blocks, 128 // ratio)
            if ratio == 2:
                entries["ring"] = compressor.make_c2_pool_observed(capacity, blocks, 1)

            def dispatch(programs, variant):
                def execute(name, rank, _case, values, _outputs, _stage):
                    program = programs[name]
                    engine.execute((ratio, variant, name), rank, lambda: program,
                                   dict(zip(program.param_names, values, strict=True)))
                return execute

            execute = dispatch(entries, "after")
            execute_old = dispatch(old_entries, "before")

            provider = SimpleNamespace(local_ranks=(0,), layouts={0: {"fresh": layout}})
            baseline_raw, candidate_raw, initial = {}, {}, {}
            for key, plane in layout["raw"].items():
                # A finite poison in FP32, BF16 and FP16 interpretations. The
                # tiny integer perturbations give each physical page a distinct payload.
                raw = torch.full(plane["shape"], -53, dtype=torch.int8)
                raw[::128] = torch.arange(len(raw[::128])) % 25 - 12
                initial[key] = raw
                for label, arrays in (("before", baseline_raw), ("after", candidate_raw)):
                    arrays[key] = raw.clone()
                    runtime.pin(arrays[key], (0,), readonly=False, name=f"raw/{ratio}/{label}/{key}")

            x = (torch.randn(capacity, 512) * 0.001).bfloat16()
            key_input = (torch.randn(capacity, 128) * 0.001).bfloat16()
            for name, tensor in (("long", x), ("key", key_input)):
                runtime.pin(tensor, (0,), readonly=True, name=f"{ratio}/{name}")
            for pages_per_group, active in ((2, 8), (32, 32), (128, 32)):
                allocations = []
                for i in range(len(groups)):
                    pages = list(range(1 + i * 128, 1 + i * 128 + pages_per_group))
                    # Cross a descending, physically nonconsecutive page boundary.
                    if pages_per_group == 2:
                        pages.reverse()
                    else:
                        pages[-1], pages[0] = pages[0], pages[-1]
                    allocations.append(pages)
                anchor = (pages_per_group - 1) * 128 - 3
                template = dict(groups=[dict(group_id=i, names=group) for i, group in enumerate(groups)],
                                hash_constants={})
                metadata = native("prefill_prepare").build_chunk_metadata(
                    template, layout, allocations, tokens=[13] * active, context=anchor,
                    tp_rank=0, tp=1, active=True, reserved=0,
                )
                global_meta = {name: item["global_metadata"] for name, item in metadata["metadata"].items()}
                published = global_meta[names["index"]]["slot_mapping"]
                dirty = index_written_pages(published)
                assert len(dirty) == 2
                padded_slots = torch.full((capacity, 2), -1, dtype=torch.int32)
                padded_slots[:active] = published
                intervals, _ = native("state").NetworkCacheProvider.verification_ranges(provider, 0, metadata)
                before_ranges = {}
                for group in metadata["groups"]:
                    for name in group["names"]:
                        planes = layout["views"][name]
                        for plane in planes if isinstance(planes, list) else [planes]:
                            extents = before_ranges.setdefault(plane["storage_key"], set())
                            if name == names["swa"]:
                                extents.update((page * 131072 + row * 1024, 1024)
                                               for page, row in group["slot_mapping"].tolist())
                            else:
                                extents.update((page * 131072, 131072) for page in group["live_block_ids"])

                def index(raw, verify, *, publish=True, all_pages=False):
                    implementation, launch = (cache, execute) if verify else (old_cache, execute_old)
                    dense = implementation.load_network_index(raw, geometry, allocations[1], launch, 0, "probe")
                    if publish:
                        payload = runtime.workspace((capacity, 128), torch.int8, 0, fill=None)
                        scales = runtime.workspace((capacity, 8), torch.float32, 0, fill=None)
                        execute("index_publish", 0, "probe",
                                [key_input, padded_slots, dense["codes"], dense["scales"], payload, scales, active], {}, "")
                    implementation.store_network_index(
                        raw, geometry, dirty if verify and not all_pages else allocations[1], dense, launch, 0, "probe",
                    )

                def write_other(raw):
                    for name, program in (("swa", "network_swa_publish"), ("long", "network_compressed_publish")):
                        slots = torch.full((capacity, 2), -1, dtype=torch.int64)
                        slots[:active] = global_meta[names[name]]["slot_mapping"]
                        matrix = cache.raw_cache_matrix(raw, geometry["own" if name == "swa" else "long"])
                        execute(program, 0, "probe", [x, slots, matrix, active], {}, "")
                    if ratio == 2:
                        native_ring = global_meta[names["ring"]]
                        positions = torch.zeros(capacity, dtype=torch.int32)
                        positions[:active] = native_ring["positions"]
                        requested = torch.full((capacity,), -1, dtype=torch.int32)
                        requested[:active] = 0
                        pooled = runtime.workspace((capacity, 512), torch.bfloat16, 0, fill=None)
                        normalized = runtime.workspace((capacity, 512), torch.bfloat16, 0, fill=None)
                        ring = cache.raw_cache_matrix(raw, geometry["ring"]).reshape(blocks, 32, 1024)
                        execute("ring", 0, "probe", [
                            x.float().contiguous(), x.float().contiguous() * 0.5,
                            native_ring["query_start_loc"], positions, requested,
                            native_ring["block_table"], torch.ones(512, dtype=torch.bfloat16),
                            ring, pooled, normalized, active,
                        ], {}, "")

                ranges = [(candidate_raw[key], value) for key, value in intervals.items()]
                with runtime.preserve_cache_ranges(0, ranges):
                    index(candidate_raw, True)
                    write_other(candidate_raw)
                    index(baseline_raw, False)
                    write_other(baseline_raw)
                    for key in sorted(initial):
                        expected = runtime.read_output(baseline_raw[key], 0)
                        actual = runtime.read_output(candidate_raw[key], 0)
                        assert not torch.equal(initial[key], expected), (ratio, key, "fixture made no write")
                        assert torch.equal(actual, expected), (ratio, key, pages_per_group, "write mismatch")
                if pages_per_group == 128:
                    full_snapshot = [(candidate_raw[key], sorted(spans)) for key, spans in before_ranges.items()]
                    with runtime.preserve_cache_ranges(0, full_snapshot):
                        index(candidate_raw, True, all_pages=True)
                        write_other(candidate_raw)
                        for key in initial:
                            assert torch.equal(runtime.read_output(candidate_raw[key], 0),
                                               runtime.read_output(baseline_raw[key], 0)), (ratio, key, "full store")
                for key in initial:
                    assert torch.equal(runtime.read_output(candidate_raw[key], 0), initial[key]), (ratio, key, "rollback")
                    runtime.copy_tensor(baseline_raw[key], initial[key], 0)
                runtime.release_values()

                samples = {}
                for name, raw, verify in (("before", baseline_raw, False), ("after", candidate_raw, True)):
                    elapsed = []
                    for _ in range(args.iterations):
                        started = perf_counter()
                        index(raw, verify, publish=False)
                        elapsed.append((perf_counter() - started) * 1000)
                        runtime.release_values()
                    samples[name] = median(elapsed)
                results.append(dict(
                    ratio=ratio, active=active, pages_per_group=pages_per_group,
                    index_copy_ms=samples,
                    backup_bytes=dict(before=sum(size for spans in before_ranges.values() for _, size in spans),
                                      after=sum(size for spans in intervals.values() for _, size in spans)),
                    byte_exact=True,
                ))
    print(json.dumps(dict(passed=True, cases=results, iterations=args.iterations)), flush=True)


if __name__ == "__main__":
    main()
