"""Time the native verifier's shared index source and four reindex consumers."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from threading import Lock
from time import perf_counter

import torch

from blockspec.pypto_dsv41_runtime import ResidentDraftAttention, native
from blockspec.pypto_dsv41_verify_cache import VerifyIndexKeys
from validate_dsv41_pypto_verify_compressed_cache import geometry_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(41132)
    cache, indexer = native("cache"), native("prefill_indexer")
    roles = [cache.resolve_layer_role(layer) for layer in range(40)]
    source = next(role for role in roles if role["candidate_write"])
    readers = [role for role in roles if role["mode"] == "reindex" and role["kv_source"] == source["physical_layer"]]
    names, _, layout = geometry_for(source, 131)
    geometry = cache.network_cache_geometry(layout, source, 131)
    capacity, requests, table_pages = 128, 2, 128
    entries = cache.make_network_cache_entries(
        capacity, dict(num_attention_heads=64, head_dim=512), geometry, source,
    )
    entries.update(indexer.make_physical_index_query_entries(capacity, {}, 131, requests, table_pages, 1))
    prefix = names["index"].removesuffix("indexer.k_cache")
    args.output.mkdir()
    report = []
    with native("prefill_resident").ResidentNetworkRuntime((0,), dict(ep=1, dp=1, tp=1)) as runtime:
        engine = ResidentDraftAttention.__new__(ResidentDraftAttention)
        engine.resident, engine.rank_ids, engine.native = runtime, {0: 0}, native
        engine.compile_lock, engine.programs, engine.output = Lock(), {}, args.output
        raw = {name: torch.randint(-8, 8, plane["shape"], dtype=torch.int8, generator=generator)
               for name, plane in layout["raw"].items()}
        views = native("prefill_compressed_runtime").physical_cache_views(raw, layout)
        views[names["index"]][1].fill_(0.03125)
        for name, tensor in raw.items():
            runtime.pin(tensor, (0,), readonly=True, name=name)
        query = torch.randint(-48, 48, (32, 32, 128), dtype=torch.int8, generator=generator)
        cosine = torch.zeros((32, 64), dtype=torch.bfloat16)
        operands = dict(index_wqb=None, index_wqb_scale=None, index_weights=None)

        for history in (31, 1025, 8193):
            positions = torch.arange(history - 16, history).repeat(2)
            lengths = torch.tensor([history, history])
            required_pages = (history + 127) // 128
            pages = sorted(set(range(1, required_pages + 1)) | set(range(2, required_pages + 2)))
            long_table = torch.stack((torch.arange(1, required_pages + 1),
                                      torch.arange(required_pages + 1, 1, -1))).int()
            window = torch.tensor([[129], [130]], dtype=torch.int32)
            meta = {names["index"]: dict(cache_seq_lens=lengths), names["long"]: dict(block_table=long_table),
                    names["swa"]: dict(block_table=window)}
            original = dict(cp_token_range=(0, 32, 32, 32), positions=positions,
                            cache_metadata=dict(actual_metadata={k: dict(global_metadata=v) for k, v in meta.items()}),
                            metadata=dict(query_start_loc=torch.tensor([0, 16, 32])))

            def run(reuse):
                timings, observed, counts = [], [], Counter()
                index_keys = VerifyIndexKeys(roles) if reuse else None
                hooks = dict(copy_tensor=runtime.copy_tensor, verify_index_cache=reuse, verify_index_keys=index_keys)

                for role in (source, *readers):
                    # Run actual index gather, score/Top-K and sparse controls. Inject
                    # deterministic post-projection queries that differ across layers.
                    layer = role["physical_layer"]
                    q_codes = query.roll(layer, dims=2)

                    def execute(name, rank, _case, values, outputs, _stage):
                        if name == "index_query_quant":
                            outputs["codes"][:32].copy_(q_codes)
                            outputs["scales"][:32].fill_(0.03125)
                        elif name == "index_weights_scale":
                            outputs["weights"][:32].fill_(0.03125)
                        elif name not in ("index_wqb", "index_weights_proj", "index_query_rope"):
                            program = entries[name]
                            counts[name] += 1
                            engine.execute(name, rank, lambda: program,
                                           dict(zip(program.param_names, values, strict=True)))

                    def load_index():
                        dense = cache.load_network_index(raw, geometry, pages, execute, 0, "fresh")
                        return dense["codes"], dense["scales"]

                    begin = perf_counter()
                    codes, scales = load_index() if role is source or not reuse else (None, None)
                    selection = indexer.run_index_query_selection(
                        dict(original=original, total=32, role=role, runtime=hooks),
                        None, None, cosine, cosine, operands, execute, 0, "fresh",
                        capacity, requests, table_pages, prefix, prefix, codes, scales, load_index=load_index,
                    )
                    timings.append(perf_counter() - begin)
                    outputs = [runtime.read_control(selection[key][:32], 0)
                               for key in ("selected", "lengths", "compressed", "window")]
                    if role is source:
                        outputs.append(runtime.read_output(selection["scores"][:32], 0))
                    observed.append(outputs)
                    runtime.release_values(index_keys.finish_layer(layer, {0: 0}) if reuse else ())
                return sum(timings), observed, counts

            _, reference, before_counts = run(False)
            _, replayed, after_counts = run(True)
            assert len(readers) == 4 and after_counts["index_select"] == before_counts["index_select"] == 10
            assert after_counts["index_key_gather"] == 2 and before_counts["index_key_gather"] == 10
            for a, b in zip(reference, replayed, strict=True):
                for left, right in zip(a, b, strict=True):
                    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
            before, after = [], []
            for _ in range(args.iterations):
                before.append(run(False)[0])
                after.append(run(True)[0])
            report.append(dict(history=history, requests=requests, readers=len(readers), byte_exact=True,
                               calls_before=before_counts, calls_after=after_counts,
                               before_ms=median(before) * 1000, after_ms=median(after) * 1000))
    print(json.dumps(dict(passed=True, cases=report)), flush=True)


if __name__ == "__main__":
    main()
