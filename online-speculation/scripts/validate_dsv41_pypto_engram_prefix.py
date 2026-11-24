"""Check the native HOST Engram lookup when committing a verified token prefix."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from models.deepseek_v4_1_flash_a3.engram_host import HostEngram, make_layout


def main():
    torch.set_num_threads(2)
    torch.manual_seed(41131)
    topology = dict(dp=4, tp=4, ep=16)
    slots = (1, 14)
    layout = make_layout(compressed_vocab_size=1024, vocab_size=65537)

    class Table:
        def __init__(self, data):
            self.data = data
            self.reads = 0

        def gather(self, rows):
            self.reads += 1
            return self.data[rows.long().remainder(len(self.data))]

    tables = {
        "codes": Table(torch.randint(-127, 128, (4096, 256), dtype=torch.int8)),
        "scales": Table(torch.rand((4096, 8), dtype=torch.float32) / 128),
    }
    host = HostEngram.__new__(HostEngram)
    host.checkpoint = str(Path("/test-checkpoint").resolve())
    host._closed, host.arrays = False, {}
    host.constants = dict(token_map=torch.arange(1024))
    host.layout = layout
    host.contracts = {
        layer: dict(topology=topology, model_config={"image_token_id": 1019},
                    engram_shards=[dict(head_range=(6 * r, 6 * (r + 1))) for r in range(4)])
        for layer in slots
    }
    host._table = lambda _layer, _rank, kind: tables[kind]

    def request(slot, anchor, seed):
        generator = torch.Generator().manual_seed(seed)
        return dict(request_slot=slot, generation=1,
                    history_tokens=torch.randint(0, 1018, (anchor,), generator=generator).tolist(),
                    token_ids=torch.randint(0, 1020, (32,), generator=generator).tolist())

    def lookup(rows, capacity, verified=None):
        engines, metadata = [], []
        for dp, requests in enumerate(rows):
            normalized = []
            for row in requests:
                start, count = len(row["history_tokens"]), len(row["token_ids"])
                normalized.append(dict(row, initial_context_length=start, live_token_mask=[1] * count,
                                       position_ids=list(range(start, start + count))))
            positions = torch.tensor([pos for row in normalized for pos in row["position_ids"]], dtype=torch.long)
            starts = [0]
            for row in normalized:
                starts.append(starts[-1] + len(row["token_ids"]))
            active = starts[-1]
            physical = (active + 3) // 4
            for lane in range(4):
                slot = dict(cp_token_range=(physical * lane, physical * (lane + 1), physical, 4 * physical),
                            global_metadata=dict(query_start_loc=torch.tensor(starts, dtype=torch.int32)))
                metadata.append(dict(positions=positions, query_start_loc=torch.tensor(starts),
                                     metadata={"model.layers.1.self_attn.swa_cache": slot}))
            engines.append(dict(dp_rank=dp, active=active, requests=normalized))
        config = dict(topology=topology, capacity=capacity, input_contract_sha256="fixture",
                      checkpoint=host.checkpoint)
        results = host.lookup(dict(input_contract=dict(DP_engines=engines, topology=topology)),
                              cache=dict(cache_ABI=metadata), config=config, verified=verified)
        records = results["cases"]["fresh"]
        reference = [dict(requests=engine["requests"], hashes=row["hashes"], full=row["full"])
                     for engine, row in zip(engines, records[::4], strict=True)]
        return records, reference

    def prefix(row, length):
        return dict(row, token_ids=row["token_ids"][:length])

    def exact(before, after):
        for left, right in zip(before, after, strict=True):
            for name in ("full_input_ids", "positions", "hashes", "mask", "local_mask"):
                assert torch.equal(left[name], right[name]), name
            assert left["cp_token_range"] == right["cp_token_range"]
            for name in ("full", "local"):
                for layer in slots:
                    assert torch.equal(left[name][layer], right[name][layer]), (name, layer)

    single = request(1, 127, 8)
    full_dp = [[request(0, 254 + dp, 31 + dp)] for dp in range(4)]
    full_dp[2][0]["token_ids"][0] = 1019
    packed = [request(slot, 1023 + slot, 50 + slot) for slot in range(4)]
    cases = (
        ("single", [[single], [], [], []], [[prefix(single, 1)], [], [], []], 16),
        ("4dp", full_dp, [[prefix(row[0], size)] for row, size in zip(full_dp, (1, 2, 3, 8), strict=True)], 128),
        ("reordered", [packed, [], [], []], [[prefix(packed[i], i + 1) for i in (3, 2, 0, 1)], [], [], []], 16),
    )
    result = []
    for name, verify, accept, capacity in cases:
        _, verification = lookup(verify, 128)
        before, _ = lookup(accept, capacity)
        reads = tables["codes"].reads
        after, _ = lookup(accept, capacity, verification)
        assert tables["codes"].reads == reads
        exact(before, after)
        samples = {"before": [], "after": []}
        for _ in range(12):
            for mode in samples:
                start = perf_counter()
                lookup(accept, capacity, verification if mode == "after" else None)
                samples[mode].append((perf_counter() - start) * 1e3)
        result.append(dict(case=name, capacity=capacity, byte_exact=True,
                           wall_ms={mode: median(times) for mode, times in samples.items()}))

    _, verification = lookup([[single], [], [], []], 128)
    changed_history = single["history_tokens"].copy()
    changed_history[-1] = (changed_history[-1] + 1) % 1024
    for change in ({"generation": 2}, {"history_tokens": changed_history}, {"token_ids": [999]}):
        invalid = [[dict(prefix(single, 1), **change)], [], [], []]
        before, _ = lookup(invalid, 16)
        reads = tables["codes"].reads
        after, _ = lookup(invalid, 16, verification)
        assert tables["codes"].reads > reads
        exact(before, after)
    print(json.dumps(dict(passed=True, cases=result, mismatched_ancestry_falls_back=True)), flush=True)


if __name__ == "__main__":
    main()
