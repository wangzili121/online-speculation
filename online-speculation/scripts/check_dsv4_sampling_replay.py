#!/usr/bin/env python3
"""Separate within-service repeatability from changes in batching or sampling.

Different engines need not produce the same tokens for the same seed. This
checks one running service, preserving raw records for failure diagnosis.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from validate_dsv4_service import _completion, _signature
from blockspec.reporting import dumps
from blockspec.vllm_ascend.artifacts import write_private_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8901")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    base = {"prompt": "A random color is", "max_tokens": args.max_tokens,
            "temperature": 0.8, "top_p": 0.95}
    cases = [{**base, "seed": seed} for seed in range(4)]

    def complete(case):
        return _completion(args.base_url, args.model, case)

    complete(cases[0])  # Exclude compilation and allocator warmup.
    records = {}
    matches = {}

    def record(name, rows, reference=None):
        records[name] = rows
        if reference is not None:
            matches[name] = [_signature(a) == _signature(b)
                             for a, b in zip(rows, reference)]
        write_private_json(args.output, {
            "schema": "dsv4-sampling-replay-v1", "model": args.model,
            "matches": matches, "records": records,
        })

    sequential = [complete(case) for case in cases]
    record("sequential", sequential)
    record("sequential_replay", [complete(case) for case in cases], sequential)
    for name, batch, reference in (
        ("concurrent", cases, sequential),
        ("concurrent_reordered", cases[::-1], sequential[::-1]),
        ("concurrent_same_seed", [cases[1]] * 4, [sequential[1]] * 4),
    ):
        with ThreadPoolExecutor(max_workers=4) as pool:
            record(name, list(pool.map(complete, batch)), reference)
    record("post_concurrency", [complete(case) for case in cases], sequential)
    print(dumps({"matches": matches}, indent=2), flush=True)
    return 0 if all(all(rows) for rows in matches.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
