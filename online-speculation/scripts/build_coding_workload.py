"""Freeze the Orthrus high-concurrency mixed-length coding workload."""

import argparse

from transformers import AutoTokenizer

from blockspec import reporting as report
from blockspec.vllm_ascend.workload import (build_records, collect_code,
                                            write_manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    buckets = ((6144, 10240, 0.50), (24576, 36863, 0.35),
               (36864, 39936, 0.15))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source, files = collect_code(args.source)
    records = build_records(
        tokenizer, source, count=args.count, buckets=buckets,
        max_new_tokens=args.max_new_tokens, seed=args.seed)
    digest = write_manifest(args.output, records)
    lengths = sorted(record["actual_prompt_tokens"] for record in records)
    print(report.dumps({
        "requests": len(records),
        "source_files": files,
        "manifest_sha256": digest,
        "prompt_tokens": {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int((len(lengths) - 1) * 0.95)],
            "max": lengths[-1],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
