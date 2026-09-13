"""Derive a shorter-output tuning manifest without changing any prompt."""

import argparse

from blockspec import reporting as report
from blockspec.vllm_ascend.workload import read_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    records = read_manifest(args.input)
    for record in records:
        record["max_tokens"] = args.max_new_tokens
    digest = write_manifest(args.output, records)
    print(report.dumps({"requests": len(records),
                        "max_new_tokens": args.max_new_tokens,
                        "manifest_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
