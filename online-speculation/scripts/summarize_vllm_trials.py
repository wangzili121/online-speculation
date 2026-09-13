"""Summarize and rank completed Orthrus vLLM trials."""

import argparse

from blockspec.vllm_ascend.results import (format_trials, load_trials, rank_trials,
                                           write_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--require-no-preemption", action="store_true")
    args = parser.parse_args()
    rows = load_trials(args.root)
    ranked = rank_trials(rows, require_no_preemption=args.require_no_preemption)
    if args.output_csv:
        write_csv(args.output_csv, ranked)
    print(format_trials(ranked))


if __name__ == "__main__":
    main()
