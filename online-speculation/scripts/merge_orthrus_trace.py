"""Convert Orthrus JSONL stage events into a Chrome/Perfetto trace."""

import argparse

from blockspec.vllm_ascend.trace import merge_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"wrote {merge_events(args.input, args.output)} trace events")


if __name__ == "__main__":
    main()
