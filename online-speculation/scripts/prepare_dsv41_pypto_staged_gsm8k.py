#!/usr/bin/env python3
"""Tokenize prepared GSM8K partitions for the 1122 resident serving runner."""

import argparse
from itertools import islice
import json
from pathlib import Path


def tokenize_workload(source, tokenizer, *, limit, new_tokens, context, dp_size=4):
    rows = []
    with source.open() as stream:
        for index, line in enumerate(islice(stream, limit)):
            sample = json.loads(line)
            ids = tokenizer.encode(sample["prompt"]).ids
            if not ids or len(ids) + new_tokens > context:
                raise ValueError(f"Prompt {sample['id']} and its generation budget exceed context {context}")
            rows.append(dict(id=sample["id"], dp=index % dp_size, input_ids=ids, new_tokens=new_tokens))
    if len(rows) != limit:
        raise ValueError(f"{source} contains {len(rows)} entries, expected at least {limit}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Output of prepare_dsv41_gsm8k.py")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--warmup-limit", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=512)
    args = parser.parse_args()
    if (min(args.train_limit, args.eval_limit, args.warmup_limit, args.new_tokens, args.warmup_tokens) < 1
            or args.context <= max(args.new_tokens, args.warmup_tokens)):
        parser.error("Use positive split sizes and a generation budget smaller than context")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    splits = {
        "online": ("tuning", args.train_limit, args.new_tokens),
        "heldout": ("quality_test", args.eval_limit, args.new_tokens),
        "warmup": ("warmup", args.warmup_limit, args.warmup_tokens),
    }
    workloads = {
        name: tokenize_workload(
            args.dataset / f"{split}.jsonl", tokenizer, limit=limit, new_tokens=tokens, context=args.context,
        )
        for name, (split, limit, tokens) in splits.items()
    }
    seen = set()
    for name, workload in workloads.items():
        prompts = {tuple(item["input_ids"]) for item in workload}
        if prompts & seen:
            raise ValueError(f"GSM8K {name} prompts overlap another partition after tokenization")
        seen |= prompts
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in workloads.items():
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(rows) + "\n")
        print(f"{path}  {len(rows)} prompts", flush=True)


if __name__ == "__main__":
    main()
