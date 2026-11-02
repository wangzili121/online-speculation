#!/usr/bin/env python3
"""Tokenize local text/chat JSONL or Parquet for Qwen draft distillation."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def tokenize(row, tokenizer):
    if not isinstance(row, dict):
        raise ValueError("each source row must be an object")
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
    if not isinstance(row.get("text"), str):
        raise ValueError("source rows require text or messages")
    return tokenizer.encode(row["text"], add_special_tokens=False)


def prepare(source, output, tokenizer, *, min_tokens, max_tokens, max_records):
    selected, skipped = [], 0
    with output.open("x", encoding="utf-8") as stream:
        try:
            for index, row in enumerate(rows(source)):
                ids = tokenize(row, tokenizer)[:max_tokens]
                if len(ids) < min_tokens:
                    skipped += 1
                    continue
                stream.write(json.dumps(dict(input_ids=ids, source_row=index)) + "\n")
                selected.append(dict(source_row=index, tokens=len(ids)))
                if max_records and len(selected) >= max_records:
                    break
        except BaseException:
            output.unlink()
            raise
    if not selected:
        output.unlink()
        raise ValueError("no source rows meet min_tokens")
    return dict(records=selected, skipped_short=skipped, source_sha256=sha256(source),
                tokens_sha256=sha256(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="dataset ID, revision, config and split for provenance")
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=0, help="0 processes the complete file")
    args = parser.parse_args()
    if not 0 < args.min_tokens <= args.max_tokens or args.max_records < 0:
        parser.error("require 0 < min_tokens <= max_tokens and max_records >= 0")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args.input, args.output, tokenizer, min_tokens=args.min_tokens,
                       max_tokens=args.max_tokens, max_records=args.max_records)
    manifest.update(source=args.source, model=str(args.model), min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens, chat_template_sha256=hashlib.sha256(
                        json.dumps(tokenizer.chat_template, sort_keys=True).encode()).hexdigest())
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest | {"records": len(manifest["records"])}), flush=True)


if __name__ == "__main__":
    main()
