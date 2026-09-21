#!/usr/bin/env python3
"""Prepare packed offline or completed-traffic data for DSV4 Orthrus fitting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blockspec.dsv4_data import (  # noqa: E402
    PackedTokenWriter,
    file_sha256,
    prepare_records,
)
from blockspec import reporting as report  # noqa: E402


class JsonlRows:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


def _tokenizer(
    path: Path, offline: bool, tokenizer_mode: str, chat_template: Path | None
):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("data preparation requires the dsv4-train dependencies") from error
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=offline,
        trust_remote_code=False,
        use_fast=True,
    )
    if chat_template is not None:
        tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
    if tokenizer_mode == "auto":
        tokenizer_mode = "hf" if tokenizer.chat_template else "deepseek_v4"
    if tokenizer_mode == "deepseek_v4":
        try:
            from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
        except ImportError as error:
            raise SystemExit(
                "deepseek_v4 tokenization requires the pinned vLLM image"
            ) from error
        tokenizer = get_deepseek_v4_tokenizer(tokenizer)
    elif not tokenizer.chat_template:
        raise SystemExit("HF tokenizer mode requires --chat-template or tokenizer metadata")
    return tokenizer, tokenizer_mode


def _paper_sources(args):
    source = Path(args.source).expanduser()
    metadata = {"source": args.source, "requested_revision": args.revision}
    if source.is_dir() and all((source / f"{name}.jsonl").is_file() for name in args.domains):
        datasets = {name: JsonlRows(source / f"{name}.jsonl") for name in args.domains}
        metadata["source_type"] = "domain_jsonl"
        metadata["source_files"] = {
            name: file_sha256(source / f"{name}.jsonl") for name in args.domains
        }
        return datasets, metadata
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as error:
        raise SystemExit("paper data requires the Hugging Face datasets package") from error
    if source.is_dir():
        loaded = load_from_disk(str(source))
        if not isinstance(loaded, DatasetDict):
            raise SystemExit("a saved paper dataset must contain chat/math/code splits")
        datasets = {name: loaded[name] for name in args.domains}
        metadata["source_type"] = "datasets_save_to_disk"
    else:
        if args.revision is None:
            raise SystemExit("remote paper datasets require an immutable --revision")
        options = {
            "revision": args.revision,
            "download_mode": "reuse_dataset_if_exists",
        }
        if args.offline:
            options["download_mode"] = "reuse_cache_if_exists"
        datasets = {}
        for name in args.domains:
            positional = [args.source]
            if args.dataset_config:
                positional.append(args.dataset_config)
            datasets[name] = load_dataset(*positional, split=name, **options)
        metadata["source_type"] = "huggingface_dataset"
    metadata["fingerprints"] = {
        name: getattr(dataset, "_fingerprint", None)
        for name, dataset in datasets.items()
    }
    return datasets, metadata


def _sample_indices(length: int, count: int, *, seed: int, allow_repeat: bool):
    if length < 1:
        raise SystemExit("paper dataset contains an empty domain")
    if count > length and not allow_repeat:
        raise SystemExit(
            f"requested {count} rows from a domain with {length}; lower "
            "--examples-per-domain or pass --allow-repeat explicitly"
        )
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        cycle = list(range(length))
        rng.shuffle(cycle)
        result.extend(cycle[: count - len(result)])
    return result


def _paper_records(args, datasets):
    indices = {
        name: _sample_indices(
            len(datasets[name]),
            args.examples_per_domain,
            seed=args.seed + index,
            allow_repeat=args.allow_repeat,
        )
        for index, name in enumerate(args.domains)
    }
    for row in range(args.examples_per_domain):
        for name in args.domains:
            yield name, datasets[name][indices[name][row]]


def _online_records(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON on input line {line_number}") from error
            domain = str(record.get("domain") or "online") if isinstance(record, dict) else "online"
            yield domain, record


def _build(args, records, tokenizer, source_metadata: dict) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    writers = {
        split: PackedTokenWriter(
            temporary / f"{split}.tokens.jsonl",
            sequence_length=args.sequence_length,
            block_size=args.block_size,
        )
        for split in ("train", "validation", "test")
    }
    try:
        counts = prepare_records(
            records,
            writers,
            tokenizer=tokenizer,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            online=args.command == "online",
            all_token_anchors=getattr(args, "all_token_anchors", False),
        )
    finally:
        files = {split: writer.close() for split, writer in writers.items()}
    for split, details in files.items():
        path = temporary / details["file"]
        details["sha256"] = file_sha256(path)
        if split == "train" and details["records"] == 0:
            raise SystemExit("no complete training sequence with a valid K-token block")
    manifest = {
        "format": "orthrus-dsv4-packed-v1",
        "mode": args.command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_length": args.sequence_length,
        "block_size": args.block_size,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "tokenizer": str(args.tokenizer.resolve()) if args.tokenizer else None,
        "source": source_metadata,
        "counts": counts,
        "files": files,
    }
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
    (temporary / "manifest.json").write_text(
        encoder.encode(manifest) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    paper = commands.add_parser("paper", help="balanced Nemotron chat/math/code data")
    paper.add_argument(
        "--source", default="nvidia/Nemotron-Post-Training-Dataset-v2"
    )
    paper.add_argument("--dataset-config")
    paper.add_argument("--revision")
    paper.add_argument("--domains", nargs=3, default=("chat", "math", "code"))
    paper.add_argument("--examples-per-domain", type=int, default=200_000)
    paper.add_argument("--allow-repeat", action="store_true")
    paper.add_argument(
        "--all-token-anchors",
        action="store_true",
        help="debug only: allow anchors outside assistant responses",
    )
    paper.add_argument("--offline", action="store_true")
    paper.add_argument("--tokenizer", type=Path, required=True)
    paper.add_argument(
        "--tokenizer-mode",
        choices=("auto", "hf", "deepseek_v4"),
        default="auto",
    )
    paper.add_argument("--chat-template", type=Path)
    paper.add_argument("--output", type=Path, required=True)
    paper.add_argument("--sequence-length", type=int, default=2048)
    paper.add_argument("--block-size", type=int, default=32)
    paper.add_argument("--seed", type=int, default=42)
    paper.add_argument("--validation-fraction", type=float, default=0.01)
    paper.add_argument("--test-fraction", type=float, default=0.01)

    online = commands.add_parser("online", help="completed production requests")
    online.add_argument("--input", type=Path, required=True)
    online.add_argument("--tokenizer", type=Path)
    online.add_argument(
        "--tokenizer-mode",
        choices=("auto", "hf", "deepseek_v4"),
        default="auto",
    )
    online.add_argument("--chat-template", type=Path)
    online.add_argument("--output", type=Path, required=True)
    online.add_argument("--sequence-length", type=int, default=2048)
    online.add_argument("--block-size", type=int, default=32)
    online.add_argument("--seed", type=int, default=42)
    online.add_argument("--validation-fraction", type=float, default=0.1)
    online.add_argument("--test-fraction", type=float, default=0.0)
    return value


def main() -> int:
    args = parser().parse_args()
    if (
        args.sequence_length < args.block_size
        or args.block_size < 2
        or args.seed < 0
        or not 0 <= args.validation_fraction < 1
        or not 0 <= args.test_fraction < 1
        or args.validation_fraction + args.test_fraction >= 1
    ):
        raise SystemExit("invalid packing dimensions, seed, or split fractions")
    if args.command == "paper" and args.examples_per_domain < 1:
        raise SystemExit("--examples-per-domain must be positive")
    if getattr(args, "offline", False):
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = None
    resolved_tokenizer_mode = None
    if args.tokenizer:
        tokenizer, resolved_tokenizer_mode = _tokenizer(
            args.tokenizer,
            getattr(args, "offline", False),
            args.tokenizer_mode,
            args.chat_template,
        )
    if args.command == "paper":
        datasets, source = _paper_sources(args)
        source["available_rows"] = {name: len(rows) for name, rows in datasets.items()}
        source["examples_per_domain"] = args.examples_per_domain
        source["allow_repeat"] = args.allow_repeat
        source["assistant_only"] = not args.all_token_anchors
        source["tokenizer_mode"] = resolved_tokenizer_mode
        source["chat_template_sha256"] = (
            file_sha256(args.chat_template) if args.chat_template else None
        )
        records = _paper_records(args, datasets)
    else:
        source = {
            "input": str(args.input.resolve()),
            "sha256": file_sha256(args.input),
            "tokenizer_mode": resolved_tokenizer_mode,
            "chat_template_sha256": (
                file_sha256(args.chat_template) if args.chat_template else None
            ),
        }
        records = _online_records(args.input)
    manifest = _build(args, records, tokenizer, source)
    print(report.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
