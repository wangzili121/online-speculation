#!/usr/bin/env python3
"""Download, verify and deterministically partition GSM8K for DSV4.1 studies."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from urllib.request import urlopen


ROOT = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
SOURCES = {
    "train": (f"{ROOT}/train.jsonl", "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
    "test": (f"{ROOT}/test.jsonl", "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def acquire(name, directory, source_directory=None):
    url, expected = SOURCES[name]
    path = directory / f"{name}.jsonl"
    if source_directory is not None:
        data = (source_directory / f"{name}.jsonl").read_bytes()
    elif path.exists() and sha256(path) == expected:
        return path
    else:
        with urlopen(url, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"GSM8K {name} SHA256 does not match the pinned public file")
    path.write_bytes(data)
    return path


def records(path, split):
    result = []
    with path.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if set(row) != {"question", "answer"} or not all(isinstance(row[key], str) for key in row):
                raise ValueError(f"invalid GSM8K row {split}/{index}")
            identity = hashlib.sha256(f"{split}\0{index}\0{row['question']}".encode()).hexdigest()[:20]
            result.append({
                "id": f"gsm8k-{split}-{identity}",
                "source_split": split,
                "source_index": index,
                "question": row["question"],
                "reference_answer": row["answer"],
                "prompt": (
                    "Solve the following grade-school math problem. Show the reasoning, then end "
                    "with a line in the form #### <numeric answer>.\n\nQuestion: "
                    f"{row['question']}\n\nAnswer:"
                ),
            })
    return result


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path,
                        help="Use already downloaded train.jsonl/test.jsonl instead of network access")
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir()
    train_path = acquire("train", raw, args.source_directory)
    test_path = acquire("test", raw, args.source_directory)
    train, test = records(train_path, "train"), records(test_path, "test")
    random.Random(args.seed).shuffle(train)
    partitions = {
        "warmup": train[:128],
        "tuning": train[128:256],
        "online_eval": train[256:640],
        "quality_test": test,
    }
    identities = []
    for name, rows in partitions.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
        identities.extend(row["id"] for row in rows)
    if len(identities) != len(set(identities)):
        raise RuntimeError("GSM8K partitions overlap")
    manifest = {
        "format": "orthrus-dsv41-gsm8k-v1",
        "seed": args.seed,
        "source": {name: {"url": url, "sha256": digest} for name, (url, digest) in SOURCES.items()},
        "partitions": {
            name: {"rows": len(rows), "sha256": sha256(args.output / f"{name}.jsonl")}
            for name, rows in partitions.items()
        },
        "supervision": "Reference answers are retained for quality scoring only; online draft updates use target-model logits.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
