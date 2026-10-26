"""Create a deterministic tokenized corpus for a training-path smoke test."""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


TRAINING_TEXTS = (
    "Speculative decoding proposes several future tokens and verifies them with the target model.",
    "A bidirectional draft block shares the frozen transformer backbone with the autoregressive model.",
    "The teacher distribution supervises the draft attention while the shared parameters remain frozen.",
    "Attention maps queries to weighted values by comparing each query with the visible keys.",
    "The draft model reads committed history and predicts several masked positions in one forward pass.",
    "Only the independent draft attention projections are updated by the distillation objective.",
    "A verifier accepts a matching prefix and discards candidates after the first rejected token.",
    "Training and validation records must remain disjoint so that loss reduction can be checked honestly.",
)

VALIDATION_TEXTS = (
    "The frozen autoregressive path provides soft token probabilities for every future position.",
    "The special block mask exposes history and the current draft block without exposing answer tokens.",
    "A successful smoke test requires finite gradients and unchanged frozen model parameters.",
    "The final deployment quantizes the trained draft head only after floating point optimization.",
)


def encode(tokenizer, text, minimum_length):
    repeated = (text + "\n") * (minimum_length // 4 + 2)
    tokens = tokenizer.encode(repeated).ids
    if len(tokens) < minimum_length:
        raise RuntimeError("tokenizer produced an unexpectedly short record")
    return tokens[:minimum_length]


def write_split(path, tokenizer, texts, sequence_length):
    rows = [json.dumps({"input_ids": encode(tokenizer, text, sequence_length)}, separators=(",", ":"))
            for text in texts]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()
    if args.sequence_length < 32:
        parser.error("sequence length must be at least 32")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    write_split(args.output / "train.jsonl", tokenizer, TRAINING_TEXTS, args.sequence_length)
    write_split(args.output / "validation.jsonl", tokenizer, VALIDATION_TEXTS, args.sequence_length)


if __name__ == "__main__":
    main()
