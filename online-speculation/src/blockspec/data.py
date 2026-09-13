"""Explicit JSONL data contract: one independent sequence per record, no packing."""

import json
import hashlib
from pathlib import Path

import torch


def assert_split_files_disjoint(training_path, validation_path):
    """Reject question-group overlap and exact record overlap before fitting."""
    def identities(path):
        keys = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("group_sha256"):
                    keys.add("group:" + record["group_sha256"])
                content = record.get("input_ids", record.get("text"))
                if content is not None:
                    digest = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
                    keys.add("content:" + digest)
        return keys
    if identities(training_path).intersection(identities(validation_path)):
        raise ValueError("training and validation overlap by question or exact content")


def load_sequences(path, vocab_size, *, tokenizer=None):
    sequences = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "input_ids" in record:
                ids = record["input_ids"]
            elif "text" in record and tokenizer is not None:
                ids = tokenizer.encode(record["text"], add_special_tokens=False)
            else:
                raise ValueError(f"line {line_number}: input_ids or text + tokenizer required")
            if not isinstance(ids, list) or not all(type(t) is int and 0 <= t < vocab_size for t in ids):
                raise ValueError(f"line {line_number}: invalid vocabulary ids")
            if len(ids) < 2:
                raise ValueError(f"line {line_number}: sequence is too short")
            sequences.append(torch.tensor(ids, dtype=torch.long))
    if not sequences:
        raise ValueError("empty training data")
    return sequences


def sample_batch(sequences, *, batch_size, length, bos_id, device, generator):
    """Random contiguous crop + an explicit clean BOS; never concatenate records."""
    eligible = [x for x in sequences if len(x) >= length - 1]
    if not eligible or batch_size < 1 or length < 2:
        raise ValueError("no sufficiently long sequences or invalid batch dimensions")
    rows = []
    for _ in range(batch_size):
        index = int(torch.randint(len(eligible), (), generator=generator))
        sequence = eligible[index]
        start = int(torch.randint(len(sequence) - length + 2, (), generator=generator))
        rows.append(torch.cat((torch.tensor([bos_id]), sequence[start:start + length - 1])))
    return torch.stack(rows).to(device)
