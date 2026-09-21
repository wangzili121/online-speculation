"""Deterministic packed data for DSV4 Orthrus offline and online fitting."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any


SUCCESS_STATUSES = {"complete", "completed", "finished", "ok", "success", "succeeded"}


def split_for_group(
    group: str,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    if (
        not isinstance(group, str)
        or not group
        or not 0 <= validation_fraction < 1
        or not 0 <= test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("invalid split group or fractions")
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big"
    ) / 2**64
    train_fraction = 1 - validation_fraction - test_fraction
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def content_group(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _token_ids(value: Any, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        raise ValueError(f"{name} must be a nonempty list of nonnegative token IDs")
    return value


def _ranges(value: Any, token_count: int) -> list[tuple[int, int]]:
    if value is None:
        return [(0, token_count)]
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or type(span[0]) is not int
            or type(span[1]) is not int
            or not 0 <= span[0] < span[1] <= token_count
            for span in value
        )
    ):
        raise ValueError("anchor ranges must be nonempty half-open token spans")
    spans = sorted((int(start), int(stop)) for start, stop in value)
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("anchor ranges must not overlap")
    return spans


class PackedTokenWriter:
    """Stream samples into fixed-length rows without crossing sample boundaries."""

    def __init__(self, path: Path, *, sequence_length: int, block_size: int):
        if sequence_length < block_size or block_size < 2:
            raise ValueError("sequence length must contain a complete draft block")
        self.path = Path(path)
        self.sequence_length = int(sequence_length)
        self.block_size = int(block_size)
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self._tokens: list[int] = []
        self._ranges: list[list[int]] = []
        self.source_samples = 0
        self.source_tokens = 0
        self.supervised_tokens = 0
        self.records = 0
        self.records_without_blocks = 0
        self.trailing_tokens = 0
        self.closed = False

    def add(self, input_ids: list[int], anchor_ranges=None) -> None:
        if self.closed:
            raise RuntimeError("packed writer is closed")
        tokens = _token_ids(input_ids, "input_ids")
        spans = _ranges(anchor_ranges, len(tokens))
        self.source_samples += 1
        self.source_tokens += len(tokens)
        self.supervised_tokens += sum(stop - start for start, stop in spans)
        offset = 0
        while offset < len(tokens):
            output_start = len(self._tokens)
            take = min(self.sequence_length - output_start, len(tokens) - offset)
            stop = offset + take
            self._tokens.extend(tokens[offset:stop])
            for span_start, span_stop in spans:
                left, right = max(span_start, offset), min(span_stop, stop)
                if right - left >= self.block_size:
                    mapped = output_start + left - offset
                    self._ranges.append([mapped, mapped + right - left])
            offset = stop
            if len(self._tokens) == self.sequence_length:
                self._flush()

    def _flush(self) -> None:
        if self._ranges:
            row = {"input_ids": self._tokens, "anchor_ranges": self._ranges}
            self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.records += 1
        else:
            self.records_without_blocks += 1
        self._tokens, self._ranges = [], []

    def close(self) -> dict[str, int | str]:
        if not self.closed:
            self.trailing_tokens = len(self._tokens)
            self._handle.close()
            self.closed = True
        return {
            "file": self.path.name,
            "source_samples": self.source_samples,
            "source_tokens": self.source_tokens,
            "supervised_tokens": self.supervised_tokens,
            "records": self.records,
            "records_without_blocks": self.records_without_blocks,
            "trailing_tokens_dropped": self.trailing_tokens,
        }


def tokenize_messages(tokenizer, messages: Any, *, add_generation_prompt: bool) -> list[int]:
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        )
    ):
        raise ValueError("messages must contain role/content strings")
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(tokens, dict):
        tokens = tokens.get("input_ids")
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], list):
        tokens = tokens[0]
    return _token_ids(tokens, "chat-template output")


def assistant_ranges(tokenizer, messages: list[dict], tokens: list[int]) -> list[tuple[int, int]]:
    """Locate assistant turns using exact chat-template prefix boundaries."""
    ranges = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = tokenize_messages(
            tokenizer, messages[:index], add_generation_prompt=True
        )
        through = tokenize_messages(
            tokenizer, messages[: index + 1], add_generation_prompt=False
        )
        if through[: len(before)] != before or tokens[: len(through)] != through:
            raise ValueError("chat template does not expose stable assistant boundaries")
        if len(through) > len(before):
            ranges.append((len(before), len(through)))
    if not ranges:
        raise ValueError("conversation contains no assistant token span")
    return ranges


def paper_sample(
    record: dict, tokenizer, *, all_token_anchors: bool = False
) -> tuple[list[int], list[tuple[int, int]], str]:
    messages = record.get("messages")
    tokens = tokenize_messages(tokenizer, messages, add_generation_prompt=False)
    ranges = (
        [(0, len(tokens))]
        if all_token_anchors
        else assistant_ranges(tokenizer, messages, tokens)
    )
    group = str(record.get("id") or record.get("uuid") or content_group(messages))
    return tokens, ranges, group


def online_sample(
    record: dict, tokenizer=None
) -> tuple[list[int], list[tuple[int, int]], str]:
    if not isinstance(record, dict):
        raise ValueError("each online row must be an object")
    status = record.get("status")
    if status is not None and str(status).lower() not in SUCCESS_STATUSES:
        raise ValueError("request is not complete")
    if record.get("finished") is False or record.get("error"):
        raise ValueError("request is not complete")

    prompt = record.get("prompt_token_ids")
    output = record.get("output_token_ids")
    if prompt is None and output is None:
        prompt, output = record.get("input_ids"), record.get("output_ids")
    if prompt is not None or output is not None:
        prompt = _token_ids(prompt, "prompt_token_ids")
        output = _token_ids(output, "output_token_ids")
        tokens = prompt + output
        output_start = len(prompt)
    else:
        if tokenizer is None:
            raise ValueError("text traffic requires --tokenizer")
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or messages[-1].get("role") != "assistant"
        ):
            raise ValueError("completed messages must end with an assistant response")
        prompt = tokenize_messages(
            tokenizer, messages[:-1], add_generation_prompt=True
        )
        tokens = tokenize_messages(tokenizer, messages, add_generation_prompt=False)
        if tokens[: len(prompt)] != prompt:
            raise ValueError("chat template does not preserve the prompt prefix")
        output_start = len(prompt)
    anchor_start = max(0, output_start - 1)
    group = str(
        record.get("request_id")
        or record.get("id")
        or content_group({"tokens": tokens, "output_start": output_start})
    )
    return tokens, [(anchor_start, len(tokens))], group


def prepare_records(
    records: Iterable[tuple[str, dict]],
    writers: dict[str, PackedTokenWriter],
    *,
    tokenizer,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    online: bool,
    all_token_anchors: bool = False,
) -> dict:
    counts = {"accepted": 0, "rejected": 0, "domains": {}, "reasons": {}}
    for domain, record in records:
        try:
            if online:
                tokens, ranges, group = online_sample(record, tokenizer)
            else:
                tokens, ranges, group = paper_sample(
                    record, tokenizer, all_token_anchors=all_token_anchors
                )
            if not any(
                stop - start >= writers["train"].block_size
                for start, stop in ranges
            ):
                raise ValueError("sample has no complete draft block")
            split = split_for_group(
                group,
                seed=seed,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
            )
            writers[split].add(tokens, ranges)
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error)
            counts["rejected"] += 1
            counts["reasons"][reason] = counts["reasons"].get(reason, 0) + 1
            continue
        counts["accepted"] += 1
        counts["domains"][domain] = counts["domains"].get(domain, 0) + 1
    return counts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
