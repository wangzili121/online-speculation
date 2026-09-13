"""Pinned GSM8K split loader and exact numeric scorer.

Both public splits are downloaded from OpenAI's archived repository and
checked byte-for-byte.  Training code can therefore prove that it only saw
the public training split, while evaluation code defaults to the test split.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Literal, Sequence

from inference_scaling.shared.evaluation.numeric import extract_numeric_answer

GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/train.jsonl"
)
GSM8K_TRAIN_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
GSM8K_TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"

GSM8K_PROMPT_SUFFIX = (
    "\n\nSolve the problem step by step. End your response with exactly "
    "`#### <number>`, where <number> is the final numeric answer."
)

GSM8KSplit = Literal["train", "test"]
_SPLITS: dict[GSM8KSplit, tuple[str, str, int]] = {
    "train": (GSM8K_TRAIN_URL, GSM8K_TRAIN_SHA256, 7473),
    "test": (GSM8K_TEST_URL, GSM8K_TEST_SHA256, 1319),
}

@dataclass(frozen=True, slots=True)
class GSM8KProblem:
    index: int
    question: str
    gold_solution: str
    gold_answer: Fraction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_gsm8k(path: str | Path, *, split: GSM8KSplit = "test") -> Path:
    """Download an official split atomically and verify its checksum."""

    url, expected_sha256, _ = _SPLITS[split]
    destination = Path(path)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f"GSM8K {split} checksum mismatch: "
                f"expected {expected_sha256}, downloaded {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_gsm8k(
    path: str | Path,
    *,
    split: GSM8KSplit = "test",
    download: bool = True,
) -> tuple[GSM8KProblem, ...]:
    _, expected_sha256, expected_rows = _SPLITS[split]
    source = download_gsm8k(path, split=split) if download else Path(path)
    if _sha256(source) != expected_sha256:
        raise ValueError(f"{source} is not the pinned official GSM8K {split} file")
    problems: list[GSM8KProblem] = []
    with source.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            answer = extract_numeric_answer(str(item["answer"]))
            if answer is None:
                raise ValueError(f"could not parse GSM8K gold answer at row {index}")
            problems.append(
                GSM8KProblem(
                    index=index,
                    question=str(item["question"]),
                    gold_solution=str(item["answer"]),
                    gold_answer=answer,
                )
            )
    if len(problems) != expected_rows:
        raise ValueError(
            f"expected {expected_rows:,} GSM8K {split} rows, found {len(problems):,}"
        )
    return tuple(problems)


def gsm8k_prompt(question: str) -> str:
    """Return the single prompt text shared by training and every baseline."""

    return question + GSM8K_PROMPT_SUFFIX


def select_problems(
    problems: Sequence[GSM8KProblem],
    count: int | None,
    *,
    seed: int,
) -> tuple[GSM8KProblem, ...]:
    """Choose a preregistered random subset and retain dataset order.

    Sampling indices before model inference prevents result-dependent cherry
    picking.  ``count=None`` selects the complete public test split.
    """

    if count is None or count == len(problems):
        return tuple(problems)
    if count <= 0 or count > len(problems):
        raise ValueError(f"count must lie in [1, {len(problems)}]")
    indices = sorted(random.Random(seed).sample(range(len(problems)), count))
    return tuple(problems[index] for index in indices)


def accuracy(correct: Iterable[bool]) -> float:
    values = tuple(bool(value) for value in correct)
    if not values:
        raise ValueError("accuracy requires at least one result")
    return sum(values) / len(values)
