"""Public math benchmark selection and isolated, answer-only evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any

MATH500_REPOSITORY = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


@dataclass(frozen=True)
class MathProblem:
    identifier: str
    question: str
    answer: str
    subject: str
    level: int


def load_math500(path: Path, *, download: bool = False) -> list[MathProblem]:
    if download and not path.is_file():
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(MATH500_REPOSITORY, "test.jsonl", repo_type="dataset",
                                     revision=MATH500_REVISION, local_dir=path.parent)
        path = Path(downloaded)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = [MathProblem(row["unique_id"], row["problem"], row["answer"],
                            row["subject"], int(row["level"])) for row in rows]
    if len({problem.identifier for problem in problems}) != len(problems):
        raise ValueError("benchmark contains duplicate problem identifiers")
    return problems


def stratified_subset(problems: list[MathProblem], *, seed: int, minimum_level: int,
                      development_count: int, test_count: int,
                      excluded_ids: tuple[str, ...] | list[str] = ()) -> dict[str, list[MathProblem]]:
    """Round-robin subject/level strata, shuffled independently of answers."""
    if development_count < 0 or test_count <= 0:
        raise ValueError("invalid development/test sizes")
    groups: dict[tuple[str, int], list[MathProblem]] = defaultdict(list)
    for problem in sorted(problems, key=lambda item: item.identifier):
        if problem.level >= minimum_level and problem.identifier not in excluded_ids:
            groups[(problem.subject, problem.level)].append(problem)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    ordered = []
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    count = development_count + test_count
    if count > len(ordered):
        raise ValueError("requested subset exceeds the eligible benchmark")
    return {"development": ordered[:development_count],
            "test": ordered[development_count:count]}


def _judge_worker(connection) -> None:
    # A single persistent process supplies a real timeout on Windows and Unix.
    # Internal signal/closure-based library timeouts are disabled in this worker.
    from math_verify import parse, verify, LatexExtractionConfig

    @lru_cache(maxsize=4096)
    def parsed(text: str, reference: bool):
        if reference:
            text = "\\boxed{" + text + "}"
        return parse(text, extraction_config=[LatexExtractionConfig()],
                     fallback_mode="no_fallback", parsing_timeout=0)

    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            operation, values = request
            try:
                result: Any
                if operation == "grade":
                    prediction, reference = values
                    gold, answer = parsed(reference, True), parsed(prediction, False)
                    result = {"correct": bool(gold and answer and verify(gold, answer, timeout_seconds=0)),
                              "parseable": bool(answer), "reference_parseable": bool(gold)}
                elif operation == "equivalent":
                    left, right = (parsed(value, False) for value in values)
                    result = bool(left and right and verify(left, right, timeout_seconds=0)
                                  and verify(right, left, timeout_seconds=0))
                elif operation == "parse":
                    value = parsed(values[0], False)
                    result = None if not value else str(value[0])
                else:
                    raise ValueError(f"unknown judge operation: {operation}")
                connection.send((True, result))
            except Exception as error:
                connection.send((False, f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class MathJudge:
    """Math-Verify for evaluation and reference-free answer equivalence."""

    def __init__(self, timeout_seconds: float = 20):
        self.timeout_seconds = timeout_seconds
        self._process: Any = None
        self._connection: Any = None

    def _request(self, operation: str, *values: str) -> Any:
        if self._process is None:
            context = mp.get_context("spawn")
            self._connection, child = context.Pipe()
            self._process = context.Process(target=_judge_worker, args=(child,), daemon=True)
            self._process.start()
            child.close()
        self._connection.send((operation, values))
        if not self._connection.poll(self.timeout_seconds):
            self.close()
            raise TimeoutError("math answer evaluation timed out")
        success, result = self._connection.recv()
        if not success:
            raise ValueError(result)
        return result

    def grade(self, prediction: str, reference: str) -> dict[str, bool]:
        return self._request("grade", prediction, reference)

    def equivalent(self, left: str, right: str) -> bool:
        return bool(self._request("equivalent", left, right))

    def answer_key(self, text: str) -> str | None:
        return self._request("parse", text)

    def close(self) -> None:
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5)
            self._process.close()
            self._connection.close()
            self._process = self._connection = None
