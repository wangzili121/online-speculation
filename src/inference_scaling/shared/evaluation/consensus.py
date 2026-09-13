"""Self-consistency rewards and deterministic consensus selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction

from inference_scaling.shared.evaluation.numeric import extract_numeric_answer
from inference_scaling.shared.types import TokenSequence


def modal_answer(answers: Sequence[Fraction | None]) -> Fraction | None:
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return None
    maximum = max(counts.values())
    # Fraction has a total order.  This makes ties independent of request order.
    return min(answer for answer, count in counts.items() if count == maximum)


def consensus_index(texts: Sequence[str], logprobs: Sequence[float]) -> int:
    """Choose a completion matching the modal answer, then the most likely one."""

    if not texts or len(texts) != len(logprobs):
        raise ValueError("texts and logprobs must be non-empty and equally sized")
    answers = [extract_numeric_answer(text) for text in texts]
    mode = modal_answer(answers)
    eligible = [index for index, answer in enumerate(answers) if answer == mode]
    if mode is None:
        eligible = list(range(len(texts)))
    return max(eligible, key=lambda index: (logprobs[index], -index))


class CumulativeConsensusReward:
    """Self-consistency reward accumulated across guidance steps."""

    def __init__(self, decoder: Callable[[TokenSequence], str]) -> None:
        self._decoder = decoder
        self._counts: Counter[Fraction] = Counter()

    @property
    def counts(self) -> Counter[Fraction]:
        return self._counts.copy()

    def __call__(
        self,
        _prompt: TokenSequence,
        generated: Sequence[TokenSequence],
    ) -> tuple[float, ...]:
        answers = [extract_numeric_answer(self._decoder(tokens)) for tokens in generated]
        self._counts.update(answer for answer in answers if answer is not None)
        if not self._counts:
            return (0.0,) * len(answers)
        maximum = max(self._counts.values())
        mode = min(answer for answer, count in self._counts.items() if count == maximum)
        return tuple(1.0 if answer == mode else 0.0 for answer in answers)
