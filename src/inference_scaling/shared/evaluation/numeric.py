"""Dataset-independent parsing and reference rewards for numeric tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from inference_scaling.shared.verifier import VerifierContext

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
_FRACTION = rf"(?:{_NUMBER})\s*/\s*(?:{_NUMBER})"
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(" + _FRACTION + "|" + _NUMBER + r")\s*\}")
_HASH_RE = re.compile(r"####\s*(" + _FRACTION + "|" + _NUMBER + r")")
_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|answer)\s*(?:is|:|=)?\s*\$?\s*("
    + _FRACTION
    + "|"
    + _NUMBER
    + r")",
    flags=re.IGNORECASE,
)
_ANY_NUMBER_RE = re.compile(_FRACTION + "|" + _NUMBER)


def _as_fraction(value: str) -> Fraction | None:
    cleaned = value.strip().replace(",", "").replace("$", "")
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", 1)
            return Fraction(Decimal(numerator.strip())) / Fraction(
                Decimal(denominator.strip())
            )
        return Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def extract_numeric_answer(text: str) -> Fraction | None:
    """Extract a final numeric value without executing generated text."""

    for pattern in (_HASH_RE, _BOXED_RE, _ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            parsed = _as_fraction(matches[-1])
            if parsed is not None:
                return parsed
    matches = _ANY_NUMBER_RE.findall(text)
    return _as_fraction(matches[-1]) if matches else None


@dataclass(frozen=True, slots=True)
class NumericReferenceVerifier:
    """Configurable reward against one bound numeric reference value."""

    expected: Fraction
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    unparseable_reward: float = 0.0

    def score(self, _prompt: str, completion: str) -> float:
        predicted = extract_numeric_answer(completion)
        if predicted is None:
            return self.unparseable_reward
        return self.correct_reward if predicted == self.expected else self.incorrect_reward


def build_numeric_reference_verifier(
    *,
    context: VerifierContext,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    unparseable_reward: float = 0.0,
) -> NumericReferenceVerifier:
    """Factory used by ``[verifier]`` Python-provider configuration."""

    if context.reference is None:
        raise ValueError("numeric reference verifier requires context.reference")
    expected = extract_numeric_answer(f"#### {context.reference}")
    if expected is None:
        raise ValueError(f"could not parse numeric reference {context.reference!r}")
    return NumericReferenceVerifier(
        expected=expected,
        correct_reward=float(correct_reward),
        incorrect_reward=float(incorrect_reward),
        unparseable_reward=float(unparseable_reward),
    )


__all__ = [
    "NumericReferenceVerifier",
    "build_numeric_reference_verifier",
    "extract_numeric_answer",
]
