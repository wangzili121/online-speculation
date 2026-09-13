"""Model-independent preference-pair construction from verifier scores."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PreferencePair:
    chosen: str
    rejected: str
    chosen_source: str
    rejected_source: str


def select_scored_preference_pair(
    *,
    candidate_texts: Sequence[str],
    candidate_rewards: Sequence[float],
    reference_text: str | None = None,
    reference_reward: float | None = None,
) -> PreferencePair | None:
    """Select the highest- and lowest-reward distinct completions.

    A dataset-provided solution may be included as one more scored completion;
    it is never assumed to be preferred without evaluation by the configured
    verifier. Equal rewards contain no preference information and return
    ``None``.
    """

    if not candidate_texts:
        raise ValueError("candidate_texts must be non-empty")
    if len(candidate_texts) != len(candidate_rewards):
        raise ValueError("candidate texts and rewards have different lengths")
    if (reference_text is None) != (reference_reward is None):
        raise ValueError("reference text and reward must be supplied together")
    rewards = tuple(float(value) for value in candidate_rewards)
    if any(not isfinite(value) for value in rewards):
        raise ValueError("preference rewards must be finite")

    entries = [
        (text, reward, f"candidate:{index}")
        for index, (text, reward) in enumerate(
            zip(candidate_texts, rewards, strict=True)
        )
    ]
    if reference_text is not None and reference_reward is not None:
        value = float(reference_reward)
        if not isfinite(value):
            raise ValueError("reference reward must be finite")
        entries.append((reference_text, value, "dataset_reference_completion"))

    minimum = min(entry[1] for entry in entries)
    maximum = max(entry[1] for entry in entries)
    if minimum == maximum:
        return None
    chosen = next(entry for entry in entries if entry[1] == maximum)
    rejected = next(entry for entry in entries if entry[1] == minimum)
    if chosen[0] == rejected[0]:
        raise RuntimeError("verifier assigned different rewards to identical texts")
    return PreferencePair(
        chosen=chosen[0],
        rejected=rejected[0],
        chosen_source=chosen[2],
        rejected_source=rejected[2],
    )


__all__ = ["PreferencePair", "select_scored_preference_pair"]
