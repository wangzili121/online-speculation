"""Execution profiles for real-model LLaDA experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping

LLaDAExecutionProfile = Literal["smoke", "full"]


def apply_execution_profile(
    config: Mapping[str, Any], profile: LLaDAExecutionProfile
) -> dict[str, Any]:
    """Return an isolated effective config for a smoke or full experiment.

    The smoke profile retains two complete diffusion blocks.  Consequently it
    exercises conditional rollouts and off-policy trajectory rescoring instead
    of reducing IS to terminal candidate selection.
    """

    if profile not in {"smoke", "full"}:
        raise ValueError(f"unknown LLaDA execution profile {profile!r}")
    effective = deepcopy(dict(config))
    if profile == "full":
        return effective

    effective["run"]["sample_count"] = 1
    effective["generation"].update(
        max_new_tokens=96,
        block_length=48,
        denoising_steps=4,
    )
    effective["exact_policy"].update(
        block_length=48,
        denoising_steps=4,
    )
    effective["search"].update(
        width=2,
        branching_factor=2,
        decision_block_size=48,
    )
    effective["best_of_n"]["samples"] = 2
    effective["mh"].update(
        decision_block_size=48,
        updates_per_stage=1,
        updates=2,
    )
    effective["conditional_is"].update(
        candidate_count=2,
        rollout_count=1,
        decision_block_size=48,
    )
    effective["replay"].update(history_rollouts=1, fresh_rollouts=1)
    effective["passk"].update(draws=2, k=[1, 2])
    return effective


__all__ = ["LLaDAExecutionProfile", "apply_execution_profile"]
