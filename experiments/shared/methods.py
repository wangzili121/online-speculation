"""Canonical method registry for experiment selection and validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MethodSpec:
    family: str
    name: str
    components: frozenset[str]
    requires_proposal: bool = False
    requires_adapter: bool = False
    paired: bool = True


def _spec(
    family: str,
    name: str,
    *components: str,
    requires_proposal: bool = False,
    requires_adapter: bool = False,
    paired: bool = True,
) -> MethodSpec:
    return MethodSpec(
        family=family,
        name=name,
        components=frozenset(components),
        requires_proposal=requires_proposal,
        requires_adapter=requires_adapter,
        paired=paired,
    )


METHOD_SPECS = (
    _spec("arllm", "base", "quality", "default_quality", "passk", "distribution", "async"),
    _spec("arllm", "beam", "quality", "default_quality"),
    _spec("arllm", "best_of_n", "quality", "default_quality", "async"),
    _spec("arllm", "mh", "quality", "default_quality", "passk"),
    _spec("arllm", "reward_mh", "quality", paired=False),
    _spec("arllm", "conditional_is", "quality", "default_quality", "is_passk", "async"),
    _spec("arllm", "iterated_conditional_is", "quality", paired=False),
    _spec(
        "arllm",
        "conditional_is_small_proposal",
        "quality",
        "default_quality",
        "is_passk",
        "async",
        requires_proposal=True,
    ),
    _spec(
        "arllm",
        "conditional_is_small_proposal_unclipped",
        "is_passk",
        requires_proposal=True,
    ),
    _spec(
        "arllm",
        "conditional_is_small_proposal_uncorrected",
        "is_passk",
        requires_proposal=True,
    ),
    _spec("arllm", "verifier_mh", "quality", "matched_target", "distribution"),
    _spec(
        "arllm",
        "verifier_conditional_is",
        "quality",
        "matched_target",
        "distribution",
    ),
    _spec(
        "arllm",
        "verifier_conditional_is_small_proposal",
        "quality",
        "matched_target",
        "distribution",
        requires_proposal=True,
    ),
    _spec(
        "arllm",
        "rl_sample",
        "quality",
        "default_quality",
        "passk",
        "distribution",
        requires_adapter=True,
    ),
    _spec("arllm", "rl_greedy", "quality", "default_quality", requires_adapter=True),
    _spec("arllm", "base_candidate_fixed", "dynamic_is"),
    _spec("arllm", "replay_aware_fixed", "dynamic_is"),
    _spec("arllm", "replay_aware_optimal", "dynamic_is"),
    _spec("dllm", "base", "quality", "default_quality", "passk", "distribution", "async"),
    _spec("dllm", "block_beam", "quality", "default_quality"),
    _spec("dllm", "best_of_n", "quality", "default_quality", "async"),
    _spec("dllm", "trajectory_power_mh", "quality", "default_quality", "passk"),
    _spec("dllm", "conditional_is", "quality", "default_quality", "is_passk", "async"),
    _spec(
        "dllm",
        "conditional_is_reduced_layer_proposal",
        "quality",
        "default_quality",
        "is_passk",
        "async",
        requires_proposal=True,
    ),
    _spec(
        "dllm",
        "conditional_is_reduced_layer_proposal_unclipped",
        "is_passk",
        "default_quality",
        requires_proposal=True,
    ),
    _spec(
        "dllm",
        "conditional_is_reduced_layer_proposal_uncorrected",
        "is_passk",
        "default_quality",
        requires_proposal=True,
    ),
    _spec(
        "dllm",
        "verifier_mh",
        "quality",
        "default_quality",
        "matched_target",
        "distribution",
    ),
    _spec(
        "dllm",
        "verifier_conditional_is",
        "quality",
        "default_quality",
        "matched_target",
        "distribution",
    ),
    _spec(
        "dllm",
        "verifier_conditional_is_reduced_layer_proposal",
        "quality",
        "default_quality",
        "matched_target",
        "distribution",
        requires_proposal=True,
    ),
    _spec(
        "dllm",
        "vrpo_sample",
        "quality",
        "aligned",
        "passk",
        "distribution",
        requires_adapter=True,
    ),
    _spec("dllm", "vrpo_greedy", "quality", "aligned", requires_adapter=True),
    _spec("dllm", "base_candidate_fixed", "dynamic_is"),
    _spec("dllm", "trajectory_replay_aware_fixed", "dynamic_is"),
    _spec("dllm", "trajectory_replay_aware_optimal", "dynamic_is"),
)


METHOD_REGISTRY = {(spec.family, spec.name): spec for spec in METHOD_SPECS}
if len(METHOD_REGISTRY) != len(METHOD_SPECS):
    raise RuntimeError("duplicate method specification")


def methods_for(family: str, component: str) -> tuple[str, ...]:
    if family not in {"arllm", "dllm"}:
        raise ValueError(f"unknown model family {family!r}")
    return tuple(
        spec.name
        for spec in METHOD_SPECS
        if spec.family == family and component in spec.components
    )


def method_spec(family: str, name: str) -> MethodSpec:
    try:
        return METHOD_REGISTRY[(family, name)]
    except KeyError as error:
        raise ValueError(f"unknown {family} method {name!r}") from error


AR_METHODS = methods_for("arllm", "quality")
AR_DEFAULT_METHODS = methods_for("arllm", "default_quality")
AR_PASSK_METHODS = methods_for("arllm", "passk")
AR_IS_PASSK_METHODS = methods_for("arllm", "is_passk")
AR_ASYNC_METHODS = methods_for("arllm", "async")
AR_DISTRIBUTION_METHODS = methods_for("arllm", "distribution")
AR_DYNAMIC_METHODS = methods_for("arllm", "dynamic_is")
AR_PAIRED_METHODS = tuple(
    spec.name
    for spec in METHOD_SPECS
    if spec.family == "arllm" and "quality" in spec.components and spec.paired
)

DLLM_METHODS = tuple(spec.name for spec in METHOD_SPECS if spec.family == "dllm")
DLLM_DEFAULT_METHODS = methods_for("dllm", "default_quality")
DLLM_ALIGNED_METHODS = methods_for("dllm", "aligned")
DLLM_DYNAMIC_METHODS = methods_for("dllm", "dynamic_is")


__all__ = [
    "AR_ASYNC_METHODS",
    "AR_DEFAULT_METHODS",
    "AR_DISTRIBUTION_METHODS",
    "AR_DYNAMIC_METHODS",
    "AR_IS_PASSK_METHODS",
    "AR_METHODS",
    "AR_PASSK_METHODS",
    "AR_PAIRED_METHODS",
    "DLLM_ALIGNED_METHODS",
    "DLLM_DEFAULT_METHODS",
    "DLLM_DYNAMIC_METHODS",
    "DLLM_METHODS",
    "METHOD_REGISTRY",
    "METHOD_SPECS",
    "MethodSpec",
    "method_spec",
    "methods_for",
]
