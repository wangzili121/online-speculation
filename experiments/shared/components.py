"""Canonical experiment-component names shared by both model families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    name: str
    families: frozenset[str]
    description: str


COMPONENT_SPECS = (
    ComponentSpec("quality", frozenset({"arllm", "dllm"}), "primary quality comparison"),
    ComponentSpec("matched_target", frozenset({"arllm", "dllm"}), "shared verifier target"),
    ComponentSpec("replay", frozenset({"arllm", "dllm"}), "rollout replay"),
    ComponentSpec("dynamic_is", frozenset({"arllm", "dllm"}), "dynamic IS extensions"),
    ComponentSpec("async", frozenset({"arllm", "dllm"}), "asynchronous execution"),
    ComponentSpec("passk", frozenset({"arllm", "dllm"}), "multi-draw pass@k"),
    ComponentSpec("ablations", frozenset({"arllm", "dllm"}), "algorithm ablations"),
    ComponentSpec("budget_curve", frozenset({"arllm", "dllm"}), "quality-cost curve"),
    ComponentSpec("length_ablation", frozenset({"arllm", "dllm"}), "generation-length ablation"),
    ComponentSpec("distribution", frozenset({"arllm", "dllm"}), "answer-distribution audit"),
    ComponentSpec("infra", frozenset({"arllm", "dllm"}), "infrastructure ablations"),
    ComponentSpec("vllm", frozenset({"arllm"}), "vLLM backend comparison"),
)

COMPONENT_REGISTRY = {spec.name: spec for spec in COMPONENT_SPECS}
COMPONENTS = tuple(COMPONENT_REGISTRY)
AR_COMPONENTS = tuple(
    spec.name for spec in COMPONENT_SPECS if "arllm" in spec.families
)
DLLM_COMPONENTS = tuple(
    spec.name for spec in COMPONENT_SPECS if "dllm" in spec.families
)

# ``full`` is the production reproduction route.  Research screens and
# ablations remain available through an explicit ``--components`` selection,
# but are not scheduled implicitly.
PRODUCTION_COMPONENTS = (
    "quality",
    "matched_target",
    "replay",
    "async",
    "passk",
    "distribution",
)
RESEARCH_COMPONENTS = tuple(
    name for name in COMPONENTS if name not in PRODUCTION_COMPONENTS
)
FULL_COMPONENTS = PRODUCTION_COMPONENTS


def components_for(family: str) -> tuple[str, ...]:
    if family == "arllm":
        return AR_COMPONENTS
    if family == "dllm":
        return DLLM_COMPONENTS
    raise ValueError(f"unknown model family {family!r}")


def validate_components(family: str, components: tuple[str, ...]) -> None:
    supported = set(components_for(family))
    unsupported = sorted(set(components) - supported)
    if unsupported:
        raise ValueError(
            f"components unsupported by {family}: {', '.join(unsupported)}"
        )


__all__ = [
    "AR_COMPONENTS",
    "COMPONENTS",
    "COMPONENT_REGISTRY",
    "COMPONENT_SPECS",
    "DLLM_COMPONENTS",
    "FULL_COMPONENTS",
    "PRODUCTION_COMPONENTS",
    "RESEARCH_COMPONENTS",
    "ComponentSpec",
    "components_for",
    "validate_components",
]
