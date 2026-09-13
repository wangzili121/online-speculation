"""Role-specific dependency checks for the two model families."""

from __future__ import annotations

from importlib import metadata
import sys
from typing import Iterable, Mapping

from packaging.requirements import Requirement

_REQUIREMENTS = {
    "arllm-inference": (
        "torch>=2.9",
        "transformers>=5.14,<6",
        "accelerate>=1.14,<2",
        "safetensors>=0.8,<1",
    ),
    "arllm-training": (
        "datasets==5.0.1",
        "peft==0.20.0",
        "pyarrow==25.0.0",
        "trl==1.9.2",
    ),
    "dllm-inference": (
        "torch>=2.7",
        "transformers==4.53.2",
        "accelerate>=1.7,<2",
        "huggingface-hub>=0.30,<1",
        "safetensors>=0.4.3,<1",
    ),
    "dllm-training": ("peft==0.20.0",),
    "vllm": (
        "torch==2.11.0",
        "vllm>=0.25,<0.27",
    ),
}


def required_distributions(
    family: str,
    *,
    stage: str,
    components: Iterable[str] = (),
) -> tuple[str, ...]:
    if family not in {"arllm", "dllm"}:
        raise ValueError(f"unknown model family {family!r}")
    if stage not in {"prepare", "train", "inference", "all"}:
        raise ValueError(f"unknown execution stage {stage!r}")
    groups: list[str] = []
    if stage in {"inference", "all"}:
        groups.append(f"{family}-inference")
    if stage in {"train", "all"}:
        groups.extend((f"{family}-inference", f"{family}-training"))
    if family == "arllm" and "vllm" in components:
        groups.append("vllm")
    requirements = [
        item
        for group in dict.fromkeys(groups)
        for item in _REQUIREMENTS[group]
    ]
    return tuple(dict.fromkeys(requirements))


def environment_issues(
    family: str,
    *,
    stage: str,
    components: Iterable[str] = (),
    installed: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> tuple[str, ...]:
    requirements = required_distributions(
        family,
        stage=stage,
        components=components,
    )
    versions = {key.lower(): value for key, value in (installed or {}).items()}
    issues: list[str] = []
    for text in requirements:
        requirement = Requirement(text)
        key = requirement.name.lower()
        if installed is None:
            try:
                version = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                version = None
        else:
            version = versions.get(key)
        if version is None:
            issues.append(f"missing {requirement}")
        elif version not in requirement.specifier:
            issues.append(
                f"{requirement.name}=={version} does not satisfy "
                f"{requirement.specifier}"
            )
    if "vllm" in set(components) and (platform_name or sys.platform) == "win32":
        issues.append("vLLM requires Linux or WSL2; native Windows is unsupported")
    return tuple(issues)


def validate_environment(
    family: str,
    *,
    stage: str,
    components: Iterable[str] = (),
) -> None:
    issues = environment_issues(family, stage=stage, components=components)
    if issues:
        details = "\n  - ".join(issues)
        variable = "AR_PYTHON" if family == "arllm" else "DLLM_PYTHON"
        raise RuntimeError(
            f"{family} environment preflight failed for {sys.executable}:\n"
            f"  - {details}\n"
            f"Select a compatible interpreter with {variable} or the corresponding "
            "CLI option."
        )


__all__ = [
    "environment_issues",
    "required_distributions",
    "validate_environment",
]
