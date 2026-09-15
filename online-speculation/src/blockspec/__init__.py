"""Shared-backbone drafting, exact correction and online continuation."""

from __future__ import annotations

from typing import Any


__all__ = [
    "DualViewConfig",
    "DualViewDecoder",
    "MaskedAttentionBranch",
    "SamplingConfig",
    "generate",
    "generate_ar",
]


def __getattr__(name: str) -> Any:
    if name == "SamplingConfig":
        from .sampling import SamplingConfig

        return SamplingConfig
    if name in {
        "DualViewConfig",
        "DualViewDecoder",
        "MaskedAttentionBranch",
        "generate",
        "generate_ar",
    }:
        from . import parallel

        return getattr(parallel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
