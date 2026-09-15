"""Orthrus compatibility hooks for the pinned vLLM-Ascend runtime."""

from __future__ import annotations

from typing import Any


__all__ = ["OrthrusDraftBatch", "OrthrusProposer"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import proposer

        return getattr(proposer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
