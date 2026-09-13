"""Orthrus compatibility hooks for the pinned vLLM-Ascend runtime."""

from .proposer import OrthrusDraftBatch, OrthrusProposer

__all__ = ["OrthrusDraftBatch", "OrthrusProposer"]
