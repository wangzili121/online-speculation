"""Shared-history parallel drafting with explicit branch and cache contracts."""

from .backbone import DualViewConfig, DualViewDecoder
from .branches import MaskedAttentionBranch
from .generation import generate, generate_ar

__all__ = ["DualViewConfig", "DualViewDecoder",
           "MaskedAttentionBranch", "generate", "generate_ar"]
