"""Shared-backbone bidirectional drafting, exact correction and online continuation."""

from .parallel import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate, generate_ar
from .sampling import SamplingConfig

__all__ = ["DualViewConfig", "DualViewDecoder", "MaskedAttentionBranch", "SamplingConfig", "generate", "generate_ar"]
