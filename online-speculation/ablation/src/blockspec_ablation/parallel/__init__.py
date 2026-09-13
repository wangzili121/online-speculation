"""Optional draft policies sharing the mainline backbone and generation loop."""

from blockspec.parallel import DualViewConfig, DualViewDecoder, generate, generate_ar
from .branches import CausalLowRankBranch, MaskedAttentionBranch

__all__ = ["DualViewConfig", "DualViewDecoder", "CausalLowRankBranch", "MaskedAttentionBranch", "generate", "generate_ar"]
