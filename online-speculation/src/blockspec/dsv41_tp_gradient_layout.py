"""TP4 gradient layout for the DSV4.1 diffusion Attention head.

Forward execution stores transposed projection operands and partitions only
query heads/output groups.  Online optimization owns the artifact layout, so
rank-local gradients must be reduced or concatenated back to the eight public
parameter tensors before AdamW and publication.
"""

from collections.abc import Mapping

import torch


PARAMETERS = (
    "wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight",
    "kv_norm.weight", "attn_sink", "wo_a.weight", "wo_b.weight",
)


class DSV41TPGradientLayout:
    """Restore one DP replica's four TP lanes to artifact parameter layout."""

    def __init__(self, *, tp=4):
        if tp != 4:
            raise ValueError("DSV4.1 Orthrus gradient layout is verified for TP4")
        self.tp = tp

    def lanes(self, rank_gradients):
        if not isinstance(rank_gradients, Mapping) or len(rank_gradients) != self.tp:
            raise ValueError("one DP replica must provide exactly four TP gradient rows")
        ranks = sorted(rank_gradients)
        dp = ranks[0] // self.tp
        if ranks != list(range(dp * self.tp, (dp + 1) * self.tp)):
            raise ValueError("TP gradients must be one complete contiguous DP partition")
        for rank in ranks:
            if set(rank_gradients[rank]) != set(PARAMETERS):
                raise ValueError(f"rank {rank} changed the draft gradient ABI")
        return tuple(rank_gradients[rank] for rank in ranks)

    @staticmethod
    def _same(values, name):
        first = values[0]
        if any(not torch.is_tensor(value) or value.shape != first.shape or value.dtype != first.dtype
               for value in values):
            raise ValueError(f"replicated TP gradient {name} changed shape or dtype")
        return torch.stack(values).sum(0)

    @staticmethod
    def _cat(values, name, dimension):
        first = values[0]
        if any(not torch.is_tensor(value) or value.ndim != first.ndim or value.dtype != first.dtype
               or any(value.shape[axis] != first.shape[axis]
                      for axis in range(first.ndim) if axis != dimension)
               for value in values):
            raise ValueError(f"sharded TP gradient {name} changed non-sharded geometry")
        return torch.cat(values, dim=dimension)

    def assemble(self, rank_gradients):
        """Return gradients in the untransposed exchange-artifact layout.

        Rank-local projection gradients follow their execution operands:
        ``wq_a/wq_b/wkv/wo_b`` are transposed, while grouped ``wo_a`` keeps a
        leading local-group dimension.  Replicated branches are summed; true
        shards are concatenated in TP-lane order.
        """
        lanes = self.lanes(rank_gradients)
        values = {name: [lane[name] for lane in lanes] for name in PARAMETERS}
        return {
            "wq_a.weight": self._same(values["wq_a.weight"], "wq_a.weight").T.contiguous(),
            "q_norm.weight": self._same(values["q_norm.weight"], "q_norm.weight"),
            "wq_b.weight": self._cat(values["wq_b.weight"], "wq_b.weight", 1).T.contiguous(),
            "wkv.weight": self._same(values["wkv.weight"], "wkv.weight").T.contiguous(),
            "kv_norm.weight": self._same(values["kv_norm.weight"], "kv_norm.weight"),
            "attn_sink": self._cat(values["attn_sink"], "attn_sink", 0),
            "wo_a.weight": self._cat(values["wo_a.weight"], "wo_a.weight", 0).flatten(0, 1),
            "wo_b.weight": self._cat(values["wo_b.weight"], "wo_b.weight", 0).T.contiguous(),
        }


def validate_artifact_shapes(values, config):
    """Validate complete optimizer/publication tensors against model config."""
    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads, dim = config["num_attention_heads"], config["head_dim"]
    groups, o_rank = config["o_groups"], config["o_lora_rank"]
    shapes = {
        "wq_a.weight": (latent, hidden),
        "q_norm.weight": (latent,),
        "wq_b.weight": (heads * dim, latent),
        "wkv.weight": (dim, hidden),
        "kv_norm.weight": (dim,),
        "attn_sink": (heads,),
        "wo_a.weight": (groups * o_rank, heads * dim // groups),
        "wo_b.weight": (hidden, groups * o_rank),
    }
    if set(values) != set(shapes):
        raise ValueError("assembled gradients changed the draft parameter ABI")
    for name, shape in shapes.items():
        if not torch.is_tensor(values[name]) or tuple(values[name].shape) != shape:
            raise ValueError(f"assembled gradient has the wrong shape: {name}")
    return True
