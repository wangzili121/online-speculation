"""Host staging for the BF16 Orthrus GLM MLA weights."""

from math import prod


def head_layout():
    """Packed fields per MLA layer in one TP8 rank, in element offsets."""
    from models.glm5_3_flash.config import D, INDEX_DIM, INDEX_H, Q_LORA

    from .pypto_glm53_mla import HEADS, LATENT, VALUE

    local_width = HEADS * VALUE
    shapes = {
        "q_a": (Q_LORA, D), "q_norm": (Q_LORA,), "q_b": (local_width, Q_LORA),
        "kv_a": (LATENT, D), "kv_norm": (LATENT,),
        "key": (local_width, LATENT), "value": (local_width, LATENT),
        "out": (D, local_width), "index_q": (Q_LORA, INDEX_H * INDEX_DIM), "index_gate": (D, INDEX_H),
    }
    offset = 0
    layout = {}
    for name, shape in shapes.items():
        elements = prod(shape)
        layout[name] = (offset, elements, shape)
        offset += elements
    return layout, offset


def initialize_rank_head(checkpoint, layer, rank):
    """One initialization slice, dequantized with the checkpoint's row scales."""
    import torch

    attention = checkpoint.attention(layer, tp_rank=rank, tp=8, mla_mode="native")
    indexer = checkpoint.indexer(layer)

    def decompressed(name):
        weights = attention[name]
        scale = attention[name + "_scale"]
        if weights.dtype != torch.int8 or scale.shape != (weights.shape[0],):
            raise ValueError(f"GLM draft initialization requires INT8 weights and a row scale for {name}")
        return (weights.float() * scale.float().unsqueeze(1)).bfloat16()

    values = dict(
        q_a=decompressed("w_q_a"), q_norm=attention["q_a_norm_weight"], q_b=decompressed("w_q_b"),
        kv_a=decompressed("w_kv_a"), kv_norm=attention["kv_a_norm_weight"], key=attention["w_k"].flatten(0, 1),
        value=attention["w_v"].flatten(0, 1), out=decompressed("w_o"),
        index_q=indexer["w_q_b"], index_gate=indexer["w_weights"],
    )
    layout, size = head_layout()
    packed = torch.empty(size, dtype=torch.bfloat16)
    for name, (offset, elements, shape) in layout.items():
        tensor = values[name]
        if tensor.shape != shape or tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid GLM draft projection {name} for layer {layer}, rank {rank}")
        packed[offset:offset + elements].copy_(tensor.reshape(-1))
    return packed
