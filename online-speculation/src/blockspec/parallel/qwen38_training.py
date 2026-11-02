"""Clean-text random-anchor distillation for the hybrid Qwen draft."""

import torch

from ..feedback import Feedback


@torch.no_grad()
def offline_feedback(model, tokens, anchor):
    """Teacher-forced AR supplies all K-1 targets, unlike reached-only online feedback."""
    width = model.config.block_size
    if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.long
            or type(anchor) is not int or not 0 <= anchor <= tokens.shape[1] - width
            or bool(((tokens < 0) | (tokens >= model.config.vocab_size)).any())):
        raise ValueError("one clean token sequence and an anchor with a complete block required")
    history = model(tokens[:, :anchor], logits_to_keep=1).cache if anchor else None
    clean = tokens[:, anchor:anchor + width]
    teacher = model(clean, cache=history).logits[0, :-1].detach()
    noisy = torch.full_like(clean, model.config.mask_token_id)
    noisy[:, 0] = clean[:, 0]
    boundary = model.draft_boundary(noisy, history)
    return Feedback(noisy, history, teacher, width - 1, boundary)
