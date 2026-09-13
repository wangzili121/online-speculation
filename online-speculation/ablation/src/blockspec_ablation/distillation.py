"""Paired clean/noisy next-token distillation; the base teacher never changes."""

from dataclasses import dataclass


import torch

from .diffusion import corrupt


LOSS_KINDS = ("l1", "tv", "forward_kl", "reverse_kl", "reverse_kl_l1")


@dataclass
class PairedBatch:
    tokens: torch.Tensor
    positions: torch.Tensor
    allowed: torch.Tensor
    adapter_mask: torch.Tensor
    eligible: torch.Tensor


def paired_batch(clean, block_size, *, noisy=None, generator=None):
    """Pack [clean, noisy] with identical position ids and causal block attention.

    Row j on BOTH sides predicts token j+1, never token j. The first token is
    treated as BOS and is kept clean. No padding/sequence packing is implicit.
    """
    if clean.ndim != 2 or clean.shape[1] < 2 or block_size < 1:
        raise ValueError("expected [batch, length>=2] tokens and positive block size")
    if noisy is None or noisy.shape != clean.shape:
        raise ValueError("explicit noise with the same shape is required")
    batch, length = clean.shape
    noisy = noisy.clone()
    noisy[:, 0] = clean[:, 0]
    j = torch.arange(length, device=clean.device)
    causal = j[None, :] <= j[:, None]
    start = (j // block_size) * block_size
    prior_clean = j[None, :] < start[:, None]
    own_noise = causal & ((j[None, :] // block_size) == (j[:, None] // block_size))
    allowed = torch.zeros(2 * length, 2 * length, dtype=torch.bool, device=clean.device)
    allowed[:length, :length] = causal
    allowed[length:, :length] = prior_clean
    allowed[length:, length:] = own_noise
    active = torch.ones_like(clean, dtype=torch.bool)
    active[:, 0] = False
    mask = torch.cat((torch.zeros_like(active), active), dim=1)
    return PairedBatch(torch.cat((clean, noisy), dim=1),
                       j.repeat(2)[None].expand(batch, -1), allowed[None, None], mask, active)




def divergence(student_logits, teacher_logits, kind="l1"):
    """Per-position divergence at temperature one (independent of generation).

    l1 is twice mathematical TV, matching the reference paper's unhalved loss.
    reverse_kl means KL(student || teacher); forward_kl means KL(teacher || student).
    reverse_kl_l1 is the paper's alpha=beta=1 warm-up, not KL alone or KL+TV.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logit shapes differ")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_q = student_logits.to(dtype).log_softmax(-1)
    log_p = teacher_logits.detach().to(dtype).log_softmax(-1)
    q, p = log_q.exp(), log_p.exp()
    if kind == "reverse_kl_l1":
        return (q * (log_q - log_p)).sum(-1) + (q - p).abs().sum(-1)
    if kind == "reverse_kl":
        return (q * (log_q - log_p)).sum(-1)
    if kind == "forward_kl":
        return (p * (log_p - log_q)).sum(-1)
    if kind in ("l1", "tv"):
        return (q - p).abs().sum(-1) * (0.5 if kind == "tv" else 1.0)
    raise ValueError(f"unknown distillation loss: {kind}")



def paired_loss(model, clean, block_size, *, kind="l1", noisy=None, generator=None):
    if noisy is None:
        prior = torch.full((model.config.vocab_size,), 1 / model.config.vocab_size,
                           device=clean.device)
        noisy = corrupt(clean, prior, 0.0, generator=generator)
    paired = paired_batch(clean, block_size, noisy=noisy)
    logits = model(paired.tokens, positions=paired.positions, allowed=paired.allowed,
                   adapter_mask=paired.adapter_mask)
    length = clean.shape[1]
    teacher, student = logits[:, :length].detach(), logits[:, length:]
    losses = divergence(student, teacher, kind)
    return losses[paired.eligible].mean()


def offline_step(model, optimizer, clean, block_size, *, kind="l1", generator=None,
                 clip_norm=1.0):
    optimizer.zero_grad(set_to_none=True)
    loss = paired_loss(model, clean, block_size, kind=kind, generator=generator)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite offline loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), clip_norm,
                                         error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm": float(norm)}
