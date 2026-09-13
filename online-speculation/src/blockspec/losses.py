"""Complete-vocabulary losses with detached teachers and FP32/BF16-safe accumulation."""

import torch


def divergence(student_logits, teacher_logits, kind="forward_kl"):
    """Per-position KL(teacher || student) or total variation at temperature one."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logit shapes differ")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_q = student_logits.to(dtype).log_softmax(-1)
    log_p = teacher_logits.detach().to(dtype).log_softmax(-1)
    q, p = log_q.exp(), log_p.exp()
    if kind == "forward_kl":
        return (p * (log_p - log_q)).sum(-1)
    if kind == "tv":
        return .5 * (q - p).abs().sum(-1)
    raise ValueError(f"unknown distillation loss: {kind}")
