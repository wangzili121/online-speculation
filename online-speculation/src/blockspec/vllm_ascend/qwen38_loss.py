"""Exact vocabulary-parallel distillation using existing tensor collectives."""

import torch


class _VocabDivergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, start, tp, kind):
        if (kind not in {"forward_kl", "tv"} or student.shape[:-1] != teacher.shape[:-1]
                or start < 0 or start + student.shape[-1] > teacher.shape[-1]):
            raise ValueError("aligned local student and complete teacher vocabulary required")
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        local = (student.to(dtype), teacher[..., start:start + student.shape[-1]].detach().to(dtype))
        # 只汇总每个位置的两项局部 LSE，再稳定合并为完整词表的归一化量。
        normalizer = tp.gather(torch.stack([value.logsumexp(-1) for value in local], dim=-1))
        normalizer = normalizer.unflatten(-1, (tp.size, 2)).logsumexp(-2)
        log_q, log_p = [value - normalizer[..., i, None] for i, value in enumerate(local)]
        q, p = log_q.exp(), log_p.exp()
        if kind == "forward_kl":
            loss = tp.reduce((p * (log_p - log_q)).sum(-1))
            derivative = q - p
        else:
            sign = (q - p).sign()
            combined = tp.reduce(torch.stack([.5 * (q - p).abs().sum(-1), (q * sign).sum(-1)]))
            loss = combined[0]
            derivative = .5 * q * (sign - combined[1, ..., None])
        # 每个 rank 持有相同 loss；只返回本地 logits 梯度，LM head 再归并 hidden 梯度。
        ctx.save_for_backward(derivative)
        ctx.dtype = student.dtype
        return loss

    @staticmethod
    def backward(ctx, gradient):
        (derivative,) = ctx.saved_tensors
        return (derivative * gradient[..., None]).to(ctx.dtype), None, None, None, None


def vocab_divergence(student, teacher, *, start, tp, kind):
    return _VocabDivergence.apply(student, teacher, start, tp, kind)
