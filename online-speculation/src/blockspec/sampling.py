"""Categorical sampling and the exact positive-residual rejection correction."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid top-k or top-p")


def probabilities(logits: Tensor, config: SamplingConfig) -> Tensor:
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return _probabilities_unchecked(logits, config)


def _probabilities_unchecked(logits: Tensor, config: SamplingConfig) -> Tensor:
    """Pure tensor transform; graph callers check finiteness at the boundary."""
    work = logits if logits.dtype == torch.float64 else logits.float()
    if config.temperature == 0:
        return torch.zeros_like(work).scatter_(-1, work.argmax(-1, keepdim=True), 1)
    work = work / config.temperature
    if config.top_k:
        keep = min(config.top_k, work.shape[-1])
        indices = work.topk(keep, dim=-1).indices
        selected = torch.zeros_like(work, dtype=torch.bool).scatter_(-1, indices, True)
        work = work.masked_fill(~selected, -torch.inf)
    if config.top_p < 1:
        ordered, indices = work.sort(dim=-1, descending=True)
        masses = ordered.softmax(-1)
        # Retain the crossing token; probability before this token is < top_p.
        remove = (masses.cumsum(-1) - masses) >= config.top_p
        remove[..., 0] = False
        removed = torch.zeros_like(remove).scatter_(-1, indices, remove)
        work = work.masked_fill(removed, -torch.inf)
    return work.softmax(-1)


def greedy_tokens(logits):
    """Exact greedy specialization: no vocabulary-sized one-hot allocation."""
    if logits.ndim < 1 or logits.shape[-1] < 1 or not torch.isfinite(logits).all():
        raise ValueError("nonempty finite logits required")
    return logits.argmax(-1)


def sample_logits(logits, config=SamplingConfig(), generator=None):
    if config.temperature == 0:
        return greedy_tokens(logits)
    return draw(probabilities(logits, config), generator)


def validate_distribution(p):
    if p.ndim < 1 or p.shape[-1] < 1 or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError("probabilities must sum to one")


def draw(p: Tensor, generator=None) -> Tensor:
    validate_distribution(p)
    return torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).reshape(p.shape[:-1])


def residual(p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    validate_distribution(p)
    validate_distribution(q)
    if p.shape != q.shape:
        raise ValueError("target and proposal shapes differ")
    positive = (p - q).clamp_min(0)
    mass = positive.sum(-1, keepdim=True)
    # When mass=0 rejection is impossible; the returned fallback is never used
    # by a correct rejection decision. Keeping it normalized helps diagnostics.
    corrected = torch.where(mass > 0, positive / mass.clamp_min(torch.finfo(p.dtype).tiny), p)
    return corrected, mass.squeeze(-1)


@dataclass
class Verification:
    tokens: list[int]
    accepted: int
    rejected_at: int | None
    supervised: int


def verify_greedy(proposed, target_ids):
    """The p/q correction for two one-hot laws reduces to equality + replacement."""
    if (proposed.ndim != 1 or target_ids.shape != (proposed.numel() + 1,)
            or proposed.dtype not in (torch.int32, torch.int64)
            or target_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("integer proposals[n] and greedy targets[n+1] required")
    candidates, targets = proposed.tolist(), target_ids.tolist()
    if any(t < 0 for t in candidates + targets):
        raise ValueError("negative token id")
    for i, token in enumerate(candidates):
        if token != targets[i]:
            return Verification(candidates[:i] + [targets[i]], i, i, i + 1)
    return Verification(candidates + [targets[-1]], len(candidates), None, len(candidates))


def verify_linear(proposed: Tensor, q: Tensor, p: Tensor, *, generator=None,
                  acceptance_uniforms: Tensor | None = None) -> Verification:
    """Verify n proposals; p has n+1 rows, the last for the all-accept bonus.

    Returned tokens do not include the clean root generated in the draft pass.
    The first rejected row is a valid training row; later rows are not on-policy.
    """
    n = proposed.numel()
    if proposed.ndim != 1 or q.ndim != 2 or p.shape != (n + 1, q.shape[-1]) or q.shape[0] != n:
        raise ValueError("expected proposals[n], q[n,V], p[n+1,V]")
    validate_distribution(p)
    validate_distribution(q)
    if ((proposed < 0) | (proposed >= q.shape[-1])).any():
        raise ValueError("proposal token outside vocabulary")
    if acceptance_uniforms is None:
        acceptance_uniforms = torch.rand(n, device=p.device, generator=generator)
    if (acceptance_uniforms.shape != (n,) or not torch.isfinite(acceptance_uniforms).all()
            or ((acceptance_uniforms < 0) | (acceptance_uniforms >= 1)).any()):
        raise ValueError("acceptance draws must lie in [0,1)")
    output = []
    for i in range(n):
        token = int(proposed[i])
        denominator = q[i, token]
        if denominator <= 0:
            raise ValueError("a token cannot have been sampled from a zero proposal probability")
        if acceptance_uniforms[i] < torch.minimum(p[i, token] / denominator, denominator.new_tensor(1)):
            output.append(token)
        else:
            correction, mass = residual(p[i], q[i])
            if mass <= 0:
                raise RuntimeError("rejection with zero residual mass indicates numerical inconsistency")
            output.append(int(draw(correction, generator)))
            return Verification(output, i, i, i + 1)
    output.append(int(draw(p[-1], generator)))
    return Verification(output, n, None, n)
