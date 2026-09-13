"""Conditional full-answer work under exact speculative rejection correction.

The output is an AR trajectory. Given its next token x, an acceptance flag has
probability min(1, q(x)/p(x)). A visit-mass recurrence integrates those flags,
including the resulting changes in later block starts. This module implements
the masked-attention branch, whose input includes one committed anchor.
"""

from dataclasses import dataclass
import time

import torch
from torch import Tensor
from torch.nn import functional as F

from blockspec.sampling import SamplingConfig, probabilities
from blockspec.state import trim_cache


class TraceSupportError(ValueError):
    """A delivered token has zero mass in a recomputed, truncated target law."""


@dataclass
class VirtualWork:
    draft_visits: Tensor
    tail_visits: Tensor
    output_tokens: int

    @property
    def draft_forwards(self):
        return self.draft_visits.sum()

    @property
    def verifier_forwards(self):
        return self.draft_forwards

    @property
    def tail_ar_forwards(self):
        return self.tail_visits.sum()

    @property
    def decode_forwards(self):
        return 2 * self.draft_forwards + self.tail_ar_forwards

    def cost(self, round_cost=2., tail_cost=1.):
        """Scalar or per-start operation costs; exclude the common prefill."""
        return (self.draft_visits * round_cost + self.tail_visits * tail_cost).sum()


def expected_work(acceptance, *, output_tokens, output_budget, block_size):
    """Expected draft/verifier/AR-tail visits conditional on a delivered answer.

Row j-1 describes a round starting after j committed output tokens; column i
describes proposal i+1. Only tokens preceding the terminal output affect future
visits. The requested cap controls block length, including early-EOS rounds.
The first output comes from a common AR prefill. This recurrence is differentiable.
"""
    if (any(type(v) is not int for v in (output_tokens, output_budget, block_size))
            or not 0 <= output_tokens <= output_budget or block_size < 2
            or acceptance.shape != (max(0, output_tokens - 1), block_size - 1)
            or not acceptance.is_floating_point() or not torch.isfinite(acceptance).all()
            or ((acceptance < 0) | (acceptance > 1)).any()):
        raise ValueError("finite conditional probabilities and consistent output/block sizes required")
    zero = acceptance.sum() * 0
    if output_tokens <= 1:
        empty = acceptance.sum(1)
        return VirtualWork(empty, empty, output_tokens)
    visits = [zero for _ in range(output_tokens)]
    visits[1] = zero + 1
    draft, tail = [], []
    for j in range(1, output_tokens):
        mass = visits[j]
        remaining = output_budget - j
        if remaining == 1:
            draft.append(zero)
            tail.append(mass)
            continue
        draft.append(mass)
        tail.append(zero)
        proposals = min(block_size, remaining + 1) - 1
        survival = mass
        for offset in range(1, min(proposals, output_tokens - j - 1) + 1):
            gamma = acceptance[j - 1, offset - 1]
            visits[j + offset] = visits[j + offset] + survival * (1 - gamma)
            survival = survival * gamma
        following = j + proposals + 1
        if following < output_tokens:
            visits[following] = visits[following] + survival
    return VirtualWork(torch.stack(draft), torch.stack(tail), output_tokens)


def inverse_acceptance(proposal_logp, target_logp):
    """P(accept | delivered x); delivered tokens must have positive target mass."""
    if torch.isneginf(target_logp).any():
        raise TraceSupportError("delivered tokens require positive recomputed target mass")
    if (proposal_logp.shape != target_logp.shape or not torch.isfinite(target_logp).all()
            or torch.isnan(proposal_logp).any() or torch.isposinf(proposal_logp).any()
            or (proposal_logp > 0).any() or (target_logp > 0).any()):
        raise ValueError("matching log probabilities with positive target mass required")
    return (proposal_logp - target_logp).clamp_max(0).exp()


@dataclass
class TraceScores:
    acceptance: Tensor
    proposal_logp: Tensor
    target_logp: Tensor
    valid: Tensor
    output_budget: int
    output_tokens: int
    block_size: int

    def work(self):
        # One transfer avoids a device launch for each scalar DP transition.
        return expected_work(self.acceptance.cpu(), output_tokens=self.output_tokens,
                             output_budget=self.output_budget, block_size=self.block_size)


def _observed_logp(model, hidden, targets, sampling, chunk_rows):
    parts = []
    for start in range(0, len(hidden), chunk_rows):
        logits = F.linear(hidden[start:start + chunk_rows], model.head.weight)
        indices = targets[start:start + chunk_rows, None]
        if sampling.temperature > 0 and sampling.top_k == 0 and sampling.top_p == 1:
            work = logits if logits.dtype == torch.float64 else logits.float()
            logp = (work / sampling.temperature).log_softmax(-1).gather(1, indices)
        else:
            logp = probabilities(logits, sampling).gather(1, indices).log()
        parts.append(logp[:, 0])
    return torch.cat(parts)


def trace_layout(clean, starts, length, mask_token_id):
    """Isolated anchor+mask blocks; future masks may extend beyond an early EOS."""
    if (clean.ndim != 2 or clean.shape[0] != 1 or clean.dtype != torch.long
            or starts.ndim != 1 or starts.dtype != torch.long or len(starts) < 1
            or starts.device != clean.device or type(length) is not int or length < 2
            or ((starts < 0) | (starts >= clean.shape[1])).any()):
        raise ValueError("batch-one clean tokens, valid anchors and block length >= 2 required")
    positions = starts[:, None] + torch.arange(length, device=clean.device)
    tokens = clean.new_full(positions.shape, mask_token_id)
    tokens[:, 0] = clean[0, starts]
    clean_visible = torch.arange(clean.shape[1], device=clean.device)[None] < starts.repeat_interleave(length)[:, None]
    blocks = torch.arange(starts.numel(), device=clean.device).repeat_interleave(length)
    isolated = blocks[:, None] == blocks[None]
    allowed = torch.cat((clean_visible, isolated), 1)[None, None]
    return tokens.reshape(1, -1), positions.reshape(1, -1), allowed


@torch.no_grad()
def score_trace(model, prompt, outputs, *, output_budget, block_size=None,
                sampling=SamplingConfig(1.), eos_id=None, chunk_rows=32, anchors_per_pass=64):
    """Score every possible round start in a completed AR answer, in masked batches.

Full-vocabulary normalization is row-chunked. Missing q entries are neutral and
marked by `valid`; terminal-token acceptance cannot change future work. Runtime
costs, training, and publication remain the caller's responsibility.
"""
    block_size = model.config.block_size if block_size is None else block_size
    if (prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1 or prompt.dtype != torch.long
            or outputs.ndim != 1 or outputs.dtype != torch.long or outputs.device != prompt.device
            or type(output_budget) is not int or not 0 <= outputs.numel() <= output_budget
            or any(type(v) is not int or v < 1 for v in (block_size, chunk_rows, anchors_per_pass))
            or block_size < 2 or model.backend not in ("eager", "sdpa")
            or ((outputs < 0) | (outputs >= model.config.vocab_size)).any()):
        raise ValueError("completed batch-one output and positive masked evaluation sizes required")
    n = outputs.numel()
    if ((n < output_budget and (n == 0 or eos_id is None or int(outputs[-1]) != eos_id))
            or (eos_id is not None and (outputs[:-1] == eos_id).any())):
        raise ValueError("a completed answer must reach its cap or first EOS")
    shape = (max(0, n - 1), block_size - 1)
    q_log = torch.zeros(shape, device=prompt.device, dtype=torch.float64)
    valid = torch.zeros(shape, device=prompt.device, dtype=torch.bool)
    if n <= 1:
        return TraceScores(q_log + 1, q_log, q_log, valid, output_budget, n, block_size)
    clean = torch.cat((prompt, outputs[None]), 1)
    teacher = model(clean, compute_logits=False)
    p_log = _observed_logp(model, teacher.hidden[0, prompt.shape[1]:-1], outputs[1:], sampling, chunk_rows)
    target = q_log.clone()
    # The block input has one anchor, hence remaining+1. A one-token remainder
    # follows the ordinary AR tail path in the actual generator.
    groups = {}
    for j in range(1, n - 1):
        if output_budget - j > 1:
            groups.setdefault(min(block_size, output_budget - j + 1), []).append(j)
    for length, starts in groups.items():
        for first in range(0, len(starts), anchors_per_pass):
            js = torch.tensor(starts[first:first + anchors_per_pass], device=prompt.device)
            anchors = prompt.shape[1] + js - 1
            inputs, positions, allowed = trace_layout(clean, anchors, length, model.config.mask_token_id)
            student = model(inputs, view="draft", cache=teacher.cache, positions=positions,
                            allowed=allowed, compute_logits=False)
            hidden = student.hidden.reshape(len(js), length, -1)[:, :-1]
            output_rows = js[:, None] + torch.arange(length - 1, device=prompt.device)[None]
            used = output_rows < n - 1
            actual = output_rows[used]
            q = _observed_logp(model, hidden[used], outputs[actual], sampling, chunk_rows)
            row = (js - 1)[:, None].expand(-1, length - 1)[used]
            col = torch.arange(length - 1, device=prompt.device)[None].expand(len(js), -1)[used]
            q_log[row, col] = q.double()
            target[row, col] = p_log[actual - 1].double()
            valid[row, col] = True
    acceptance = torch.ones_like(q_log)
    acceptance[valid] = inverse_acceptance(q_log[valid], target[valid])
    return TraceScores(acceptance, q_log, target, valid, output_budget, n, block_size)


def calibration_starts(visits, count):
    """Equal-mass midpoint quadrature over actual conditional round visits."""
    if (visits.ndim != 1 or type(count) is not int or count < 1
            or not torch.isfinite(visits).all() or (visits < 0).any()):
        raise ValueError("finite nonnegative visits and positive calibration count required")
    total = visits.sum()
    if total == 0:
        return []
    quantiles = (torch.arange(count, device=visits.device, dtype=visits.dtype) + .5) * total / count
    return (torch.searchsorted(visits.cumsum(0), quantiles) + 1).tolist()


@dataclass
class TraceTiming:
    work: VirtualWork
    predicted_seconds: float
    measured_seconds: float
    score_seconds: float
    calibration_seconds: float
    round_seconds: list[float]
    tail_seconds: float


@torch.no_grad()
def estimate_trace(branch, prompt, outputs, *, output_budget, prefill_seconds, sampler, generator,
                   sampling=SamplingConfig(1.), eos_id=None, anchors_per_pass=128, chunk_rows=32,
                   calibration_points=3):
    """Full-answer screening estimate with measured per-round kernel costs.

    The work law is exact for the supplied p/q. Kernel timing uses visit-weighted
    quadrature and serves as an empirical approximation. A separate generator
    owns calibration draws. All measurement and calibration time is returned.
    """
    if (branch.name != "masked_attention" or branch.input_budget_extra != 1 or not branch.initial_ar_token
            or branch.default_block_size != branch.model.config.block_size or sampler.config != sampling
            or not isinstance(generator, torch.Generator) or not 0 <= prefill_seconds < float("inf")
            or type(calibration_points) is not int or calibration_points < 1):
        raise ValueError("masked branch, matching sampling, finite prefill time and calibration settings required")
    def sync():
        if prompt.device.type == "cuda":
            torch.cuda.synchronize(prompt.device)

    sync()
    start = time.perf_counter()
    scores = score_trace(branch.model, prompt, outputs, output_budget=output_budget, sampling=sampling,
                          eos_id=eos_id, anchors_per_pass=anchors_per_pass, chunk_rows=chunk_rows)
    work = scores.work()
    sync()
    scoring_end = time.perf_counter()
    starts = calibration_starts(work.draft_visits, calibration_points)
    round_seconds, tail_seconds = [], 0.
    if starts or work.tail_ar_forwards > 0:
        clean = torch.cat((prompt, outputs[None]), 1)
        teacher = branch.model(clean, compute_logits=False)
        for j in starts:
            anchor_index = prompt.shape[1] + j - 1
            anchor = clean[:, anchor_index:anchor_index + 1]
            cache = trim_cache(teacher.cache, anchor_index)
            length = min(branch.default_block_size, output_budget - j + 1)
            sync()
            marker = time.perf_counter()
            proposal = branch.draft(anchor, cache, length, sampling, generator, sampler=sampler)
            logits, verified_cache = branch.ar(proposal.verifier_inputs, cache=proposal.cache)
            verified, _ = sampler.verify(proposal, logits[0], generator)
            # Include the cache and anchor bookkeeping used by the generation loop.
            kept_cache = trim_cache(verified_cache, anchor_index + len(verified.tokens))
            next_anchor = prompt.new_tensor([[verified.tokens[-1]]])
            sync()
            round_seconds.append(time.perf_counter() - marker)
            del proposal, logits, verified_cache, kept_cache, next_anchor
        if work.tail_ar_forwards > 0:
            anchor_index = prompt.shape[1] + outputs.numel() - 2
            cache = trim_cache(teacher.cache, anchor_index)
            sync()
            marker = time.perf_counter()
            logits, _ = branch.ar(clean[:, anchor_index:anchor_index + 1], cache=cache, logits_to_keep=1)
            sampler.sample_ar(logits[0, -1], generator)
            sync()
            tail_seconds = time.perf_counter() - marker
    average_round = sum(round_seconds) / len(round_seconds) if round_seconds else 0.
    predicted = prefill_seconds + float(work.cost(average_round, tail_seconds))
    sync()
    end = time.perf_counter()
    return TraceTiming(work, predicted, end - start, scoring_end - start, end - scoring_end,
                        round_seconds, tail_seconds)
