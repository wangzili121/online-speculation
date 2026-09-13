"""Rao–Blackwell prefix-overlap objective for a factorized parallel draft."""

from dataclasses import dataclass
import math

import torch

from blockspec.feedback import Feedback
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.online import SuffixConfig, SuffixLearner


def prefix_overlap(q, p, candidates, proposal_taken, *, eos_id=None):
    """Estimate the number of additional output positions reachable in a round.

    Row k analytically integrates its final categorical draw. Earlier candidate
    draws use exact importance weights from the immutable generation proposal.
    The caller selects min(candidate_count, remaining_budget - 1) rows.
    """
    if (q.ndim != 2 or q.shape != p.shape or len(q) < 1
            or candidates.shape != (len(q),) or proposal_taken.shape != candidates.shape):
        raise ValueError("aligned full-vocabulary rows, candidates and saved probabilities required")
    if candidates.dtype != torch.long or ((candidates < 0) | (candidates >= q.shape[1])).any():
        raise ValueError("integer candidates within the vocabulary required")
    if (not torch.isfinite(proposal_taken).all() or (proposal_taken <= 0).any()
            or (proposal_taken > 1).any()):
        raise ValueError("positive saved proposal probabilities required")
    if eos_id is not None and not 0 <= eos_id < q.shape[1]:
        raise ValueError("EOS must be within the vocabulary")
    target, denominator = p.detach(), proposal_taken.detach()
    overlap = torch.minimum(q, target)
    live = overlap if eos_id is None else overlap * (
        torch.arange(q.shape[1], device=q.device) != eos_id)[None]
    selected = overlap.gather(1, candidates[:, None]).squeeze(1) / denominator
    if eos_id is not None:
        selected = selected * (candidates != eos_id)
    reach = torch.cat((selected.new_ones(1), selected[:-1].cumprod(0)))
    return (reach * live.sum(-1)).sum()


@dataclass(frozen=True)
class PrefixConfig(SuffixConfig):
    loss: str = "prefix_overlap"
    temperature: float = 1.

    def __post_init__(self):
        # Reuse the common optimizer and replay-window validation.
        SuffixConfig(self.last_layers, self.stride, self.replay_blocks, self.learning_rate, self.clip_norm)
        if (not math.isfinite(self.temperature) or self.temperature <= 0
                or self.loss != "prefix_overlap"):
            raise ValueError("positive full-vocabulary temperature and prefix-overlap objective required")


@dataclass
class PrefixFeedback(Feedback):
    candidates: torch.Tensor | None = None
    proposal_taken: torch.Tensor | None = None
    version: int = 0

    def detached(self, *, cache_start=0):
        owned = super().detached(cache_start=cache_start)
        return PrefixFeedback(owned.inputs, owned.cache, owned.teacher_logits, owned.valid,
                              owned.boundary, owned.fully_covered, self.candidates.detach().clone(),
                              self.proposal_taken.detach().clone(), self.version)


class PrefixLearner(SuffixLearner):
    def observe(self, feedback, *, may_update=True):
        if not isinstance(feedback, PrefixFeedback) or feedback.version != self.version:
            raise ValueError("feedback must carry the current immutable proposal version")
        return super().observe(feedback, may_update=may_update)

    def backward(self):
        if not self.replay or any(item.version != self.version for item in self.replay):
            raise ValueError("fresh on-version replay blocks required for the local gradient estimator")
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.
        with torch.enable_grad():
            for item in self.replay:
                weights = {name: p.to(self.execution[name].dtype) for name, p in self.master.items()}
                # Preserve the actual draft output-head shape before selecting objective rows.
                output = self.model.forward_suffix(item.boundary, cache=item.cache, draft_weights=weights)
                q = (output.logits[0, :item.valid].float() / self.config.temperature).softmax(-1)
                p = (item.teacher_logits.float() / self.config.temperature).softmax(-1)
                loss = -prefix_overlap(q, p, item.candidates, item.proposal_taken,
                                       eos_id=self.model.config.eos_token_id) / len(self.replay)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite prefix objective")
                loss.backward()
                total += float(loss.detach())
        return total


class PrefixOnlineFeedback(OnlineFeedback):
    def __init__(self, *, learner, output_budget):
        super().__init__(learner=learner)
        self.output_budget = output_budget

    def begin(self, prompt):
        super().begin(prompt)
        self.emitted = self.last_commit = 0

    def commit(self, tokens):
        self.last_commit = len(tokens)
        self.emitted += self.last_commit

    def observe(self, proposal, teacher_logits, target, *, used, fully_covered, done):
        if not proposal.collect_feedback:
            self.learner._skip_decoder_feedback(used)
            return
        remaining = self.output_budget - (self.emitted - self.last_commit)
        rows = min(proposal.candidates.numel(), remaining - 1)
        if rows < 1 or proposal.proposal is None:
            raise ValueError("a stochastic proposal with at least two remaining output positions required")
        candidates = proposal.candidates[:rows]
        taken = proposal.proposal[:rows].gather(1, candidates[:, None]).squeeze(1)
        item = PrefixFeedback(proposal.draft_inputs, proposal.draft_cache, teacher_logits[:rows],
                              rows, proposal.boundary, fully_covered, candidates, taken, self.learner.version)
        self.learner.observe(item, may_update=not done)


def feedback_factory(*, learner, output_budget):
    return (PrefixOnlineFeedback(learner=learner, output_budget=output_budget)
            if isinstance(learner, PrefixLearner) else OnlineFeedback(learner=learner))
