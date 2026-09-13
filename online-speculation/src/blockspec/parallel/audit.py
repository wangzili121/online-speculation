"""Common-prefix distribution comparisons for a fixed suffix checkpoint."""

import torch

from ..losses import divergence
from ..sampling import SamplingConfig, probabilities
from .backbone import DraftBoundary
from .sampling import ProposalSampler


METRICS = ("original_raw_kl", "learned_raw_kl", "original_raw_tv", "learned_raw_tv",
           "original_sampling_tv", "learned_sampling_tv")


def audit_summary(totals):
    values = totals.detach().double().cpu()
    total = values.sum(0)
    return {"positions": int(total[0]),
            "mean": {name: float(total[i + 1] / total[0]) if total[0] else None for i, name in enumerate(METRICS)},
            "by_depth": [{"depth": depth + 1, "positions": int(row[0]),
                          **{name: float(row[i + 1] / row[0]) if row[0] else None for i, name in enumerate(METRICS)}}
                         for depth, row in enumerate(values)]}


def paired_audit_intervals(records, *, seed=0, repeats=2000):
    """Question-cluster intervals for learned-minus-original mean divergence."""
    records = [row for row in records if row["positions"] > 0]
    if not records or repeats < 1:
        raise ValueError("nonempty paired audit and positive resampling count required")
    rows = torch.tensor([[r["positions"], *[r["positions"] * (
        r["mean"][f"learned_{metric}"] - r["mean"][f"original_{metric}"])
        for metric in ("raw_kl", "raw_tv", "sampling_tv")]] for r in records], dtype=torch.float64)
    indices = torch.randint(len(rows), (repeats, len(rows)), generator=torch.Generator().manual_seed(seed))
    total, draws = rows.sum(0), rows[indices].sum(1)
    return {metric: {"difference": float(total[i + 1] / total[0].clamp_min(1)),
                     "paired_request_ci95": torch.quantile(draws[:, i + 1] / draws[:, 0].clamp_min(1),
                                                            torch.tensor([.025, .975], dtype=torch.float64)).tolist()}
            for i, metric in enumerate(("raw_kl", "raw_tv", "sampling_tv"))}


class AuditSampler(ProposalSampler):
    """Own the actual draft logits for a subsequent replay alignment check."""

    def propose(self, logits, generator, **kwargs):
        self.last_logits = logits.detach().clone()
        return super().propose(logits, generator, **kwargs)


class SuffixAudit:
    """Replay both suffixes on the original drafter's actually reached prefixes.

    Inspection runs outside throughput timing. It consumes no random draws,
    publishes no weights and retains only per-depth scalar statistics.
    """

    needs_decoder_feedback = True
    updates = update_seconds = coverage_skips = 0

    def __init__(self, model, start_layer, alternative, block_size, sampling=SamplingConfig(), *, recorded_logits=None):
        if not 0 <= start_layer < len(model.layers) or block_size < 2:
            raise ValueError("valid suffix and block size required")
        self.model, self.capture_layer, self.sampling = model, start_layer, sampling
        self.recorded_logits = recorded_logits
        self.max_replay_logit_error = 0.
        expected = {name: p for name, p in model.named_parameters() if name.startswith("layers.")
                    and int(name.split(".")[1]) >= start_layer and ".attention.draft." in name}
        if alternative.keys() != expected.keys() or any(
                p.shape != expected[name].shape or p.dtype != expected[name].dtype
                or p.device != expected[name].device or not torch.isfinite(p).all()
                for name, p in alternative.items()):
            raise ValueError("complete finite alternative suffix in the model's execution precision required")
        self.alternative = {name: p.detach().clone() for name, p in alternative.items()}
        self.totals = next(model.parameters()).new_zeros(block_size - 1, len(METRICS) + 1, dtype=torch.float64)
        self.feedback_blocks = 0

    def clear_replay(self):
        pass

    @torch.no_grad()
    def observe(self, feedback, *, may_update=True):
        n = feedback.valid
        if (not isinstance(feedback.boundary, DraftBoundary) or feedback.boundary.start_layer != self.capture_layer
                or not 0 < n <= len(self.totals) or feedback.teacher_logits.shape != (n, self.model.config.vocab_size)):
            raise ValueError("aligned reached-prefix feedback required for audit")
        cache = None if feedback.cache is None else feedback.cache[self.capture_layer:]
        # Retain the original full-block output-head shape, then select reached rows.
        original = self.model.forward_suffix(feedback.boundary, cache=cache).logits[0, :n]
        learned = self.model.forward_suffix(feedback.boundary, cache=cache,
                                            draft_weights=self.alternative).logits[0, :n]
        if self.recorded_logits is not None:
            recorded = self.recorded_logits()[:n]
            self.max_replay_logit_error = max(self.max_replay_logit_error, float((original.float() - recorded.float()).abs().max()))
            if not torch.equal(original, recorded):
                raise AssertionError("full-block replay must reproduce the actual original draft logits")
        teacher = feedback.teacher_logits
        target = probabilities(teacher, self.sampling)
        original_tv = .5 * (probabilities(original, self.sampling) - target).abs().sum(-1)
        learned_tv = .5 * (probabilities(learned, self.sampling) - target).abs().sum(-1)
        values = torch.stack((divergence(original, teacher, "forward_kl"),
                              divergence(learned, teacher, "forward_kl"),
                              divergence(original, teacher, "tv"), divergence(learned, teacher, "tv"),
                              original_tv, learned_tv), -1)
        if not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite common-prefix audit")
        self.totals[:n, 0] += 1
        self.totals[:n, 1:] += values.double()
        self.feedback_blocks += 1
