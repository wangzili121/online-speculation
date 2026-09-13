"""Prefix-conditioned mixtures of frozen drafts and request-local continuations."""

from dataclasses import dataclass
import math
import time

import torch


class SuffixLookup:
    """Incremental longest-suffix index with an observed continuation budget."""

    def __init__(self, tokens=(), *, minimum=2, maximum=8, lookahead=2):
        if (type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum
                or type(lookahead) is not int or lookahead < 2):
            raise ValueError("ordered positive suffix lengths required")
        self.minimum, self.maximum = minimum, maximum
        self.lookahead = lookahead
        self.tokens, self.index = [], {}
        self.extend(tokens)

    def extend(self, tokens):
        for token in tokens:
            self.tokens.append(int(token))
            start = len(self.tokens) - self.lookahead
            for n in range(self.minimum, min(self.maximum, start) + 1):
                self.index[tuple(self.tokens[start - n:start])] = start

    def find(self, length):
        if type(length) is not int or length < 2:
            raise ValueError("root and at least one proposal required")
        for n in range(min(self.maximum, len(self.tokens)), self.minimum - 1, -1):
            start = self.index.get(tuple(self.tokens[-n:]))
            if start is not None:
                return self.tokens[start:start + length], n
        return [], 0


@dataclass
class CopyFeedback:
    baseline: torch.Tensor
    copied: torch.Tensor
    weights: torch.Tensor
    active: torch.Tensor
    group: int = 0


def copy_mixture(baseline, weights, copied, exponential):
    """Parallel scan equals a conditional copy-until-divergence proposal walk.

    Independent exponential rows supply both hypothetical categorical draws.
    A row's branch is selected using preceding rows only. The root is exact.
    """
    race = lambda q: (q / exponential.clamp_min(torch.finfo(q.dtype).tiny)).argmax(-1)
    valid = copied[1:] >= 0
    amount = weights * valid
    mixed = baseline.clone()
    mixed[1:] *= 1 - amount[:, None]
    mixed[1:].scatter_add_(-1, copied[1:].clamp_min(0)[:, None], amount[:, None])
    raw_tokens, mixed_tokens = race(baseline), race(mixed)
    matches = mixed_tokens[:-1] == copied[:-1]
    active = matches.long().cumprod(0).bool() & valid
    gate = torch.cat((active.new_zeros(1), active))
    q = torch.where(gate[:, None], mixed, baseline)
    tokens = torch.where(gate, mixed_tokens, raw_tokens)
    return tokens, q, CopyFeedback(baseline[1:], copied[1:], amount, active)


def copy_tv_gradient(feedback, teacher):
    """TV derivative for (1-lambda) q0 + lambda delta_c on reached rows."""
    n = len(teacher)
    q0, copied = feedback.baseline[:n], feedback.copied[:n].clamp_min(0)
    amount, active = feedback.weights[:n], feedback.active[:n]
    q = q0 * (1 - amount[:, None])
    q.scatter_add_(-1, copied[:, None], amount[:, None])
    deficit = q < teacher
    gradient = (q0 * deficit).sum(-1) - deficit.gather(-1, copied[:, None])[:, 0].to(q0.dtype)
    return gradient * active, q


class ContinuationMix:
    """Online bounded coefficients for suffix lengths 2, 3 and at least 4.

    Learned coefficients survive requests. Each request gets its own text index.
    The base model and diffusion adapter remain frozen throughout.
    """

    kind = "continuation"
    temperatures = ()

    def __init__(self, block_size, top_k, *, learning_rate=.5, interval=8, adaptive=True,
                 diagnostics=False, device="cpu", dtype=torch.float32, start_depth=1):
        if (type(block_size) is not int or block_size < 2 or type(top_k) is not int or top_k < 0
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or type(interval) is not int or interval < 1 or dtype not in (torch.float32, torch.float64)
                or type(start_depth) is not int or not 1 <= start_depth < block_size):
            raise ValueError("valid block, sampling support, precision and update cadence required")
        self.block_size, self.top_k = block_size, top_k
        self.learning_rate, self.interval = learning_rate, interval
        self.adaptive, self.diagnostics = adaptive, diagnostics
        self.start_depth = start_depth
        self.weights = torch.zeros(3, block_size - 1, device=device, dtype=dtype)
        self.gradient, self.counts, self.steps = [torch.zeros_like(self.weights) for _ in range(3)]
        self.totals = self.weights.new_zeros(block_size - 1, 3)
        self.updates = self.feedback_blocks = 0
        self.update_seconds = 0.
        self.memory = SuffixLookup(lookahead=self.start_depth + 1)
        self.group, self.copied = 0, []

    def begin_request(self, tokens):
        self.memory = SuffixLookup(tokens, lookahead=self.start_depth + 1)
        self.copied = []

    def commit(self, tokens):
        self.memory.extend(tokens)

    def lookup(self, length):
        self.copied, match = self.memory.find(length)
        self.group = min(match - 2, 2) if match else 0
        if len(self.copied) <= self.start_depth:
            self.copied = []
        return self.copied

    @torch.no_grad()
    def draft(self, baseline, generator=None):
        copied = self.lookup(len(baseline))
        exponential = torch.empty_like(baseline).exponential_(generator=generator)
        if not copied:
            return (baseline / exponential.clamp_min(torch.finfo(baseline.dtype).tiny)).argmax(-1), baseline, None
        ids = torch.tensor(copied + [-1] * (len(baseline) - len(copied)), device=baseline.device)
        tokens, q, feedback = copy_mixture(baseline, self.weights[self.group, :len(baseline) - 1], ids, exponential)
        feedback.group = self.group
        return tokens, q, feedback

    @torch.no_grad()
    def observe(self, feedback, teacher, *, root=None):
        n = len(teacher)
        if (feedback is None or n < self.start_depth or not (self.adaptive or self.diagnostics)
                or (root is not None and self.copied and root != self.copied[0])):
            return
        gradient, q = copy_tv_gradient(feedback, teacher)
        active = feedback.active[:n].clone()
        active[:self.start_depth - 1] = False
        gradient[:self.start_depth - 1] = 0
        if self.diagnostics:
            before = .5 * (feedback.baseline[:n] - teacher).abs().sum(-1)
            after = .5 * (q - teacher).abs().sum(-1)
            self.totals[:n] += torch.stack((active, before * active, after * active), -1)
        self.feedback_blocks += 1
        if self.adaptive:
            self.gradient[feedback.group, :n] += gradient
            self.counts[feedback.group, :n] += active
            if self.feedback_blocks % self.interval == 0:
                self.step()

    @torch.no_grad()
    def step(self):
        if self.weights.is_cuda:
            torch.cuda.synchronize(self.weights.device)
        start = time.perf_counter()
        active = self.counts > 0
        rate = self.learning_rate / (self.steps + 1).sqrt()
        self.weights.sub_(rate * self.gradient / self.counts.clamp_min(1)).clamp_(0, 1)
        self.steps += active
        self.gradient.zero_()
        self.counts.zero_()
        self.updates += 1
        if self.weights.is_cuda:
            torch.cuda.synchronize(self.weights.device)
        self.update_seconds += time.perf_counter() - start

    def _config(self):
        return {"block_size": self.block_size, "top_k": self.top_k, "learning_rate": self.learning_rate,
                "interval": self.interval, "dtype": str(self.weights.dtype), "kind": self.kind,
                "start_depth": self.start_depth}

    def state_dict(self):
        return {"config": self._config(), "tensors": {name: getattr(self, name).detach().cpu().clone()
                for name in ("weights", "gradient", "counts", "steps")},
                "updates": self.updates, "feedback_blocks": self.feedback_blocks, "update_seconds": self.update_seconds}

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("config") != self._config():
            raise ValueError("matching continuation mixture configuration required")
        tensors = state.get("tensors", {})
        if set(tensors) != {"weights", "gradient", "counts", "steps"}:
            raise ValueError("exact continuation state tensor keys required")
        for name, value in tensors.items():
            if (not isinstance(value, torch.Tensor) or value.shape != self.weights.shape
                    or value.dtype != self.weights.dtype or not torch.isfinite(value).all()):
                raise ValueError("finite matching continuation tensors required")
        if ((tensors["weights"] < 0).any() or (tensors["weights"] > 1).any()
                or any((tensors[k][:, :self.start_depth - 1] != 0).any() for k in tensors)
                or any((tensors[k] < 0).any() for k in ("counts", "steps"))
                or any(type(state.get(k)) is not int or state[k] < 0 for k in ("updates", "feedback_blocks"))
                or not math.isfinite(state.get("update_seconds", float("nan"))) or state["update_seconds"] < 0):
            raise ValueError("bounded weights and nonnegative state counters required")
        for name, value in tensors.items():
            getattr(self, name).copy_(value)
        self.updates, self.feedback_blocks = state["updates"], state["feedback_blocks"]
        self.update_seconds = state["update_seconds"]
        self.totals.zero_()

    def metrics(self):
        values = self.totals.cpu()
        count = values[:, 0].clamp_min(1)
        return {"kind": self.kind, "start_depth": self.start_depth,
                "learned_coefficients": 3 * (self.block_size - self.start_depth), "weights": self.weights.cpu().tolist(),
                "depth_observations": values[:, 0].tolist(), "depth_base_tv": (values[:, 1] / count).tolist(),
                "depth_mixed_tv": (values[:, 2] / count).tolist(), "updates": self.updates,
                "feedback_blocks": self.feedback_blocks, "update_seconds": self.update_seconds}
