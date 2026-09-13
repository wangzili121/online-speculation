"""Online convex mixing of sparse proposal temperatures on a frozen drafter."""

from dataclasses import dataclass
import math
import time

import torch


def project_simplex(values):
    """Euclidean projection of each last-axis vector onto nonnegative unit sum."""
    ordered = values.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(-1) - 1
    ranks = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    rho = (ordered - cumulative / ranks > 0).sum(-1, keepdim=True) - 1
    threshold = cumulative.gather(-1, rho) / (rho + 1).to(values.dtype)
    return (values - threshold).clamp_min(0)


@dataclass
class CalibrationFeedback:
    indices: torch.Tensor
    experts: torch.Tensor
    mixed: torch.Tensor
    baseline: torch.Tensor


def mix_rows(baseline, weights, powers, identity_index, top_k, protected_rows=1):
    """Pure tensor proposal map, shared by eager and graph execution."""
    compact, indices = baseline[protected_rows:].topk(min(top_k, baseline.shape[-1]), dim=-1)
    experts = compact[:, None, :].pow(powers)
    experts = experts / experts.sum(-1, keepdim=True).clamp_min(torch.finfo(experts.dtype).tiny)
    experts[:, identity_index] = compact
    mixed = (experts * weights[:len(compact), :, None]).sum(1)
    q = torch.zeros_like(baseline)
    q[:protected_rows] = baseline[:protected_rows]
    q[protected_rows:].scatter_(-1, indices, mixed)
    return q, CalibrationFeedback(indices, experts, mixed, compact)


class OverlapMix:
    """Per-depth simplex updates minimize TV on the actually reached prefixes.

    Temperatures act on the original truncated proposal's positive support.
    The identity expert is exact. The target law and clean root are unchanged.
    Stored proposal snapshots are consumed before an online update.
    """

    def __init__(self, block_size, top_k, *, temperatures=(.5, .75, 1., 1.25, 1.5),
                 learning_rate=.5, interval=8, adaptive=True, fixed_temperature=1.,
                 diagnostics=False, device="cpu", dtype=torch.float32, protected_rows=1):
        if (type(block_size) is not int or block_size < 2 or type(top_k) is not int or top_k < 1
                or not temperatures or len(set(temperatures)) != len(temperatures)
                or any(not math.isfinite(t) or t <= 0 for t in temperatures)
                or 1. not in temperatures or fixed_temperature not in temperatures
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or type(interval) is not int or interval < 1 or type(protected_rows) is not int
                or protected_rows not in (0, 1)):
            raise ValueError("block >=2, positive bounded support, distinct positive temperatures including identity required")
        self.block_size, self.top_k = block_size, top_k
        self.protected_rows = protected_rows
        self.temperatures = tuple(temperatures)
        self.learning_rate, self.interval, self.adaptive = learning_rate, interval, adaptive
        self.diagnostics = diagnostics
        self.fixed_index = self.temperatures.index(fixed_temperature)
        self.identity_index = self.temperatures.index(1.)
        self.weights = torch.zeros(block_size - 1, len(temperatures), device=device, dtype=dtype)
        self.powers = self.weights.new_tensor(temperatures).reciprocal()[None, :, None]
        self.gradient = torch.zeros_like(self.weights)
        self.counts = self.weights.new_zeros(block_size - 1, 1)
        self.steps = torch.zeros_like(self.counts)
        self.totals = self.weights.new_zeros(block_size - 1, len(temperatures) + 3)
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.weights.zero_()
        self.weights[:, self.fixed_index] = 1
        self.gradient.zero_()
        self.counts.zero_()
        self.steps.zero_()
        self.totals.zero_()
        self.updates = self.feedback_blocks = 0
        self.update_seconds = 0.

    @torch.no_grad()
    def propose(self, baseline):
        if (baseline.ndim != 2 or not self.protected_rows < len(baseline) <= self.block_size - 1 + self.protected_rows
                or baseline.device != self.weights.device or baseline.dtype != self.weights.dtype):
            raise ValueError("matching floating rows and protected-prefix layout required")
        return mix_rows(baseline, self.weights, self.powers, self.identity_index, self.top_k, self.protected_rows)

    @torch.no_grad()
    def observe(self, feedback, teacher):
        n = len(teacher)
        if not n or not (self.adaptive or self.diagnostics):
            return
        if n > len(feedback.indices):
            raise ValueError("teacher feedback must be an actually reached proposal prefix")
        p = teacher.gather(-1, feedback.indices[:n])
        experts, q = feedback.experts[:n], feedback.mixed[:n]
        if self.diagnostics:
            expert_tv = 1 - torch.minimum(p[:, None, :], experts).sum(-1)
            tv = 1 - torch.minimum(p, q).sum(-1)
            # Columns: count, actual pre-update mixture TV, support mass, expert TVs.
            self.totals[:n] += torch.cat((torch.ones_like(tv[:, None]), tv[:, None],
                                         p.sum(-1, keepdim=True), expert_tv), -1)
        self.feedback_blocks += 1
        if self.adaptive:
            subgradient = -(experts * (q < p)[:, None, :]).sum(-1)
            self.gradient[:n] += subgradient
            self.counts[:n] += 1
            if self.feedback_blocks % self.interval == 0:
                self.step()

    @torch.no_grad()
    def step(self):
        if self.weights.is_cuda:
            torch.cuda.synchronize(self.weights.device)
        start = time.perf_counter()
        valid = self.counts > 0
        gradient = self.gradient / self.counts.clamp_min(1)
        rate = self.learning_rate / (self.steps + 1).sqrt()
        updated = project_simplex(self.weights - rate * gradient)
        self.weights.copy_(torch.where(valid, updated, self.weights))
        self.steps += valid
        self.gradient.zero_()
        self.counts.zero_()
        self.updates += 1
        if self.weights.is_cuda:
            torch.cuda.synchronize(self.weights.device)
        self.update_seconds += time.perf_counter() - start

    def _state_config(self):
        return {"block_size": self.block_size, "top_k": self.top_k,
                "temperatures": self.temperatures, "learning_rate": self.learning_rate,
                "interval": self.interval, "dtype": str(self.weights.dtype), "protected_rows": self.protected_rows}

    def state_dict(self):
        """Owned portable state; restoring keeps the destination tensor storage."""
        return {"config": self._state_config(),
                "tensors": {name: getattr(self, name).detach().cpu().clone()
                            for name in ("weights", "gradient", "counts", "steps")},
                "updates": self.updates, "feedback_blocks": self.feedback_blocks,
                "update_seconds": self.update_seconds}

    @torch.no_grad()
    def load_state_dict(self, state):
        """Restore learned parameters and cadence for frozen or live comparisons."""
        config = dict(state.get("config", {}))
        config.setdefault("protected_rows", 1)
        if config != self._state_config():
            raise ValueError("calibration state must match shape, temperatures, precision and update policy")
        tensors = state.get("tensors", {})
        if set(tensors) != {"weights", "gradient", "counts", "steps"}:
            raise ValueError("exact calibration tensor keys required")
        for name in ("weights", "gradient", "counts", "steps"):
            value, destination = tensors.get(name), getattr(self, name)
            if (not isinstance(value, torch.Tensor) or value.shape != destination.shape
                    or value.dtype != destination.dtype or not torch.isfinite(value).all()):
                raise ValueError("finite matching calibration state tensors required")
        w = tensors["weights"]
        if ((w < 0).any() or not torch.allclose(w.sum(-1), torch.ones_like(w[:, 0]), atol=1e-6, rtol=1e-6)
                or any((tensors[name] < 0).any() for name in ("counts", "steps"))
                or any(type(state.get(name)) is not int or state[name] < 0 for name in ("updates", "feedback_blocks"))
                or not math.isfinite(state.get("update_seconds", float("nan"))) or state["update_seconds"] < 0):
            raise ValueError("normalized mixture and nonnegative calibration counters required")
        for name, value in tensors.items():
            getattr(self, name).copy_(value)
        self.updates, self.feedback_blocks = state["updates"], state["feedback_blocks"]
        self.update_seconds = state["update_seconds"]
        self.totals.zero_()

    def metrics(self):
        values = self.totals.detach().cpu()
        count = values[:, :1].clamp_min(1)
        return {"temperatures": self.temperatures, "weights": self.weights.detach().cpu().tolist(),
                "depth_observations": values[:, 0].tolist(),
                "depth_preupdate_tv": (values[:, 1] / count[:, 0]).tolist(),
                "depth_support_mass": (values[:, 2] / count[:, 0]).tolist(),
                "depth_expert_tv": (values[:, 3:] / count).tolist(),
                "updates": self.updates, "feedback_blocks": self.feedback_blocks,
                "update_seconds": self.update_seconds}
