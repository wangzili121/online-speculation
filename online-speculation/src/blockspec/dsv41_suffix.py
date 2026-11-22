"""Contracts for replaying and updating a DSV4.1 Orthrus layer suffix.

This module intentionally separates *which* layers are trainable from *how*
the serving backend executes their frozen surroundings.  The PyPTO backend
captures one boundary before the first selected layer and replays every layer
through layer 39; otherwise changing an early draft layer would leave stale
downstream hidden states.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch


_ATTENTION_PARAMETERS = (
    "wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight",
    "kv_norm.weight", "attn_sink", "wo_a.weight", "wo_b.weight",
)


@dataclass(frozen=True)
class DSV41SuffixPlan:
    """Static trainable-layer contract for the 40-layer DSV4.1 text model."""

    last_layers: int
    total_layers: int = 40

    def __post_init__(self):
        if type(self.last_layers) is not int or not 1 <= self.last_layers <= self.total_layers:
            raise ValueError("last_layers must select a nonempty DSV4.1 suffix")
        if self.total_layers != 40:
            raise ValueError("this contract targets the 40-layer DSV4.1 text backbone")

    @property
    def first_layer(self):
        return self.total_layers - self.last_layers

    @property
    def layers(self):
        return tuple(range(self.first_layer, self.total_layers))

    @property
    def reindex_layers(self):
        # DSV4.1 C1A owns a fresh indexer every four layers starting at 20.
        return tuple(layer for layer in self.layers if layer >= 20 and (layer - 20) % 4 == 0)

    def trainable_names(self):
        """Draft Attention parameters; C1A indexers stay frozen online.

        Top-K selection is discrete.  Reusing the forward selection gives an
        exact VJP for the selected Attention path; changing indexer weights
        requires a separate indexer-distillation objective and is deliberately
        not hidden inside the online TV/KL update.
        """
        return tuple(
            f"layers.{layer}.attn.{name}"
            for layer in self.layers
            for name in _ATTENTION_PARAMETERS
        )


@dataclass(frozen=True)
class DSV41SuffixReplay:
    """One authoritative K32 replay boundary plus backend-owned layer state."""

    plan: DSV41SuffixPlan
    boundary: object
    layers: tuple[object, ...]
    student_logits: object | None = None

    def __post_init__(self):
        if self.boundary is None:
            raise ValueError("suffix replay boundary must be a tensor or backend-owned handle")
        if len(self.layers) != self.plan.last_layers:
            raise ValueError("suffix replay must contain one state per selected layer")
        if self.student_logits is not None and (
            not torch.is_tensor(self.student_logits) or self.student_logits.ndim != 2
            or not self.student_logits.is_floating_point()
            or not 1 <= self.student_logits.shape[0] < 32
        ):
            raise ValueError("captured student logits must be K32 full-vocabulary rows")


class DSV41SuffixLearner:
    """Backend-neutral last-N optimizer with a true suffix replay.

    ``replay_layer`` must recompute both the trainable draft Attention and the
    frozen mHC/MoE tail for a layer.  Therefore gradients from the final logits
    reach every selected Attention, while target-model parameters remain absent
    from the optimizer.  PyPTO supplies this callback with explicit VJPs;
    Torch tests may use ordinary autograd.
    """

    def __init__(
        self,
        parameters: Mapping[str, torch.nn.Parameter],
        *,
        last_layers: int,
        replay_layer: Callable[[int, torch.Tensor, object, Mapping[str, torch.Tensor]], torch.Tensor],
        project_logits: Callable[[torch.Tensor], torch.Tensor],
        loss: Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor],
        learning_rate=1e-5,
    ):
        self.plan = DSV41SuffixPlan(last_layers)
        self.last_layers = last_layers
        wanted = set(self.plan.trainable_names())
        if set(parameters) != wanted:
            missing, extra = sorted(wanted - set(parameters)), sorted(set(parameters) - wanted)
            raise ValueError(f"suffix parameter mismatch: missing={missing[:3]} extra={extra[:3]}")
        if any(not isinstance(value, torch.nn.Parameter) for value in parameters.values()):
            raise TypeError("suffix weights must be trainable Parameters")
        self.master = dict(parameters)
        self.replay_layer = replay_layer
        self.project_logits = project_logits
        self.loss_function = loss
        self.optimizer = torch.optim.AdamW(self.master.values(), lr=learning_rate, weight_decay=0)
        self.version = 0

    def logits(self, replay: DSV41SuffixReplay, *, reached: int):
        if replay.plan != self.plan or not 1 <= reached < 32:
            raise ValueError("replay plan or reached K32 rows do not match the learner")
        hidden = replay.boundary
        for layer, state in zip(self.plan.layers, replay.layers, strict=True):
            prefix = f"layers.{layer}.attn."
            weights = {name.removeprefix(prefix): value for name, value in self.master.items()
                       if name.startswith(prefix)}
            hidden = self.replay_layer(layer, hidden, state, weights)
        return self.project_logits(hidden[:reached])

    @torch.inference_mode(False)
    @torch.enable_grad()
    def update_rows(self, replay, teacher, *, kind="tv", learning_rate=None):
        if not torch.is_tensor(teacher) or teacher.ndim != 2 or teacher.shape[0] < 1:
            raise ValueError("one or more teacher-logit rows are required")
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        logits = self.logits(replay, reached=teacher.shape[0])
        loss = self.loss_function(logits, teacher.float(), kind)
        self.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite DSV4.1 suffix loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.master.values(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.version += 1
        return float(loss.detach()), float(norm), self.version

    def publication_state(self):
        """Return the complete suffix as one versioned publication payload."""
        return {name: value.detach().bfloat16().cpu().contiguous() for name, value in self.master.items()}
