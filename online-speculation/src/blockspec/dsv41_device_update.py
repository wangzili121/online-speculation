"""Full last-N update orchestration for the native DSV4.1 PyPTO backend."""

from dataclasses import dataclass
import math

import torch

from .dsv41_suffix import DSV41SuffixPlan, DSV41SuffixReplay


@dataclass(frozen=True)
class SuffixGradient:
    """Gradients crossing a DSV4.1 layer boundary."""

    hidden: object
    pre_mix: object


class DSV41DeviceSuffixLearner:
    """Compose PyPTO leaves into one exact fixed-route suffix update.

    The backend owns resident tensors and launches.  This object fixes the
    algorithmic order: forward the complete suffix, form a full-vocabulary
    loss, reverse frozen output/mHC/MoE code, reverse trainable Attention, DP
    reduce the parameter gradients, update FP32 masters, then expose one
    versioned publication payload.
    """

    def __init__(self, backend, *, last_layers, learning_rate=1e-5):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("positive finite learning rate required")
        self.backend = backend
        self.plan = DSV41SuffixPlan(last_layers)
        self.last_layers = last_layers
        self.learning_rate = learning_rate
        self.version = 0

    def _forward(self, replay):
        hidden = replay.boundary
        states = []
        for layer, captured in zip(self.plan.layers, replay.layers, strict=True):
            hidden, state = self.backend.forward_layer(layer, hidden, captured)
            states.append(state)
        logits, output_state = self.backend.project_logits(hidden)
        return logits, output_state, states

    def update_rows(self, replay: DSV41SuffixReplay, teacher, *, kind="tv", learning_rate=None):
        if replay.plan != self.plan:
            raise ValueError("device learner received a different last-N replay plan")
        if (
            not torch.is_tensor(teacher) or teacher.ndim != 2 or not teacher.is_floating_point()
            or not 1 <= teacher.shape[0] < 32 or not torch.isfinite(teacher).all()
        ):
            raise ValueError("finite contiguous verifier rows required")
        rate = self.learning_rate if learning_rate is None else learning_rate
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("positive finite update rate required")
        logits, output_state, states = self._forward(replay)
        loss, gradient = self.backend.loss_vjp(logits, teacher, teacher.shape[0], kind)
        boundary = self.backend.project_logits_vjp(gradient, output_state)
        parameter_gradients = {}
        for layer, captured, state in reversed(tuple(zip(
            self.plan.layers, replay.layers, states, strict=True,
        ))):
            # ``reverse_layer`` launches, in order, final/next-layer input VJP,
            # FFN hc_post, fixed-route MoE, FFN norm, hc_pre/Sinkhorn,
            # Attention hc_post, and C1A Attention VJP.
            boundary, gradients = self.backend.reverse_layer(layer, captured, state, boundary)
            prefix = f"layers.{layer}.attn."
            expected = {name.removeprefix(prefix) for name in self.plan.trainable_names()
                        if name.startswith(prefix)}
            if set(gradients) != expected:
                raise ValueError(f"layer {layer} returned an incomplete draft gradient")
            parameter_gradients.update({prefix + name: value for name, value in gradients.items()})
        # Only the request-owning DP contributes nonzero rows.  All DP replicas
        # participate so each TP lane obtains the same gradient and head version.
        reduced = self.backend.dp_all_reduce(parameter_gradients)
        expected = set(self.plan.trainable_names())
        if set(reduced) != expected:
            raise ValueError("DP reduction changed the trainable suffix ABI")
        norm, clip = self.backend.gradient_norm_and_clip(reduced, maximum=1.)
        self.version += 1
        self.backend.adamw(reduced, rate=rate, step=self.version, clip=clip)
        return float(self.backend.read_scalar(loss)), float(self.backend.read_scalar(norm)), self.version

    def update_zero(self, *, learning_rate, update_number):
        """Join the owner's DP reduction from a replica without teacher rows.

        Every DP replica must enter the same HCCL calls.  Non-owning replicas
        contribute exactly zero parameter gradients, then receive the owner's
        reduced gradient and apply the identical FP32 optimizer step.
        """
        if update_number != self.version + 1:
            raise ValueError("zero-gradient participant received an out-of-order update")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("positive finite update rate required")
        gradients = self.backend.zero_gradients(self.plan.layers)
        if set(gradients) != set(self.plan.trainable_names()):
            raise ValueError("zero-gradient participant has an incomplete suffix ABI")
        reduced = self.backend.dp_all_reduce(gradients)
        norm, clip = self.backend.gradient_norm_and_clip(reduced, maximum=1.)
        self.version = update_number
        self.backend.adamw(reduced, rate=learning_rate, step=self.version, clip=clip)
        return 0., float(self.backend.read_scalar(norm)), self.version

    def publication_state(self):
        values = self.backend.publication_state(self.plan.layers)
        if set(values) != set(self.plan.trainable_names()):
            raise ValueError("device publication is not the complete selected suffix")
        return values
