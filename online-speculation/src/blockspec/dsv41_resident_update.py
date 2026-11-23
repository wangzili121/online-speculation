"""Strict orchestration of a DSV4.1 resident last-N reverse pass.

The PyPTO leaves remain small, independently testable kernels.  This module
defines their production ordering and passes the exact states captured by the
serving forward to those leaves.  It deliberately has no fallback that
reconstructs routing, mHC gates, or Attention selection from a newer model
version.
"""

from collections.abc import Mapping

import torch

from .dsv41_device_update import SuffixGradient
from .dsv41_suffix import DSV41SuffixPlan
from .dsv41_tp_gradient_layout import DSV41TPGradientLayout


class DSV41ResidentUpdateBackend:
    """Compose rank-local PyPTO VJPs, collectives, optimizer and publication.

    ``primitives`` is the resident execution owner.  It launches the already
    compiled PyPTO leaves and owns FP32 master weights/moments.  Keeping this
    orchestration separate from the leaves lets the serving integration audit
    the exact update order and prevents a Torch/autograd fallback in service.
    """

    def __init__(self, primitives, *, last_layers):
        self.primitives = primitives
        self.plan = DSV41SuffixPlan(last_layers)
        self.tp_layout = DSV41TPGradientLayout()
        # The implemented reverse kernel is C1A (ratio-1), which covers layers
        # 20..39.  Do not silently apply it to C2A or the two initial SWA
        # layers; the first GSM8K scan (last 1/2/4/8) is wholly inside C1A.
        if self.plan.first_layer < 20:
            raise ValueError("resident online update currently supports the C1A suffix (last-N <= 20)")
        required = {
            "loss_vjp", "lm_head_vjp", "reverse_frozen_tail",
            "reverse_draft_attention", "tp_assemble_gradients",
            "dp_all_reduce", "gradient_norm_and_clip", "adamw",
            "zero_gradients", "publication_state", "read_scalar",
        }
        missing = sorted(name for name in required if not callable(getattr(primitives, name, None)))
        if missing:
            raise TypeError(f"resident update primitives missing {missing}")

    @staticmethod
    def _rank_rows(value, *, label):
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{label} must contain one row per owner TP rank")
        if any(type(rank) is not int for rank in value):
            raise ValueError(f"{label} rank keys must be integers")
        return value

    def captured_output_state(self, layer):
        """Use the final captured hidden/pre-mix pair from the same draft pass."""
        rows = self._rank_rows(layer.ffn, label="captured output state")
        return {
            rank: {
                "hidden": row["output_hidden"],
                "pre_mix": row["output_pre_mix"],
                **({
                    "token_begin": row["token_begin"],
                    "token_end": row["token_end"],
                } if "token_begin" in row else {}),
            }
            for rank, row in rows.items()
        }

    def forward_layer(self, layer, hidden, captured):
        # Online serving normally reuses captured student logits.  A replay
        # forward is intentionally explicit rather than silently using Torch.
        replay = getattr(self.primitives, "forward_layer", None)
        if not callable(replay):
            raise RuntimeError("resident suffix replay forward is not configured")
        return replay(layer, hidden, captured)

    def project_logits(self, hidden):
        project = getattr(self.primitives, "project_logits", None)
        if not callable(project):
            raise RuntimeError("resident LM-head forward is not configured")
        return project(hidden)

    def loss_vjp(self, logits, teacher, valid, kind):
        if (
            not torch.is_tensor(logits) or not torch.is_tensor(teacher)
            or logits.ndim != 2 or teacher.ndim != 2
            or logits.shape[1] != teacher.shape[1]
            or logits.shape[0] < teacher.shape[0]
            or valid != teacher.shape[0]
        ):
            raise ValueError("student and verifier logits must be aligned full-vocabulary rows")
        # A captured K32 draft owns 31 student rows, while delayed online
        # feedback deliberately supervises only the authoritative prefix that
        # has become available (normally 1..7 rows).  Never pad the teacher or
        # train rows beyond that corrected prefix.
        return self.primitives.loss_vjp(logits[:valid], teacher, valid, kind)

    def project_logits_vjp(self, gradient, state):
        rows = self._rank_rows(state, label="LM-head replay state")
        result = self.primitives.lm_head_vjp(gradient, rows)
        if not isinstance(result, SuffixGradient):
            raise TypeError("LM-head VJP must return hidden and pre-mix gradients")
        return result

    def reverse_layer(self, layer, captured, state, boundary, *, synchronization=None):
        """Reverse one complete C1A layer with fixed forward decisions.

        The frozen tail consumes captured mHC and MoE state first.  Its input
        gradient then enters the trainable draft Attention VJP.  TP shards are
        assembled only after every owner rank has produced its local result.
        """
        if layer != captured.layer or layer not in self.plan.layers:
            raise ValueError("captured layer does not match the selected suffix")
        attention = self._rank_rows(captured.attention, label="Attention replay")
        ffn = self._rank_rows(captured.ffn, label="mHC/MoE replay")
        if set(attention) != set(ffn):
            raise ValueError("Attention and mHC/MoE captures cover different TP ranks")

        if isinstance(state, Mapping) and set(state) == set(attention):
            states = state
        else:
            states = {rank: state for rank in attention}
        if isinstance(boundary.hidden, Mapping) and isinstance(boundary.pre_mix, Mapping):
            if set(boundary.hidden) != set(attention) or set(boundary.pre_mix) != set(attention):
                raise ValueError("suffix gradient does not cover the owner TP ranks")
            gradients = {
                rank: SuffixGradient(boundary.hidden[rank], boundary.pre_mix[rank])
                for rank in attention
            }
        else:
            gradients = {rank: boundary for rank in attention}

        ranks = tuple(sorted(attention))
        if synchronization is None:
            tails = {
                rank: self.primitives.reverse_frozen_tail(
                    layer=layer, rank=rank, captured=ffn[rank],
                    state=states[rank], output_gradient=gradients[rank],
                )
                for rank in ranks
            }
        else:
            reverse = getattr(self.primitives, "reverse_frozen_tail_owner_batch", None)
            if not callable(reverse):
                raise RuntimeError(
                    "synchronized DSV4.1 reverse requires the resident primitive "
                    "hook reverse_frozen_tail_owner_batch"
                )
            tails = reverse(
                layer=layer, ranks=ranks, captured=ffn, states=states,
                output_gradients=gradients, synchronization=synchronization,
            )
            if not isinstance(tails, Mapping) or set(tails) != set(ranks):
                raise ValueError("owner frozen-tail batch must return every TP rank")

        rank_inputs, rank_parameters = {}, {}
        for rank in ranks:
            tail = tails[rank]
            if not isinstance(tail, SuffixGradient):
                raise TypeError("frozen-tail VJP must return a suffix gradient")
            incoming, parameters = self.primitives.reverse_draft_attention(
                layer=layer, rank=rank, captured=attention[rank],
                output_gradient=tail,
            )
            if not isinstance(incoming, SuffixGradient) or not isinstance(parameters, Mapping):
                raise TypeError("draft Attention VJP returned an invalid result")
            rank_inputs[rank], rank_parameters[rank] = incoming, dict(parameters)

        # Validate complete lane ownership and the public eight-parameter ABI
        # before a backend performs device collectives/concatenation.
        self.tp_layout.lanes(rank_parameters)
        assembled = self.primitives.tp_assemble_gradients(layer, rank_parameters)
        prefix = f"layers.{layer}.attn."
        expected = {
            name.removeprefix(prefix) for name in self.plan.trainable_names()
            if name.startswith(prefix)
        }
        if set(assembled) != expected:
            raise ValueError(f"layer {layer} TP assembly changed the draft parameter ABI")
        return SuffixGradient(
            {rank: value.hidden for rank, value in rank_inputs.items()},
            {rank: value.pre_mix for rank, value in rank_inputs.items()},
        ), assembled

    def participate_reverse(self, layers, synchronization):
        """Enter remote routed-expert VJPs in the owner's exact layer order."""
        layers = tuple(layers)
        if layers != self.plan.layers:
            raise ValueError("reverse participant received a different last-N suffix")
        participate = getattr(self.primitives, "participate_frozen_tail_peer", None)
        if not callable(participate):
            raise RuntimeError(
                "synchronized DSV4.1 reverse requires the resident primitive "
                "hook participate_frozen_tail_peer"
            )
        for layer in reversed(layers):
            participate(layer=layer, synchronization=synchronization)

    def dp_all_reduce(self, gradients):
        return self.primitives.dp_all_reduce(gradients)

    def zero_gradients(self, layers):
        return self.primitives.zero_gradients(tuple(layers), self.plan.trainable_names())

    def gradient_norm_and_clip(self, gradients, maximum):
        return self.primitives.gradient_norm_and_clip(gradients, maximum)

    def adamw(self, gradients, *, rate, step, clip):
        return self.primitives.adamw(gradients, rate=rate, step=step, clip=clip)

    def read_scalar(self, value):
        return self.primitives.read_scalar(value)

    def publication_state(self, layers):
        if tuple(layers) != self.plan.layers:
            raise ValueError("publication requested a different last-N suffix")
        return self.primitives.publication_state(self.plan.layers)
