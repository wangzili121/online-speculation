"""Serving bridge from native K32 verification to DSV4.1 online updates."""

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from .dsv41_suffix import DSV41SuffixPlan, DSV41SuffixReplay


@dataclass(frozen=True)
class DSV41CapturedLayer:
    """Device replay state retained for one selected layer and one K32 block.

    The values intentionally use the native layer names.  The learner can run
    the exact reverse path without reconstructing routing or Sinkhorn choices
    from a later model version.
    """

    layer: int
    attention: Mapping[str, object]
    ffn: Mapping[str, object]

    def __post_init__(self):
        attention = {
            "readonly_hidden", "hidden", "incoming", "pre", "post", "comb",
            "collapsed", "normed", "changed", "cos", "sin", "history", "selection",
        }
        ffn = {
            "changed", "pre", "post", "comb", "collapsed", "normed",
            "router_input", "router_logits", "route_ids", "route_weights",
            "shared", "combined", "output_hidden", "output_pre_mix",
        }
        attention_rows = self.attention.values() if self.attention and all(
            isinstance(key, int) for key in self.attention
        ) else (self.attention,)
        ffn_rows = self.ffn.values() if self.ffn and all(
            isinstance(key, int) for key in self.ffn
        ) else (self.ffn,)
        if not all(attention <= set(row) for row in attention_rows):
            raise ValueError(f"incomplete Attention replay at layer {self.layer}")
        if not all(ffn <= set(row) for row in ffn_rows):
            raise ValueError(f"incomplete mHC/MoE replay at layer {self.layer}")


@dataclass(frozen=True)
class DSV41ServingReplay:
    """A suffix boundary with exact native states and request ownership."""

    request_id: str
    prefix: tuple[int, ...]
    owner: tuple[int, int, int]
    plan: DSV41SuffixPlan
    boundary: object
    layers: tuple[DSV41CapturedLayer, ...]

    def __post_init__(self):
        if not self.request_id or not self.prefix or len(self.owner) != 3:
            raise ValueError("serving replay requires request, prefix and DP owner")
        if tuple(layer.layer for layer in self.layers) != self.plan.layers:
            raise ValueError("serving replay does not cover the selected suffix")

    def suffix(self):
        return DSV41SuffixReplay(self.plan, self.boundary, self.layers)


class DSV41ServingOnline:
    """Join native replay capture, verifier rows, optimizer and publication.

    The runner calls ``capture_due`` before a draft dispatch, then registers
    the completed replay.  ``verified`` consumes authoritative target rows in
    the exact order delivered to the request.  No rejected candidate branch
    can become training data.
    """

    def __init__(self, controller):
        self.controller = controller
        self.pending = {}

    @property
    def plan(self):
        return DSV41SuffixPlan(self.controller.config.last_layers)

    def capture_due(self):
        next_round = self.controller.rounds + 1
        return self.controller.config.should_capture(
            next_round, self.controller.completed_updates,
        )

    def register_replay(self, replay: DSV41ServingReplay):
        if replay.plan != self.plan:
            raise ValueError("captured replay and online last-N policy disagree")
        if replay.request_id in self.pending:
            raise ValueError("request already owns an online replay")
        captured = self.controller.begin_round(
            replay.request_id, replay.prefix, replay.suffix(),
        )
        if captured:
            self.pending[replay.request_id] = replay
        return captured

    def verified(self, request_id, prefix, candidate_inputs, delivered, logits, *, finished=False):
        """Consume verifier logits for tokens actually committed to the target."""
        if request_id not in self.pending:
            return None
        prefix = tuple(int(token) for token in prefix)
        delivered = tuple(int(token) for token in delivered)
        candidate_inputs = None if candidate_inputs is None else tuple(int(token) for token in candidate_inputs)
        if not delivered:
            raise ValueError("verification must deliver a correction, bonus or accepted token")
        if (
            not torch.is_tensor(logits) or logits.ndim != 2
            or logits.shape[0] < len(delivered) or not logits.is_floating_point()
        ):
            raise ValueError("full finite verifier-logit rows are required for online feedback")
        teacher = logits[:len(delivered)].detach().cpu()
        if not torch.isfinite(teacher).all():
            raise ValueError("non-finite verifier rows cannot train the draft")
        # Inputs are always rebuilt from the corrected branch.  Candidate IDs
        # are checked only for the rows that were genuinely accepted.
        inputs = (prefix[-1], *delivered[:-1])
        accepted = max(0, len(delivered) - 1)
        if candidate_inputs is not None and candidate_inputs[:accepted] != delivered[:accepted]:
            raise ValueError("delivered prefix is not the accepted candidate prefix")
        record = self.controller.observe(
            request_id, prefix, inputs, delivered, teacher, finished=finished,
        )
        if record is not None or finished:
            self.pending.pop(request_id, None)
        return record

    def continued(self, request_id, prefix, delivered, logits, *, finished=False):
        """Add delayed teacher rows from later target-only decoding."""
        return self.verified(request_id, prefix, None, delivered, logits, finished=finished)

    def finish(self, request_id):
        self.pending.pop(request_id, None)
        return self.controller.finish_request(request_id)

    def summary(self):
        return self.controller.summary() | {"pending_replays": len(self.pending)}
