"""Serving bridge from native K32 verification to DSV4.1 online updates."""

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from .dsv41_online_profile import active_profile
from .dsv41_suffix import DSV41SuffixPlan, DSV41SuffixReplay


@dataclass(frozen=True)
class DSV41SequenceSpan:
    """Intersection of a global K32 request and a rank's physical SP rows."""

    local_start: int
    block_start: int
    rows: int

    @classmethod
    def from_record(cls, record, begin, end):
        rank_start = record["cp_token_range"][0]
        rank_end = rank_start + record["SP_valid_rows"]
        start, stop = max(begin, rank_start), min(end, rank_end)
        if start >= stop:
            return cls(0, 0, 0)
        return cls(start - rank_start, start - begin, stop - start)


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
            "qkv", "contribution", "lse",
        }
        ffn = {
            "changed", "pre", "post", "comb", "collapsed", "normed",
            "router_input", "router_logits", "route_ids", "route_weights",
            "shared", "expert_outputs", "combined", "output_hidden", "output_pre_mix",
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
    student_logits: object | None = None
    release: object | None = None

    def __post_init__(self):
        if not self.request_id or not self.prefix or len(self.owner) != 3:
            raise ValueError("serving replay requires request, prefix and DP owner")
        if tuple(layer.layer for layer in self.layers) != self.plan.layers:
            raise ValueError("serving replay does not cover the selected suffix")

    def suffix(self):
        return DSV41SuffixReplay(self.plan, self.boundary, self.layers,
                                 self.student_logits, owner=self.owner, release=self.release)

    def with_student_logits(self, value):
        """Attach the draft distribution emitted by this exact captured pass."""
        return DSV41ServingReplay(
            self.request_id, self.prefix, self.owner, self.plan,
            self.boundary, self.layers, value, self.release,
        )


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
        self.remote_pending = None

    @property
    def plan(self):
        return DSV41SuffixPlan(self.controller.config.last_layers)

    def capture_due(self):
        due = self.controller.begin_verifier_round()
        profile = active_profile()
        return (due and not self.pending and self.remote_pending is None
                and (profile is None or profile.can_capture()))

    @staticmethod
    def _capture_rows(requests):
        return [
            {"owner": list(owner), "request_id": request_id}
            for owner, (request_id, _) in sorted(requests.items())
        ]

    def select_capture(self, requests, exchange=None):
        """Select one globally ordered request and mark the peer snapshot busy."""
        local = self._capture_rows(requests)
        peer = [] if exchange is None else exchange(
            f"online-capture/{self.controller.rounds}", local,
        )
        candidates = [(*row["owner"], row["request_id"], "local") for row in local]
        candidates += [(*row["owner"], row["request_id"], "peer") for row in peer]
        if not candidates:
            return {}
        chosen = min(candidates)
        owner, request_id, location = tuple(chosen[:3]), chosen[3], chosen[4]
        if location == "peer":
            self.remote_pending = {"owner": list(owner), "request_id": request_id}
            return {}
        return {owner: requests[owner]}

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
        elif replay.release is not None:
            replay.release()
        return captured

    def verified(
        self, request_id, prefix, candidate_inputs, delivered, logits, *,
        finished=False, defer=False,
    ):
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
        method = self.controller.prepare if defer else self.controller.observe
        record = method(request_id, prefix, inputs, delivered, teacher, finished=finished)
        if record is not None or finished:
            self.pending.pop(request_id, None)
        return record

    def continued(self, request_id, prefix, delivered, logits, *, finished=False, defer=False):
        """Add delayed teacher rows from later target-only decoding."""
        return self.verified(
            request_id, prefix, None, delivered, logits, finished=finished, defer=defer,
        )

    def apply(self, prepared):
        """Apply a prepared update after the cluster synchronization point."""
        return self.controller.apply(prepared)

    def synchronize(self, prepared, exchange):
        """Rendezvous both hosts before entering DP reduction and publication."""
        if prepared is not None and prepared is not self.controller.prepared:
            raise ValueError("serving tried to synchronize a foreign prepared update")
        update_number = self.controller.completed_updates + 1
        local = None if prepared is None else prepared.contract()
        peer = exchange(f"online-update/{update_number}", local)
        if local is None and peer is None:
            return None
        if local is not None and peer is not None:
            raise ValueError("only one host may own an online feedback block")
        contract = local if local is not None else peer
        if contract["update_number"] != update_number:
            raise ValueError("cluster online update versions disagree")
        def apply_update():
            if prepared is not None:
                return self.controller.apply(prepared, synchronization=contract)
            return self.controller.apply_remote(contract)

        profile = active_profile()
        if profile is None:
            record = apply_update()
        else:
            with profile.update(contract, owner_is_local=prepared is not None) as measured:
                record = apply_update()
                measured.update(
                    update_seconds=record["update_seconds"],
                    publication_seconds=record["publication_seconds"],
                    loss=record["loss"] if prepared is not None else None,
                    gradient_norm=record["gradient_norm"],
                )
        self.remote_pending = None
        return record

    def finish(self, request_id, *, defer=False):
        replay = self.pending.pop(request_id, None)
        prepared = self.controller.finish_request(request_id)
        if prepared is None and replay is not None and replay.release is not None:
            replay.release()
        if prepared is None or defer:
            return prepared
        return self.controller.apply(prepared)

    def summary(self):
        return self.controller.summary() | {
            "pending_replays": len(self.pending),
            "remote_pending": self.remote_pending,
        }
