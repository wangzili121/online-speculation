"""Online-feedback policy for the DSV4.1 Orthrus suffix learner.

The inference block remains K32.  A feedback block may supervise fewer rows;
later rows are collected only after the target has continued along the
corrected, authoritative prefix.
"""

from dataclasses import dataclass
import json
import math
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import torch

from .dsv41_suffix import DSV41SuffixPlan


def read_workload(path, limit=None):
    """Read unique JSONL request IDs while preserving scoring metadata."""
    entries = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("prompt"), str):
                raise ValueError(f"Invalid workload row {line_number}")
            entries.append(row)
            if limit is not None and len(entries) == limit:
                break
    if not entries:
        raise ValueError("The workload contains no requests")
    if len({row["id"] for row in entries}) != len(entries):
        raise ValueError("Workload IDs must be unique")
    return entries


@dataclass(frozen=True)
class DSV41OnlineConfig:
    """Conservative starting policy transferred from the Qwen3.8 study."""

    last_layers: int = 1
    teacher_rows: int = 7
    learning_rate: float = 1e-5
    warmup_updates: int = 8
    warm_update_stride: int = 16
    warm_update_steps: int = 256
    steady_update_stride: int = 128
    publish_every: int = 4
    objective: str = "tv"

    def __post_init__(self):
        integers = (
            self.last_layers,
            self.teacher_rows,
            self.warmup_updates,
            self.warm_update_stride,
            self.warm_update_steps,
            self.steady_update_stride,
            self.publish_every,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("positive integer DSV4.1 online dimensions required")
        if self.last_layers > 40:
            raise ValueError("DSV4.1 last-N cannot exceed the 40-layer text backbone")
        if not 1 <= self.teacher_rows < 32:
            raise ValueError("DSV4.1 teacher rows must fit inside the K32 draft block")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("a positive finite DSV4.1 online learning rate is required")
        if self.objective not in ("tv", "forward_kl"):
            raise ValueError("DSV4.1 online objective must be TV or forward KL")

    def update_interval(self, completed_updates: int) -> int:
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("completed update count must be nonnegative")
        return (
            self.warm_update_stride
            if completed_updates < self.warm_update_steps
            else self.steady_update_stride
        )

    def learning_rate_for(self, update_number: int) -> float:
        """Return the rate for a one-based optimizer update number."""
        if type(update_number) is not int or update_number < 1:
            raise ValueError("update number must be positive")
        return self.learning_rate * min(1.0, update_number / self.warmup_updates)

    def should_capture(self, verifier_round: int, completed_updates: int) -> bool:
        if type(verifier_round) is not int or verifier_round < 1:
            raise ValueError("verifier round must be positive")
        return verifier_round % self.update_interval(completed_updates) == 0

    def should_publish(self, completed_updates: int) -> bool:
        if type(completed_updates) is not int or completed_updates < 1:
            raise ValueError("completed update count must be positive")
        return completed_updates % self.publish_every == 0

    @classmethod
    def from_study(cls, path, *, overrides=None):
        """Load the candidate policy from the versioned GSM8K study file.

        Serving must not silently accept misspelled or newly added knobs: a
        configuration drift on only one host would deadlock collectives or
        publish a different head version.
        """
        document = json.loads(Path(path).read_text())
        if document.get("format") != "orthrus-dsv41-gsm8k-study-v1":
            raise ValueError("unsupported DSV4.1 online study format")
        values = document.get("candidate_default")
        expected = set(cls.__dataclass_fields__)
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError("candidate_default must define the complete online policy")
        values = dict(values)
        if overrides:
            unknown = set(overrides) - expected
            if unknown:
                raise ValueError(f"unknown DSV4.1 online overrides: {sorted(unknown)}")
            values.update(overrides)
        return cls(**values)

    def contract(self):
        """Stable cluster-visible policy used in startup identity checks."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class DSV41PreparedUpdate:
    """One authoritative feedback block, ready for a synchronized update."""

    request_id: str
    replay: object
    teacher: torch.Tensor
    update_number: int
    learning_rate: float
    objective: str

    def __post_init__(self):
        if (
            not self.request_id or not torch.is_tensor(self.teacher)
            or self.teacher.ndim != 2 or not self.teacher.is_floating_point()
            or not self.teacher.isfinite().all()
        ):
            raise ValueError("prepared update requires finite full-vocabulary teacher rows")
        if self.update_number < 1 or not math.isfinite(self.learning_rate):
            raise ValueError("prepared update has an invalid optimizer step")

    def contract(self):
        replay_owner = getattr(self.replay, "owner", None)
        plan = getattr(self.replay, "plan", None)
        if plan is None:
            raise RuntimeError("prepared update has no synchronized suffix plan")
        return {
            "request_id": self.request_id,
            "owner": None if replay_owner is None else list(replay_owner),
            "teacher_rows": self.teacher.shape[0],
            "vocabulary": self.teacher.shape[1],
            "update_number": self.update_number,
            "learning_rate": self.learning_rate,
            "objective": self.objective,
            "reverse": {
                "protocol": "dsv41-synchronized-reverse-v1",
                "layers": list(plan.layers),
            },
        }


class DSV41DelayedTeacherBlock:
    """Collect target rows along the corrected prefix for one K32 snapshot."""

    def __init__(self, request_id, prefix, replay, *, width=8):
        prefix = tuple(int(token) for token in prefix)
        if not request_id or not prefix or type(width) is not int or not 2 <= width <= 32:
            raise ValueError("request, authoritative prefix and width 2..32 required")
        self.request_id = request_id
        self.anchor = len(prefix) - 1
        self.width = width
        self.prefix = prefix
        self.replay = replay
        self.rows = []
        self.closed = False

    @property
    def full(self):
        return len(self.rows) == self.width - 1

    @property
    def ready(self):
        return self.full or self.closed and bool(self.rows)

    def append_verified(self, request_id, prefix, inputs, delivered, teacher_rows):
        """Append only rows actually verified on the corrected target branch."""
        if self.closed or self.full:
            raise ValueError("teacher block is already finished")
        prefix = tuple(int(token) for token in prefix)
        inputs = tuple(int(token) for token in inputs)
        delivered = tuple(int(token) for token in delivered)
        if request_id != self.request_id or prefix != self.prefix:
            raise ValueError("teacher rows do not continue the captured request prefix")
        if not torch.is_tensor(teacher_rows) or teacher_rows.ndim != 2 or not teacher_rows.is_floating_point():
            raise ValueError("teacher rows must be a floating [rows, vocabulary] tensor")
        if (
            not delivered
            or len(inputs) != len(delivered)
            or teacher_rows.shape[0] != len(delivered)
            or inputs != (prefix[-1], *delivered[:-1])
        ):
            raise ValueError("unverified or wrong-branch teacher rows cannot train the draft")
        count = min(len(delivered), self.width - 1 - len(self.rows))
        self.rows.extend(row.detach().cpu().clone() for row in teacher_rows[:count])
        self.prefix += delivered[:count]
        return count

    def finish(self):
        """Keep a real short block at EOS/budget exhaustion; never invent rows."""
        self.closed = True
        return self.ready

    def teacher(self):
        if not self.ready:
            raise RuntimeError("teacher block is not ready for an update")
        return torch.stack(self.rows)


class DSV41OnlineController:
    """Engine-neutral capture/update/publication state machine.

    The engine owns tensor capture and safe publication.  This class owns the
    schedule and refuses wrong-prefix feedback before it reaches the learner.
    """

    def __init__(self, learner, config=DSV41OnlineConfig(), *, publisher: Callable | None = None):
        if not hasattr(learner, "update_rows"):
            raise TypeError("DSV4.1 online learner must implement update_rows")
        if getattr(learner, "last_layers", config.last_layers) != config.last_layers:
            raise ValueError("online policy and suffix learner select different last-N layers")
        self.learner = learner
        self.config = config
        self.publisher = publisher
        self.rounds = self.completed_updates = self.publications = 0
        self.update_seconds = self.publication_seconds = 0.0
        self.blocks = {}
        self.prepared = None
        self.records = []

    def begin_verifier_round(self):
        """Advance the globally shared draft-round schedule once."""
        self.rounds += 1
        return not self.blocks and self.prepared is None and self.config.should_capture(
            self.rounds, self.completed_updates,
        )

    def begin_round(self, request_id, prefix, replay):
        """Register the single request selected for this verifier round."""
        if (
            self.rounds < 1 or self.blocks or self.prepared is not None
            or not self.config.should_capture(self.rounds, self.completed_updates)
        ):
            return False
        self.blocks[request_id] = DSV41DelayedTeacherBlock(
            request_id, prefix, replay, width=self.config.teacher_rows + 1,
        )
        return True

    def prepare(self, request_id, prefix, inputs, delivered, teacher_rows, *, finished=False):
        """Collect corrected rows without entering a device collective.

        Serving calls this while processing verifier output.  The returned
        object is deliberately side-effect free with respect to model weights,
        so both hosts can rendezvous before either enters DP all-reduce.
        """
        block = self.blocks.get(request_id)
        if block is None:
            return None
        block.append_verified(request_id, prefix, inputs, delivered, teacher_rows)
        if finished:
            block.finish()
        if not block.ready:
            return None
        del self.blocks[request_id]
        update_number = self.completed_updates + 1
        self.prepared = DSV41PreparedUpdate(
            request_id=request_id,
            replay=block.replay,
            teacher=block.teacher(),
            update_number=update_number,
            learning_rate=self.config.learning_rate_for(update_number),
            objective=self.config.objective,
        )
        return self.prepared

    def apply(self, prepared: DSV41PreparedUpdate, *, synchronization=None):
        """Execute one update after every participating host is ready."""
        if not isinstance(prepared, DSV41PreparedUpdate):
            raise TypeError("DSV4.1 online apply requires a prepared update")
        if prepared is not self.prepared:
            raise ValueError("prepared DSV4.1 update is stale or foreign")
        update_number = self.completed_updates + 1
        if prepared.update_number != update_number:
            raise ValueError("prepared DSV4.1 update is stale or out of order")
        if synchronization is not None and synchronization != prepared.contract():
            raise ValueError("owner synchronized reverse contract does not match its replay")
        started = perf_counter()
        options = {
            "kind": prepared.objective,
            "learning_rate": prepared.learning_rate,
        }
        if synchronization is not None:
            options["synchronization"] = synchronization
        try:
            loss, norm, learner_version = self.learner.update_rows(
                prepared.replay, prepared.teacher, **options,
            )
        finally:
            release = getattr(prepared.replay, "release", None)
            if release is not None:
                release()
        update_seconds = perf_counter() - started
        publication_seconds = 0.0
        published = self.config.should_publish(update_number)
        if published:
            if self.publisher is None:
                raise RuntimeError("an online publication boundary has no engine publisher")
            started = perf_counter()
            self.publisher(self.learner, update_number)
            publication_seconds = perf_counter() - started
            self.publications += 1
        self.completed_updates = update_number
        self.update_seconds += update_seconds
        self.publication_seconds += publication_seconds
        record = {
            "round": self.rounds,
            "update": update_number,
            "learner_version": learner_version,
            "teacher_rows": prepared.teacher.shape[0],
            "learning_rate": prepared.learning_rate,
            "loss": loss,
            "gradient_norm": norm,
            "published": published,
            "update_seconds": update_seconds,
            "publication_seconds": publication_seconds,
        }
        self.records.append(record)
        self.prepared = None
        return record

    def apply_remote(self, contract):
        """Participate in remote EP reverse, then apply the DP-reduced update."""
        required = {
            "request_id", "owner", "teacher_rows", "vocabulary",
            "update_number", "learning_rate", "objective", "reverse",
        }
        if not isinstance(contract, dict) or set(contract) != required:
            raise ValueError("remote DSV4.1 update contract has a different ABI")
        update_number = self.completed_updates + 1
        reverse = contract["reverse"]
        expected_layers = list(DSV41SuffixPlan(self.config.last_layers).layers)
        if (
            contract["update_number"] != update_number
            or not 1 <= contract["teacher_rows"] < 32
            or contract["vocabulary"] < 1
            or contract["objective"] != self.config.objective
            or contract["learning_rate"] != self.config.learning_rate_for(update_number)
            or not isinstance(reverse, dict)
            or set(reverse) != {"protocol", "layers"}
            or reverse["protocol"] != "dsv41-synchronized-reverse-v1"
            or reverse["layers"] != expected_layers
        ):
            raise ValueError("remote DSV4.1 update contract is stale or inconsistent")
        participate = getattr(self.learner, "update_participant", None)
        if not callable(participate):
            raise RuntimeError(
                "dual-node DSV4.1 online update requires learner.update_participant; "
                "zero-gradient remote shortcuts are not allowed"
            )
        started = perf_counter()
        loss, norm, learner_version = participate(
            contract,
            learning_rate=contract["learning_rate"], update_number=update_number,
        )
        update_seconds = perf_counter() - started
        publication_seconds = 0.0
        published = self.config.should_publish(update_number)
        if published:
            if self.publisher is None:
                raise RuntimeError("an online publication boundary has no engine publisher")
            started = perf_counter()
            self.publisher(self.learner, update_number)
            publication_seconds = perf_counter() - started
            self.publications += 1
        self.completed_updates = update_number
        self.update_seconds += update_seconds
        self.publication_seconds += publication_seconds
        record = {
            "round": self.rounds,
            "update": update_number,
            "learner_version": learner_version,
            "teacher_rows": contract["teacher_rows"],
            "learning_rate": contract["learning_rate"],
            "loss": loss,
            "gradient_norm": norm,
            "published": published,
            "remote_owner": contract["owner"],
            "update_seconds": update_seconds,
            "publication_seconds": publication_seconds,
        }
        self.records.append(record)
        return record

    def observe(self, request_id, prefix, inputs, delivered, teacher_rows, *, finished=False):
        """Compatibility path for a single process without cluster rendezvous."""
        prepared = self.prepare(
            request_id, prefix, inputs, delivered, teacher_rows, finished=finished,
        )
        return None if prepared is None else self.apply(prepared)

    def finish_request(self, request_id):
        """Close a short real feedback block and prepare its device update.

        No row is fabricated at EOS.  Rows already observed on the corrected
        target branch remain valid supervision and are applied through the
        same synchronized path as a full block.
        """
        block = self.blocks.get(request_id)
        if block is None:
            return None
        block.finish()
        if not block.ready:
            del self.blocks[request_id]
            return None
        del self.blocks[request_id]
        if self.prepared is not None:
            raise RuntimeError("request completion collided with another prepared update")
        update_number = self.completed_updates + 1
        self.prepared = DSV41PreparedUpdate(
            request_id=request_id,
            replay=block.replay,
            teacher=block.teacher(),
            update_number=update_number,
            learning_rate=self.config.learning_rate_for(update_number),
            objective=self.config.objective,
        )
        return self.prepared

    def summary(self):
        return {
            "verifier_rounds": self.rounds,
            "updates": self.completed_updates,
            "publications": self.publications,
            "update_seconds": self.update_seconds,
            "publication_seconds": self.publication_seconds,
            "active_feedback_blocks": len(self.blocks),
            "prepared_update": None if self.prepared is None else self.prepared.update_number,
            "last_update": self.records[-1] if self.records else None,
        }
