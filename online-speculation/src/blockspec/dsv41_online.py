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

import torch


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
        if self.last_layers != 1:
            raise ValueError("the verified DSV4.1 replay currently supports last-1 only")
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
        self.learner = learner
        self.config = config
        self.publisher = publisher
        self.rounds = self.completed_updates = self.publications = 0
        self.blocks = {}
        self.records = []

    def begin_round(self, request_id, prefix, replay):
        """Count one verifier round and optionally capture a fresh block."""
        self.rounds += 1
        if request_id in self.blocks or not self.config.should_capture(self.rounds, self.completed_updates):
            return False
        self.blocks[request_id] = DSV41DelayedTeacherBlock(
            request_id, prefix, replay, width=self.config.teacher_rows + 1,
        )
        return True

    def observe(self, request_id, prefix, inputs, delivered, teacher_rows, *, finished=False):
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
        rate = self.config.learning_rate_for(update_number)
        loss, norm, learner_version = self.learner.update_rows(
            block.replay, block.teacher(), kind=self.config.objective, learning_rate=rate,
        )
        self.completed_updates = update_number
        published = self.config.should_publish(update_number)
        if published:
            if self.publisher is None:
                raise RuntimeError("an online publication boundary has no engine publisher")
            self.publisher(self.learner, update_number)
            self.publications += 1
        record = {
            "round": self.rounds,
            "update": update_number,
            "learner_version": learner_version,
            "teacher_rows": len(block.rows),
            "learning_rate": rate,
            "loss": loss,
            "gradient_norm": norm,
            "published": published,
        }
        self.records.append(record)
        return record

    def finish_request(self, request_id):
        block = self.blocks.get(request_id)
        if block is None:
            return None
        block.finish()
        if not block.ready:
            del self.blocks[request_id]
            return None
        # No new teacher row is fabricated at request completion.  The caller
        # should have passed the final real row to observe before this method.
        del self.blocks[request_id]
        return block

    def summary(self):
        return {
            "verifier_rounds": self.rounds,
            "updates": self.completed_updates,
            "publications": self.publications,
            "active_feedback_blocks": len(self.blocks),
            "last_update": self.records[-1] if self.records else None,
        }
