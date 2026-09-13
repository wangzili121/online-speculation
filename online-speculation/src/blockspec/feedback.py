"""Owned reached-prefix feedback and frozen-boundary replay protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from .state import Cache


class ReplayBoundary(Protocol):
    start_layer: int

    def detached(self) -> ReplayBoundary: ...


@dataclass
class Feedback:
    inputs: Tensor
    cache: Cache | None
    teacher_logits: Tensor
    valid: int
    boundary: ReplayBoundary | None = None
    fully_covered: bool = False

    def detached(self, *, cache_start=0):
        cache = None if self.cache is None else tuple((k.detach(), v.detach()) for k, v in self.cache[cache_start:])
        return Feedback(self.inputs.detach().clone(), cache, self.teacher_logits.detach().clone(), self.valid,
                        None if self.boundary is None else self.boundary.detached(), self.fully_covered)
