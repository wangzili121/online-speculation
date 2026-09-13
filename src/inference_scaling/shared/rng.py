"""Order-independent random streams for asynchronous experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedStream:
    root_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("root_seed must be an integer")
        if not 0 <= self.root_seed < 2**128:
            raise ValueError("root_seed must lie in [0, 2**128)")

    @staticmethod
    def _path_part(value: object) -> list[object]:
        if value is None:
            return ["none", None]
        if isinstance(value, bool):
            return ["bool", value]
        if isinstance(value, int):
            return ["int", str(value)]
        if isinstance(value, float):
            if not isfinite(value):
                raise ValueError("seed path floats must be finite")
            return ["float", value.hex()]
        if isinstance(value, str):
            return ["str", value]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        raise TypeError(
            "seed path entries must be None, bool, int, float, str, or bytes"
        )

    def derive(self, *path: object) -> int:
        encoded_path = json.dumps(
            [self._path_part(part) for part in path],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.blake2b(
            encoded_path,
            digest_size=8,
            key=self.root_seed.to_bytes(16, "little", signed=False),
        ).digest()
        return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)

    def generator(self, *path: object) -> np.random.Generator:
        return np.random.default_rng(self.derive(*path))
