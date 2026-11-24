"""Two-phase publication contract for a dual-node DSV4.1 draft head."""

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import re


_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PublicationReceipt:
    phase: str
    version: int
    digest: str
    ranks: tuple[int, ...]

    def __post_init__(self):
        if self.phase not in ("prepared", "activated"):
            raise ValueError("publication phase must be prepared or activated")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("publication version must be positive")
        if not isinstance(self.digest, str) or _DIGEST.fullmatch(self.digest) is None:
            raise ValueError("publication digest must be a lowercase SHA256")
        if (not self.ranks or len(set(self.ranks)) != len(self.ranks)
                or any(type(rank) is not int or rank < 0 for rank in self.ranks)):
            raise ValueError("publication ranks must be unique nonnegative integers")

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("publication receipt must be a mapping")
        return cls(value["phase"], value["version"], value["digest"], tuple(value["ranks"]))

    def payload(self):
        return asdict(self) | {"ranks": list(self.ranks)}


def validate_pair(local, peer, *, current_version, expected_ranks=range(16), phase="prepared"):
    """Reject partial, stale or mixed-weight publication before a live switch."""
    local = local if isinstance(local, PublicationReceipt) else PublicationReceipt.parse(local)
    peer = peer if isinstance(peer, PublicationReceipt) else PublicationReceipt.parse(peer)
    expected = set(expected_ranks)
    if local.phase != phase or peer.phase != phase:
        raise ValueError(f"both publication receipts must be {phase}")
    if local.version != peer.version or local.version != current_version + 1:
        raise ValueError("publication versions are stale or disagree")
    if local.digest != peer.digest:
        raise ValueError("nodes staged different draft weights")
    if set(local.ranks) & set(peer.ranks) or set(local.ranks) | set(peer.ranks) != expected:
        raise ValueError("publication receipts do not cover every rank exactly once")
    return local.version, local.digest


class AtomicHeadPublisher:
    """Stage, cross-check and activate a head at a verifier boundary.

    The engine supplies device-specific callbacks. ``stage`` writes an inactive
    slot and returns its receipt; ``activate`` only flips the live version
    pointer. ``rollback`` restores the previous pointer and discards the slot.
    """

    def __init__(
        self,
        *,
        exchange: Callable[[str, dict], dict],
        stage: Callable[[object, int], PublicationReceipt],
        activate: Callable[[int], PublicationReceipt],
        rollback: Callable[[int], None],
        commit: Callable[[int], None] | None = None,
        current_version=0,
        expected_ranks: Iterable[int] = range(16),
    ):
        if type(current_version) is not int or current_version < 0:
            raise ValueError("current publication version must be nonnegative")
        self.exchange, self.stage, self.activate, self.rollback = exchange, stage, activate, rollback
        self.commit = commit
        self.current_version = current_version
        self.expected_ranks = tuple(expected_ranks)

    def __call__(self, learner, update_number):
        version = self.current_version + 1
        if update_number < version:
            raise ValueError("optimizer update cannot precede the head version")
        prepared = self.stage(learner, version)
        try:
            peer = self.exchange(f"online-head/{version}/prepared", prepared.payload())
            digest = validate_pair(
                prepared, peer, current_version=self.current_version,
                expected_ranks=self.expected_ranks, phase="prepared",
            )[1]
            activated = self.activate(version)
            peer = self.exchange(f"online-head/{version}/activated", activated.payload())
            validate_pair(
                activated, peer, current_version=self.current_version,
                expected_ranks=self.expected_ranks, phase="activated",
            )
            if activated.digest != digest:
                raise ValueError("activated head does not match the staged head")
            if self.commit is not None:
                self.commit(version)
        except BaseException:
            self.rollback(version)
            raise
        self.current_version = version
        return version
