"""Replay records, behavior policies, and enforceable data-pool lifecycle."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite, log

import numpy as np

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.rollout_broker import (
    AsyncRolloutBroker,
    PartialRollout,
    RolloutBrokerSnapshot,
)
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    TokenSequence,
)
from inference_scaling.shared.verifier import TokenReward


@dataclass(frozen=True, slots=True)
class ReplayKey:
    prompt: TokenSequence
    generated_prefix: TokenSequence
    candidate: TokenSequence
    reward_version: str

    @property
    def rollout_prefix(self) -> TokenSequence:
        return self.prompt + self.generated_prefix + self.candidate


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    record_id: str
    key: ReplayKey
    completion: TokenSequence
    reward: float
    behavior_id: str
    behavior_logprob: float

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id cannot be empty")
        if not self.behavior_id:
            raise ValueError("behavior_id cannot be empty")
        if not isfinite(self.reward):
            raise ValueError("stored reward must be finite")
        if not isfinite(self.behavior_logprob):
            raise ValueError("a generated completion must have finite behavior log-probability")


@dataclass(frozen=True, slots=True)
class ReplaySampleRequest:
    key: ReplayKey
    max_new_tokens: int
    seed: int
    record_id: str

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.record_id:
            raise ValueError("record_id cannot be empty")


@dataclass(frozen=True, slots=True)
class BehaviorPolicy:
    behavior_id: str
    backend: AutoregressiveBackend
    sampling: SamplingConfig

    @classmethod
    def for_backend(
        cls,
        backend: AutoregressiveBackend,
        sampling: SamplingConfig,
        *,
        label: str | None = None,
    ) -> "BehaviorPolicy":
        identifier = label or f"{backend.model_id}|{sampling.policy_id}"
        return cls(identifier, backend, sampling)


@dataclass(frozen=True, slots=True)
class BrokeredReplayState:
    """Replay metadata paired with a resumable, not-yet-finished rollout."""

    request: ReplaySampleRequest
    rollout: PartialRollout

    def __post_init__(self) -> None:
        if self.request.max_new_tokens <= 0:
            raise ValueError("brokered replay requires a positive rollout length")
        expected_id = f"replay:{self.request.record_id}"
        if self.rollout.request.request_id != expected_id:
            raise ValueError("brokered replay request id does not match its record")
        if self.rollout.request.prefix != self.request.key.rollout_prefix:
            raise ValueError("brokered replay rollout has the wrong prefix")
        if self.rollout.request.max_new_tokens != self.request.max_new_tokens:
            raise ValueError("brokered replay rollout has the wrong token budget")
        if self.rollout.request.seed != self.request.seed:
            raise ValueError("brokered replay rollout has the wrong seed")

    @classmethod
    def start(
        cls,
        policy: BehaviorPolicy,
        request: ReplaySampleRequest,
        *,
        priority: float = 0.0,
    ) -> "BrokeredReplayState":
        if request.max_new_tokens <= 0:
            raise ValueError("brokered replay requires a positive rollout length")
        generation = GenerationRequest(
            prefix=request.key.rollout_prefix,
            max_new_tokens=request.max_new_tokens,
            sampling=policy.sampling,
            seed=request.seed,
            request_id=f"replay:{request.record_id}",
        )
        return cls(
            request,
            PartialRollout.from_request(generation, priority=float(priority)),
        )


@dataclass(frozen=True, slots=True)
class BrokeredReplayResult:
    records: tuple[ReplayRecord, ...]
    partial: tuple[BrokeredReplayState, ...]
    snapshot: RolloutBrokerSnapshot


class BehaviorRegistry:
    def __init__(self, policies: Iterable[BehaviorPolicy] = ()) -> None:
        self._policies: dict[str, BehaviorPolicy] = {}
        for policy in policies:
            self.register(policy)

    def register(self, policy: BehaviorPolicy) -> None:
        existing = self._policies.get(policy.behavior_id)
        if existing is not None:
            if (
                existing.backend.model_id != policy.backend.model_id
                or existing.sampling != policy.sampling
            ):
                raise ValueError(f"behavior id {policy.behavior_id!r} has conflicting definitions")
            return
        self._policies[policy.behavior_id] = policy

    def get(self, behavior_id: str) -> BehaviorPolicy:
        try:
            return self._policies[behavior_id]
        except KeyError as error:
            raise KeyError(f"unknown behavior policy {behavior_id!r}") from error

    def __contains__(self, behavior_id: str) -> bool:
        return behavior_id in self._policies


@dataclass(frozen=True, slots=True)
class FrozenReplayClaim:
    """Metadata-only claim created before evaluation completions are revealed."""

    claim_id: str
    key: ReplayKey
    behavior_counts: tuple[tuple[str, int], ...]

    @property
    def count(self) -> int:
        return sum(count for _, count in self.behavior_counts)


class InMemoryReplayStore:
    """In-memory store that makes evaluation records single-use by construction."""

    def __init__(self) -> None:
        self._evaluation: dict[ReplayKey, list[ReplayRecord]] = defaultdict(list)
        self._design: list[ReplayRecord] = []
        self._reserved: dict[str, tuple[ReplayRecord, ...]] = {}
        self._record_ids: set[str] = set()
        self._claim_counter = 0

    def _check_new(self, record: ReplayRecord) -> None:
        if record.record_id in self._record_ids:
            raise ValueError(f"duplicate replay record id {record.record_id!r}")
        self._record_ids.add(record.record_id)

    def add_evaluation(self, record: ReplayRecord) -> None:
        self._check_new(record)
        self._evaluation[record.key].append(record)

    def add_design(self, record: ReplayRecord) -> None:
        self._check_new(record)
        self._design.append(record)

    def inventory(self, key: ReplayKey) -> dict[str, int]:
        """Expose policy versions and counts, but no completion or reward fields."""

        return dict(Counter(record.behavior_id for record in self._evaluation.get(key, ())))

    def freeze_claims(
        self,
        keys: Sequence[ReplayKey],
        max_records_per_candidate: int,
    ) -> tuple[FrozenReplayClaim, ...]:
        if max_records_per_candidate < 0:
            raise ValueError("max_records_per_candidate must be non-negative")
        claims: list[FrozenReplayClaim] = []
        for key in keys:
            available = self._evaluation.get(key, [])
            selected = tuple(available[:max_records_per_candidate])
            del available[: len(selected)]
            self._claim_counter += 1
            claim_id = f"claim-{self._claim_counter}"
            self._reserved[claim_id] = selected
            counts = tuple(sorted(Counter(record.behavior_id for record in selected).items()))
            claims.append(FrozenReplayClaim(claim_id, key, counts))
        return tuple(claims)

    def reveal_and_consume(self, claim: FrozenReplayClaim) -> tuple[ReplayRecord, ...]:
        try:
            records = self._reserved.pop(claim.claim_id)
        except KeyError as error:
            raise ValueError(f"claim {claim.claim_id!r} is unknown or already consumed") from error
        if any(record.key != claim.key for record in records):
            raise RuntimeError("replay claim key mismatch")
        actual_counts = tuple(sorted(Counter(record.behavior_id for record in records).items()))
        if actual_counts != claim.behavior_counts:
            raise RuntimeError("replay claim metadata changed after freezing")
        self._design.extend(records)
        return records

    def design_records(self, key: ReplayKey | None = None) -> tuple[ReplayRecord, ...]:
        if key is None:
            return tuple(self._design)
        return tuple(record for record in self._design if record.key == key)

    @property
    def evaluation_count(self) -> int:
        return sum(len(records) for records in self._evaluation.values())

    @property
    def design_count(self) -> int:
        return len(self._design)

    @property
    def reserved_count(self) -> int:
        return sum(len(records) for records in self._reserved.values())


def score_continuations(
    policy: BehaviorPolicy,
    key: ReplayKey,
    completions: Sequence[TokenSequence],
) -> tuple[float, ...]:
    if not completions:
        return ()
    scored = policy.backend.score_batch(
        [ScoreRequest(key.rollout_prefix, tuple(completions), policy.sampling)]
    )
    if len(scored) != len(completions):
        raise RuntimeError("behavior backend returned an invalid number of scores")
    totals: list[float] = []
    for completion, token_scores in zip(completions, scored, strict=True):
        if len(token_scores) != len(completion):
            raise RuntimeError("behavior backend returned an invalid token score shape")
        totals.append(float(sum(token_scores)))
    return tuple(totals)


def mixture_logprobabilities(
    registry: BehaviorRegistry,
    key: ReplayKey,
    behavior_counts: Mapping[str, int],
    completions: Sequence[TokenSequence],
) -> tuple[float, ...]:
    total_count = sum(behavior_counts.values())
    if total_count <= 0:
        raise ValueError("a behavior mixture requires at least one selected record")
    components: list[np.ndarray] = []
    for behavior_id, count in sorted(behavior_counts.items()):
        if count <= 0:
            raise ValueError("behavior mixture counts must be positive")
        scores = score_continuations(registry.get(behavior_id), key, completions)
        components.append(np.asarray(scores, dtype=np.float64) + log(count / total_count))
    matrix = np.stack(components, axis=0)
    return tuple(float(value) for value in np.logaddexp.reduce(matrix, axis=0))


def validate_record_probabilities(
    records: Sequence[ReplayRecord],
    registry: BehaviorRegistry,
    *,
    absolute_tolerance: float = 1e-5,
    per_token_tolerance: float = 2e-4,
) -> None:
    if absolute_tolerance < 0 or per_token_tolerance < 0:
        raise ValueError("probability validation tolerances must be non-negative")
    grouped: dict[tuple[ReplayKey, str], list[ReplayRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.key, record.behavior_id)].append(record)
    for (key, behavior_id), group in grouped.items():
        rescored = score_continuations(
            registry.get(behavior_id), key, [record.completion for record in group]
        )
        for record, score in zip(group, rescored, strict=True):
            tolerance = absolute_tolerance + per_token_tolerance * len(record.completion)
            if not isclose(
                record.behavior_logprob,
                score,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    f"stored behavior probability for {record.record_id!r} cannot be reproduced "
                    f"within tolerance {tolerance:g}"
                )


def sample_replay_record(
    policy: BehaviorPolicy,
    key: ReplayKey,
    max_new_tokens: int,
    reward: TokenReward,
    *,
    seed: int,
    record_id: str,
) -> ReplayRecord:
    return sample_replay_records(
        policy,
        [ReplaySampleRequest(key, max_new_tokens, seed, record_id)],
        reward,
    )[0]


def sample_replay_records(
    policy: BehaviorPolicy,
    requests: Sequence[ReplaySampleRequest],
    reward: TokenReward,
) -> tuple[ReplayRecord, ...]:
    """Generate heterogeneous replay completions in one backend batch."""

    generation_requests: list[GenerationRequest] = []
    generated_positions: list[int] = []
    for index, request in enumerate(requests):
        if request.max_new_tokens == 0:
            continue
        generation_requests.append(
            GenerationRequest(
                prefix=request.key.rollout_prefix,
                max_new_tokens=request.max_new_tokens,
                sampling=policy.sampling,
                seed=request.seed,
                request_id=f"replay:{request.record_id}",
            )
        )
        generated_positions.append(index)
    samples = policy.backend.sample_batch(generation_requests) if generation_requests else []
    if len(samples) != len(generation_requests):
        raise RuntimeError("replay backend returned an invalid number of samples")
    samples_by_position = dict(zip(generated_positions, samples, strict=True))

    records: list[ReplayRecord] = []
    for index, request in enumerate(requests):
        sample = samples_by_position.get(index)
        if sample is None:
            completion: TokenSequence = ()
            behavior_logprob = 0.0
        else:
            if (
                sample.model_id != policy.backend.model_id
                or sample.policy_id != policy.sampling.policy_id
            ):
                raise RuntimeError("replay sample does not match its declared behavior policy")
            completion = sample.token_ids
            behavior_logprob = sample.logprob
        full_generated = request.key.generated_prefix + request.key.candidate + completion
        records.append(
            ReplayRecord(
                record_id=request.record_id,
                key=request.key,
                completion=completion,
                reward=float(reward(request.key.prompt, full_generated)),
                behavior_id=policy.behavior_id,
                behavior_logprob=behavior_logprob,
            )
        )
    return tuple(records)


def sample_replay_records_brokered(
    policy: BehaviorPolicy,
    work: Sequence[ReplaySampleRequest | BrokeredReplayState],
    reward: TokenReward,
    broker: AsyncRolloutBroker,
    *,
    completion_target: int | None = None,
    on_record: Callable[[ReplayRecord], None] | None = None,
) -> BrokeredReplayResult:
    """Generate replay records with bounded partial-rollout preservation.

    A returned partial state can be passed to a later call unchanged.  Only
    completed trajectories become :class:`ReplayRecord` objects or trigger
    ``on_record``; unfinished tokens are scheduling state, never statistical
    observations.  This makes the callback safe to connect directly to a
    frozen-design streaming IS estimator.
    """

    if broker.backend is not policy.backend:
        raise ValueError("rollout broker must use the behavior policy backend")
    states = tuple(
        item
        if isinstance(item, BrokeredReplayState)
        else BrokeredReplayState.start(policy, item)
        for item in work
    )
    if not states:
        raise ValueError("brokered replay requires at least one rollout")
    by_request_id: dict[str, BrokeredReplayState] = {}
    for state in states:
        generation = state.rollout.request
        if generation.sampling != policy.sampling:
            raise ValueError("brokered replay state uses the wrong behavior policy")
        if state.rollout.complete:
            raise ValueError("completed replay state must not be resubmitted")
        if generation.request_id in by_request_id:
            raise ValueError("brokered replay record ids must be unique")
        by_request_id[generation.request_id] = state

    result = broker.run_until(
        tuple(state.rollout for state in states),
        completion_target=completion_target,
    )
    records: list[ReplayRecord] = []
    for sample in result.completed:
        try:
            state = by_request_id[sample.request_id]
        except KeyError as error:
            raise RuntimeError("broker completed an unknown replay request") from error
        if (
            sample.model_id != policy.backend.model_id
            or sample.policy_id != policy.sampling.policy_id
        ):
            raise RuntimeError("brokered replay sample used the wrong behavior policy")
        request = state.request
        full_generated = (
            request.key.generated_prefix + request.key.candidate + sample.token_ids
        )
        record = ReplayRecord(
            record_id=request.record_id,
            key=request.key,
            completion=sample.token_ids,
            reward=float(reward(request.key.prompt, full_generated)),
            behavior_id=policy.behavior_id,
            behavior_logprob=sample.logprob,
        )
        records.append(record)
        if on_record is not None:
            on_record(record)

    partial = tuple(
        BrokeredReplayState(
            by_request_id[state.request.request_id].request,
            state,
        )
        for state in result.partial
    )
    return BrokeredReplayResult(tuple(records), partial, result.snapshot)
