"""Exact rollout-reuse and scheduling accelerations for reward-based MH.

The mechanisms in this module alter either proposal scheduling or a proposal
whose forward and reverse probabilities are evaluated explicitly.  None of the
paths clips a Hastings ratio.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite, log

import numpy as np

from inference_scaling.arllm.algorithms.mh import (
    RewardMHChainResult,
    RewardMHStep,
    _draw_suffix,
    _is_base_proposal,
    _sample_exact_length,
    _sample_exact_lengths,
    _score_one,
    _validate_proposal,
)
from inference_scaling.arllm.config import RewardMHConfig, SamplingConfig
from inference_scaling.shared.mh import decide_metropolis_hastings
from inference_scaling.shared.rng import SeedStream
from inference_scaling.arllm.types import AutoregressiveBackend, TokenSequence
from inference_scaling.shared.verifier import TokenReward


RewardFunction = TokenReward


@dataclass(frozen=True, slots=True)
class DelayedRewardMHStep:
    step: int
    cut: int
    proposed_suffix_length: int
    current_reward: float
    proposed_reward: float | None
    current_surrogate_reward: float
    proposed_surrogate_reward: float
    stage_one_log_acceptance: float
    stage_one_accepted: bool
    stage_two_log_acceptance: float | None
    exact_reward_evaluated: bool
    accepted: bool
    suffix_schedule: str
    suffix_probability: float
    proposed_token_changes: int
    accepted_token_changes: int


@dataclass(frozen=True, slots=True)
class DelayedRewardMHResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    reward: float
    surrogate_reward: float
    base_token_logprobs: tuple[float, ...]
    proposal_token_logprobs: tuple[float, ...]
    trace: tuple[DelayedRewardMHStep, ...]
    chain_id: int
    exact_reward_evaluations: int
    surrogate_reward_evaluations: int

    @property
    def attempts(self) -> int:
        return len(self.trace)

    @property
    def accepted(self) -> int:
        return sum(step.accepted for step in self.trace)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    @property
    def exact_reward_fraction(self) -> float:
        denominator = 1 + self.attempts
        return self.exact_reward_evaluations / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class PrefetchSnapshot:
    used_proposals: int
    prefetched_proposals: int
    unused_prefetched_proposals: int
    reward_evaluations: int


@dataclass(frozen=True, slots=True)
class PrefetchedRewardMHResult:
    chain: RewardMHChainResult
    snapshot: PrefetchSnapshot


@dataclass(frozen=True, slots=True)
class ReplayProposalMHStep:
    step: int
    cut: int
    proposed_suffix_length: int
    current_reward: float
    proposed_reward: float
    old_proposal_logprob: float
    new_proposal_logprob: float
    log_acceptance: float
    accepted: bool
    proposal_source: str
    suffix_schedule: str
    suffix_probability: float
    proposed_token_changes: int
    accepted_token_changes: int


@dataclass(frozen=True, slots=True)
class ReplayProposalMHResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    reward: float
    base_token_logprobs: tuple[float, ...]
    trace: tuple[ReplayProposalMHStep, ...]
    chain_id: int
    proposal_logprob: float

    @property
    def attempts(self) -> int:
        return len(self.trace)

    @property
    def accepted(self) -> int:
        return sum(step.accepted for step in self.trace)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0


@dataclass(frozen=True, slots=True)
class ReplayProposalSnapshot:
    stored_suffixes: int
    stored_prefix_lengths: int
    base_draws: int
    history_draws: int
    logprob_queries: int


@dataclass(frozen=True, slots=True)
class ReplayProposalDraw:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    proposal_logprob: float
    source: str


@dataclass(frozen=True, slots=True)
class _StandardState:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    proposal_token_logprobs: tuple[float, ...]
    reward: float | None


@dataclass(frozen=True, slots=True)
class _StandardProposal:
    source_state: _StandardState
    cut: int
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    proposal_token_logprobs: tuple[float, ...]
    suffix_schedule: str
    suffix_probability: float

    @property
    def sequence(self) -> TokenSequence:
        return self.source_state.token_ids[: self.cut] + self.token_ids


def _finite_reward(
    reward: RewardFunction, prompt: TokenSequence, sequence: TokenSequence
) -> float:
    value = float(reward(prompt, sequence))
    if not isfinite(value):
        raise ValueError("reward must be finite")
    return value


def _initialize_standard_state(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: RewardMHConfig,
    proposal: SamplingConfig,
    reward: RewardFunction,
    seeds: SeedStream,
    chain_id: int,
) -> _StandardState:
    tokens, proposal_logs, cached_base_logs = _sample_exact_length(
        backend,
        prefix=prompt,
        length=config.total_length,
        sampling=proposal,
        seed=seeds.derive("reward_mh", chain_id, "initialize"),
        request_id=f"reward-mh:{chain_id}:initialize",
    )
    base_logs = (
        proposal_logs
        if _is_base_proposal(proposal)
        else cached_base_logs or _score_one(backend, prompt, tokens, None)
    )
    if any(not isfinite(value) for value in base_logs):
        raise ValueError("proposal generated a sequence outside the base model support")
    return _StandardState(
        tokens,
        tuple(base_logs),
        tuple(proposal_logs),
        _finite_reward(reward, prompt, tokens),
    )


def _sample_standard_proposals(
    backend: AutoregressiveBackend,
    states: Sequence[_StandardState],
    *,
    prompt: TokenSequence,
    config: RewardMHConfig,
    proposal: SamplingConfig,
    seeds: SeedStream,
    chain_id: int,
    step_index: int,
) -> tuple[_StandardProposal, ...]:
    if not states:
        return ()
    cut, suffix_length, suffix_probability = _draw_suffix(
        stage_length=config.total_length,
        schedule=config.suffix_schedule,
        rng=seeds.generator("reward_mh", chain_id, step_index, "cut"),
    )
    prefixes = tuple(prompt + state.token_ids[:cut] for state in states)
    tokens, proposal_logs, base_logs = _sample_exact_lengths(
        backend,
        prefixes=prefixes,
        lengths=(suffix_length,) * len(states),
        sampling=proposal,
        seeds=(
            seeds.derive("reward_mh", chain_id, step_index, "proposal"),
        )
        * len(states),
        request_ids=(f"reward-mh:{chain_id}:step:{step_index}",) * len(states),
    )
    return tuple(
        _StandardProposal(
            state,
            cut,
            sampled,
            base,
            proposed,
            config.suffix_schedule,
            suffix_probability,
        )
        for state, sampled, base, proposed in zip(
            states, tokens, base_logs, proposal_logs, strict=True
        )
    )


def _accept_standard_proposal(
    state: _StandardState,
    proposed: _StandardProposal,
    proposed_reward: float,
    *,
    config: RewardMHConfig,
    uniform: float,
) -> tuple[_StandardState, float, bool]:
    if state.reward is None:
        raise RuntimeError("current MH state is missing its exact reward")
    cut = proposed.cut
    decision = decide_metropolis_hastings(
        current_target_log_density=(
            sum(state.base_token_logprobs[cut:])
            + state.reward / config.reward_temperature
        ),
        proposed_target_log_density=(
            sum(proposed.base_token_logprobs)
            + proposed_reward / config.reward_temperature
        ),
        forward_proposal_log_probability=sum(proposed.proposal_token_logprobs),
        reverse_proposal_log_probability=sum(state.proposal_token_logprobs[cut:]),
        uniform=uniform,
    )
    log_acceptance = decision.log_acceptance
    accepted = decision.accepted
    if not accepted:
        return state, log_acceptance, False
    return (
        _StandardState(
            proposed.sequence,
            state.base_token_logprobs[:cut] + proposed.base_token_logprobs,
            state.proposal_token_logprobs[:cut] + proposed.proposal_token_logprobs,
            proposed_reward,
        ),
        log_acceptance,
        True,
    )


def run_reward_mh_chain_delayed(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: RewardMHConfig,
    proposal: SamplingConfig,
    reward: RewardFunction,
    surrogate_reward: RewardFunction,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> DelayedRewardMHResult:
    """Run exact two-stage delayed-acceptance suffix MH.

    Stage one uses the fixed surrogate target.  Stage two corrects the complete
    difference between the surrogate and exact reward, so early rejection saves
    exact reward work without changing the final target distribution.
    """

    _validate_proposal(proposal)
    state = _initialize_standard_state(
        backend, prompt, config, proposal, reward, seeds, chain_id
    )
    assert state.reward is not None
    current_surrogate = _finite_reward(surrogate_reward, prompt, state.token_ids)
    exact_evaluations = 1
    surrogate_evaluations = 1
    trace: list[DelayedRewardMHStep] = []

    for step_index in range(config.updates):
        proposed = _sample_standard_proposals(
            backend,
            (state,),
            prompt=prompt,
            config=config,
            proposal=proposal,
            seeds=seeds,
            chain_id=chain_id,
            step_index=step_index,
        )[0]
        proposed_surrogate = _finite_reward(
            surrogate_reward, prompt, proposed.sequence
        )
        surrogate_evaluations += 1
        cut = proposed.cut
        stage_one_decision = decide_metropolis_hastings(
            current_target_log_density=(
                sum(state.base_token_logprobs[cut:])
                + current_surrogate / config.reward_temperature
            ),
            proposed_target_log_density=(
                sum(proposed.base_token_logprobs)
                + proposed_surrogate / config.reward_temperature
            ),
            forward_proposal_log_probability=sum(proposed.proposal_token_logprobs),
            reverse_proposal_log_probability=sum(state.proposal_token_logprobs[cut:]),
            uniform=float(
                seeds.generator(
                    "reward_mh", chain_id, step_index, "delayed-stage-one"
                ).random()
            ),
        )
        stage_one_log_acceptance = stage_one_decision.log_acceptance
        stage_one_accepted = stage_one_decision.accepted
        proposed_reward: float | None = None
        stage_two_log_acceptance: float | None = None
        accepted = False
        if stage_one_accepted:
            proposed_reward = _finite_reward(reward, prompt, proposed.sequence)
            exact_evaluations += 1
            stage_two_decision = decide_metropolis_hastings(
                current_target_log_density=(
                    state.reward - current_surrogate
                )
                / config.reward_temperature,
                proposed_target_log_density=(
                    proposed_reward - proposed_surrogate
                )
                / config.reward_temperature,
                uniform=float(
                    seeds.generator(
                        "reward_mh", chain_id, step_index, "delayed-stage-two"
                    ).random()
                ),
            )
            stage_two_log_acceptance = stage_two_decision.log_acceptance
            accepted = stage_two_decision.accepted
        previous_reward = state.reward
        previous_surrogate = current_surrogate
        proposed_token_changes = sum(
            old != new
            for old, new in zip(
                state.token_ids[cut:], proposed.token_ids, strict=True
            )
        )
        if accepted:
            assert proposed_reward is not None
            state = _StandardState(
                proposed.sequence,
                state.base_token_logprobs[:cut] + proposed.base_token_logprobs,
                state.proposal_token_logprobs[:cut]
                + proposed.proposal_token_logprobs,
                proposed_reward,
            )
            current_surrogate = proposed_surrogate
        trace.append(
            DelayedRewardMHStep(
                step=step_index,
                cut=cut,
                proposed_suffix_length=config.total_length - cut,
                current_reward=previous_reward,
                proposed_reward=proposed_reward,
                current_surrogate_reward=previous_surrogate,
                proposed_surrogate_reward=proposed_surrogate,
                stage_one_log_acceptance=stage_one_log_acceptance,
                stage_one_accepted=stage_one_accepted,
                stage_two_log_acceptance=stage_two_log_acceptance,
                exact_reward_evaluated=stage_one_accepted,
                accepted=accepted,
                suffix_schedule=proposed.suffix_schedule,
                suffix_probability=proposed.suffix_probability,
                proposed_token_changes=proposed_token_changes,
                accepted_token_changes=(proposed_token_changes if accepted else 0),
            )
        )

    assert state.reward is not None
    return DelayedRewardMHResult(
        prompt=prompt,
        token_ids=state.token_ids,
        reward=state.reward,
        surrogate_reward=current_surrogate,
        base_token_logprobs=state.base_token_logprobs,
        proposal_token_logprobs=state.proposal_token_logprobs,
        trace=tuple(trace),
        chain_id=chain_id,
        exact_reward_evaluations=exact_evaluations,
        surrogate_reward_evaluations=surrogate_evaluations,
    )


def run_reward_mh_chain_prefetched(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: RewardMHConfig,
    proposal: SamplingConfig,
    reward: RewardFunction,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> PrefetchedRewardMHResult:
    """Overlap exact reward evaluation with a one-step accept/reject proposal tree.

    For the next MH step, proposals are generated for both possible current
    states.  Once the present reward finishes, the chain consumes only the branch
    selected by the ordinary Hastings decision.  The unused branch is accounting
    overhead, not an additional chain sample.
    """

    _validate_proposal(proposal)
    state = _initialize_standard_state(
        backend, prompt, config, proposal, reward, seeds, chain_id
    )
    current_proposal = _sample_standard_proposals(
        backend,
        (state,),
        prompt=prompt,
        config=config,
        proposal=proposal,
        seeds=seeds,
        chain_id=chain_id,
        step_index=0,
    )[0]
    trace: list[RewardMHStep] = []
    prefetched_count = 1
    used_count = 0
    reward_evaluations = 1

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mh-reward") as executor:
        for step_index in range(config.updates):
            if current_proposal.source_state.token_ids != state.token_ids:
                raise RuntimeError("prefetched MH proposal does not match the selected branch")
            reward_future = executor.submit(
                _finite_reward, reward, prompt, current_proposal.sequence
            )
            next_proposals: tuple[_StandardProposal, ...] = ()
            if step_index + 1 < config.updates:
                accept_state = _StandardState(
                    current_proposal.sequence,
                    state.base_token_logprobs[: current_proposal.cut]
                    + current_proposal.base_token_logprobs,
                    state.proposal_token_logprobs[: current_proposal.cut]
                    + current_proposal.proposal_token_logprobs,
                    None,
                )
                next_proposals = _sample_standard_proposals(
                    backend,
                    (state, accept_state),
                    prompt=prompt,
                    config=config,
                    proposal=proposal,
                    seeds=seeds,
                    chain_id=chain_id,
                    step_index=step_index + 1,
                )
                prefetched_count += len(next_proposals)
            proposed_reward = float(reward_future.result())
            reward_evaluations += 1
            assert state.reward is not None
            previous_reward = state.reward
            proposed_token_changes = sum(
                old != new
                for old, new in zip(
                    state.token_ids[current_proposal.cut :],
                    current_proposal.token_ids,
                    strict=True,
                )
            )
            uniform = float(
                seeds.generator("reward_mh", chain_id, step_index, "accept").random()
            )
            state, log_acceptance, accepted = _accept_standard_proposal(
                state,
                current_proposal,
                proposed_reward,
                config=config,
                uniform=uniform,
            )
            used_count += 1
            trace.append(
                RewardMHStep(
                    step=step_index,
                    cut=current_proposal.cut,
                    proposed_suffix_length=config.total_length - current_proposal.cut,
                    current_reward=previous_reward,
                    proposed_reward=proposed_reward,
                    log_acceptance=log_acceptance,
                    accepted=accepted,
                    suffix_schedule=current_proposal.suffix_schedule,
                    suffix_probability=current_proposal.suffix_probability,
                    proposed_token_changes=proposed_token_changes,
                    accepted_token_changes=(
                        proposed_token_changes if accepted else 0
                    ),
                )
            )
            if next_proposals:
                current_proposal = next_proposals[1 if accepted else 0]

    assert state.reward is not None
    chain = RewardMHChainResult(
        prompt=prompt,
        token_ids=state.token_ids,
        reward=state.reward,
        base_token_logprobs=state.base_token_logprobs,
        proposal_token_logprobs=state.proposal_token_logprobs,
        trace=tuple(trace),
        chain_id=chain_id,
    )
    return PrefetchedRewardMHResult(
        chain,
        PrefetchSnapshot(
            used_proposals=used_count,
            prefetched_proposals=prefetched_count,
            unused_prefetched_proposals=prefetched_count - used_count,
            reward_evaluations=reward_evaluations,
        ),
    )


class FrozenReplaySuffixProposal:
    """A frozen defensive mixture of base suffixes and empirical replay suffixes."""

    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        history_mixture: float = 0.25,
        sampling: SamplingConfig | None = None,
    ) -> None:
        if not 0 <= history_mixture < 1:
            raise ValueError("history_mixture must lie in [0, 1)")
        self.backend = backend
        self.sampling = sampling or SamplingConfig()
        _validate_proposal(self.sampling)
        if not _is_base_proposal(self.sampling):
            raise ValueError("defensive replay proposal currently requires base sampling")
        self.history_mixture = float(history_mixture)
        self._suffixes: dict[tuple[TokenSequence, int], Counter[TokenSequence]] = defaultdict(
            Counter
        )
        self._frozen = False
        self._base_draws = 0
        self._history_draws = 0
        self._logprob_queries = 0
        self._lock = threading.RLock()

    def observe_suffix(self, prefix: TokenSequence, suffix: TokenSequence) -> None:
        if self._frozen:
            raise RuntimeError("replay proposal is frozen")
        if not suffix:
            raise ValueError("replay proposal suffix cannot be empty")
        self._suffixes[(tuple(prefix), len(suffix))][tuple(suffix)] += 1

    def observe_sequence(self, prompt: TokenSequence, sequence: TokenSequence) -> None:
        values = tuple(sequence)
        for cut in range(len(values)):
            self.observe_suffix(tuple(prompt) + values[:cut], values[cut:])

    def observe_sequences(
        self, prompt: TokenSequence, sequences: Iterable[TokenSequence]
    ) -> None:
        for sequence in sequences:
            self.observe_sequence(prompt, sequence)

    def freeze(self) -> None:
        self._frozen = True

    @staticmethod
    def _history_probability(
        counts: Counter[TokenSequence] | None, suffix: TokenSequence
    ) -> float:
        if not counts:
            return 0.0
        return counts.get(tuple(suffix), 0) / sum(counts.values())

    def _mixture_logprob(
        self,
        base_logprob: float,
        counts: Counter[TokenSequence] | None,
        suffix: TokenSequence,
    ) -> float:
        if not counts or self.history_mixture == 0:
            return float(base_logprob)
        history_probability = self._history_probability(counts, suffix)
        terms = [log(1.0 - self.history_mixture) + float(base_logprob)]
        if history_probability > 0:
            terms.append(log(self.history_mixture) + log(history_probability))
        return float(np.logaddexp.reduce(np.asarray(terms, dtype=np.float64)))

    def draw(
        self,
        prefix: TokenSequence,
        length: int,
        *,
        seed: int,
        request_id: str,
    ) -> ReplayProposalDraw:
        if not self._frozen:
            raise RuntimeError("replay proposal must be frozen before sampling")
        if length <= 0:
            raise ValueError("replay proposal length must be positive")
        counts = self._suffixes.get((tuple(prefix), int(length)))
        component_rng = SeedStream(seed).generator("replay-proposal-component")
        use_history = bool(counts) and float(component_rng.random()) < self.history_mixture
        if use_history:
            assert counts is not None
            support = tuple(sorted(counts))
            masses = np.asarray([counts[value] for value in support], dtype=np.float64)
            masses /= masses.sum()
            index = int(
                SeedStream(seed)
                .generator("replay-proposal-history")
                .choice(len(support), p=masses)
            )
            suffix = support[index]
            base_token_logprobs = _score_one(self.backend, prefix, suffix, None)
            source = "history"
            with self._lock:
                self._history_draws += 1
        else:
            suffix, base_token_logprobs, cached_base = _sample_exact_length(
                self.backend,
                prefix=prefix,
                length=length,
                sampling=self.sampling,
                seed=SeedStream(seed).derive("replay-proposal-base"),
                request_id=request_id,
            )
            if cached_base is not None:
                base_token_logprobs = cached_base
            source = "base"
            with self._lock:
                self._base_draws += 1
        base_total = float(sum(base_token_logprobs))
        return ReplayProposalDraw(
            token_ids=tuple(suffix),
            base_token_logprobs=tuple(base_token_logprobs),
            proposal_logprob=self._mixture_logprob(base_total, counts, tuple(suffix)),
            source=source,
        )

    def logprob(
        self,
        prefix: TokenSequence,
        suffix: TokenSequence,
        *,
        base_logprob: float | None = None,
    ) -> float:
        if not self._frozen:
            raise RuntimeError("replay proposal must be frozen before scoring")
        if not suffix:
            raise ValueError("replay proposal suffix cannot be empty")
        if base_logprob is None:
            base_logprob = float(sum(_score_one(self.backend, prefix, suffix, None)))
        counts = self._suffixes.get((tuple(prefix), len(suffix)))
        with self._lock:
            self._logprob_queries += 1
        return self._mixture_logprob(float(base_logprob), counts, tuple(suffix))

    def snapshot(self) -> ReplayProposalSnapshot:
        with self._lock:
            return ReplayProposalSnapshot(
                stored_suffixes=sum(sum(counts.values()) for counts in self._suffixes.values()),
                stored_prefix_lengths=len(self._suffixes),
                base_draws=self._base_draws,
                history_draws=self._history_draws,
                logprob_queries=self._logprob_queries,
            )


def run_reward_mh_chain_replay_proposal(
    proposal: FrozenReplaySuffixProposal,
    prompt: TokenSequence,
    config: RewardMHConfig,
    reward: RewardFunction,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> ReplayProposalMHResult:
    """Run reward MH with exact forward/reverse defensive replay probabilities."""

    proposal.freeze()
    initial = proposal.draw(
        prompt,
        config.total_length,
        seed=seeds.derive("reward_mh", chain_id, "initialize"),
        request_id=f"reward-mh-replay:{chain_id}:initialize",
    )
    tokens = initial.token_ids
    base_logs = initial.base_token_logprobs
    current_reward = _finite_reward(reward, prompt, tokens)
    trace: list[ReplayProposalMHStep] = []

    for step_index in range(config.updates):
        cut, suffix_length, suffix_probability = _draw_suffix(
            stage_length=config.total_length,
            schedule=config.suffix_schedule,
            rng=seeds.generator("reward_mh", chain_id, step_index, "cut"),
        )
        retained = tokens[:cut]
        prefix = prompt + retained
        old_suffix = tokens[cut:]
        old_p = float(sum(base_logs[cut:]))
        old_q = proposal.logprob(prefix, old_suffix, base_logprob=old_p)
        draw = proposal.draw(
            prefix,
            config.total_length - cut,
            seed=seeds.derive("reward_mh", chain_id, step_index, "proposal"),
            request_id=f"reward-mh-replay:{chain_id}:step:{step_index}",
        )
        proposed_sequence = retained + draw.token_ids
        proposed_reward = _finite_reward(reward, prompt, proposed_sequence)
        new_p = float(sum(draw.base_token_logprobs))
        decision = decide_metropolis_hastings(
            current_target_log_density=(
                old_p + current_reward / config.reward_temperature
            ),
            proposed_target_log_density=(
                new_p + proposed_reward / config.reward_temperature
            ),
            forward_proposal_log_probability=draw.proposal_logprob,
            reverse_proposal_log_probability=old_q,
            uniform=float(
                seeds.generator("reward_mh", chain_id, step_index, "accept").random()
            ),
        )
        log_acceptance = decision.log_acceptance
        accepted = decision.accepted
        previous_reward = current_reward
        proposed_token_changes = sum(
            old != new
            for old, new in zip(old_suffix, draw.token_ids, strict=True)
        )
        if accepted:
            tokens = proposed_sequence
            base_logs = base_logs[:cut] + draw.base_token_logprobs
            current_reward = proposed_reward
        trace.append(
            ReplayProposalMHStep(
                step=step_index,
                cut=cut,
                proposed_suffix_length=suffix_length,
                current_reward=previous_reward,
                proposed_reward=proposed_reward,
                old_proposal_logprob=old_q,
                new_proposal_logprob=draw.proposal_logprob,
                log_acceptance=log_acceptance,
                accepted=accepted,
                proposal_source=draw.source,
                suffix_schedule=config.suffix_schedule,
                suffix_probability=suffix_probability,
                proposed_token_changes=proposed_token_changes,
                accepted_token_changes=(proposed_token_changes if accepted else 0),
            )
        )

    final_q = proposal.logprob(
        prompt,
        tokens,
        base_logprob=float(sum(base_logs)),
    )
    return ReplayProposalMHResult(
        prompt=prompt,
        token_ids=tokens,
        reward=current_reward,
        base_token_logprobs=base_logs,
        trace=tuple(trace),
        chain_id=chain_id,
        proposal_logprob=final_q,
    )


def run_reward_mh_chains_replay_proposal(
    proposal: FrozenReplaySuffixProposal,
    prompt: TokenSequence,
    config: RewardMHConfig,
    reward: RewardFunction,
    seeds: SeedStream,
    *,
    chains: int,
) -> tuple[ReplayProposalMHResult, ...]:
    if chains <= 0:
        raise ValueError("chains must be positive")
    proposal.freeze()
    return tuple(
        run_reward_mh_chain_replay_proposal(
            proposal,
            prompt,
            config,
            reward,
            seeds,
            chain_id=chain_id,
        )
        for chain_id in range(chains)
    )
