"""Measure exact rollout-reuse accelerations for IS and reward-based MH.

Every comparison fixes one public GSM8K prompt, model, dtype, random seed and
logical budget.  Controlled verifier delays are labelled explicitly.  They are
used to expose overlap and early-rejection behavior, not as a task-quality
metric.
"""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides, require_full_scope

import argparse
import hashlib
import json
import threading
import time
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from experiments.arllm.benchmark_rollout_infra import (
    ROOT,
    _BackendFactory,
    _decode_arm,
    _delta,
    _history_batches,
    _machine,
    _measure,
    _prompt_tokens,
    _snapshot,
    _speculation,
    _warmup,
)
from inference_scaling.arllm.acceleration import sample_batch_with_callback
from inference_scaling.arllm.algorithms.mh import run_reward_mh_chain
from inference_scaling.arllm.algorithms.mh_acceleration import (
    FrozenReplaySuffixProposal,
    run_reward_mh_chain_delayed,
    run_reward_mh_chain_prefetched,
    run_reward_mh_chain_replay_proposal,
)
from inference_scaling.experimental.arllm.streaming_is import (
    FrozenStreamingISEstimator,
    ordinary_importance_log_weight,
)
from inference_scaling.arllm.config import RewardMHConfig, SamplingConfig
from inference_scaling.shared.evaluation import load_gsm8k, select_problems
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    ReplayKey,
    ReplayRecord,
    ReplaySampleRequest,
    sample_replay_records,
    sample_replay_records_brokered,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.arllm.rollout_broker import AsyncRolloutBroker
from inference_scaling.arllm.types import GenerationRequest, SequenceSample, TokenSequence


def _save(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _token_hash(tokens: Sequence[int]) -> str:
    raw = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _zero_cost() -> dict[str, Any]:
    return {"telemetry": {"wall_seconds": 0.0}, "main_model": {}}


def _sequence_reward(_prompt: TokenSequence, generated: TokenSequence) -> float:
    if not generated:
        return 0.0
    return float(sum(int(token) % 7 == 0 for token in generated)) / len(generated)


def _frozen_estimator(
    record_ids: Sequence[str], candidate_by_id: dict[str, int]
) -> FrozenStreamingISEstimator:
    estimator = FrozenStreamingISEstimator(2)
    estimator.add_history("history:candidate:0", 0, 0.0)
    estimator.add_history("history:candidate:1", 1, 0.0)
    estimator.freeze(
        tuple(
            tuple(
                record_id
                for record_id in record_ids
                if candidate_by_id[record_id] == candidate
            )
            for candidate in range(2)
        )
    )
    return estimator


def _consume_record(
    estimator: FrozenStreamingISEstimator,
    record: ReplayRecord,
    candidate_by_id: dict[str, int],
) -> None:
    estimator.consume_fresh(
        record.record_id,
        candidate_by_id[record.record_id],
        ordinary_importance_log_weight(
            reward=record.reward,
            reward_temperature=1.0,
            target_logprob=record.behavior_logprob,
            behavior_logprob=record.behavior_logprob,
        ),
    )


def _broker_arm(
    factory: _BackendFactory,
    *,
    name: str,
    prompt: TokenSequence,
    chunk_tokens: int,
    seed: int,
    preserve_partials: bool,
) -> dict[str, Any]:
    backend = factory.create(None)
    try:
        _warmup(backend, backend.tokenizer, backend.tokenizer.eos_token_id)
        sampling = SamplingConfig()
        policy = BehaviorPolicy.for_backend(backend, sampling, label="base-fixed-length")
        lengths = (chunk_tokens, chunk_tokens) + (3 * chunk_tokens,) * 6
        key = ReplayKey(prompt, (), (), "controlled-token-reward-v1")
        requests = tuple(
            ReplaySampleRequest(
                key,
                length,
                SeedStream(seed).derive("broker", index),
                f"broker:{index}",
            )
            for index, length in enumerate(lengths)
        )
        candidate_by_id = {
            request.record_id: index % 2 for index, request in enumerate(requests)
        }
        estimator = _frozen_estimator(
            tuple(request.record_id for request in requests), candidate_by_id
        )
        before = _snapshot(backend)

        def run() -> dict[str, Any]:
            records: list[ReplayRecord] = []
            if preserve_partials:
                broker = AsyncRolloutBroker(
                    backend,
                    chunk_tokens=chunk_tokens,
                    max_batch_size=len(requests),
                )
                first = sample_replay_records_brokered(
                    policy,
                    requests,
                    _sequence_reward,
                    broker,
                    completion_target=2,
                    on_record=lambda record: _consume_record(
                        estimator, record, candidate_by_id
                    ),
                )
                second = sample_replay_records_brokered(
                    policy,
                    first.partial,
                    _sequence_reward,
                    broker,
                    on_record=lambda record: _consume_record(
                        estimator, record, candidate_by_id
                    ),
                )
                records.extend(first.records)
                records.extend(second.records)
                scheduling = {
                    "first_wave": asdict(first.snapshot),
                    "second_wave": asdict(second.snapshot),
                    "partial_tokens_preserved": first.snapshot.partial_tokens_preserved,
                    "discarded_partial_tokens": 0,
                }
            else:
                first_requests = tuple(
                    GenerationRequest(
                        prompt,
                        min(chunk_tokens, request.max_new_tokens),
                        sampling,
                        request.seed,
                        f"discard:first:{request.record_id}",
                    )
                    for request in requests
                )
                first_samples = backend.sample_batch(first_requests)
                for request, sample in zip(requests[:2], first_samples[:2], strict=True):
                    record = ReplayRecord(
                        request.record_id,
                        request.key,
                        sample.token_ids,
                        _sequence_reward((), sample.token_ids),
                        policy.behavior_id,
                        sample.logprob,
                    )
                    records.append(record)
                    _consume_record(estimator, record, candidate_by_id)
                restarted = sample_replay_records(policy, requests[2:], _sequence_reward)
                records.extend(restarted)
                for record in restarted:
                    _consume_record(estimator, record, candidate_by_id)
                scheduling = {
                    "first_wave": {
                        "completed_rollouts": 2,
                        "partial_rollouts": 6,
                    },
                    "second_wave": {"completed_rollouts": 6},
                    "partial_tokens_preserved": 0,
                    "discarded_partial_tokens": 6 * chunk_tokens,
                }
            return {
                "records": len(records),
                "completion_tokens": sum(len(record.completion) for record in records),
                "record_ids": [record.record_id for record in records],
                "scheduling": scheduling,
                "is_estimator": asdict(estimator.snapshot()),
            }

        result, telemetry = _measure(run)
        after = _snapshot(backend)
        return {
            "name": name,
            "online": {
                "telemetry": telemetry,
                "main_model": _delta(before, after),
                **result,
            },
        }
    finally:
        factory.release(backend)


def _streaming_is_arm(
    factory: _BackendFactory,
    *,
    name: str,
    prompt: TokenSequence,
    chunk_tokens: int,
    seed: int,
    verifier_delay_seconds: float,
    streaming: bool,
) -> dict[str, Any]:
    backend = factory.create(None)
    try:
        _warmup(backend, backend.tokenizer, backend.tokenizer.eos_token_id)
        sampling = SamplingConfig()
        lengths = tuple(
            chunk_tokens * (index % 4 + 1)
            for index in range(12)
        )
        requests = tuple(
            GenerationRequest(
                prompt,
                length,
                sampling,
                SeedStream(seed).derive("streaming-is", index),
                f"streaming-is:{index}",
            )
            for index, length in enumerate(lengths)
        )
        candidate_by_id = {
            request.request_id: index % 2 for index, request in enumerate(requests)
        }
        estimator = _frozen_estimator(
            tuple(request.request_id for request in requests), candidate_by_id
        )
        update_times: list[float] = []
        update_lock = threading.Lock()
        started_at = 0.0

        def verify(sample: SequenceSample) -> None:
            if verifier_delay_seconds:
                time.sleep(verifier_delay_seconds)
            value = ordinary_importance_log_weight(
                reward=_sequence_reward(sample.prefix, sample.token_ids),
                reward_temperature=1.0,
                target_logprob=sample.logprob,
                behavior_logprob=sample.logprob,
            )
            estimator.consume_fresh(
                sample.request_id,
                candidate_by_id[sample.request_id],
                value,
            )
            with update_lock:
                update_times.append(time.perf_counter() - started_at)

        before = _snapshot(backend)

        def run() -> dict[str, Any]:
            nonlocal started_at
            started_at = time.perf_counter()
            futures: list[Future[None]] = []
            futures_lock = threading.Lock()
            completion_order: list[str] = []
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="infra-verifier") as pool:
                if streaming:
                    def completed(_index: int, sample: SequenceSample) -> None:
                        completion_order.append(sample.request_id)
                        future = pool.submit(verify, sample)
                        with futures_lock:
                            futures.append(future)

                    samples = sample_batch_with_callback(backend, requests, completed)
                else:
                    samples = backend.sample_batch(requests)
                    completion_order.extend(sample.request_id for sample in samples)
                    futures.extend(pool.submit(verify, sample) for sample in samples)
                with futures_lock:
                    submitted = tuple(futures)
                for future in submitted:
                    future.result()
            return {
                "sequences": len(samples),
                "completion_order": completion_order,
                "first_estimator_update_seconds": min(update_times),
                "is_estimator": asdict(estimator.snapshot()),
                "output_hashes": [_token_hash(sample.token_ids) for sample in samples],
            }

        result, telemetry = _measure(run)
        after = _snapshot(backend)
        return {
            "name": name,
            "verifier_delay_seconds_per_sequence": verifier_delay_seconds,
            "online": {
                "telemetry": telemetry,
                "main_model": _delta(before, after),
                **result,
            },
        }
    finally:
        factory.release(backend)


def _speculation_arms(
    factory: _BackendFactory,
    *,
    prompt: TokenSequence,
    gold: Any,
    maximum: int,
    history_rollouts: int,
    draft_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    sampling = SamplingConfig(eos_token_id=factory.tokenizer.eos_token_id)
    histories = _history_batches(
        (prompt,),
        count=history_rollouts,
        length=maximum,
        sampling=sampling,
        seeds=SeedStream(seed),
    )
    evaluations = [
        [
            GenerationRequest(
                prompt,
                maximum,
                sampling,
                SeedStream(seed).derive("stochastic-tree", draw),
                f"stochastic-tree:{draw}",
            )
        ]
        for draw in range(4)
    ]
    deterministic = _speculation(
        maximum_batch=1,
        maximum_draft_tokens=draft_tokens,
        dynamic=False,
    )
    stochastic = replace(deterministic, stochastic_tree=True)
    arms: list[dict[str, Any]] = []
    traces: dict[str, dict[str, TokenSequence]] = {}
    for name, config in (
        ("no_history_draft", None),
        ("deterministic_history_draft", deterministic),
        ("stochastic_history_draft_exact", stochastic),
    ):
        arm, trace = _decode_arm(
            factory,
            name=name,
            speculation=config,
            dynamic_vllm=False,
            history_batches=histories,
            evaluation_batches=evaluations,
            gold_by_prompt={prompt: gold},
        )
        arms.append(arm)
        traces[name] = trace
    baseline = traces["no_history_draft"]
    for arm in arms:
        trace = traces[arm["name"]]
        common = tuple(sorted(set(baseline).intersection(trace)))
        arm["online"]["exact_token_trace_match_fraction_vs_no_draft"] = (
            sum(trace[key] == baseline[key] for key in common) / len(common)
            if common
            else None
        )
    return arms


def _mh_standard_or_prefetch_arm(
    factory: _BackendFactory,
    *,
    name: str,
    prompt: TokenSequence,
    config: RewardMHConfig,
    seed: int,
    reward_delay_seconds: float,
    prefetch: bool,
) -> dict[str, Any]:
    backend = factory.create(None)
    try:
        _warmup(backend, backend.tokenizer, backend.tokenizer.eos_token_id)
        calls = 0
        calls_lock = threading.Lock()

        def reward(_prompt: TokenSequence, sequence: TokenSequence) -> float:
            nonlocal calls
            with calls_lock:
                calls += 1
            if reward_delay_seconds:
                time.sleep(reward_delay_seconds)
            return _sequence_reward((), sequence)

        before = _snapshot(backend)
        if prefetch:
            result, telemetry = _measure(
                lambda: run_reward_mh_chain_prefetched(
                    backend,
                    prompt,
                    config,
                    SamplingConfig(),
                    reward,
                    SeedStream(seed),
                )
            )
            chain = result.chain
            scheduling = asdict(result.snapshot)
        else:
            chain, telemetry = _measure(
                lambda: run_reward_mh_chain(
                    backend,
                    prompt,
                    config,
                    SamplingConfig(),
                    reward,
                    SeedStream(seed),
                )
            )
            scheduling = {
                "used_proposals": config.updates,
                "prefetched_proposals": 0,
                "unused_prefetched_proposals": 0,
                "reward_evaluations": calls,
            }
        after = _snapshot(backend)
        return {
            "name": name,
            "reward_delay_seconds_per_evaluation": reward_delay_seconds,
            "online": {
                "telemetry": telemetry,
                "main_model": _delta(before, after),
                "reward_evaluations": calls,
                "scheduling": scheduling,
                "acceptance_rate": chain.acceptance_rate,
                "final_reward": chain.reward,
                "token_ids": list(chain.token_ids),
                "token_hash": _token_hash(chain.token_ids),
                "trace": [asdict(step) for step in chain.trace],
            },
        }
    finally:
        factory.release(backend)


def _mh_delayed_arm(
    factory: _BackendFactory,
    *,
    name: str,
    prompt: TokenSequence,
    config: RewardMHConfig,
    seed: int,
    reward_delay_seconds: float,
    delayed: bool,
) -> dict[str, Any]:
    backend = factory.create(None)
    try:
        _warmup(backend, backend.tokenizer, backend.tokenizer.eos_token_id)
        calls = 0

        def surrogate(_prompt: TokenSequence, sequence: TokenSequence) -> float:
            return float(bool(sequence) and int(sequence[-1]) % 2 == 0)

        def exact(prompt_tokens: TokenSequence, sequence: TokenSequence) -> float:
            nonlocal calls
            calls += 1
            if reward_delay_seconds:
                time.sleep(reward_delay_seconds)
            return surrogate(prompt_tokens, sequence)

        before = _snapshot(backend)
        if delayed:
            result, telemetry = _measure(
                lambda: run_reward_mh_chain_delayed(
                    backend,
                    prompt,
                    config,
                    SamplingConfig(),
                    exact,
                    surrogate,
                    SeedStream(seed),
                )
            )
            exact_evaluations = result.exact_reward_evaluations
            surrogate_evaluations = result.surrogate_reward_evaluations
            token_ids = result.token_ids
            acceptance_rate = result.acceptance_rate
            final_reward = result.reward
            early_rejections = sum(
                not step.exact_reward_evaluated for step in result.trace
            )
            trace = [asdict(step) for step in result.trace]
        else:
            result, telemetry = _measure(
                lambda: run_reward_mh_chain(
                    backend,
                    prompt,
                    config,
                    SamplingConfig(),
                    exact,
                    SeedStream(seed),
                )
            )
            exact_evaluations = calls
            surrogate_evaluations = 0
            token_ids = result.token_ids
            acceptance_rate = result.acceptance_rate
            final_reward = result.reward
            early_rejections = 0
            trace = [asdict(step) for step in result.trace]
        after = _snapshot(backend)
        return {
            "name": name,
            "reward_delay_seconds_per_evaluation": reward_delay_seconds,
            "surrogate": "exact token predicate without the controlled delay",
            "online": {
                "telemetry": telemetry,
                "main_model": _delta(before, after),
                "exact_reward_evaluations": exact_evaluations,
                "surrogate_reward_evaluations": surrogate_evaluations,
                "early_rejections": early_rejections,
                "acceptance_rate": acceptance_rate,
                "final_reward": final_reward,
                "token_ids": list(token_ids),
                "token_hash": _token_hash(token_ids),
                "trace": trace,
            },
        }
    finally:
        factory.release(backend)


def _mh_replay_arm(
    factory: _BackendFactory,
    *,
    name: str,
    prompt: TokenSequence,
    config: RewardMHConfig,
    history_rollouts: int,
    seed: int,
    replay: bool,
    chains: int = 4,
) -> dict[str, Any]:
    backend = factory.create(None)
    try:
        _warmup(backend, backend.tokenizer, backend.tokenizer.eos_token_id)

        def reward(_prompt: TokenSequence, sequence: TokenSequence) -> float:
            return float(bool(sequence) and int(sequence[-1]) % 2 == 0)

        if replay:
            history_requests = tuple(
                GenerationRequest(
                    prompt,
                    config.total_length,
                    SamplingConfig(),
                    SeedStream(seed).derive("mh-replay-history", index),
                    f"mh-replay-history:{index}",
                )
                for index in range(history_rollouts)
            )
            cache_before = _snapshot(backend)
            histories, cache_telemetry = _measure(
                lambda: backend.sample_batch(history_requests)
            )
            cache_after = _snapshot(backend)
            proposal = FrozenReplaySuffixProposal(
                backend,
                history_mixture=0.7,
            )
            proposal.observe_sequences(prompt, (sample.token_ids for sample in histories))
            before = _snapshot(backend)
            results, telemetry = _measure(
                lambda: tuple(
                    run_reward_mh_chain_replay_proposal(
                        proposal,
                        prompt,
                        config,
                        reward,
                        SeedStream(seed),
                        chain_id=chain_id,
                    )
                    for chain_id in range(chains)
                )
            )
            after = _snapshot(backend)
            proposal_snapshot = asdict(proposal.snapshot())
            source_counts = {
                source: sum(
                    step.proposal_source == source
                    for result in results
                    for step in result.trace
                )
                for source in ("base", "history")
            }
            cache = {
                "telemetry": cache_telemetry,
                "main_model": _delta(cache_before, cache_after),
                "sequences": len(histories),
                "tokens": sum(len(sample.token_ids) for sample in histories),
            }
        else:
            before = _snapshot(backend)
            results, telemetry = _measure(
                lambda: tuple(
                    run_reward_mh_chain(
                        backend,
                        prompt,
                        config,
                        SamplingConfig(),
                        reward,
                        SeedStream(seed),
                        chain_id=chain_id,
                    )
                    for chain_id in range(chains)
                )
            )
            after = _snapshot(backend)
            proposal_snapshot = None
            source_counts = {"base": config.updates * chains, "history": 0}
            cache = _zero_cost()
        attempts = sum(result.attempts for result in results)
        accepted = sum(result.accepted for result in results)
        return {
            "name": name,
            "cache_build": cache,
            "online": {
                "telemetry": telemetry,
                "main_model": _delta(before, after),
                "chains": chains,
                "acceptance_rate": accepted / attempts if attempts else 0.0,
                "mean_final_reward": sum(result.reward for result in results) / chains,
                "token_ids": [list(result.token_ids) for result in results],
                "token_hashes": [_token_hash(result.token_ids) for result in results],
                "proposal_sources": source_counts,
                "proposal_snapshot": proposal_snapshot,
                "traces": [
                    [asdict(step) for step in result.trace] for result in results
                ],
            },
        }
    finally:
        factory.release(backend)


def _prefetch_group(
    factory: _BackendFactory,
    *,
    prompt: TokenSequence,
    chunk_tokens: int,
    seed: int,
    slow_delay: float,
) -> list[dict[str, Any]]:
    config = RewardMHConfig(
        total_length=2 * chunk_tokens,
        block_size=2 * chunk_tokens,
        steps_per_block=4,
        reward_temperature=0.5,
    )
    arms: list[dict[str, Any]] = []
    for label, delay in (("cheap", 0.0), ("delayed", slow_delay)):
        ordinary = _mh_standard_or_prefetch_arm(
            factory,
            name=f"ordinary_mh_{label}_reward",
            prompt=prompt,
            config=config,
            seed=SeedStream(seed).derive("prefetch", label),
            reward_delay_seconds=delay,
            prefetch=False,
        )
        prefetched = _mh_standard_or_prefetch_arm(
            factory,
            name=f"proposal_tree_prefetch_{label}_reward",
            prompt=prompt,
            config=config,
            seed=SeedStream(seed).derive("prefetch", label),
            reward_delay_seconds=delay,
            prefetch=True,
        )
        prefetched["online"]["path_match_vs_ordinary"] = (
            prefetched["online"]["token_ids"] == ordinary["online"]["token_ids"]
            and prefetched["online"]["trace"] == ordinary["online"]["trace"]
        )
        arms.extend((ordinary, prefetched))
    return arms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/gsm8k_3090_aligned.toml"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm", "vllm-sync"),
        default="transformers",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--section",
        choices=("broker", "streaming", "speculation", "mh", "all"),
        default="all",
    )
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--history-rollouts", type=int, default=8)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--verifier-delay", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260812)
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.chunk_tokens <= 0 or args.max_new_tokens < 4 * args.chunk_tokens:
        raise ValueError("max-new-tokens must cover four positive chunks")
    if args.history_rollouts <= 0 or args.draft_tokens <= 0:
        raise ValueError("history-rollouts and draft-tokens must be positive")
    if args.verifier_delay < 0:
        raise ValueError("verifier-delay must be non-negative")
    if args.backend != "transformers" and args.section in {"speculation", "all"}:
        raise ValueError(
            "the stochastic empirical token-tree arm is a Transformers-specific "
            "ablation; vLLM uses its native exact suffix proposer"
        )

    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    apply_model_output_overrides(config, args)
    require_full_scope(config, "benchmark_is_mh_reuse")
    problem = select_problems(
        load_gsm8k(args.data), 1, seed=int(config["run"]["subset_seed"])
    )[0]
    factory = _BackendFactory(config, args.backend, args.dtype)
    try:
        prompt = _prompt_tokens(factory.tokenizer, problem.question, config)
        report: dict[str, Any] = {
            "schema_version": 1,
            "created_at_unix": time.time(),
            "machine": _machine(),
            "setting": {
                "backend": args.backend,
                "dtype": args.dtype,
                "dataset": "pinned OpenAI GSM8K test split",
                "problem_index": problem.index,
                "model": str(config["models"]["base"]),
                "chunk_tokens": args.chunk_tokens,
                "max_new_tokens": args.max_new_tokens,
                "history_rollouts": args.history_rollouts,
                "draft_tokens": args.draft_tokens,
                "controlled_verifier_delay_seconds": args.verifier_delay,
                "streaming_is_sequences": 12,
                "streaming_is_verifier_workers": 2,
                "mh_replay_chains": 4,
                "seed": args.seed,
            },
            "broker_arms": [],
            "streaming_is_arms": [],
            "stochastic_draft_arms": [],
            "mh_prefetch_arms": [],
            "mh_delayed_acceptance_arms": [],
            "mh_replay_proposal_arms": [],
            "accounting": {
                "main_model_flops": "2 * parameter_count * measured forward token slots",
                "cache_build_online_and_drain": "reported separately",
                "controlled_reward_delay": (
                    "diagnostic latency only; not GSM8K quality or model FLOPs"
                ),
                "excluded_from_main_model_flops": [
                    "CPU token-tree operations",
                    "reward sleep and parsing",
                    "scheduler bookkeeping",
                    "model loading",
                ],
            },
        }
        if args.section in {"broker", "all"}:
            report["broker_arms"] = [
                _broker_arm(
                    factory,
                    name="discard_partial_rollouts",
                    prompt=prompt,
                    chunk_tokens=args.chunk_tokens,
                    seed=args.seed,
                    preserve_partials=False,
                ),
                _broker_arm(
                    factory,
                    name="resume_partial_rollouts",
                    prompt=prompt,
                    chunk_tokens=args.chunk_tokens,
                    seed=args.seed,
                    preserve_partials=True,
                ),
            ]
            _save(report, args.output)
        if args.section in {"streaming", "all"}:
            for label, delay in (("cheap", 0.0), ("delayed", args.verifier_delay)):
                report["streaming_is_arms"].extend(
                    (
                        _streaming_is_arm(
                            factory,
                            name=f"wait_batch_then_verify_{label}",
                            prompt=prompt,
                            chunk_tokens=args.chunk_tokens,
                            seed=SeedStream(args.seed).derive("streaming", label),
                            verifier_delay_seconds=delay,
                            streaming=False,
                        ),
                        _streaming_is_arm(
                            factory,
                            name=f"stream_completion_into_is_{label}",
                            prompt=prompt,
                            chunk_tokens=args.chunk_tokens,
                            seed=SeedStream(args.seed).derive("streaming", label),
                            verifier_delay_seconds=delay,
                            streaming=True,
                        ),
                    )
                )
            _save(report, args.output)
        if args.section in {"speculation", "all"}:
            report["stochastic_draft_arms"] = _speculation_arms(
                factory,
                prompt=prompt,
                gold=problem.gold_answer,
                maximum=args.max_new_tokens,
                history_rollouts=args.history_rollouts,
                draft_tokens=args.draft_tokens,
                seed=args.seed,
            )
            _save(report, args.output)
        if args.section in {"mh", "all"}:
            report["mh_prefetch_arms"] = _prefetch_group(
                factory,
                prompt=prompt,
                chunk_tokens=args.chunk_tokens,
                seed=args.seed,
                slow_delay=args.verifier_delay,
            )
            delayed_config = RewardMHConfig(
                total_length=2 * args.chunk_tokens,
                block_size=2 * args.chunk_tokens,
                steps_per_block=8,
                reward_temperature=0.15,
            )
            delayed_seed = SeedStream(args.seed).derive("delayed-acceptance")
            report["mh_delayed_acceptance_arms"] = [
                _mh_delayed_arm(
                    factory,
                    name="ordinary_mh_expensive_reward",
                    prompt=prompt,
                    config=delayed_config,
                    seed=delayed_seed,
                    reward_delay_seconds=args.verifier_delay,
                    delayed=False,
                ),
                _mh_delayed_arm(
                    factory,
                    name="delayed_acceptance_exact",
                    prompt=prompt,
                    config=delayed_config,
                    seed=delayed_seed,
                    reward_delay_seconds=args.verifier_delay,
                    delayed=True,
                ),
            ]
            replay_config = RewardMHConfig(
                total_length=2 * args.chunk_tokens,
                block_size=2 * args.chunk_tokens,
                steps_per_block=8,
                reward_temperature=0.3,
            )
            replay_seed = SeedStream(args.seed).derive("mh-replay")
            report["mh_replay_proposal_arms"] = [
                _mh_replay_arm(
                    factory,
                    name="base_suffix_proposal",
                    prompt=prompt,
                    config=replay_config,
                    history_rollouts=args.history_rollouts,
                    seed=replay_seed,
                    replay=False,
                ),
                _mh_replay_arm(
                    factory,
                    name="frozen_replay_mixture_proposal",
                    prompt=prompt,
                    config=replay_config,
                    history_rollouts=args.history_rollouts,
                    seed=replay_seed,
                    replay=True,
                ),
            ]
            _save(report, args.output)
        _save(report, args.output)
    finally:
        factory.close()


if __name__ == "__main__":
    main()
