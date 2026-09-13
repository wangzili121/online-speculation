"""Measure off-policy rollout replay against a fresh-only matched baseline.

The reward comes from the independently configured verifier.  The default GSM8K
profile selects a deterministic numeric-reference verifier so replayed values
can be checked exactly; another verifier can be selected without changing the
replay implementation.
"""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import hashlib
import json
import statistics
import tomllib
from pathlib import Path
from typing import Any


from experiments.arllm.scoped_execution import fixed_experiment_reward, with_output_scope
from experiments.arllm.runtime import validate_model_artifacts
from experiments.shared.artifacts import load_jsonl as _load_records

from experiments.arllm.gsm8k_reproduction import (
    _configured_verifier_reward,
    _fingerprint,
    _implementation_hashes,
    _fraction_text,
    _load_backend,
    _prompt_tokens,
    _snapshot_delta,
    _timed,
)
from inference_scaling.arllm.algorithms.base_replay import _score_base, base_replay_step
from inference_scaling.arllm.algorithms.conditional_is import _sample_candidates
from inference_scaling.arllm.backends import (
    BACKEND_CHOICES,
    ScoreCachingBackend,
    close_backend,
    set_backend_override,
)
from inference_scaling.arllm.config import BaseReplayConfig, SamplingConfig
from inference_scaling.shared.evaluation import extract_numeric_answer, load_gsm8k, select_problems
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    ReplaySampleRequest,
    sample_replay_records,
    validate_record_probabilities,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import replace_verifier_from_file
from inference_scaling.arllm.types import GenerationRequest

IMPLEMENTATION_FILES = (
    "experiments/arllm/gsm8k_replay_benchmark.py",
    "src/inference_scaling/arllm/algorithms/base_replay.py",
    "src/inference_scaling/arllm/algorithms/conditional_is.py",
    "src/inference_scaling/arllm/backends/cache.py",
    "src/inference_scaling/arllm/backends/loader.py",
    "src/inference_scaling/arllm/backends/transformers_backend.py",
    "src/inference_scaling/arllm/backends/vllm_backend.py",
    "src/inference_scaling/arllm/replay.py",
    "src/inference_scaling/arllm/config.py",
    "src/inference_scaling/arllm/types.py",
)


@with_output_scope
def _run_fresh(
    backend,
    prompt,
    reward,
    reward_version: str,
    config: dict[str, Any],
    seeds: SeedStream,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    section = config["conditional_is"]
    replay = config["replay"]
    total_rollouts = int(replay["history_rollouts"]) + int(replay["fresh_rollouts"])
    algorithm = BaseReplayConfig(
        candidate_count=int(section["candidate_count"]),
        block_size=int(section["block_size"]),
        total_length=int(config["generation"]["max_new_tokens"]),
        reward_temperature=float(section["reward_temperature"]),
        max_history_per_candidate=0,
        fresh_rollouts=total_rollouts,
        truncation=float(replay["truncation"]),
    )
    sampling = SamplingConfig(temperature=float(config.get("sampling", {}).get("temperature", 1.0)), eos_token_id=backend.tokenizer.eos_token_id)
    registry = BehaviorRegistry()
    store = InMemoryReplayStore()
    generated: list[int] = []
    steps = []
    step_index = 0
    while len(generated) < algorithm.total_length:
        step = base_replay_step(
            base_backend=backend,
            registry=registry,
            store=store,
            prompt=prompt,
            generated_prefix=tuple(generated),
            config=algorithm,
            base_sampling=sampling,
            reward=reward,
            reward_version=reward_version,
            seeds=seeds,
            step_index=step_index,
        )
        generated.extend(step.selected.token_ids)
        steps.append(step)
        eos = backend.tokenizer.eos_token_id
        if eos in step.selected.token_ids:
            generated = generated[: generated.index(eos) + 1]
            break
        step_index += 1
    return tuple(generated), {
        "steps": len(steps),
        "history_used": sum(
            candidate.estimate.history_count for step in steps for candidate in step.candidates
        ),
        "fresh_used": sum(
            candidate.estimate.fresh_count for step in steps for candidate in step.candidates
        ),
    }


@with_output_scope
def _run_warm(
    backend,
    proposal_backend,
    prompt,
    reward,
    reward_version: str,
    config: dict[str, Any],
    seeds: SeedStream,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    section = config["conditional_is"]
    replay = config["replay"]
    algorithm = BaseReplayConfig(
        candidate_count=int(section["candidate_count"]),
        block_size=int(section["block_size"]),
        total_length=int(config["generation"]["max_new_tokens"]),
        reward_temperature=float(section["reward_temperature"]),
        max_history_per_candidate=int(replay["history_rollouts"]),
        fresh_rollouts=int(replay["fresh_rollouts"]),
        truncation=float(replay["truncation"]),
    )
    sampling = SamplingConfig(temperature=float(config.get("sampling", {}).get("temperature", 1.0)), eos_token_id=backend.tokenizer.eos_token_id)
    proposal_sampling = SamplingConfig(eos_token_id=proposal_backend.tokenizer.eos_token_id)
    cached_base = ScoreCachingBackend(backend)
    cached_proposal = ScoreCachingBackend(proposal_backend)
    behavior = BehaviorPolicy.for_backend(
        cached_proposal,
        proposal_sampling,
        label="small-model-history",
    )
    registry = BehaviorRegistry()
    registry.register(behavior)
    store = InMemoryReplayStore()
    generated: list[int] = []
    steps = []
    cache_build_seconds = 0.0
    online_seconds = 0.0
    candidates_reproduced = True
    candidate_draws_reused = 0
    history_generated = 0
    step_index = 0
    cache_base_deltas: list[dict[str, int | float]] = []
    cache_proposal_deltas: list[dict[str, int | float]] = []
    online_base_deltas: list[dict[str, int | float]] = []
    online_proposal_deltas: list[dict[str, int | float]] = []

    while len(generated) < algorithm.total_length:
        remaining = algorithm.total_length - len(generated)
        block_length = min(algorithm.block_size, remaining)
        cache_base_before = backend.snapshot()
        cache_proposal_before = proposal_backend.snapshot()

        def build_history(
            generated_prefix=tuple(generated),
            block_length=block_length,
            remaining=remaining,
            step_index=step_index,
        ):
            candidates = _sample_candidates(
                cached_base,
                prompt + generated_prefix,
                algorithm.candidate_count,
                block_length,
                sampling,
                seeds,
                step_index,
            )
            requests: list[ReplaySampleRequest] = []
            rollout_length = max(0, remaining - block_length)
            for candidate_index, candidate in enumerate(candidates):
                if rollout_length == 0 or candidate.token_ids[-1] == backend.tokenizer.eos_token_id:
                    continue
                key = ReplayKey(
                    prompt,
                    generated_prefix,
                    candidate.token_ids,
                    reward_version,
                )
                for history_index in range(algorithm.max_history_per_candidate):
                    requests.append(
                        ReplaySampleRequest(
                            key=key,
                            max_new_tokens=rollout_length,
                            seed=seeds.derive(
                                "replay-history",
                                step_index,
                                candidate_index,
                                history_index,
                            ),
                            record_id=(
                                f"history:{step_index}:{candidate_index}:{history_index}:"
                                f"{seeds.derive('history-id', step_index, candidate_index, history_index)}"
                            ),
                        )
                    )
            records = sample_replay_records(behavior, requests, reward)
            # A reusable rollout cache stores all immutable probability terms once.
            # The online decision then reads exact cache entries rather than running
            # the base and behavior models again for historical completions.
            validate_record_probabilities(records, registry)
            records_by_key: dict[ReplayKey, list[tuple[int, ...]]] = {}
            for record in records:
                records_by_key.setdefault(record.key, []).append(record.completion)
            for key, completions in records_by_key.items():
                _score_base(cached_base, key, completions, sampling)
            for record in records:
                store.add_evaluation(record)
            return candidates, len(records)

        (cached_candidates, generated_count), build_seconds = _timed(build_history)
        cache_base_after = backend.snapshot()
        cache_proposal_after = proposal_backend.snapshot()
        cache_build_seconds += build_seconds
        history_generated += generated_count
        online_before = backend.snapshot()
        online_proposal_before = proposal_backend.snapshot()
        (step, decision_seconds) = _timed(
            lambda generated_prefix=tuple(generated),
            step_index=step_index,
            candidate_samples=cached_candidates: base_replay_step(
                base_backend=cached_base,
                registry=registry,
                store=store,
                prompt=prompt,
                generated_prefix=generated_prefix,
                config=algorithm,
                base_sampling=sampling,
                reward=reward,
                reward_version=reward_version,
                seeds=seeds,
                step_index=step_index,
                candidate_samples=candidate_samples,
            )
        )
        online_seconds += decision_seconds
        online_after = backend.snapshot()
        online_proposal_after = proposal_backend.snapshot()
        candidates_reproduced &= [item.token_ids for item in cached_candidates] == [
            item.token_ids for item in step.candidates
        ]
        candidate_draws_reused += len(cached_candidates)
        generated.extend(step.selected.token_ids)
        steps.append(step)
        cache_base_deltas.append(_snapshot_delta(cache_base_before, cache_base_after))
        cache_proposal_deltas.append(
            _snapshot_delta(cache_proposal_before, cache_proposal_after)
        )
        online_base_deltas.append(_snapshot_delta(online_before, online_after))
        online_proposal_deltas.append(
            _snapshot_delta(online_proposal_before, online_proposal_after)
        )
        eos = backend.tokenizer.eos_token_id
        if eos in step.selected.token_ids:
            generated = generated[: generated.index(eos) + 1]
            break
        step_index += 1

    history_used = sum(
        candidate.estimate.history_count for step in steps for candidate in step.candidates
    )
    fresh_used = sum(
        candidate.estimate.fresh_count for step in steps for candidate in step.candidates
    )

    def total(field: str, deltas: list[dict[str, int | float]]) -> int:
        return sum(int(delta[field]) for delta in deltas)

    return tuple(generated), {
        "steps": len(steps),
        "cache_build_seconds": cache_build_seconds,
        "online_seconds": online_seconds,
        "end_to_end_seconds": cache_build_seconds + online_seconds,
        "history_generated": history_generated,
        "history_used": history_used,
        "fresh_used": fresh_used,
        "rollout_reuse_rate": history_used / (history_used + fresh_used)
        if history_used + fresh_used
        else 0.0,
        "candidates_reproduced": candidates_reproduced,
        "candidate_draws_reused": candidate_draws_reused,
        "evaluation_records_remaining": store.evaluation_count,
        "design_records": store.design_count,
        "cache_build_base_forward_token_slots": total(
            "generation_forward_token_slots", cache_base_deltas
        )
        + total("score_forward_token_slots", cache_base_deltas),
        "cache_build_proposal_forward_token_slots": total(
            "generation_forward_token_slots", cache_proposal_deltas
        )
        + total("score_forward_token_slots", cache_proposal_deltas),
        "cache_build_estimated_dense_forward_flops": total(
            "estimated_dense_forward_flops", cache_base_deltas
        )
        + total("estimated_dense_forward_flops", cache_proposal_deltas),
        "cache_build_base_estimated_dense_forward_flops": total(
            "estimated_dense_forward_flops", cache_base_deltas
        ),
        "cache_build_proposal_estimated_dense_forward_flops": total(
            "estimated_dense_forward_flops", cache_proposal_deltas
        ),
        "online_base_forward_token_slots": total(
            "generation_forward_token_slots", online_base_deltas
        )
        + total("score_forward_token_slots", online_base_deltas),
        "online_proposal_forward_token_slots": total(
            "generation_forward_token_slots", online_proposal_deltas
        )
        + total("score_forward_token_slots", online_proposal_deltas),
        "online_forward_token_slots": total(
            "generation_forward_token_slots", online_base_deltas
        )
        + total("score_forward_token_slots", online_base_deltas)
        + total("generation_forward_token_slots", online_proposal_deltas)
        + total("score_forward_token_slots", online_proposal_deltas),
        "online_base_estimated_dense_forward_flops": total(
            "estimated_dense_forward_flops", online_base_deltas
        ),
        "online_proposal_estimated_dense_forward_flops": total(
            "estimated_dense_forward_flops", online_proposal_deltas
        ),
        "online_estimated_dense_forward_flops": total(
            "estimated_dense_forward_flops", online_base_deltas
        )
        + total("estimated_dense_forward_flops", online_proposal_deltas),
        "base_score_cache": {
            "entries": cached_base.snapshot().entries,
            "hits": cached_base.snapshot().hits,
            "misses": cached_base.snapshot().misses,
        },
        "proposal_score_cache": {
            "entries": cached_proposal.snapshot().entries,
            "hits": cached_proposal.snapshot().hits,
            "misses": cached_proposal.snapshot().misses,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        help="optional second path for the aggregate summary only",
    )
    parser.add_argument("--tag", default="default")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verifier-config", type=Path)
    add_model_output_arguments(parser)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    apply_model_output_overrides(config, args)
    replace_verifier_from_file(config, args.verifier_config)
    set_backend_override(config, args.backend)
    if args.limit is not None:
        config["run"]["sample_count"] = args.limit
    problems = select_problems(
        load_gsm8k(args.data),
        int(config["run"]["sample_count"]),
        seed=int(config["run"]["subset_seed"]),
    )
    run_dir = (
        args.output_root
        / str(config["run"]["name"])
        / f"replay-comparison-{args.tag}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    input_artifacts = validate_model_artifacts(config, {"base", "proposal"})
    actual_base_hash = input_artifacts["weight_sha256"]["base"]
    actual_proposal_hash = input_artifacts["weight_sha256"]["proposal"]
    effective = {
        "config": config,
        "tag": args.tag,
        "problem_indices": [problem.index for problem in problems],
        "implementation_sha256": _implementation_hashes(
            Path(__file__).resolve().parents[2],
            entrypoints=IMPLEMENTATION_FILES,
        ),
        "input_weight_sha256": {
            "base": actual_base_hash,
            "proposal": actual_proposal_hash,
        },
        "input_metadata_sha256": input_artifacts["metadata_sha256"],
    }
    fingerprint = _fingerprint(effective)
    manifest = {
        "schema_version": 4,
        "fingerprint": fingerprint,
        "benchmark": "GSM8K verifier-assisted rollout replay performance",
        "effective": effective,
    }
    previous = manifest
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError(
                f"{run_dir} contains a different replay benchmark; choose a new --tag"
            )
    elif records_path.is_file():
        raise ValueError(f"{run_dir} contains records without a manifest")
    completed = {int(item["problem_index"]) for item in _load_records(records_path)}
    pending = [problem for problem in problems if problem.index not in completed]
    backend = None
    proposal = None
    if pending:
        backend = _load_backend(str(config["models"]["base"]), config, role="base")
        proposal = _load_backend(str(config["models"]["proposal"]), config, role="proposal")
        if backend.tokenizer.get_vocab() != proposal.tokenizer.get_vocab():
            raise ValueError("base and replay proposal tokenizers must match")
        manifest["models"] = {
            "base": {
                "path": str(config["models"]["base"]),
                "parameter_count": backend.parameter_count,
            },
            "proposal": {
                "path": str(config["models"]["proposal"]),
                "parameter_count": proposal.parameter_count,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prompt = _prompt_tokens(backend, pending[0], config)
        backend.sample_batch(
            [
                # Warm model kernels; the request is excluded by per-method snapshots.
                GenerationRequest(
                    prompt,
                    2,
                    SamplingConfig(eos_token_id=backend.tokenizer.eos_token_id),
                    int(config["run"]["seed"]),
                    "warmup",
                )
            ]
        )
    else:
        manifest = previous

    with records_path.open("a", encoding="utf-8", buffering=1) as sink:
        for ordinal, problem in enumerate(pending, 1):
            prompt = _prompt_tokens(backend, problem, config)
            verifier_reward, reward_version = fixed_experiment_reward(backend, problem, config, _configured_verifier_reward)
            seed = SeedStream(
                SeedStream(int(config["run"]["seed"])).derive("replay", problem.index)
            )

            fresh_before = backend.snapshot()
            (fresh_tokens, fresh_info), fresh_seconds = _timed(
                lambda prompt=prompt,
                reward=verifier_reward,
                reward_version=reward_version,
                seed=seed: _run_fresh(
                    backend, prompt, reward, reward_version, config, seed
                )
            )
            fresh_after = backend.snapshot()
            warm_base_before = backend.snapshot()
            warm_proposal_before = proposal.snapshot()
            (warm_tokens, warm_info), warm_total_seconds = _timed(
                lambda prompt=prompt,
                reward=verifier_reward,
                reward_version=reward_version,
                seed=seed: _run_warm(
                    backend,
                    proposal,
                    prompt,
                    reward,
                    reward_version,
                    config,
                    seed,
                )
            )
            warm_base_after = backend.snapshot()
            warm_proposal_after = proposal.snapshot()
            fresh_delta = _snapshot_delta(fresh_before, fresh_after)
            fresh_forward_slots = int(
                fresh_delta["generation_forward_token_slots"]
            ) + int(fresh_delta["score_forward_token_slots"])
            fresh_flops = int(fresh_delta["estimated_dense_forward_flops"])
            warm_online_flops = int(
                warm_info["online_estimated_dense_forward_flops"]
            )
            warm_end_to_end_flops = warm_online_flops + int(
                warm_info["cache_build_estimated_dense_forward_flops"]
            )
            online_speedup = fresh_seconds / warm_info["online_seconds"]
            end_to_end_speedup = fresh_seconds / warm_total_seconds
            fresh_text = fresh_info["output_segments"]["content_text"]
            warm_text = warm_info["output_segments"]["content_text"]
            record = {
                "schema_version": 4,
                "manifest_fingerprint": fingerprint,
                "problem_index": problem.index,
                "question_sha256": hashlib.sha256(problem.question.encode()).hexdigest(),
                "gold_answer": _fraction_text(problem.gold_answer),
                "fresh": {
                    "seconds": fresh_seconds,
                    "prediction": _fraction_text(extract_numeric_answer(fresh_text)),
                    "correct": extract_numeric_answer(fresh_text) == problem.gold_answer,
                    "output": fresh_text,
                    "backend_delta": fresh_delta,
                    "forward_token_slots": fresh_forward_slots,
                    "estimated_dense_forward_flops": fresh_flops,
                    **fresh_info,
                },
                "warm_replay": {
                    "measured_total_seconds": warm_total_seconds,
                    "prediction": _fraction_text(extract_numeric_answer(warm_text)),
                    "correct": extract_numeric_answer(warm_text) == problem.gold_answer,
                    "output": warm_text,
                    "base_backend_delta": _snapshot_delta(warm_base_before, warm_base_after),
                    "proposal_backend_delta": _snapshot_delta(
                        warm_proposal_before, warm_proposal_after
                    ),
                    **warm_info,
                },
                "comparisons": {
                    "fresh_over_warm_online_flop_factor": (
                        fresh_flops / warm_online_flops
                    ),
                    "fresh_over_warm_one_shot_flop_factor": (
                        fresh_flops / warm_end_to_end_flops
                    ),
                    "fresh_over_warm_online_wall_time_factor": online_speedup,
                    "fresh_over_warm_one_shot_wall_time_factor": end_to_end_speedup,
                },
            }
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"[{ordinal}/{len(pending)}] gsm8k_index={problem.index} "
                f"fresh_over_warm_online_wall={online_speedup:.3f} "
                f"reuse={warm_info['rollout_reuse_rate']:.3f}",
                flush=True,
            )

    selected_indices = {problem.index for problem in problems}
    records = [
        item
        for item in _load_records(records_path)
        if int(item["problem_index"]) in selected_indices
    ]
    online_speedups = [
        item["comparisons"]["fresh_over_warm_online_wall_time_factor"]
        for item in records
    ]
    end_to_end_speedups = [
        item["comparisons"]["fresh_over_warm_one_shot_wall_time_factor"]
        for item in records
    ]
    fresh_flops = sum(
        int(item["fresh"]["estimated_dense_forward_flops"]) for item in records
    )
    warm_online_flops = sum(
        int(item["warm_replay"]["online_estimated_dense_forward_flops"])
        for item in records
    )
    warm_cache_flops = sum(
        int(item["warm_replay"]["cache_build_estimated_dense_forward_flops"])
        for item in records
    )
    warm_online_base_flops = sum(
        int(item["warm_replay"]["online_base_estimated_dense_forward_flops"])
        for item in records
    )
    warm_online_proposal_flops = sum(
        int(item["warm_replay"]["online_proposal_estimated_dense_forward_flops"])
        for item in records
    )
    warm_cache_base_flops = sum(
        int(item["warm_replay"]["cache_build_base_estimated_dense_forward_flops"])
        for item in records
    )
    warm_cache_proposal_flops = sum(
        int(item["warm_replay"]["cache_build_proposal_estimated_dense_forward_flops"])
        for item in records
    )
    base_cache_hits = sum(
        int(item["warm_replay"]["base_score_cache"]["hits"])
        for item in records
    )
    base_cache_misses = sum(
        int(item["warm_replay"]["base_score_cache"]["misses"])
        for item in records
    )
    proposal_cache_hits = sum(
        int(item["warm_replay"]["proposal_score_cache"]["hits"])
        for item in records
    )
    proposal_cache_misses = sum(
        int(item["warm_replay"]["proposal_score_cache"]["misses"])
        for item in records
    )
    summary = {
        "schema_version": 4,
        "benchmark": "GSM8K verifier-assisted rollout replay performance",
        "manifest_fingerprint": fingerprint,
        "profile": str(config["run"]["name"]),
        "problem_indices": [problem.index for problem in problems],
        "input_weight_sha256": effective["input_weight_sha256"],
        "implementation_sha256": effective["implementation_sha256"],
        "settings": {
            "generation": config["generation"],
            "sampling": config.get("sampling", {"temperature": 1.0}),
            "conditional_is": config["conditional_is"],
            "replay": config["replay"],
        },
        "examples": len(records),
        "comparison_contract": {
            "fresh_over_warm_online_flop_factor": (
                "fresh-only H+F base-rollout estimated dense FLOPs divided by warm "
                "online F-base-rollout plus fresh behavior-scoring estimated dense "
                "FLOPs; immutable history base/behavior scores are cache hits, and "
                "history construction is excluded because it occurred before the "
                "repeated query; values above one are a reduction"
            ),
            "fresh_over_warm_one_shot_flop_factor": (
                "fresh-only estimated dense FLOPs divided by small-model history "
                "construction, one-time history base/behavior scoring, and warm online "
                "estimated dense FLOPs; values below one mean the first query costs more"
            ),
            "fresh_over_warm_online_wall_time_factor": (
                "fresh-only replay with H+F newly generated base rollouts divided by "
                "warm replay with H pre-existing, pre-scored off-policy rollouts and F "
                "fresh base rollouts; candidate count, candidate source, block size, "
                "output length, and H+F are fixed; values above one are a speedup"
            ),
            "fresh_over_warm_one_shot_wall_time_factor": (
                "same fresh-only runtime divided by cache construction plus warm online "
                "runtime; values below one mean the first query is slower"
            ),
        },
        "fresh_total_estimated_dense_forward_flops": fresh_flops,
        "warm_online_total_estimated_dense_forward_flops": warm_online_flops,
        "warm_online_base_estimated_dense_forward_flops": warm_online_base_flops,
        "warm_online_proposal_estimated_dense_forward_flops": (
            warm_online_proposal_flops
        ),
        "cache_build_total_estimated_dense_forward_flops": warm_cache_flops,
        "cache_build_base_estimated_dense_forward_flops": warm_cache_base_flops,
        "cache_build_proposal_estimated_dense_forward_flops": (
            warm_cache_proposal_flops
        ),
        "candidate_draws_reused": sum(
            int(item["warm_replay"]["candidate_draws_reused"]) for item in records
        ),
        "aggregate_fresh_over_warm_online_flop_factor": fresh_flops
        / warm_online_flops,
        "aggregate_fresh_over_warm_one_shot_flop_factor": (
            fresh_flops / (warm_online_flops + warm_cache_flops)
        ),
        "compute_definition": (
            "2 * each model's parameter count * observed forward token slots; base and "
            "small proposal contributions are calculated separately"
        ),
        "mean_fresh_over_warm_online_wall_time_factor": statistics.fmean(
            online_speedups
        ),
        "median_fresh_over_warm_online_wall_time_factor": statistics.median(
            online_speedups
        ),
        "mean_fresh_over_warm_one_shot_wall_time_factor": statistics.fmean(
            end_to_end_speedups
        ),
        "mean_rollout_reuse_rate": statistics.fmean(
            item["warm_replay"]["rollout_reuse_rate"] for item in records
        ),
        "history_score_cache": {
            "base_hits": base_cache_hits,
            "base_misses": base_cache_misses,
            "base_hit_rate": base_cache_hits / (base_cache_hits + base_cache_misses)
            if base_cache_hits + base_cache_misses
            else 0.0,
            "proposal_hits": proposal_cache_hits,
            "proposal_misses": proposal_cache_misses,
            "proposal_hit_rate": proposal_cache_hits
            / (proposal_cache_hits + proposal_cache_misses)
            if proposal_cache_hits + proposal_cache_misses
            else 0.0,
        },
        "fresh_total_seconds": sum(item["fresh"]["seconds"] for item in records),
        "warm_online_total_seconds": sum(
            item["warm_replay"]["online_seconds"] for item in records
        ),
        "cache_build_total_seconds": sum(
            item["warm_replay"]["cache_build_seconds"] for item in records
        ),
    }
    serialized_summary = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    summary_path.write_text(serialized_summary, encoding="utf-8")
    if args.aggregate_output is not None:
        args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_output.write_text(serialized_summary, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    close_backend(proposal)
    close_backend(backend)


if __name__ == "__main__":
    main()
