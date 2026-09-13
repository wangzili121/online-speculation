"""Compare fresh LLaDA rollouts with corrected early-exit trajectory replay."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import sys
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dllm.gsm8k_reproduction import (
    IMPLEMENTATION_FILES as QUALITY_IMPLEMENTATION_FILES,
    _configured_verifier_reward,
)
from experiments.shared.paired_protocol import load_pairing
from experiments.shared.artifacts import load_jsonl as _load_records
from experiments.shared.statistics import wilson_interval
from experiments.dllm.profiles import apply_execution_profile
from experiments.dllm.runtime import (
    capped_generation_length,
    checkpoint_metadata_hashes,
    implementation_hashes,
    json_fingerprint,
    llada_snapshot_delta,
    sampling_from_section,
    validate_llada_weights,
)
from inference_scaling.dllm.backends import load_llada_backend
from inference_scaling.dllm.config import diffusion_decision_stage_lengths
from inference_scaling.dllm.replay import (
    build_diffusion_replay_history,
    select_diffusion_candidates_with_replay,
)
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence
from inference_scaling.shared.verifier import (
    TokenBatchReward,
    replace_verifier_from_file,
)

IMPLEMENTATION_FILES = (
    *QUALITY_IMPLEMENTATION_FILES,
    "experiments/dllm/gsm8k_replay_benchmark.py",
    "src/inference_scaling/dllm/replay.py",
    "src/inference_scaling/shared/importance.py",
)


def _sample_candidates(
    backend: Any,
    *,
    prefix: TokenSequence,
    length: int,
    count: int,
    sampling: Any,
    seeds: SeedStream,
    stage_index: int,
):
    requests = [
        DiffusionGenerationRequest(
            prefix=prefix,
            generation_length=length,
            sampling=sampling,
            seed=seeds.derive("dllm-replay-candidate", stage_index, candidate_index),
            request_id=f"dllm-replay-candidate:{stage_index}:{candidate_index}",
        )
        for candidate_index in range(count)
    ]
    candidates = backend.sample_batch(requests)
    if len(candidates) != count:
        raise RuntimeError("backend returned an invalid replay candidate count")
    return tuple(candidates)


def _run_fresh(
    backend: Any,
    prompt: TokenSequence,
    reward_batch: TokenBatchReward,
    config: dict[str, Any],
    seed: int,
) -> tuple[TokenSequence, dict[str, Any]]:
    generation_sampling = sampling_from_section(config["generation"])
    exact_sampling = sampling_from_section(config["exact_policy"])
    maximum = int(config["generation"]["max_new_tokens"])
    total_length = capped_generation_length(
        prompt_length=len(prompt), maximum=maximum, sampling=generation_sampling
    )
    conditional = config["conditional_is"]
    replay = config["replay"]
    stage_lengths = diffusion_decision_stage_lengths(
        prompt_length=len(prompt),
        total_length=total_length,
        decision_block_size=int(conditional["decision_block_size"]),
        sampling=generation_sampling,
    )
    seeds = SeedStream(seed)
    generated: TokenSequence = ()
    selections = []
    total_fresh = int(replay["history_rollouts"]) + int(replay["fresh_rollouts"])
    for stage_index, candidate_length in enumerate(stage_lengths):
        candidates = _sample_candidates(
            backend,
            prefix=prompt + generated,
            length=candidate_length,
            count=int(conditional["candidate_count"]),
            sampling=generation_sampling,
            seeds=seeds,
            stage_index=stage_index,
        )
        rollout_length = total_length - len(generated) - candidate_length
        selection = select_diffusion_candidates_with_replay(
            target_backend=backend,
            behavior_backend=None,
            prompt=prompt,
            generated_prefix=generated,
            candidates=candidates,
            histories=None,
            rollout_length=rollout_length,
            fresh_count=total_fresh,
            target_sampling=exact_sampling,
            behavior_sampling=None,
            reward_batch=reward_batch,
            reward_temperature=float(conditional["reward_temperature"]),
            truncation=float(replay["truncation"]),
            seed=seeds.derive("dllm-replay-fresh-select", stage_index),
        )
        selections.append(selection)
        generated += selection.selected.sample.token_ids
    return generated, {
        "decision_stages": len(selections),
        "history_used": 0,
        "fresh_used": sum(
            candidate.estimate.fresh_count
            for selection in selections
            for candidate in selection.candidates
        ),
    }


def _run_warm(
    backend: Any,
    proposal: Any,
    prompt: TokenSequence,
    reward_batch: TokenBatchReward,
    config: dict[str, Any],
    seed: int,
) -> tuple[TokenSequence, dict[str, Any]]:
    generation_sampling = sampling_from_section(config["generation"])
    exact_sampling = sampling_from_section(config["exact_policy"])
    maximum = int(config["generation"]["max_new_tokens"])
    total_length = capped_generation_length(
        prompt_length=len(prompt), maximum=maximum, sampling=generation_sampling
    )
    conditional = config["conditional_is"]
    replay = config["replay"]
    stage_lengths = diffusion_decision_stage_lengths(
        prompt_length=len(prompt),
        total_length=total_length,
        decision_block_size=int(conditional["decision_block_size"]),
        sampling=generation_sampling,
    )
    seeds = SeedStream(seed)
    generated: TokenSequence = ()
    selections = []
    build_seconds = 0.0
    online_seconds = 0.0
    build_base_deltas = []
    build_proposal_deltas = []
    online_base_deltas = []
    online_proposal_deltas = []
    candidates_reproduced = True
    history_generated = 0
    for stage_index, candidate_length in enumerate(stage_lengths):
        rollout_length = total_length - len(generated) - candidate_length
        build_base_before = backend.snapshot()
        build_proposal_before = proposal.snapshot()
        started = time.perf_counter()
        cached_candidates = _sample_candidates(
            backend,
            prefix=prompt + generated,
            length=candidate_length,
            count=int(conditional["candidate_count"]),
            sampling=generation_sampling,
            seeds=seeds,
            stage_index=stage_index,
        )
        histories = build_diffusion_replay_history(
            target_backend=backend,
            behavior_backend=proposal,
            prompt=prompt,
            generated_prefix=generated,
            candidates=cached_candidates,
            rollout_length=rollout_length,
            count_per_candidate=int(replay["history_rollouts"]),
            target_sampling=exact_sampling,
            behavior_sampling=exact_sampling,
            reward_batch=reward_batch,
            seed=seeds.derive("dllm-replay-history", stage_index),
        )
        build_seconds += time.perf_counter() - started
        build_base_deltas.append(
            llada_snapshot_delta(build_base_before, backend.snapshot())
        )
        build_proposal_deltas.append(
            llada_snapshot_delta(build_proposal_before, proposal.snapshot())
        )
        history_generated += sum(len(history.records) for history in histories)

        online_base_before = backend.snapshot()
        online_proposal_before = proposal.snapshot()
        started = time.perf_counter()
        online_candidates = _sample_candidates(
            backend,
            prefix=prompt + generated,
            length=candidate_length,
            count=int(conditional["candidate_count"]),
            sampling=generation_sampling,
            seeds=seeds,
            stage_index=stage_index,
        )
        candidates_reproduced &= [candidate.token_ids for candidate in cached_candidates] == [
            candidate.token_ids for candidate in online_candidates
        ]
        selection = select_diffusion_candidates_with_replay(
            target_backend=backend,
            behavior_backend=proposal,
            prompt=prompt,
            generated_prefix=generated,
            candidates=online_candidates,
            histories=histories,
            rollout_length=rollout_length,
            fresh_count=int(replay["fresh_rollouts"]),
            target_sampling=exact_sampling,
            behavior_sampling=exact_sampling,
            reward_batch=reward_batch,
            reward_temperature=float(conditional["reward_temperature"]),
            truncation=float(replay["truncation"]),
            seed=seeds.derive("dllm-replay-online-select", stage_index),
        )
        online_seconds += time.perf_counter() - started
        online_base_deltas.append(
            llada_snapshot_delta(online_base_before, backend.snapshot())
        )
        online_proposal_deltas.append(
            llada_snapshot_delta(online_proposal_before, proposal.snapshot())
        )
        selections.append(selection)
        generated += selection.selected.sample.token_ids

    def sum_field(deltas: Sequence[dict[str, Any]], field: str) -> float:
        return sum(float(delta[field]) for delta in deltas)

    history_used = sum(
        candidate.estimate.history_count
        for selection in selections
        for candidate in selection.candidates
    )
    fresh_used = sum(
        candidate.estimate.fresh_count
        for selection in selections
        for candidate in selection.candidates
    )
    return generated, {
        "decision_stages": len(selections),
        "cache_build_seconds": build_seconds,
        "online_seconds": online_seconds,
        "end_to_end_seconds": build_seconds + online_seconds,
        "history_generated": history_generated,
        "history_used": history_used,
        "fresh_used": fresh_used,
        "rollout_reuse_rate": history_used / (history_used + fresh_used),
        "candidates_reproduced": candidates_reproduced,
        "cache_build_main_flops": sum_field(
            build_base_deltas, "estimated_active_flops"
        ),
        "cache_build_proposal_flops": sum_field(
            build_proposal_deltas, "estimated_active_flops"
        ),
        "online_main_flops": sum_field(online_base_deltas, "estimated_active_flops"),
        "online_proposal_flops": sum_field(
            online_proposal_deltas, "estimated_active_flops"
        ),
        "cache_build_main_token_slots": sum_field(
            build_base_deltas, "model_token_slots"
        ),
        "cache_build_proposal_token_slots": sum_field(
            build_proposal_deltas, "model_token_slots"
        ),
        "online_main_token_slots": sum_field(online_base_deltas, "model_token_slots"),
        "online_proposal_token_slots": sum_field(
            online_proposal_deltas, "model_token_slots"
        ),
    }


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty replay benchmark")
    count = len(records)
    fresh_correct = sum(record["fresh_only"]["correct"] for record in records)
    warm_correct = sum(record["warm_replay"]["correct"] for record in records)

    def total(arm: str, field: str) -> float:
        return sum(float(record[arm][field]) for record in records)

    fresh_seconds = total("fresh_only", "seconds")
    warm_online_seconds = total("warm_replay", "online_seconds")
    warm_end_seconds = total("warm_replay", "end_to_end_seconds")
    fresh_flops = total("fresh_only", "main_flops")
    warm_online_flops = total("warm_replay", "online_main_flops") + total(
        "warm_replay", "online_proposal_flops"
    )
    warm_end_flops = warm_online_flops + total(
        "warm_replay", "cache_build_main_flops"
    ) + total("warm_replay", "cache_build_proposal_flops")
    history = total("warm_replay", "history_used")
    fresh = total("warm_replay", "fresh_used")
    return {
        "benchmark": "GSM8K verifier-assisted LLaDA rollout replay",
        "examples": count,
        "fresh_only": {
            "correct": fresh_correct,
            "accuracy": fresh_correct / count,
            "accuracy_wilson_95": wilson_interval(fresh_correct, count),
            "seconds": fresh_seconds,
            "estimated_active_flops": fresh_flops,
        },
        "warm_replay": {
            "correct": warm_correct,
            "accuracy": warm_correct / count,
            "accuracy_wilson_95": wilson_interval(warm_correct, count),
            "online_seconds": warm_online_seconds,
            "end_to_end_seconds": warm_end_seconds,
            "online_estimated_active_flops": warm_online_flops,
            "end_to_end_estimated_active_flops": warm_end_flops,
            "rollout_reuse_rate": history / (history + fresh),
        },
        "comparisons": {
            "online_wall_clock_speedup_vs_fresh_only": (
                fresh_seconds / warm_online_seconds
            ),
            "end_to_end_wall_clock_speedup_vs_fresh_only": (
                fresh_seconds / warm_end_seconds
            ),
            "online_compute_reduction_vs_fresh_only": 1 - warm_online_flops / fresh_flops,
            "end_to_end_compute_reduction_vs_fresh_only": 1 - warm_end_flops / fresh_flops,
        },
        "reward_scope": verifier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dllm/gsm8k")
    )
    parser.add_argument("--tag", default="paired")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verifier-config", type=Path)
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    config = apply_execution_profile(config, args.profile)
    replace_verifier_from_file(config, args.verifier_config)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        config["run"]["sample_count"] = args.limit
    actual_hashes = validate_llada_weights(config)
    problems = select_problems(
        load_gsm8k(args.data),
        int(config["run"]["sample_count"]),
        seed=int(config["run"]["subset_seed"]),
    )
    effective = {
        "config": config,
        "tag": args.tag,
        "execution_profile": args.profile,
        "model_weight_sha256": actual_hashes,
        "model_metadata_sha256": checkpoint_metadata_hashes(
            Path(str(config["model"]["path"]))
        ),
        "problem_indices": [problem.index for problem in problems],
        "implementation_sha256": implementation_hashes(
            REPOSITORY_ROOT,
            entrypoints=IMPLEMENTATION_FILES,
        ),
    }
    fingerprint = json_fingerprint(effective)
    run_dir = args.output_root / args.tag / "replay"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = {"fingerprint": fingerprint, **effective}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError(f"existing replay run has another fingerprint: {run_dir}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    records = _load_records(records_path)
    completed = {int(record["problem_index"]) for record in records}
    backend = load_llada_backend(config, "base")
    proposal = load_llada_backend(config, "proposal", base_backend=backend)
    seeds = SeedStream(int(config["run"]["seed"]))
    try:
        with records_path.open("a", encoding="utf-8", buffering=1) as output:
            for problem in problems:
                if problem.index in completed:
                    continue
                prompt = backend.encode_chat(gsm8k_prompt(problem.question))
                problem_seed = seeds.derive("dllm-replay-benchmark", problem.index)
                reward_batch = _configured_verifier_reward(
                    backend, problem, config
                ).batch

                fresh_before = backend.snapshot()
                started = time.perf_counter()
                fresh_tokens, fresh_info = _run_fresh(
                    backend, prompt, reward_batch, config, problem_seed
                )
                fresh_seconds = time.perf_counter() - started
                fresh_delta = llada_snapshot_delta(fresh_before, backend.snapshot())

                warm_started = time.perf_counter()
                warm_tokens, warm_info = _run_warm(
                    backend, proposal, prompt, reward_batch, config, problem_seed
                )
                measured_warm_seconds = time.perf_counter() - warm_started
                fresh_prediction = extract_numeric_answer(backend.decode(fresh_tokens))
                warm_prediction = extract_numeric_answer(backend.decode(warm_tokens))
                record = {
                    "fingerprint": fingerprint,
                    "problem_index": problem.index,
                    "fresh_only": {
                        "prediction": (
                            str(fresh_prediction) if fresh_prediction is not None else None
                        ),
                        "correct": fresh_prediction == problem.gold_answer,
                        "seconds": fresh_seconds,
                        "main_flops": fresh_delta["estimated_active_flops"],
                        "main_token_slots": fresh_delta["model_token_slots"],
                        **fresh_info,
                    },
                    "warm_replay": {
                        "prediction": (
                            str(warm_prediction) if warm_prediction is not None else None
                        ),
                        "correct": warm_prediction == problem.gold_answer,
                        "measured_end_to_end_seconds": measured_warm_seconds,
                        **warm_info,
                    },
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                completed.add(problem.index)
                print(
                    f"replay {len(completed)}/{len(problems)} index={problem.index} "
                    f"fresh={int(record['fresh_only']['correct'])} "
                    f"warm={int(record['warm_replay']['correct'])}",
                    flush=True,
                )
        summary = {
            "fingerprint": fingerprint,
            **summarize(records, verifier=config["verifier"]),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        del proposal
        del backend
        gc.collect()
        if str(config["runtime"]["device"]).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
