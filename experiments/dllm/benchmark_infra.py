"""Measure dLLM scheduling, replay, IS, MH, and SMC infrastructure arms."""

from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dllm.profiles import apply_execution_profile
from experiments.dllm.runtime import (
    capped_generation_length,
    checkpoint_metadata_hashes,
    empty_llada_compute,
    implementation_hashes,
    json_fingerprint,
    llada_snapshot_delta,
    sampling_from_section,
    validate_llada_weights,
    validate_runtime_device,
)
from experiments.shared.paired_protocol import load_pairing
from experiments.shared.artifacts import load_jsonl as _load_records
from inference_scaling.dllm.algorithms import (
    run_conditional_diffusion_is,
    run_diffusion_replay_mixture_mh,
    run_diffusion_reward_mh,
    run_diffusion_reward_mh_delayed,
    run_diffusion_smc_rollout_forest,
    run_progressive_diffusion_is,
)
from inference_scaling.dllm.backends import load_llada_backend
from inference_scaling.dllm.config import DiffusionISConfig, DiffusionMHConfig
from inference_scaling.dllm.dynamic_is import run_dynamic_diffusion_is
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.config import SMCForestConfig
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import (
    VerifierContext,
    build_token_verifier_reward,
    replace_verifier_from_file,
    verifier_spec_from_config,
)

IMPLEMENTATION_FILES = (
    "experiments/dllm/benchmark_infra.py",
    "experiments/dllm/runtime.py",
    "src/inference_scaling/dllm/algorithms/mh.py",
    "src/inference_scaling/dllm/algorithms/mh_acceleration.py",
    "src/inference_scaling/dllm/algorithms/progressive_is.py",
    "src/inference_scaling/dllm/algorithms/smc_forest.py",
    "src/inference_scaling/dllm/dynamic_is.py",
    "src/inference_scaling/dllm/replay.py",
    "src/inference_scaling/shared/budget.py",
    "src/inference_scaling/shared/mh.py",
)

ASYNC_FAMILIES = (
    "block_continuous_batching",
    "resume_committed_blocks",
    "streaming_reward",
    "llada_batched_transformers_backend",
)
ALL_FAMILIES = ASYNC_FAMILIES + (
    "history_block_trajectory_cache",
    "mh_continuation_prefetch",
    "delayed_acceptance_mh",
    "trajectory_replay_mixture_mh",
    "progressive_is",
    "block_smc_rollout_forest",
)
COMPARISON_ARMS = {
    "block_continuous_batching": ("sequential", "batched"),
    "resume_committed_blocks": ("restart_after_interruption", "resume_committed"),
    "streaming_reward": ("deferred_reward", "overlapped_reward"),
    "llada_batched_transformers_backend": ("sequential", "batched"),
    "history_block_trajectory_cache": ("fresh_only", "trajectory_replay"),
    "mh_continuation_prefetch": ("sequential_proposals", "prefetched_proposals"),
    "delayed_acceptance_mh": ("ordinary_exact", "delayed_exact"),
    "trajectory_replay_mixture_mh": ("base_proposal", "replay_mixture_online"),
    "progressive_is": ("fixed_rollouts", "progressive_rollouts"),
    "block_smc_rollout_forest": ("fresh_forest", "reused_forest"),
}


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _measure(
    backend: Any,
    proposal: Any,
    device: str,
    operation: Callable[[], tuple[Sequence[int], dict[str, Any]]],
) -> dict[str, Any]:
    main_before = backend.snapshot()
    proposal_before = proposal.snapshot() if proposal is not None else None
    _synchronize(device)
    started = time.perf_counter()
    token_ids, diagnostics = operation()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "output_token_ids": [int(token) for token in token_ids],
        "seconds": elapsed,
        "main_compute": llada_snapshot_delta(main_before, backend.snapshot()),
        "proposal_compute": (
            llada_snapshot_delta(proposal_before, proposal.snapshot())
            if proposal is not None and proposal_before is not None
            else empty_llada_compute()
        ),
        "diagnostics": diagnostics,
    }


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family in sorted({record["family"] for record in records}):
        family_records = [record for record in records if record["family"] == family]
        arm_names = tuple(family_records[0]["arms"])
        arms = {}
        for arm_name in arm_names:
            arms[arm_name] = {
                "seconds": sum(record["arms"][arm_name]["seconds"] for record in family_records),
                "main_estimated_active_flops": sum(
                    record["arms"][arm_name]["main_compute"]["estimated_active_flops"]
                    for record in family_records
                ),
                "proposal_estimated_active_flops": sum(
                    record["arms"][arm_name]["proposal_compute"]["estimated_active_flops"]
                    for record in family_records
                ),
                "main_forward_calls": sum(
                    record["arms"][arm_name]["main_compute"]["forward_calls"]
                    for record in family_records
                ),
            }
        baseline_name, optimized_name = COMPARISON_ARMS[family]
        baseline = arms[baseline_name]
        optimized = arms[optimized_name]

        def optimized_over_baseline(
            field: str,
            baseline=baseline,
            optimized=optimized,
        ) -> float | None:
            denominator = float(baseline[field])
            return float(optimized[field]) / denominator if denominator else None

        def baseline_over_optimized(
            field: str,
            baseline=baseline,
            optimized=optimized,
        ) -> float | None:
            denominator = float(optimized[field])
            return float(baseline[field]) / denominator if denominator else None

        families[family] = {
            "examples": len(family_records),
            "arms": arms,
            "comparison": {
                "baseline_arm": baseline_name,
                "optimized_arm": optimized_name,
                "optimized_over_baseline": {
                    "wall_clock_factor": optimized_over_baseline("seconds"),
                    "main_flops_factor": optimized_over_baseline(
                        "main_estimated_active_flops"
                    ),
                    "main_forward_call_factor": optimized_over_baseline(
                        "main_forward_calls"
                    ),
                },
                "baseline_over_optimized": {
                    "wall_clock_speedup": baseline_over_optimized("seconds"),
                    "main_flops_factor": baseline_over_optimized(
                        "main_estimated_active_flops"
                    ),
                    "main_forward_call_factor": baseline_over_optimized(
                        "main_forward_calls"
                    ),
                },
                "output_match_on_every_example": all(
                    record["arms"][baseline_name]["output_token_ids"]
                    == record["arms"][optimized_name]["output_token_ids"]
                    for record in family_records
                ),
            },
        }
    return {"families": families}


def _problem_groups(
    *,
    backend: Any,
    proposal: Any,
    prompt: tuple[int, ...],
    gold_answer: Any,
    config: dict[str, Any],
    device: str,
    seed: int,
    families: Sequence[str],
) -> dict[str, dict[str, Any]]:
    exact_sampling = sampling_from_section(config["exact_policy"])
    generation_sampling = sampling_from_section(config["generation"])
    total_length = capped_generation_length(
        prompt_length=len(prompt),
        maximum=int(config["generation"]["max_new_tokens"]),
        sampling=generation_sampling,
    )
    block_length = int(config["conditional_is"]["decision_block_size"])
    candidate_count = int(config["conditional_is"]["candidate_count"])
    rollout_count = int(config["conditional_is"]["rollout_count"])
    seeds = SeedStream(seed)

    verifier_spec = verifier_spec_from_config(config)
    verifier_reward = build_token_verifier_reward(
        config,
        context=VerifierContext(
            prompt=backend.decode(prompt),
            reference=(str(gold_answer) if verifier_spec.requires_reference else None),
        ),
        decoder=backend.decode,
    )
    exact_reward_batch = verifier_reward.batch

    def surrogate_reward_batch(_prompt, continuations):
        return [
            float(extract_numeric_answer(backend.decode(tokens)) is not None)
            for tokens in continuations
        ]

    results: dict[str, dict[str, Any]] = {}

    if "block_continuous_batching" in families or "llada_batched_transformers_backend" in families:
        requests = [
            DiffusionGenerationRequest(
                prefix=prompt,
                generation_length=block_length,
                sampling=exact_sampling,
                seed=seeds.derive("infra", "batch", index),
                request_id=f"infra-batch:{index}",
            )
            for index in range(4)
        ]

        def sequential_batch():
            samples = [backend.sample_batch((request,))[0] for request in requests]
            return tuple(token for sample in samples for token in sample.token_ids), {
                "requests": len(requests),
                "submission": "one request per backend call",
            }

        def grouped_batch():
            samples = backend.sample_batch(requests)
            return tuple(token for sample in samples for token in sample.token_ids), {
                "requests": len(requests),
                "submission": "one grouped backend call",
            }

        arms = {
            "sequential": _measure(backend, None, device, sequential_batch),
            "batched": _measure(backend, None, device, grouped_batch),
        }
        if "block_continuous_batching" in families:
            results["block_continuous_batching"] = arms
        if "llada_batched_transformers_backend" in families:
            results["llada_batched_transformers_backend"] = arms

    if "resume_committed_blocks" in families:
        block_count = total_length // block_length

        def generate_blocks(resume: bool):
            first_request = DiffusionGenerationRequest(
                prompt,
                block_length,
                exact_sampling,
                seeds.derive("infra", "resume", 0),
                "infra-resume:0",
            )
            first = backend.sample_batch((first_request,))[0].token_ids
            generated = first if resume else ()
            start = 1 if resume else 0
            for block_index in range(start, block_count):
                request = DiffusionGenerationRequest(
                    prompt + generated,
                    block_length,
                    exact_sampling,
                    seeds.derive("infra", "resume", block_index),
                    f"infra-resume:{block_index}",
                )
                generated += backend.sample_batch((request,))[0].token_ids
            return generated, {
                "committed_blocks_before_interruption": 1,
                "recovery": "continue" if resume else "restart",
            }

        results["resume_committed_blocks"] = {
            "restart_after_interruption": _measure(
                backend, None, device, lambda: generate_blocks(False)
            ),
            "resume_committed": _measure(
                backend, None, device, lambda: generate_blocks(True)
            ),
        }

    if "streaming_reward" in families:
        requests = [
            DiffusionGenerationRequest(
                prompt,
                block_length,
                exact_sampling,
                seeds.derive("infra", "stream", index),
                f"infra-stream:{index}",
            )
            for index in range(4)
        ]
        chunks = tuple(requests[index : index + 2] for index in range(0, 4, 2))

        def deferred_reward():
            samples = [sample for chunk in chunks for sample in backend.sample_batch(chunk)]
            rewards = exact_reward_batch(prompt, [sample.token_ids for sample in samples])
            return tuple(token for sample in samples for token in sample.token_ids), {
                "rewards": rewards,
                "reward_schedule": "after_all_generation",
            }

        def overlapped_reward():
            samples = []
            futures = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                for chunk in chunks:
                    chunk_samples = backend.sample_batch(chunk)
                    samples.extend(chunk_samples)
                    futures.append(
                        executor.submit(
                            exact_reward_batch,
                            prompt,
                            [sample.token_ids for sample in chunk_samples],
                        )
                    )
                rewards = [value for future in futures for value in future.result()]
            return tuple(token for sample in samples for token in sample.token_ids), {
                "rewards": rewards,
                "reward_schedule": "overlap_cpu_reward_with_next_model_chunk",
            }

        results["streaming_reward"] = {
            "deferred_reward": _measure(backend, None, device, deferred_reward),
            "overlapped_reward": _measure(backend, None, device, overlapped_reward),
        }

    configured_clip = config["conditional_is"].get("importance_log_ratio_clip")
    is_config = DiffusionISConfig(
        candidate_count=candidate_count,
        rollout_count=rollout_count,
        block_size=block_length,
        total_length=total_length,
        reward_temperature=float(config["conditional_is"]["reward_temperature"]),
        importance_log_ratio_clip=(
            float(configured_clip) if configured_clip is not None else None
        ),
    )
    if "history_block_trajectory_cache" in families:
        replay = config["replay"]

        def dynamic_arm(arm):
            result = run_dynamic_diffusion_is(
                arm=arm,
                target_backend=backend,
                auxiliary_backend=proposal,
                prompt=prompt,
                config=is_config,
                sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                history_rollouts=int(replay["history_rollouts"]),
                fresh_rollouts=int(replay["fresh_rollouts"]),
                truncation=float(replay["truncation"]),
                design_rollouts=int(config["dynamic_is"]["design_rollouts"]),
                seed=seeds.derive("infra", "history", arm),
            )
            return result.token_ids, {
                "history_used": sum(
                    candidate.estimate.history_count
                    for step in result.steps
                    for candidate in step.selection.candidates
                ),
                "fresh_used": sum(
                    candidate.estimate.fresh_count
                    for step in result.steps
                    for candidate in step.selection.candidates
                ),
            }

        results["history_block_trajectory_cache"] = {
            "fresh_only": _measure(
                backend,
                proposal,
                device,
                lambda: dynamic_arm("base_candidate_fixed"),
            ),
            "trajectory_replay": _measure(
                backend,
                proposal,
                device,
                lambda: dynamic_arm("trajectory_replay_aware_fixed"),
            ),
        }

    mh_config = DiffusionMHConfig(
        total_length=total_length,
        updates=int(config["mh"]["updates"]),
        reward_temperature=float(config["mh"]["reward_temperature"]),
    )
    if "mh_continuation_prefetch" in families:
        def mh_arm(batch_size):
            result = run_diffusion_reward_mh(
                backend=backend,
                prompt=prompt,
                config=mh_config,
                sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                proposal_batch_size=batch_size,
                seed=seeds.derive("infra", "mh-prefetch"),
            )
            return result.final.token_ids, {"acceptance_rate": result.acceptance_rate}

        results["mh_continuation_prefetch"] = {
            "sequential_proposals": _measure(
                backend, None, device, lambda: mh_arm(1)
            ),
            "prefetched_proposals": _measure(
                backend, None, device, lambda: mh_arm(None)
            ),
        }

    if "delayed_acceptance_mh" in families:
        def ordinary_exact():
            result = run_diffusion_reward_mh(
                backend=backend,
                prompt=prompt,
                config=mh_config,
                sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                seed=seeds.derive("infra", "mh-delayed"),
            )
            return result.final.token_ids, {
                "exact_reward_evaluations": mh_config.updates + 1,
                "acceptance_rate": result.acceptance_rate,
            }

        def delayed_exact():
            result = run_diffusion_reward_mh_delayed(
                backend=backend,
                prompt=prompt,
                config=mh_config,
                sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                surrogate_reward_batch=surrogate_reward_batch,
                seed=seeds.derive("infra", "mh-delayed"),
            )
            return result.final.token_ids, {
                "exact_reward_evaluations": result.exact_reward_evaluations,
                "surrogate_reward_evaluations": result.surrogate_reward_evaluations,
                "acceptance_rate": result.acceptance_rate,
            }

        results["delayed_acceptance_mh"] = {
            "ordinary_exact": _measure(backend, None, device, ordinary_exact),
            "delayed_exact": _measure(backend, None, device, delayed_exact),
        }

    if "trajectory_replay_mixture_mh" in families:
        history_requests = [
            DiffusionGenerationRequest(
                prompt,
                total_length,
                exact_sampling,
                seeds.derive("infra", "mh-history", index),
                f"infra-mh-history:{index}",
            )
            for index in range(max(2, mh_config.updates + 1))
        ]
        history_holder = {}

        def build_history():
            history = tuple(backend.sample_batch(history_requests))
            history_holder["samples"] = history
            return tuple(token for sample in history for token in sample.token_ids), {
                "cached_trajectories": len(history)
            }

        cache_build = _measure(backend, None, device, build_history)

        def base_proposal():
            result = run_diffusion_reward_mh(
                backend=backend,
                prompt=prompt,
                config=mh_config,
                sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                seed=seeds.derive("infra", "mh-mixture"),
            )
            return result.final.token_ids, {"acceptance_rate": result.acceptance_rate}

        def replay_mixture():
            result = run_diffusion_replay_mixture_mh(
                backend=backend,
                prompt=prompt,
                config=mh_config,
                sampling=exact_sampling,
                history=history_holder["samples"],
                history_probability=0.5,
                reward_batch=exact_reward_batch,
                seed=seeds.derive("infra", "mh-mixture"),
            )
            return result.final.token_ids, {
                "acceptance_rate": result.acceptance_rate,
                "base_draws": result.base_draws,
                "history_draws": result.history_draws,
                "cache_build": cache_build,
            }

        results["trajectory_replay_mixture_mh"] = {
            "base_proposal": _measure(backend, None, device, base_proposal),
            "replay_mixture_online": _measure(
                backend, None, device, replay_mixture
            ),
        }

    if "progressive_is" in families:
        def fixed_rollouts():
            result = run_conditional_diffusion_is(
                base_backend=backend,
                prompt=prompt,
                config=is_config,
                base_sampling=exact_sampling,
                rollout_sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                seed=seeds.derive("infra", "progressive-fixed"),
            )
            return result.token_ids, {
                "evaluation_rollouts": sum(
                    len(candidate.rollouts)
                    for step in result.steps
                    for candidate in step.candidates
                )
            }

        def progressive_rollouts():
            result = run_progressive_diffusion_is(
                backend=backend,
                prompt=prompt,
                config=is_config,
                sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                pilot_rollouts_per_candidate=2,
                evaluation_rollout_budget=candidate_count * rollout_count,
                seed=seeds.derive("infra", "progressive-adaptive"),
            )
            return result.token_ids, {
                "pilot_rollouts": sum(step.pilot_rollouts for step in result.steps),
                "evaluation_rollouts": sum(
                    allocation.fresh_count
                    for step in result.steps
                    for allocation in step.allocations
                ),
            }

        results["progressive_is"] = {
            "fixed_rollouts": _measure(backend, None, device, fixed_rollouts),
            "progressive_rollouts": _measure(
                backend, None, device, progressive_rollouts
            ),
        }

    if "block_smc_rollout_forest" in families:
        smc_common = {
            "particle_count": max(2, min(3, candidate_count)),
            "branch_factor": 2,
            "rollout_count": max(1, min(2, rollout_count)),
            "block_size": block_length,
            "total_length": total_length,
            "reward_temperature": is_config.reward_temperature,
        }

        def smc_arm(reuse):
            result = run_diffusion_smc_rollout_forest(
                backend=backend,
                prompt=prompt,
                config=SMCForestConfig(
                    **smc_common, reuse_rollout_forest=reuse
                ),
                sampling=exact_sampling,
                reward_batch=exact_reward_batch,
                seed=seeds.derive("infra", "smc"),
            )
            return result.token_ids, {
                "reused_rollouts": result.reused_rollouts,
                "fresh_rollouts": result.fresh_rollouts,
            }

        results["block_smc_rollout_forest"] = {
            "fresh_forest": _measure(
                backend, None, device, lambda: smc_arm(False)
            ),
            "reused_forest": _measure(
                backend, None, device, lambda: smc_arm(True)
            ),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--section", choices=("async", "all"), default="all")
    parser.add_argument("--tag", default="paired")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dllm/gsm8k")
    )
    parser.add_argument("--verifier-config", type=Path)
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    config = apply_execution_profile(config, args.profile)
    replace_verifier_from_file(config, args.verifier_config)
    limit = args.limit or 1
    if limit <= 0:
        raise ValueError("--limit must be positive")
    config["run"]["sample_count"] = limit
    device = validate_runtime_device(config)
    weight_hashes = validate_llada_weights(config)
    problems = select_problems(
        load_gsm8k(args.data),
        limit,
        seed=int(config["run"]["subset_seed"]),
    )
    families = ASYNC_FAMILIES if args.section == "async" else ALL_FAMILIES
    effective = {
        "config": config,
        "section": args.section,
        "tag": args.tag,
        "families": families,
        "problem_indices": [problem.index for problem in problems],
        "model_weight_sha256": weight_hashes,
        "model_metadata_sha256": checkpoint_metadata_hashes(
            Path(str(config["model"]["path"]))
        ),
        "implementation_sha256": implementation_hashes(
            REPOSITORY_ROOT,
            entrypoints=IMPLEMENTATION_FILES,
        ),
    }
    fingerprint = json_fingerprint(effective)
    run_dir = args.output_root / args.tag / "components" / (
        "async" if args.section == "async" else "infra"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = {"fingerprint": fingerprint, **effective}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError(f"existing infra run has another fingerprint: {run_dir}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    records = _load_records(records_path)
    completed = {
        (int(record["problem_index"]), str(record["family"])) for record in records
    }
    backend = load_llada_backend(config, "base")
    proposal = (
        None
        if args.section == "async"
        else load_llada_backend(config, "proposal", base_backend=backend)
    )
    try:
        with records_path.open("a", encoding="utf-8", buffering=1) as output:
            for problem in problems:
                pending = [
                    family
                    for family in families
                    if (problem.index, family) not in completed
                ]
                if not pending:
                    continue
                prompt = backend.encode_chat(gsm8k_prompt(problem.question))
                groups = _problem_groups(
                    backend=backend,
                    proposal=proposal,
                    prompt=prompt,
                    gold_answer=problem.gold_answer,
                    config=config,
                    device=device,
                    seed=SeedStream(int(config["run"]["seed"])).derive(
                        "dllm-infra", problem.index
                    ),
                    families=pending,
                )
                for family, arms in groups.items():
                    record = {
                        "fingerprint": fingerprint,
                        "problem_index": problem.index,
                        "family": family,
                        "arms": arms,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    completed.add((problem.index, family))
                    print(f"infra index={problem.index} family={family}", flush=True)
        report = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "section": args.section,
            "measurement_scope": (
                "wall clock, full-model and early-exit active FLOPs, forward calls, "
                "reuse counts, exact reward calls, and deterministic path checks"
            ),
            "reward_scope": config["verifier"],
            **_aggregate(records),
        }
        summary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if proposal is not None:
            del proposal
        del backend
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
