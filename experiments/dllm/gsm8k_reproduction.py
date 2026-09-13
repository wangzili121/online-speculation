"""Run the paired LLaDA GSM8K quality methods with resumable records."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.shared.paired_protocol import load_pairing
from experiments.shared.artifacts import load_jsonl as _load_records
from experiments.shared.config_overrides import apply_config_overrides
from experiments.shared.statistics import wilson_interval
from experiments.dllm.profiles import apply_execution_profile
from experiments.dllm.runtime import (
    adapter_hashes as _adapter_hashes,
    capped_generation_length as _capped_generation_length,
    checkpoint_metadata_hashes,
    empty_llada_compute as _zero_compute,
    implementation_hashes as _implementation_hashes,
    json_fingerprint as _fingerprint,
    llada_snapshot_delta as _snapshot_delta,
    sampling_from_section as _sampling,
    validate_llada_weights,
    validate_runtime_device,
)
from inference_scaling.dllm.algorithms import (
    run_conditional_diffusion_is,
    run_diffusion_block_beam,
    run_diffusion_reward_mh,
    run_diffusion_trajectory_power_mh,
)
from inference_scaling.dllm.backends import load_llada_backend
from inference_scaling.dllm.dynamic_is import run_dynamic_diffusion_is
from inference_scaling.dllm.config import (
    DiffusionBlockBeamConfig,
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
)
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.evaluation import (
    CumulativeConsensusReward,
    GSM8KProblem,
    consensus_index,
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    modal_answer,
    select_problems,
)
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence
from inference_scaling.shared.verifier import (
    TokenVerifierReward,
    VerifierContext,
    build_token_verifier_reward,
    replace_verifier_from_file,
    verifier_spec_from_config,
)
from experiments.shared.methods import DLLM_DYNAMIC_METHODS, DLLM_METHODS

METHODS = DLLM_METHODS
DYNAMIC_METHODS = DLLM_DYNAMIC_METHODS
IMPLEMENTATION_FILES = (
    "experiments/dllm/gsm8k_reproduction.py",
    "experiments/dllm/profiles.py",
    "experiments/dllm/runtime.py",
    "experiments/shared/config_overrides.py",
    "experiments/shared/statistics.py",
    "src/inference_scaling/dllm/algorithms/is_sampling.py",
    "src/inference_scaling/dllm/algorithms/mh.py",
    "src/inference_scaling/dllm/algorithms/search.py",
    "src/inference_scaling/dllm/backends/llada.py",
    "src/inference_scaling/dllm/backends/loader.py",
    "src/inference_scaling/dllm/config.py",
    "src/inference_scaling/dllm/dynamic_is.py",
    "src/inference_scaling/shared/budget.py",
    "src/inference_scaling/shared/evaluation/numeric.py",
    "src/inference_scaling/shared/verifier.py",
)


def _configured_verifier_reward(
    backend: Any,
    problem: GSM8KProblem,
    config: dict[str, Any],
) -> TokenVerifierReward:
    spec = verifier_spec_from_config(config)
    return build_token_verifier_reward(
        config,
        context=VerifierContext(
            prompt=gsm8k_prompt(problem.question),
            reference=(str(problem.gold_answer) if spec.requires_reference else None),
            metadata={"benchmark": "gsm8k", "problem_index": problem.index},
        ),
        decoder=backend.decode,
    )


def _sample_one(
    backend: Any,
    prompt: TokenSequence,
    *,
    generation_length: int,
    sampling: DiffusionSamplingConfig,
    seed: int,
    request_id: str,
) -> TokenSequence:
    samples = backend.sample_batch(
        [
            DiffusionGenerationRequest(
                prefix=prompt,
                generation_length=generation_length,
                sampling=sampling,
                seed=seed,
                request_id=request_id,
            )
        ]
    )
    if len(samples) != 1:
        raise RuntimeError("backend returned an invalid number of LLaDA samples")
    return samples[0].token_ids


def _run_best_of_n(
    backend: Any,
    prompt: TokenSequence,
    *,
    generation_length: int,
    sampling: DiffusionSamplingConfig,
    samples: int,
    seeds: SeedStream,
    problem_index: int,
) -> tuple[TokenSequence, dict[str, Any]]:
    requests = [
        DiffusionGenerationRequest(
            prefix=prompt,
            generation_length=generation_length,
            sampling=sampling,
            seed=seeds.derive("dllm-best-of-n", problem_index, draw),
            request_id=f"dllm-best-of-n:{problem_index}:{draw}",
        )
        for draw in range(samples)
    ]
    candidates = backend.sample_batch(requests)
    if len(candidates) != samples:
        raise RuntimeError("backend returned an invalid number of Best-of-N samples")
    texts = [backend.decode(candidate.token_ids) for candidate in candidates]
    answers = [extract_numeric_answer(text) for text in texts]
    tie_break_scores = [
        float(candidate.trajectory_logprob)
        if candidate.trajectory_logprob is not None
        else 0.0
        for candidate in candidates
    ]
    selected_index = consensus_index(texts, tie_break_scores)
    mode = modal_answer(answers)
    return candidates[selected_index].token_ids, {
        "candidate_count": samples,
        "selected_index": selected_index,
        "selector": "modal_numeric_answer_then_trajectory_score_then_draw_order",
        "trajectory_score_available": all(
            candidate.trajectory_logprob is not None for candidate in candidates
        ),
        "modal_answer": str(mode) if mode is not None else None,
    }


def _conditional_diagnostics(result: Any) -> dict[str, Any]:
    rollout_weights: list[list[float]] = []
    rewards: list[float] = []
    raw_corrections: list[float] = []
    applied_corrections: list[float] = []
    for step in result.steps:
        for candidate in step.candidates:
            rollout_weights.append([rollout.log_weight for rollout in candidate.rollouts])
            for rollout in candidate.rollouts:
                rewards.append(float(rollout.reward))
                if rollout.raw_log_importance_ratio is not None:
                    raw_corrections.append(float(rollout.raw_log_importance_ratio))
                    applied_corrections.append(
                        float(rollout.applied_log_importance_ratio)
                    )
    return {
        "decision_stages": len(result.steps),
        "rollout_evaluations": sum(len(values) for values in rollout_weights),
        "mean_rollout_ess": (
            statistics.fmean(
                importance_effective_sample_size(values) for values in rollout_weights
            )
            if rollout_weights
            else 0.0
        ),
        "mean_rollout_reward": statistics.fmean(rewards) if rewards else 0.0,
        "importance_corrected_rollouts": len(raw_corrections),
        "mean_absolute_raw_log_importance_correction": (
            statistics.fmean(abs(value) for value in raw_corrections)
            if raw_corrections
            else 0.0
        ),
        "mean_absolute_applied_log_importance_correction": (
            statistics.fmean(abs(value) for value in applied_corrections)
            if applied_corrections
            else 0.0
        ),
        "clipped_rollout_corrections": sum(
            raw != applied
            for raw, applied in zip(
                raw_corrections, applied_corrections, strict=True
            )
        ),
    }


def run_method(
    method: str,
    backend: Any,
    problem: GSM8KProblem,
    prompt: TokenSequence,
    config: dict[str, Any],
    *,
    seed: int,
    proposal_backend: Any | None = None,
) -> tuple[TokenSequence, dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"unknown LLaDA method {method!r}")
    generation_budget = int(config["generation"]["max_new_tokens"])
    generation_sampling = _sampling(config["generation"])
    exact_sampling = _sampling(config["exact_policy"])
    generation_length = _capped_generation_length(
        prompt_length=len(prompt),
        maximum=generation_budget,
        sampling=generation_sampling,
    )
    seeds = SeedStream(seed)

    if method in {"base", "vrpo_sample", "vrpo_greedy"}:
        sampling = generation_sampling
        if method == "vrpo_greedy":
            sampling = replace(sampling, temperature=0.0)
        tokens = _sample_one(
            backend,
            prompt,
            generation_length=generation_length,
            sampling=sampling,
            seed=seeds.derive(method, problem.index),
            request_id=f"{method}:{problem.index}",
        )
        return tokens, {"sampling_policy": sampling.policy_id}

    if method == "block_beam":
        search = config["search"]
        result = run_diffusion_block_beam(
            backend=backend,
            prompt=prompt,
            config=DiffusionBlockBeamConfig(
                total_length=generation_length,
                decision_block_size=int(search["decision_block_size"]),
                width=int(search["width"]),
                branching_factor=int(search["branching_factor"]),
            ),
            sampling=exact_sampling,
            seed=seeds.derive(method, problem.index),
        )
        return result.best.token_ids, {
            "width": int(search["width"]),
            "branching_factor": int(search["branching_factor"]),
            "decision_stages": len(result.stages),
            "sampled_proposals": sum(stage.proposals for stage in result.stages),
            "selected_trajectory_logprob": result.best.trajectory_logprob,
            "search_kind": "sampled_reverse_trajectory_block_beam",
        }

    if method == "best_of_n":
        return _run_best_of_n(
            backend,
            prompt,
            generation_length=generation_length,
            sampling=generation_sampling,
            samples=int(config["best_of_n"]["samples"]),
            seeds=seeds,
            problem_index=problem.index,
        )

    if method in DYNAMIC_METHODS:
        if proposal_backend is None:
            raise ValueError("dynamic dLLM IS requires the shared early-exit backend")
        conditional = config["conditional_is"]
        replay = config["replay"]
        dynamic = config.get("dynamic_is", {})
        importance_clip = conditional.get("importance_log_ratio_clip")
        result = run_dynamic_diffusion_is(
            arm=method,
            target_backend=backend,
            auxiliary_backend=proposal_backend,
            prompt=prompt,
            config=DiffusionISConfig(
                candidate_count=int(conditional["candidate_count"]),
                rollout_count=int(conditional["rollout_count"]),
                block_size=int(conditional["decision_block_size"]),
                total_length=generation_length,
                reward_temperature=float(conditional["reward_temperature"]),
                importance_log_ratio_clip=(
                    float(importance_clip) if importance_clip is not None else None
                ),
            ),
            sampling=exact_sampling,
            reward_batch=CumulativeConsensusReward(backend.decode),
            history_rollouts=int(replay["history_rollouts"]),
            fresh_rollouts=int(replay["fresh_rollouts"]),
            truncation=float(replay["truncation"]),
            auxiliary_probability=float(dynamic.get("auxiliary_probability", 0.5)),
            history_cost=float(dynamic.get("history_cost", 0.05)),
            fresh_cost=float(dynamic.get("fresh_cost", 1.0)),
            design_rollouts=int(dynamic.get("design_rollouts", 2)),
            seed=seeds.derive(method, problem.index),
        )
        outer_log_ratios = [
            draw.outer_log_ratio for step in result.steps for draw in step.draws
        ]
        history_used = sum(
            candidate.estimate.history_count
            for step in result.steps
            for candidate in step.selection.candidates
        )
        fresh_used = sum(
            candidate.estimate.fresh_count
            for step in result.steps
            for candidate in step.selection.candidates
        )
        return result.token_ids, {
            "target": "exact_base_reverse_trajectory_times_conditional_rollout_weight",
            "dynamic_arm": method,
            "candidate_proposal": (
                "exact_base_reverse_trajectory"
                if method == "base_candidate_fixed"
                else "defensive_base_early_exit_mixture"
            ),
            "outer_importance_correction": method != "base_candidate_fixed",
            "outer_importance_ess": importance_effective_sample_size(
                outer_log_ratios
            ),
            "decision_stages": len(result.steps),
            "design_rollouts": sum(step.design_rollouts for step in result.steps),
            "evaluation_history_generated": sum(
                step.evaluation_history_rollouts for step in result.steps
            ),
            "history_used": history_used,
            "fresh_used": fresh_used,
            "rollout_reuse_rate": (
                history_used / (history_used + fresh_used)
                if history_used + fresh_used
                else 0.0
            ),
            "allocations": [
                [asdict(allocation) for allocation in step.allocations]
                for step in result.steps
            ],
            "candidate_sources": [
                [draw.source for draw in step.draws] for step in result.steps
            ],
        }

    if method == "trajectory_power_mh":
        mh = config["mh"]
        result = run_diffusion_trajectory_power_mh(
            backend=backend,
            prompt=prompt,
            config=DiffusionPowerMHConfig(
                total_length=generation_length,
                decision_block_size=int(mh["decision_block_size"]),
                updates_per_stage=int(mh["updates_per_stage"]),
                alpha=float(mh["alpha"]),
            ),
            sampling=exact_sampling,
            seed=seeds.derive(method, problem.index),
        )
        return result.final.token_ids, {
            "target": "exact_reverse_trajectory_probability_to_alpha",
            "alpha": result.alpha,
            "decision_block_size": int(mh["decision_block_size"]),
            "updates_per_stage": int(mh["updates_per_stage"]),
            "updates": len(result.steps),
            "accepted": sum(step.accepted for step in result.steps),
            "acceptance_rate": result.acceptance_rate,
        }

    if method == "verifier_mh":
        mh = config["mh"]
        verifier_reward = _configured_verifier_reward(backend, problem, config)
        result = run_diffusion_reward_mh(
            backend=backend,
            prompt=prompt,
            config=DiffusionMHConfig(
                total_length=generation_length,
                updates=int(mh["updates"]),
                reward_temperature=float(mh["reward_temperature"]),
            ),
            sampling=generation_sampling,
            reward=verifier_reward,
            seed=seeds.derive(method, problem.index),
        )
        return result.final.token_ids, {
            "target": "base_reverse_process_times_exp_configured_verifier_over_temperature",
            "reward_source": "verifier",
            "verifier": verifier_reward.describe(),
            "updates": len(result.steps),
            "accepted": sum(step.accepted for step in result.steps),
            "acceptance_rate": result.acceptance_rate,
            "final_reward": result.final_reward,
            "uses_test_gold_oracle": verifier_reward.verifier.spec.requires_reference,
        }

    conditional = config["conditional_is"]
    reduced = "reduced_layer_proposal" in method
    if reduced and proposal_backend is None:
        raise ValueError("reduced-layer conditional IS requires its proposal backend")
    verifier = method.startswith("verifier_")
    verifier_reward = (
        _configured_verifier_reward(backend, problem, config) if verifier else None
    )
    uncorrected = method.endswith("_uncorrected")
    unclipped = method.endswith("_unclipped")
    rollout_backend = proposal_backend if reduced else backend
    assert rollout_backend is not None
    apply_correction = reduced and not uncorrected
    configured_clip = conditional.get("importance_log_ratio_clip")
    clip = (
        None
        if not apply_correction or unclipped or configured_clip is None
        else float(configured_clip)
    )
    reward_batch = None if verifier else CumulativeConsensusReward(backend.decode)
    result = run_conditional_diffusion_is(
        base_backend=backend,
        prompt=prompt,
        config=DiffusionISConfig(
            candidate_count=int(conditional["candidate_count"]),
            rollout_count=int(conditional["rollout_count"]),
            block_size=int(conditional["decision_block_size"]),
            total_length=generation_length,
            reward_temperature=(
                float(config["mh"]["reward_temperature"])
                if verifier
                else float(conditional["reward_temperature"])
            ),
            importance_log_ratio_clip=clip,
        ),
        base_sampling=generation_sampling,
        rollout_backend=rollout_backend,
        rollout_sampling=exact_sampling,
        target_rollout_backend=backend,
        target_rollout_sampling=exact_sampling,
        apply_importance_correction=apply_correction,
        reward=verifier_reward,
        reward_batch=reward_batch,
        seed=seeds.derive(method, problem.index),
    )
    diagnostics = _conditional_diagnostics(result)
    diagnostics.update(
        {
            "candidate_source": "full_llada_moe",
            "rollout_source": (
                "shared_prefix_layer_llada" if reduced else "full_llada_moe"
            ),
            "reward_source": "verifier" if verifier else "self_consistency",
            "verifier": verifier_reward.describe() if verifier_reward else None,
            "uses_test_gold_oracle": bool(
                verifier_reward
                and verifier_reward.verifier.spec.requires_reference
            ),
            "apply_importance_correction": apply_correction,
            "importance_log_ratio_clip": clip,
            "target": (
                "base_candidates_with_uncorrected_proposal_rollout_weight"
                if uncorrected
                else "base_candidates_with_target_reverse_trajectory_rollout_weight"
            ),
        }
    )
    return result.token_ids, diagnostics


_wilson = wilson_interval


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty LLaDA experiment")
    correct = sum(bool(record["correct"]) for record in records)

    def total(role: str, field: str) -> float:
        return sum(float(record[role][field]) for record in records)

    main_flops = total("main_compute", "estimated_active_flops")
    proposal_flops = total("proposal_compute", "estimated_active_flops")
    return {
        "method": records[0]["method"],
        "examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "accuracy_wilson_95": _wilson(correct, len(records)),
        "wall_clock_seconds": sum(float(record["elapsed_seconds"]) for record in records),
        "main_generation_forward_token_slots": total(
            "main_compute", "sample_model_token_slots"
        ),
        "main_exact_rescoring_forward_token_slots": total(
            "main_compute", "score_model_token_slots"
        ),
        "proposal_generation_forward_token_slots": total(
            "proposal_compute", "sample_model_token_slots"
        ),
        "proposal_exact_rescoring_forward_token_slots": total(
            "proposal_compute", "score_model_token_slots"
        ),
        "main_estimated_active_flops": main_flops,
        "proposal_estimated_active_flops": proposal_flops,
        "total_estimated_active_flops": main_flops + proposal_flops,
        "total_estimated_active_petaflops": (main_flops + proposal_flops) / 1e15,
        "compute_definition": (
            "2 * active model parameters * model-input token slots; generation and "
            "exact trajectory rescoring are recorded separately for the full LLaDA-MoE "
            "model and the shared early-exit proposal"
        ),
    }


def _synchronize_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def _run_draw(
    *,
    method: str,
    draw_index: int,
    tag: str,
    output_root: Path,
    profile: str,
    config: dict[str, Any],
    problems: Sequence[GSM8KProblem],
    actual_hashes: dict[str, str],
    model_metadata_hashes: dict[str, str],
    adapter_hashes: dict[str, str] | None,
    implementation_hashes: dict[str, str],
    backend: Any,
    proposal_backend: Any | None,
    device: str,
) -> None:
    effective = {
        "config": config,
        "method": method,
        "tag": tag,
        "draw_index": draw_index,
        "execution_profile": profile,
        "model_weight_sha256": actual_hashes,
        "model_metadata_sha256": model_metadata_hashes,
        "adapter_sha256": adapter_hashes,
        "problem_indices": [problem.index for problem in problems],
        "implementation_sha256": implementation_hashes,
    }
    fingerprint = _fingerprint(effective)
    run_dir = output_root / tag / method / f"draw-{draw_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = {"fingerprint": fingerprint, **effective}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError(f"existing run has a different fingerprint: {run_dir}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    prior_records = _load_records(records_path)
    completed = {int(record["problem_index"]) for record in prior_records}
    seeds = SeedStream(int(config["run"]["seed"]))
    with records_path.open("a", encoding="utf-8", buffering=1) as output:
        for problem in problems:
            if problem.index in completed:
                continue
            prompt = backend.encode_chat(gsm8k_prompt(problem.question))
            main_before = backend.snapshot()
            proposal_before = (
                proposal_backend.snapshot() if proposal_backend is not None else None
            )
            _synchronize_cuda(device)
            started = time.perf_counter()
            tokens, diagnostics = run_method(
                method,
                backend,
                problem,
                prompt,
                config,
                seed=seeds.derive(
                    "dllm-gsm8k", method, draw_index, problem.index
                ),
                proposal_backend=proposal_backend,
            )
            _synchronize_cuda(device)
            elapsed = time.perf_counter() - started
            main_compute = _snapshot_delta(main_before, backend.snapshot())
            proposal_compute = (
                _snapshot_delta(proposal_before, proposal_backend.snapshot())
                if proposal_backend is not None and proposal_before is not None
                else _zero_compute()
            )
            text = backend.decode(tokens)
            prediction = extract_numeric_answer(text)
            record = {
                "fingerprint": fingerprint,
                "method": method,
                "draw_index": draw_index,
                "problem_index": problem.index,
                "gold_answer": str(problem.gold_answer),
                "prediction": str(prediction) if prediction is not None else None,
                "correct": prediction == problem.gold_answer,
                "prompt_tokens": len(prompt),
                "selected_output_tokens": len(tokens),
                "output_token_ids": list(tokens),
                "output_text": text,
                "elapsed_seconds": elapsed,
                "main_compute": main_compute,
                "proposal_compute": proposal_compute,
                "diagnostics": diagnostics,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            prior_records.append(record)
            completed.add(problem.index)
            print(
                f"{method} draw={draw_index} {len(completed)}/{len(problems)} "
                f"index={problem.index} correct={int(record['correct'])} "
                f"seconds={elapsed:.3f}",
                flush=True,
            )
    ordered = sorted(prior_records, key=lambda record: int(record["problem_index"]))
    summary = {"fingerprint": fingerprint, **summarize(ordered)}
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dllm/gsm8k")
    )
    parser.add_argument("--tag", default="paired")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    draws = parser.add_mutually_exclusive_group()
    draws.add_argument("--draw-index", type=int)
    draws.add_argument(
        "--draws",
        type=int,
        help="run draw indices [0, draws) while keeping the model loaded",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--verifier-config",
        type=Path,
        help="standalone TOML file whose [verifier] table replaces the default",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override an existing TOML field; repeat for multiple fields",
    )
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    config = apply_execution_profile(config, args.profile)
    config = apply_config_overrides(config, args.config_overrides)
    replace_verifier_from_file(config, args.verifier_config)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        config["run"]["sample_count"] = args.limit
    if args.draws is not None:
        if args.draws <= 0:
            raise ValueError("--draws must be positive")
        draw_indices = tuple(range(args.draws))
    else:
        draw_index = 0 if args.draw_index is None else args.draw_index
        if draw_index < 0:
            raise ValueError("--draw-index must be non-negative")
        draw_indices = (draw_index,)
    device = validate_runtime_device(config)
    actual_hashes = validate_llada_weights(config)
    model_metadata_hashes = checkpoint_metadata_hashes(
        Path(str(config["model"]["path"]))
    )
    problems = select_problems(
        load_gsm8k(args.data),
        int(config["run"]["sample_count"]),
        seed=int(config["run"]["subset_seed"]),
    )
    role = "aligned" if args.method.startswith("vrpo_") else "base"
    adapter_hashes = (
        _adapter_hashes(Path(str(config["alignment"]["adapter"])))
        if role == "aligned"
        else None
    )
    backend = load_llada_backend(config, role)
    proposal_backend = (
        load_llada_backend(config, "proposal", base_backend=backend)
        if "reduced_layer_proposal" in args.method or args.method in DYNAMIC_METHODS
        else None
    )
    implementation_hashes = _implementation_hashes(
        REPOSITORY_ROOT,
        entrypoints=IMPLEMENTATION_FILES,
    )
    try:
        for draw_index in draw_indices:
            _run_draw(
                method=args.method,
                draw_index=draw_index,
                tag=args.tag,
                output_root=args.output_root,
                profile=args.profile,
                config=config,
                problems=problems,
                actual_hashes=actual_hashes,
                model_metadata_hashes=model_metadata_hashes,
                adapter_hashes=adapter_hashes,
                implementation_hashes=implementation_hashes,
                backend=backend,
                proposal_backend=proposal_backend,
                device=device,
            )
    finally:
        del proposal_backend
        del backend
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
