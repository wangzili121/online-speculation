"""Run the GSM8K comparison with resumable per-example records."""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
import tomllib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import transformers

from inference_scaling.arllm.algorithms import (
    run_conditional_is,
    run_mh_chain,
    run_reward_mh_chain,
)
from inference_scaling.experimental.arllm.iterated_is import (
    run_iterated_conditional_is,
)
from inference_scaling.arllm.backends import (
    BACKEND_CHOICES,
    AbsorbingEOSBackend,
    ScoreCachingBackend,
    close_backend,
    configured_backend,
    load_backend_from_config,
    set_backend_override,
)
from inference_scaling.arllm.config import (
    ConditionalISConfig,
    IteratedConditionalISConfig,
    MHConfig,
    RewardMHConfig,
    SamplingConfig,
)
from inference_scaling.arllm.rewards import (
    ConsilienceReward,
)
from inference_scaling.arllm.reward_factory import (
    MODEL_REWARD_SOURCES, model_reward_from_config, reward_temperature_from_config,
)
from inference_scaling.arllm.backends.reference import ReferencePolicyBackend
from inference_scaling.arllm.scope import SamplingScope
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
from inference_scaling.shared.verifier import (
    TokenVerifierReward,
    VerifierContext,
    build_token_verifier_reward,
    replace_verifier_from_file,
    verifier_spec_from_config,
)
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, TokenSequence

from experiments.shared.artifacts import (
    dataclass_snapshot_delta as _snapshot_delta,
    file_sha256,
    implementation_hashes as _implementation_hashes,
    json_fingerprint as _fingerprint,
    load_jsonl as _load_records,
)
from experiments.shared.statistics import wilson_interval

_file_sha256 = file_sha256

from experiments.arllm.runtime import model_metadata, set_rl_adapter_override, validate_model_artifacts
from inference_scaling.shared.generation import generation_config_for_prompt
from inference_scaling.shared.prompting import render_prompt
from experiments.shared.methods import AR_METHODS

METHODS = AR_METHODS
REWARD_SOURCES = (
    "self_consistency",
    "frozen_consensus",
    "log_probability",
    "sequence_log_probability",
    "negative_entropy",
    "self_certainty",
    "consilience",
    "verifier",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_FILES = (
    "experiments/arllm/gsm8k_reproduction.py",
    "src/inference_scaling/arllm/algorithms/conditional_is.py",
    "src/inference_scaling/experimental/arllm/iterated_is.py",
    "src/inference_scaling/arllm/algorithms/mh.py",
    "src/inference_scaling/arllm/backends/absorbing.py",
    "src/inference_scaling/arllm/backends/cache.py",
    "src/inference_scaling/arllm/backends/transformers_backend.py",
    "src/inference_scaling/arllm/backends/vllm_backend.py",
    "src/inference_scaling/arllm/backends/loader.py",
    "src/inference_scaling/shared/evaluation/consensus.py",
    "src/inference_scaling/shared/evaluation/gsm8k.py",
    "src/inference_scaling/shared/evaluation/numeric.py",
    "src/inference_scaling/shared/verifier.py",
    "src/inference_scaling/experimental/shared/iterated_sir.py",
    "src/inference_scaling/experimental/shared/rqmc.py",
    "src/inference_scaling/arllm/config.py",
    "src/inference_scaling/arllm/rewards.py",
    "src/inference_scaling/arllm/types.py",
)


def _installed_package_version(name: str) -> str | None:
    """Return an optional runtime dependency version without masking load errors."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _answer_counts(values: Sequence[Fraction | None]) -> dict[str, int]:
    """Return JSON-stable diagnostic keys, including unparseable candidates."""

    keys = (
        _fraction_text(value) if value is not None else "<unparseable>"
        for value in values
    )
    return dict(sorted(Counter(keys).items()))


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    _cuda_sync()
    started = time.perf_counter()
    result = call()
    _cuda_sync()
    return result, time.perf_counter() - started


def _prompt_tokens(backend: Any, problem: GSM8KProblem, config: dict[str, Any] | None = None) -> TokenSequence:
    messages = [{"role": "user", "content": gsm8k_prompt(problem.question)}]
    rendered = render_prompt(backend.tokenizer, messages, config or {})
    return backend.encode(str(rendered), add_special_tokens=False)


def _configured_verifier_reward(
    backend: Any,
    problem: GSM8KProblem,
    config: dict[str, Any],
) -> TokenVerifierReward:
    """Bind the selected verifier to one item without exposing it to algorithms."""

    spec = verifier_spec_from_config(config)
    context = VerifierContext(
        prompt=gsm8k_prompt(problem.question),
        reference=(
            _fraction_text(problem.gold_answer) if spec.requires_reference else None
        ),
        metadata={"benchmark": "gsm8k", "problem_index": problem.index},
    )
    return build_token_verifier_reward(config, context=context, decoder=backend.decode)


def _load_backend(
    path: str,
    config: dict[str, Any],
    *,
    adapter_base: str | None = None,
    role: str | None = None,
) -> Any:
    return load_backend_from_config(path, config, adapter_base=adapter_base, role=role)


def _trim_eos(tokens: TokenSequence, eos_token_id: int | None) -> TokenSequence:
    if eos_token_id is None or eos_token_id not in tokens:
        return tokens
    return tokens[: tokens.index(eos_token_id) + 1]


def _direct_generate(
    backend: Any,
    prompt: TokenSequence,
    *,
    max_new_tokens: int,
    num_beams: int,
) -> TokenSequence:
    generated = backend.direct_generate(
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return _trim_eos(generated, backend.tokenizer.eos_token_id)


def _sample_one(
    backend: Any,
    prompt: TokenSequence,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    request_id: str,
) -> TokenSequence:
    sample = backend.sample_batch(
        [
            GenerationRequest(
                prompt,
                max_new_tokens,
                SamplingConfig(
                    temperature=temperature,
                    eos_token_id=backend.tokenizer.eos_token_id,
                ),
                seed,
                request_id,
            )
        ]
    )[0]
    return sample.token_ids


def _minmax_rewards(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize confidence rewards within one decision batch."""

    if not values:
        raise ValueError("reward normalization requires at least one value")
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-12):
        return (0.0,) * len(values)
    scale = upper - lower
    return tuple((float(value) - lower) / scale for value in values)


def _confidence_rewards(
    backend: Any,
    prompt: TokenSequence,
    sequences: Sequence[TokenSequence],
    *,
    sampling: SamplingConfig,
    source: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    statistics_batch = backend.score_statistics_batch(
        [ScoreRequest(prompt, tuple(sequences), sampling)]
    )
    field = {
        "log_probability": "mean_logprob",
        "negative_entropy": "mean_negative_entropy",
        "self_certainty": "mean_self_certainty",
    }.get(source)
    if field is None:
        raise ValueError(f"{source!r} is not a confidence reward")
    raw = tuple(float(getattr(item, field)) for item in statistics_batch)
    return raw, _minmax_rewards(raw)


def _consilience_reward(
    backend: Any,
    sampling: SamplingConfig,
    config: dict[str, Any],
) -> ConsilienceReward:
    from inference_scaling.arllm.reward_factory import model_reward_from_config

    reward = model_reward_from_config(backend, config, source="consilience", sampling=sampling)
    assert isinstance(reward, ConsilienceReward)
    return reward


def _frozen_consensus_reward(
    backend: Any,
    prompt: TokenSequence,
    *,
    maximum: int,
    sampling: SamplingConfig,
    samples: int,
    seeds: SeedStream,
    problem_index: int,
) -> tuple[Callable[[TokenSequence, TokenSequence], float], dict[str, Any]]:
    """Construct a pointwise reward from an independent base-model pilot pool."""

    if samples <= 0:
        raise ValueError("frozen-consensus pilot_samples must be positive")
    requests = [
        GenerationRequest(
            prompt,
            maximum,
            sampling,
            seeds.derive("frozen_consensus", problem_index, pilot_index),
            f"frozen-consensus:{problem_index}:pilot:{pilot_index}",
        )
        for pilot_index in range(samples)
    ]
    pilots = backend.sample_batch(requests)
    answers = [
        extract_numeric_answer(backend.decode(sample.token_ids)) for sample in pilots
    ]
    reference = modal_answer(answers)

    def reward(_prompt: TokenSequence, generated: TokenSequence) -> float:
        prediction = extract_numeric_answer(backend.decode(generated))
        return float(reference is not None and prediction == reference)

    return reward, {
        "pilot_samples": samples,
        "pilot_answer_counts": _answer_counts(answers),
        "frozen_consensus_answer": _fraction_text(reference),
    }


def _run_best_of_n(
    backend: Any,
    problem: GSM8KProblem,
    prompt: TokenSequence,
    *,
    max_new_tokens: int,
    samples: int,
    temperature: float,
    seeds: SeedStream,
    problem_index: int,
    reward_source: str,
    config: dict[str, Any],
) -> tuple[TokenSequence, dict[str, Any]]:
    sampling = SamplingConfig(
        temperature=temperature,
        eos_token_id=backend.tokenizer.eos_token_id,
    )
    requests = [
        GenerationRequest(
            prompt,
            max_new_tokens,
            sampling,
            seeds.derive("best_of_n", problem_index, sample_index),
            f"best-of-n:{problem_index}:{sample_index}",
        )
        for sample_index in range(samples)
    ]
    candidates = backend.sample_batch(requests)
    texts = [backend.decode(candidate.token_ids) for candidate in candidates]
    parsed_answers = [extract_numeric_answer(text) for text in texts]
    raw_rewards: tuple[float, ...] | None = None
    verifier_reward: TokenVerifierReward | None = None
    model_reward_description: dict[str, object] | None = None
    if reward_source == "self_consistency":
        chosen = consensus_index(texts, [candidate.logprob for candidate in candidates])
        consensus = modal_answer(parsed_answers)
        selection_rewards = tuple(
            1.0 if consensus is not None and answer == consensus else 0.0
            for answer in parsed_answers
        )
    elif reward_source == "verifier":
        verifier_reward = _configured_verifier_reward(backend, problem, config)
        selection_rewards = verifier_reward.batch(
            prompt, [candidate.token_ids for candidate in candidates]
        )
        chosen = max(
            range(len(candidates)),
            key=lambda index: (
                selection_rewards[index],
                candidates[index].logprob,
                -index,
            ),
        )
    elif reward_source == "sequence_log_probability":
        model_reward = model_reward_from_config(
            backend, config, source="sequence_log_probability", sampling=sampling,
        )
        # Generation already returns exact log-probabilities under ``sampling``;
        # reusing them avoids a second full-sequence model forward pass.
        raw_rewards = tuple(
            model_reward.scale * candidate.logprob for candidate in candidates
        )
        model_reward_description = model_reward.describe()
        selection_rewards = raw_rewards
        chosen = max(
            range(len(candidates)),
            key=lambda index: (selection_rewards[index], -index),
        )
    elif reward_source == "consilience":
        model_reward = _consilience_reward(backend, sampling, config)
        raw_rewards = model_reward.batch(
            prompt,
            [candidate.token_ids for candidate in candidates],
        )
        model_reward_description = model_reward.describe()
        selection_rewards = raw_rewards
        chosen = max(
            range(len(candidates)),
            key=lambda index: (
                selection_rewards[index],
                candidates[index].logprob,
                -index,
            ),
        )
    else:
        raw_rewards, selection_rewards = _confidence_rewards(
            backend,
            prompt,
            [candidate.token_ids for candidate in candidates],
            sampling=sampling,
            source=reward_source,
        )
        chosen = max(
            range(len(candidates)),
            key=lambda index: (
                selection_rewards[index],
                candidates[index].logprob,
                -index,
            ),
        )
    return candidates[chosen].token_ids, {
        "candidate_count": samples,
        "selected_index": chosen,
        "reward_source": reward_source,
        "uses_test_gold_oracle": bool(
            verifier_reward is not None
            and verifier_reward.verifier.spec.requires_reference
        ),
        "verifier": (
            verifier_reward.describe() if verifier_reward is not None else None
        ),
        "model_reward": model_reward_description,
        "reward_normalization": (
            "per-decision min-max over candidate completions"
            if reward_source
            in {"log_probability", "negative_entropy", "self_certainty"}
            else None
        ),
        "selection_rewards": list(selection_rewards),
        "raw_reward_values": list(raw_rewards) if raw_rewards is not None else None,
        "raw_confidence_rewards": (
            list(raw_rewards)
            if raw_rewards is not None
            and reward_source
            in {
                "log_probability",
                "negative_entropy",
                "self_certainty",
                "consilience",
            }
            else None
        ),
        "answer_counts": _answer_counts(parsed_answers),
    }


def _conditional_diagnostics(result: Any) -> dict[str, Any]:
    ess: list[float] = []
    within_candidate_log_weight_dispersion: list[float] = []
    raw_corrections: list[float] = []
    applied_corrections: list[float] = []
    rollout_count = 0
    rewards: list[float] = []
    for step in result.steps:
        for candidate in step.candidates:
            weights = [rollout.log_weight for rollout in candidate.rollouts]
            ess.append(importance_effective_sample_size(weights))
            if len(weights) > 1:
                within_candidate_log_weight_dispersion.append(
                    statistics.pvariance(weights)
                )
            rollout_count += len(weights)
            rewards.extend(rollout.reward for rollout in candidate.rollouts)
            raw_corrections.extend(
                rollout.raw_log_importance_ratio
                for rollout in candidate.rollouts
                if rollout.raw_log_importance_ratio is not None
            )
            applied_corrections.extend(
                rollout.applied_log_importance_ratio
                for rollout in candidate.rollouts
                if rollout.applied_log_importance_ratio is not None
            )
    return {
        "guidance_steps": len(result.steps),
        "rollout_evaluations": rollout_count,
        "rollout_evaluations_planned": sum(
            int(
                getattr(
                    step,
                    "rollout_evaluations_planned",
                    sum(len(candidate.rollouts) for candidate in step.candidates),
                )
            )
            for step in result.steps
        ),
        "rollout_evaluations_performed": sum(
            int(
                getattr(
                    step,
                    "rollout_evaluations_performed",
                    sum(len(candidate.rollouts) for candidate in step.candidates),
                )
            )
            for step in result.steps
        ),
        "rollout_evaluations_skipped": sum(
            int(getattr(step, "rollout_evaluations_skipped", 0))
            for step in result.steps
        ),
        "rollout_evaluation_batches": sum(
            int(getattr(step, "rollout_evaluation_batches", 1)) for step in result.steps
        ),
        "exact_early_stop_steps": sum(
            bool(getattr(step, "exact_early_stop", False)) for step in result.steps
        ),
        "selection_invariant_verified_steps": sum(
            bool(getattr(step, "selection_invariant_verified", False))
            for step in result.steps
        ),
        "mean_rollout_ess": statistics.fmean(ess) if ess else 0.0,
        "mean_within_candidate_log_weight_dispersion": (
            statistics.fmean(within_candidate_log_weight_dispersion)
            if within_candidate_log_weight_dispersion
            else 0.0
        ),
        "candidate_log_weight_estimates_by_step": [
            [candidate.log_weight for candidate in step.candidates]
            for step in result.steps
        ],
        "candidate_token_ids_by_step": [
            [candidate.token_ids for candidate in step.candidates]
            for step in result.steps
        ],
        "candidate_log_weight_intervals_by_step": [
            [
                [
                    candidate.log_weight_lower_bound,
                    candidate.log_weight_upper_bound,
                ]
                for candidate in step.candidates
            ]
            for step in result.steps
        ],
        "selected_candidate_indices": [step.selected_index for step in result.steps],
        "mean_rollout_reward": statistics.fmean(rewards) if rewards else 0.0,
        "minimum_rollout_reward": min(rewards) if rewards else 0.0,
        "maximum_rollout_reward": max(rewards) if rewards else 0.0,
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
                raw_corrections,
                applied_corrections,
                strict=True,
            )
        ),
        "importance_corrected_rollout_evaluations": len(raw_corrections),
        "uncorrected_rollout_evaluations": rollout_count - len(raw_corrections),
    }


def _run_method(
    method: str,
    backend: Any,
    problem: GSM8KProblem,
    prompt: TokenSequence,
    config: dict[str, Any],
    seeds: SeedStream,
    proposal_backend: Any | None,
) -> tuple[TokenSequence, dict[str, Any]]:
    config, length_budget = generation_config_for_prompt(config, len(prompt), [backend, proposal_backend])
    source = config.get("reward", {}).get("source")
    if source is not None:
        config.setdefault("conditional_is", {})["reward"] = source
        config.setdefault("iterated_is", {})["reward"] = source
    scoped_algorithm = method in {"mh", "reward_mh", "verifier_mh"} or "conditional_is" in method
    scope = SamplingScope.from_config(backend, config, active=scoped_algorithm).for_prompt(prompt)
    if scope.scope == "thinking" and method != "mh":
        source = (
            "verifier" if method.startswith("verifier_") else
            config.get("iterated_is", {}).get("reward", "frozen_consensus")
            if method == "iterated_conditional_is" else
            config.get("conditional_is", {}).get("reward", "self_consistency")
        )
        if source not in MODEL_REWARD_SOURCES and config.get("reward", {}).get("input_scope") != "thinking":
            scope = scope.full_fallback("reward_uses_full_sequence")
        elif source == "consilience" and config.get("reward", {}).get("consilience", {}).get("scope") == "full":
            scope = scope.full_fallback("reward_uses_full_sequence")
    algorithm_backend = scope.wrap(backend, prompt)
    rollout_backend = None if proposal_backend is None else scope.wrap(proposal_backend, prompt)
    tokens, diagnostics = _run_method_impl(
        method, algorithm_backend, problem, prompt, config, seeds, rollout_backend
    )
    maximum = int(config["generation"]["max_new_tokens"])
    temperature = 1.0 if method.startswith("verifier_") else float(config.get("sampling", {}).get("temperature", 1.0))
    tokens, output = scope.finish(
        backend, prompt, tokens, max_new_tokens=maximum,
        sampling=SamplingConfig(temperature=temperature, eos_token_id=backend.tokenizer.eos_token_id),
        seed=seeds.derive(method, problem.index, "final-content"),
    )
    diagnostics["generation_budget"] = length_budget
    diagnostics["output_segments"] = output
    if diagnostics.get("reward_source") == "consilience" or (
        source == "consilience" and method not in {"mh", "base", "rl_sample", "beam", "rl_greedy"}
    ):
        selected_reward = model_reward_from_config(backend, config, source="consilience")
        assert isinstance(selected_reward, ConsilienceReward)
        output.update(selected_reward.describe_completion(prompt, tokens))
    return tokens, diagnostics


def _reference_mh_backend(backend: Any, prompt: TokenSequence, temperature: float):
    reference: Any = ScoreCachingBackend(backend)
    if temperature != 1.0:
        reference = ReferencePolicyBackend(reference, temperature=temperature)
    return AbsorbingEOSBackend(reference, backend.tokenizer.eos_token_id, absorbing_after=len(prompt))


def _run_method_impl(
    method: str, backend: Any, problem: GSM8KProblem, prompt: TokenSequence,
    config: dict[str, Any], seeds: SeedStream, proposal_backend: Any | None,
) -> tuple[TokenSequence, dict[str, Any]]:
    maximum = int(config["generation"]["max_new_tokens"])
    sampling_temperature = float(config.get("sampling", {}).get("temperature", 1.0))
    seed = seeds.derive(method, problem.index)
    if method in {"base", "rl_sample"}:
        temperature = 1.0 if method == "rl_sample" else sampling_temperature
        return (
            _sample_one(
                backend,
                prompt,
                max_new_tokens=maximum,
                temperature=temperature,
                seed=seed,
                request_id=f"{method}:{problem.index}",
            ),
            {"sampling_temperature": temperature},
        )
    if method in {"beam", "rl_greedy"}:
        beams = int(config["beam"]["num_beams"]) if method == "beam" else 1
        tokens = _direct_generate(
            backend,
            prompt,
            max_new_tokens=maximum,
            num_beams=beams,
        )
        forward_token_slots = beams * (len(prompt) + max(0, len(tokens) - 1))
        return tokens, {
            "num_beams": beams,
            "direct_generation_forward_token_slots": forward_token_slots,
            "direct_estimated_dense_forward_flops": (
                2 * backend.parameter_count * forward_token_slots
            ),
            "direct_compute_is_estimated": True,
            "direct_beam_compute_is_upper_bound": beams > 1,
        }
    if method == "best_of_n":
        return _run_best_of_n(
            backend,
            problem,
            prompt,
            max_new_tokens=maximum,
            samples=int(config["best_of_n"]["samples"]),
            temperature=sampling_temperature,
            seeds=seeds,
            problem_index=problem.index,
            reward_source=str(
                config["conditional_is"].get("reward", "self_consistency")
            ),
            config=config,
        )
    if method == "mh":
        mh = config["mh"]
        absorbing = _reference_mh_backend(backend, prompt, sampling_temperature)
        result = run_mh_chain(
            absorbing,
            prompt,
            MHConfig(
                alpha=float(mh["alpha"]),
                total_length=maximum,
                block_size=int(mh["block_size"]),
                steps_per_block=int(mh["steps_per_block"]),
                iterations=mh.get("iterations"),
                suffix_schedule=str(mh.get("suffix_schedule", "uniform")),
            ),
            SamplingConfig(temperature=1.0 / float(mh["alpha"])),
            SeedStream(seed),
        )
        return _trim_eos(result.token_ids, backend.tokenizer.eos_token_id), {
            "alpha": float(mh["alpha"]),
            "target_sampling_temperature": sampling_temperature,
            "block_size": int(mh["block_size"]),
            "steps_per_block": int(mh["steps_per_block"]),
            "iterations": mh.get("iterations"),
            "suffix_schedule": str(mh.get("suffix_schedule", "uniform")),
            "attempts": result.attempts,
            "accepted": result.accepted,
            "acceptance_rate": result.acceptance_rate,
            "mean_proposed_suffix_length": result.mean_proposed_suffix_length,
            "mean_proposed_token_changes": result.mean_proposed_token_changes,
            "mean_accepted_token_changes": result.mean_accepted_token_changes,
        }
    if method in {"verifier_mh", "reward_mh"}:
        mh = config["mh"]
        is_verifier = method == "verifier_mh"
        source = "verifier" if is_verifier else str(config["conditional_is"]["reward"])
        target_temperature = 1.0 if is_verifier else sampling_temperature
        absorbing = _reference_mh_backend(backend, prompt, target_temperature)
        reward_temperature = (
            float(config["matched_target"]["reward_temperature"])
            if is_verifier else reward_temperature_from_config(config, source=source)
        )
        if is_verifier:
            selected_reward = _configured_verifier_reward(backend, problem, config)
            reward_info = {
                "verifier": selected_reward.describe(),
                "uses_test_gold_oracle": selected_reward.verifier.spec.requires_reference,
            }
        else:
            selected_reward = model_reward_from_config(
                absorbing if source == "sequence_log_probability" else backend,
                config, source=source,
            )
            reward_info = {"model_reward": selected_reward.describe(), "uses_test_gold_oracle": False}

        result = run_reward_mh_chain(
            absorbing,
            prompt,
            RewardMHConfig(
                total_length=maximum,
                block_size=int(mh["block_size"]),
                steps_per_block=int(mh["steps_per_block"]),
                iterations=mh.get("iterations"),
                reward_temperature=reward_temperature,
                suffix_schedule=str(mh.get("suffix_schedule", "uniform")),
            ),
            SamplingConfig(),
            selected_reward,
            SeedStream(seed),
        )
        if isinstance(selected_reward, ConsilienceReward):
            reward_info["reward_scope_counts"] = selected_reward.scope_statistics()
        return _trim_eos(result.token_ids, backend.tokenizer.eos_token_id), {
            "target": "base_probability_times_exp_reward_over_temperature",
            "reward_source": source,
            "target_sampling_temperature": target_temperature,
            **reward_info,
            "reward_temperature": reward_temperature,
            "block_size": int(mh["block_size"]),
            "steps_per_block": int(mh["steps_per_block"]),
            "iterations": mh.get("iterations"),
            "suffix_schedule": str(mh.get("suffix_schedule", "uniform")),
            "updates": result.attempts,
            "accepted": result.accepted,
            "acceptance_rate": result.acceptance_rate,
            "final_reward": result.reward,
            "mean_proposed_suffix_length": result.mean_proposed_suffix_length,
            "mean_proposed_token_changes": result.mean_proposed_token_changes,
            "mean_accepted_token_changes": result.mean_accepted_token_changes,
        }
    conditional_methods = {
        "conditional_is",
        "iterated_conditional_is",
        "conditional_is_small_proposal",
        "verifier_conditional_is",
        "verifier_conditional_is_small_proposal",
    }
    if method in conditional_methods:
        conditional = config["conditional_is"]
        iterated = config.get("iterated_is", {})
        rollout_backend = backend
        if method.endswith("small_proposal"):
            if proposal_backend is None:
                raise ValueError("small-proposal method requires a proposal model")
            rollout_backend = proposal_backend
        use_matched_target = method.startswith("verifier_")
        reward_source = (
            "verifier"
            if use_matched_target
            else str(
                iterated.get("reward", "frozen_consensus")
                if method == "iterated_conditional_is"
                else conditional.get("reward", "self_consistency")
            )
        )
        if reward_source not in REWARD_SOURCES:
            raise ValueError(f"unknown reward source {reward_source!r}")
        use_verifier_reward = reward_source == "verifier"
        target_sampling_temperature = (
            1.0 if use_matched_target else sampling_temperature
        )
        reward_temperature = (
            float(config["matched_target"]["reward_temperature"])
            if use_matched_target
            else reward_temperature_from_config(config, source=reward_source)
        )
        reward_batch = None
        pointwise_reward: Callable[[TokenSequence, TokenSequence], float] | None = None
        reward_diagnostics: dict[str, Any] = {}
        if reward_source == "self_consistency":
            if method == "iterated_conditional_is":
                raise ValueError(
                    "iterated_conditional_is requires a fixed pointwise reward; "
                    "use frozen_consensus, sequence_log_probability, Consilience, "
                    "or verifier"
                )
            reward_batch = CumulativeConsensusReward(backend.decode)
        elif reward_source == "frozen_consensus":
            pointwise_reward, reward_diagnostics = _frozen_consensus_reward(
                backend,
                prompt,
                maximum=maximum,
                sampling=SamplingConfig(
                    temperature=target_sampling_temperature,
                    eos_token_id=backend.tokenizer.eos_token_id,
                ),
                samples=int(iterated.get("pilot_samples", 8)),
                seeds=seeds,
                problem_index=problem.index,
            )
        elif reward_source == "sequence_log_probability":
            model_reward = model_reward_from_config(
                backend, config, source="sequence_log_probability",
                sampling=SamplingConfig(
                    temperature=target_sampling_temperature,
                    eos_token_id=backend.tokenizer.eos_token_id,
                ),
            )
            if method == "iterated_conditional_is":
                pointwise_reward = model_reward
            else:
                reward_batch = model_reward.batch
            reward_diagnostics["model_reward"] = model_reward.describe()
        elif reward_source == "consilience":
            model_reward = _consilience_reward(
                backend,
                SamplingConfig(
                    temperature=target_sampling_temperature,
                    eos_token_id=backend.tokenizer.eos_token_id,
                ),
                config,
            )
            if method == "iterated_conditional_is":
                pointwise_reward = model_reward
            else:
                reward_batch = model_reward.batch
            reward_diagnostics["model_reward"] = model_reward.describe()
        elif reward_source != "verifier":
            if method == "iterated_conditional_is":
                raise ValueError(
                    "iterated_conditional_is currently accepts only fixed pointwise "
                    "frozen_consensus, sequence_log_probability, Consilience, or "
                    "verifier rewards"
                )

            def confidence_reward(
                reward_prompt: TokenSequence,
                generated_sequences: Sequence[TokenSequence],
            ) -> tuple[float, ...]:
                _, normalized = _confidence_rewards(
                    backend,
                    reward_prompt,
                    generated_sequences,
                    sampling=SamplingConfig(
                        temperature=target_sampling_temperature,
                        eos_token_id=backend.tokenizer.eos_token_id,
                    ),
                    source=reward_source,
                )
                return normalized

            reward_batch = confidence_reward

        verifier_reward = (
            _configured_verifier_reward(backend, problem, config)
            if use_verifier_reward
            else None
        )
        if verifier_reward is not None:
            pointwise_reward = verifier_reward
            reward_diagnostics["verifier"] = verifier_reward.describe()

        base_sampling = SamplingConfig(
            temperature=target_sampling_temperature,
            eos_token_id=backend.tokenizer.eos_token_id,
        )
        cached_base = ScoreCachingBackend(backend)
        cached_rollout = ScoreCachingBackend(rollout_backend)
        if method == "iterated_conditional_is":
            iterated_reward = pointwise_reward
            if iterated_reward is None and reward_batch is None:
                raise RuntimeError("iterated conditional IS did not construct a reward")
            result = run_iterated_conditional_is(
                cached_base,
                prompt,
                IteratedConditionalISConfig(
                    pool_size=int(iterated.get("pool_size", 3)),
                    updates=int(iterated.get("updates", 4)),
                    rollout_count=int(conditional["rollout_count"]),
                    block_size=int(conditional["block_size"]),
                    total_length=maximum,
                    reward_temperature=reward_temperature,
                ),
                iterated_reward,
                SeedStream(seed),
                base_sampling=base_sampling,
                rollout_backend=cached_rollout,
                rollout_sampling=base_sampling,
                reward_batch=reward_batch,
            )
        else:
            result = run_conditional_is(
                cached_base,
                prompt,
                ConditionalISConfig(
                    candidate_count=int(conditional["candidate_count"]),
                    rollout_count=int(conditional["rollout_count"]),
                    block_size=int(conditional["block_size"]),
                    total_length=maximum,
                    reward_temperature=reward_temperature,
                    importance_log_ratio_clip=(
                        float(conditional["importance_log_ratio_clip"])
                        if method.endswith("small_proposal")
                        and bool(conditional.get("apply_importance_correction", True))
                        and conditional.get("importance_log_ratio_clip") is not None
                        else None
                    ),
                    apply_importance_correction=bool(
                        conditional.get("apply_importance_correction", True)
                    ),
                    rollout_design=str(conditional.get("rollout_design", "iid")),
                    exact_rollout_early_stop=bool(
                        conditional.get("exact_rollout_early_stop", False)
                    ),
                    rollout_log_weight_bounds=(
                        (
                            float(conditional["rollout_log_weight_lower_bound"]),
                            float(conditional["rollout_log_weight_upper_bound"]),
                        )
                        if bool(conditional.get("exact_rollout_early_stop", False))
                        else None
                    ),
                    rollout_evaluation_batch_size=int(
                        conditional.get("rollout_evaluation_batch_size", 1)
                    ),
                ),
                pointwise_reward,
                SeedStream(seed),
                base_sampling=base_sampling,
                rollout_backend=cached_rollout,
                rollout_sampling=base_sampling,
                reward_batch=reward_batch,
            )
        diagnostics = _conditional_diagnostics(result)
        if reward_source == "consilience":
            reward_diagnostics["reward_scope_counts"] = model_reward.scope_statistics()
        diagnostics.update(reward_diagnostics)
        diagnostics["rollout_design"] = (
            "iid"
            if method == "iterated_conditional_is"
            else str(conditional.get("rollout_design", "iid"))
        )
        diagnostics["rollout_ess_is_descriptive"] = (
            diagnostics["rollout_design"] != "iid"
        )
        diagnostics["configured_candidate_count"] = int(conditional["candidate_count"])
        diagnostics["configured_rollout_count"] = int(conditional["rollout_count"])
        diagnostics["configured_block_size"] = int(conditional["block_size"])
        diagnostics["exact_rollout_early_stop_enabled"] = bool(
            conditional.get("exact_rollout_early_stop", False)
        )
        diagnostics["rollout_evaluation_batch_size"] = int(
            conditional.get("rollout_evaluation_batch_size", 1)
        )
        diagnostics["declared_rollout_log_weight_bounds"] = (
            [
                float(conditional["rollout_log_weight_lower_bound"]),
                float(conditional["rollout_log_weight_upper_bound"]),
            ]
            if bool(conditional.get("exact_rollout_early_stop", False))
            else None
        )
        if method == "iterated_conditional_is":
            diagnostics.update(
                {
                    "pool_size": int(iterated.get("pool_size", 3)),
                    "updates_per_block": int(iterated.get("updates", 4)),
                    "fresh_candidate_evaluations": result.fresh_candidate_evaluations,
                    "reused_pool_entries": result.reused_pool_entries,
                    "finite_pool_target_invariant": True,
                }
            )
        diagnostics["proposal_model"] = rollout_backend.model_id
        diagnostics["candidate_source"] = "base_model"
        reward_target_name = {
            "self_consistency": "cumulative_consensus",
            "frozen_consensus": "independent_pilot_frozen_consensus",
            "log_probability": "normalized_mean_log_probability",
            "sequence_log_probability": "model_sequence_log_probability",
            "negative_entropy": "normalized_negative_entropy",
            "self_certainty": "normalized_self_certainty",
            "consilience": "model_consilience",
            "verifier": "configured_verifier",
        }[reward_source]
        target_description = (
            f"base_probability_times_exp_{reward_target_name}_over_temperature"
        )
        if method.endswith("small_proposal") and not bool(
            conditional.get("apply_importance_correction", True)
        ):
            target_description = (
                "base_candidates_reweighted_by_proposal_expected_exp_"
                f"{reward_target_name}_over_temperature"
            )
        elif (
            method.endswith("small_proposal")
            and conditional.get("importance_log_ratio_clip") is not None
        ):
            target_description = (
                "clipped_finite_rollout_approximation_to_" + target_description
            )
        diagnostics["target"] = target_description
        diagnostics["reward_temperature"] = reward_temperature
        diagnostics["reward_source"] = reward_source
        diagnostics["reward_normalization"] = (
            "per-guidance-step min-max over all candidate rollouts"
            if reward_source
            in {"log_probability", "negative_entropy", "self_certainty"}
            else None
        )
        diagnostics["importance_log_ratio_clip"] = (
            float(conditional["importance_log_ratio_clip"])
            if method.endswith("small_proposal")
            and bool(conditional.get("apply_importance_correction", True))
            and conditional.get("importance_log_ratio_clip") is not None
            else None
        )
        diagnostics["apply_importance_correction"] = bool(
            conditional.get("apply_importance_correction", True)
        )
        diagnostics["sampling_temperature"] = target_sampling_temperature
        diagnostics["uses_test_gold_oracle"] = bool(
            verifier_reward is not None
            and verifier_reward.verifier.spec.requires_reference
        )
        diagnostics["uses_matched_reward_temperature"] = use_matched_target
        return result.token_ids, diagnostics
    raise ValueError(f"unknown method {method!r}")


def _summary(
    records: Sequence[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    count = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    low, high = wilson_interval(correct, count)
    elapsed = [float(record["elapsed_seconds"]) for record in records]
    output_lengths = [int(record["output_tokens"]) for record in records]
    generated = sum(
        int(record["backend_delta"].get("generated_tokens", 0)) for record in records
    )
    scored = sum(
        int(record["backend_delta"].get("scored_tokens", 0)) for record in records
    )
    proposal_generated = sum(
        int(record.get("proposal_backend_delta", {}).get("generated_tokens", 0))
        for record in records
    )
    base_generation_slots = sum(
        int(record["backend_delta"].get("generation_forward_token_slots", 0))
        for record in records
    )
    base_shared_prefill_saved = sum(
        int(record["backend_delta"].get("shared_prefill_tokens_saved", 0))
        for record in records
    )
    base_score_slots = sum(
        int(record["backend_delta"].get("score_forward_token_slots", 0))
        for record in records
    )
    base_flops = sum(
        int(record["backend_delta"].get("estimated_dense_forward_flops", 0))
        for record in records
    )
    proposal_generation_slots = sum(
        int(
            record.get("proposal_backend_delta", {}).get(
                "generation_forward_token_slots", 0
            )
        )
        for record in records
    )
    proposal_shared_prefill_saved = sum(
        int(
            record.get("proposal_backend_delta", {}).get(
                "shared_prefill_tokens_saved", 0
            )
        )
        for record in records
    )
    proposal_score_slots = sum(
        int(
            record.get("proposal_backend_delta", {}).get("score_forward_token_slots", 0)
        )
        for record in records
    )
    proposal_flops = sum(
        int(
            record.get("proposal_backend_delta", {}).get(
                "estimated_dense_forward_flops", 0
            )
        )
        for record in records
    )
    direct_slots = sum(
        int(
            record.get("diagnostics", {}).get(
                "direct_generation_forward_token_slots", 0
            )
        )
        for record in records
    )
    direct_flops = sum(
        int(
            record.get("diagnostics", {}).get("direct_estimated_dense_forward_flops", 0)
        )
        for record in records
    )
    total_forward_slots = (
        base_generation_slots
        + base_score_slots
        + proposal_generation_slots
        + proposal_score_slots
        + direct_slots
    )
    total_flops = base_flops + proposal_flops + direct_flops
    return {
        "schema_version": 3,
        "manifest_fingerprint": manifest["fingerprint"],
        "method": manifest["method"],
        "tag": manifest["tag"],
        "examples": count,
        "correct": correct,
        "accuracy": correct / count,
        "accuracy_wilson_95": [low, high],
        "sum_example_seconds": sum(elapsed),
        "mean_example_seconds": statistics.fmean(elapsed),
        "median_example_seconds": statistics.median(elapsed),
        "mean_selected_output_tokens": statistics.fmean(output_lengths),
        "median_selected_output_tokens": statistics.median(output_lengths),
        "maximum_selected_output_tokens": max(output_lengths),
        "base_generated_tokens": generated,
        "proposal_generated_tokens": proposal_generated,
        "base_scored_tokens": scored,
        "base_generation_forward_token_slots": base_generation_slots,
        "base_shared_prefill_tokens_saved": base_shared_prefill_saved,
        "base_score_forward_token_slots": base_score_slots,
        "proposal_generation_forward_token_slots": proposal_generation_slots,
        "proposal_shared_prefill_tokens_saved": proposal_shared_prefill_saved,
        "proposal_score_forward_token_slots": proposal_score_slots,
        "direct_generation_forward_token_slots": direct_slots,
        "total_forward_token_slots": total_forward_slots,
        "total_shared_prefill_tokens_saved": (
            base_shared_prefill_saved + proposal_shared_prefill_saved
        ),
        "estimated_dense_forward_flops": total_flops,
        "estimated_dense_forward_petaflops": total_flops / 1e15,
        "compute_definition": (
            "forward token slots count every non-cached model-input position actually "
            "submitted by the benchmark, including repeated prompt/scoring positions; "
            "dominant dense FLOPs = 2 * model parameter count * forward token slots, "
            "summed separately for the 1.5B and 0.5B models"
        ),
        "compute_exclusions": (
            "quadratic attention, elementwise kernels, tokenization, sampling, and host work"
        ),
    }


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    apply_model_output_overrides(config, args)
    set_backend_override(config, args.backend)
    if getattr(args, "vllm_mh_fused_logprobs", False):
        if args.method != "mh":
            raise ValueError("--vllm-mh-fused-logprobs requires --method mh")
        config.setdefault("vllm", {}).setdefault("base", {})[
            "mh_fused_logprobs"
        ] = True
    set_rl_adapter_override(config, getattr(args, "rl_adapter", None))
    replace_verifier_from_file(config, getattr(args, "verifier_config", None))
    if args.limit is not None:
        config["run"]["sample_count"] = args.limit
    if args.sampling_temperature is not None:
        config.setdefault("sampling", {})["temperature"] = args.sampling_temperature
    if args.num_beams is not None:
        config["beam"]["num_beams"] = args.num_beams
    if args.best_of_n_samples is not None:
        config["best_of_n"]["samples"] = args.best_of_n_samples
    if args.conditional_reward is not None:
        config["conditional_is"]["reward"] = args.conditional_reward
        config.setdefault("iterated_is", {})["reward"] = args.conditional_reward
        config.setdefault("reward", {})["source"] = args.conditional_reward
    if args.reward_temperature is not None:
        config["conditional_is"]["reward_temperature"] = args.reward_temperature
        config.setdefault("reward", {})["temperature"] = args.reward_temperature
    if getattr(args, "logprob_reward_scale", None) is not None:
        config["conditional_is"]["logprob_reward_scale"] = args.logprob_reward_scale
        config.setdefault("reward", {})["logprob_scale"] = args.logprob_reward_scale
    for argument, setting in (
        ("consilience_top_k", "consilience_top_k"),
        ("consilience_window_fraction", "consilience_window_fraction"),
        ("consilience_window_tokens", "consilience_window_tokens"),
        ("consilience_skip_fraction", "consilience_skip_fraction"),
        ("consilience_initial_penalty", "consilience_initial_penalty"),
        ("consilience_reward_scale", "consilience_reward_scale"),
        ("consilience_reasoning_end_text", "consilience_reasoning_end_text"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            config["conditional_is"][setting] = value
            if setting == "consilience_reasoning_end_text":
                config.setdefault("output", {})["thinking_end_text"] = value
            else:
                name = "scale" if setting == "consilience_reward_scale" else setting.removeprefix("consilience_")
                config.setdefault("reward", {}).setdefault("consilience", {})[name] = value
    if args.importance_log_ratio_clip is not None:
        value = args.importance_log_ratio_clip.strip().lower()
        parsed_clip = None if value == "none" else float(value)
        if parsed_clip is not None and parsed_clip <= 0:
            raise ValueError("importance log-ratio clip must be positive or 'none'")
        config["conditional_is"]["importance_log_ratio_clip"] = parsed_clip
    if args.disable_importance_correction:
        if not args.method.endswith("small_proposal"):
            raise ValueError(
                "--disable-importance-correction requires a small-proposal method"
            )
        config["conditional_is"]["apply_importance_correction"] = False
        config["conditional_is"]["importance_log_ratio_clip"] = None
    if args.mh_alpha is not None:
        config["mh"]["alpha"] = args.mh_alpha
    if args.mh_steps is not None:
        config["mh"]["steps_per_block"] = args.mh_steps
    if getattr(args, "mh_suffix_schedule", None) is not None:
        config["mh"]["suffix_schedule"] = args.mh_suffix_schedule
    if args.candidate_count is not None:
        config["conditional_is"]["candidate_count"] = args.candidate_count
    if args.rollout_count is not None:
        config["conditional_is"]["rollout_count"] = args.rollout_count
    if getattr(args, "rollout_design", None) is not None:
        if args.method == "iterated_conditional_is":
            raise ValueError(
                "--rollout-design is not implemented for iterated conditional IS"
            )
        config["conditional_is"]["rollout_design"] = args.rollout_design
    if getattr(args, "exact_rollout_early_stop", False):
        if args.method == "iterated_conditional_is":
            raise ValueError(
                "exact rollout early stopping is not implemented for iterated conditional IS"
            )
        lower = getattr(args, "rollout_log_weight_lower", None)
        upper = getattr(args, "rollout_log_weight_upper", None)
        if lower is None or upper is None:
            raise ValueError(
                "exact rollout early stopping requires lower and upper log-weight bounds"
            )
        config["conditional_is"]["exact_rollout_early_stop"] = True
        config["conditional_is"]["rollout_log_weight_lower_bound"] = lower
        config["conditional_is"]["rollout_log_weight_upper_bound"] = upper
        config["conditional_is"]["rollout_evaluation_batch_size"] = int(
            getattr(args, "rollout_evaluation_batch_size", None) or 1
        )
    elif any(
        value is not None
        for value in (
            getattr(args, "rollout_log_weight_lower", None),
            getattr(args, "rollout_log_weight_upper", None),
            getattr(args, "rollout_evaluation_batch_size", None),
        )
    ):
        raise ValueError(
            "rollout bounds and evaluation batch size require --exact-rollout-early-stop"
        )
    if getattr(args, "iterated_pool_size", None) is not None:
        config.setdefault("iterated_is", {})["pool_size"] = args.iterated_pool_size
    if getattr(args, "iterated_updates", None) is not None:
        config.setdefault("iterated_is", {})["updates"] = args.iterated_updates
    if getattr(args, "consensus_pilot_samples", None) is not None:
        config.setdefault("iterated_is", {})["pilot_samples"] = (
            args.consensus_pilot_samples
        )
    if args.block_size is not None:
        if args.method in {"mh", "verifier_mh", "reward_mh"}:
            config["mh"]["block_size"] = args.block_size
        else:
            config["conditional_is"]["block_size"] = args.block_size


def _model_metadata(config: dict[str, Any], method: str) -> dict[str, Any]:
    return model_metadata(config, "rl" if method.startswith("rl_") else "base")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/arllm.toml"))
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        help="override runtime.backend before the experiment fingerprint is computed",
    )
    parser.add_argument(
        "--vllm-mh-fused-logprobs",
        action="store_true",
        help=(
            "return proposal and base token probabilities from one vLLM decode; "
            "requires --backend vllm-sync and vLLM 0.26.x"
        ),
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--sampling-scope", choices=("full", "thinking"))
    parser.add_argument("--thinking-mode", choices=("auto", "enabled", "disabled"))
    parser.add_argument("--thinking-format", choices=("auto", "tags", "json", "xml"))
    parser.add_argument("--thinking-path", help="structured thinking field, e.g. response.reasoning")
    parser.add_argument("--content-path", help="structured final-content field, e.g. response.answer")
    parser.add_argument("--thinking-start-text")
    parser.add_argument("--thinking-end-text")
    parser.add_argument("--starts-in-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--rl-adapter", type=Path)
    parser.add_argument(
        "--verifier-config",
        type=Path,
        help="standalone TOML file whose [verifier] table replaces the default",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--sampling-temperature", type=float)
    parser.add_argument("--num-beams", type=int)
    parser.add_argument("--best-of-n-samples", type=int)
    parser.add_argument(
        "--conditional-reward", "--reward",
        choices=REWARD_SOURCES,
        help=(
            "reward used by Best-of-N and conditional methods; verifier uses "
            "the separately configured [verifier] component"
        ),
    )
    parser.add_argument("--reward-temperature", type=float)
    parser.add_argument(
        "--logprob-reward-scale",
        type=float,
        help=(
            "c in r=c*log p(y|x); combined with reward temperature tau this "
            "targets the power 1+c/tau"
        ),
    )
    parser.add_argument("--consilience-top-k", type=int)
    parser.add_argument("--consilience-window-fraction", type=float)
    parser.add_argument("--consilience-window-tokens", type=int)
    parser.add_argument("--consilience-skip-fraction", type=float)
    parser.add_argument("--consilience-initial-penalty", type=float)
    parser.add_argument("--consilience-reward-scale", type=float)
    parser.add_argument(
        "--consilience-reasoning-end-text",
        help=(
            "optional model-specific delimiter after the reasoning phase; "
            "the delimiter and final-answer tokens are excluded from the reward"
        ),
    )
    parser.add_argument(
        "--importance-log-ratio-clip",
        help="positive symmetric clip or 'none' for exact untruncated weights",
    )
    parser.add_argument(
        "--disable-importance-correction",
        action="store_true",
        help=(
            "skip base-model rescoring of small-model rollouts and use proposal-model "
            "continuation reward weighting as a biased lookahead signal"
        ),
    )
    parser.add_argument("--mh-alpha", type=float)
    parser.add_argument("--mh-steps", type=int)
    parser.add_argument(
        "--mh-suffix-schedule",
        choices=("uniform", "inverse_length", "multiscale"),
    )
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--rollout-count", type=int)
    parser.add_argument(
        "--rollout-design",
        choices=("iid", "scrambled_sobol", "arithmetic_lattice"),
    )
    parser.add_argument("--exact-rollout-early-stop", action="store_true")
    parser.add_argument("--rollout-log-weight-lower", type=float)
    parser.add_argument("--rollout-log-weight-upper", type=float)
    parser.add_argument("--rollout-evaluation-batch-size", type=int)
    parser.add_argument("--iterated-pool-size", type=int)
    parser.add_argument("--iterated-updates", type=int)
    parser.add_argument("--consensus-pilot-samples", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument(
        "--draw-index",
        type=int,
        default=0,
        help="independent sampling replicate included in the request-level seed",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    _apply_overrides(config, args)
    if (
        str(config["runtime"]["device"]).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    all_problems = load_gsm8k(args.data)
    problems = select_problems(
        all_problems,
        int(config["run"]["sample_count"]),
        seed=int(config["run"]["subset_seed"]),
    )
    roles = {"base"}
    if args.method.startswith("rl_"):
        roles.add("rl")
    if args.method.endswith("small_proposal"):
        roles.add("proposal")
    input_artifacts = validate_model_artifacts(config, roles)
    input_weight_hashes = input_artifacts["weight_sha256"]
    actual_base_hash = input_weight_hashes["base"]
    actual_adapter_hash = input_weight_hashes.get("rl_adapter")
    actual_proposal_hash = input_weight_hashes.get("proposal")
    effective = {
        "config": config,
        "method": args.method,
        "tag": args.tag,
        "draw_index": args.draw_index,
        "input_weight_sha256": input_weight_hashes,
        "input_metadata_sha256": input_artifacts["metadata_sha256"],
        "input_adapter_sha256": input_artifacts["adapter_sha256"],
        "implementation_sha256": _implementation_hashes(
            REPOSITORY_ROOT,
            entrypoints=IMPLEMENTATION_FILES,
        ),
        "problem_indices": [problem.index for problem in problems],
    }
    fingerprint = _fingerprint(effective)
    run_dir = (
        args.output_root / str(config["run"]["name"]) / f"{args.method}-{args.tag}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "method": args.method,
        "tag": args.tag,
        "effective": effective,
        "dataset": {
            "name": "GSM8K official test split",
            "path": str(args.data),
            "rows_in_public_split": len(all_problems),
        },
        "model": _model_metadata(config, args.method),
        "configured_verifier": (
            verifier_spec_from_config(config).as_dict()
            if "verifier" in config
            else None
        ),
        "proposal_model": (
            model_metadata(config, "proposal")
            if args.method.endswith("small_proposal")
            else None
        ),
        "environment": {
            "backend": configured_backend(config),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vllm": (
                _installed_package_version("vllm")
                if configured_backend(config).startswith("vllm")
                else None
            ),
        },
    }
    previous = manifest
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError(
                f"{run_dir} contains a different experiment; choose a new --tag"
            )
    elif records_path.is_file():
        raise ValueError(f"{run_dir} contains records without a manifest")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    existing_records = _load_records(records_path)
    completed = {int(record["problem_index"]) for record in existing_records}
    pending = [problem for problem in problems if problem.index not in completed]
    if not pending:
        manifest = previous
        selected = [
            record
            for record in existing_records
            if int(record["problem_index"]) in {problem.index for problem in problems}
        ]
        summary = _summary(selected, manifest)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    model_key = "rl" if args.method.startswith("rl_") else "base"
    adapter_base = None
    if model_key == "rl" and config["models"].get("rl_kind") == "peft_adapter":
        adapter_base = str(config["models"]["rl_base"])
    backend = None
    proposal_backend = None
    try:
        backend = _load_backend(
            str(config["models"][model_key]),
            config,
            adapter_base=adapter_base, role=model_key,
        )
        proposal_backend = None
        if args.method.endswith("small_proposal"):
            proposal_backend = _load_backend(str(config["models"]["proposal"]), config, role="proposal")
            if backend.tokenizer.get_vocab() != proposal_backend.tokenizer.get_vocab():
                raise ValueError(
                    "base and proposal tokenizers do not have identical vocabularies"
                )

        manifest["model"]["parameter_count"] = backend.parameter_count
        manifest["model"]["verified_base_weight_sha256"] = actual_base_hash
        if model_key == "rl" and config["models"].get("rl_kind") == "peft_adapter":
            assert actual_adapter_hash is not None
            manifest["model"]["adapter_weight_sha256"] = actual_adapter_hash
            manifest["model"]["base_weight_sha256"] = str(
                input_weight_hashes.get("rl_base", actual_base_hash)
            )
        if proposal_backend is not None and manifest["proposal_model"] is not None:
            assert actual_proposal_hash is not None
            manifest["proposal_model"]["parameter_count"] = proposal_backend.parameter_count
            manifest["proposal_model"]["verified_weight_sha256"] = actual_proposal_hash
        manifest["compute_accounting"] = {
            "primary_units": ["forward_token_slots", "estimated_dense_forward_flops"],
            "flop_formula": "2 * model_parameter_count * forward_token_slots",
            "wall_time_role": "hardware-dependent supplemental measurement",
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        warm_prompt = _prompt_tokens(backend, pending[0], config)
        _sample_one(
            backend,
            warm_prompt,
            max_new_tokens=2,
            temperature=1.0,
            seed=int(config["run"]["seed"]),
            request_id="warmup",
        )
        seeds = SeedStream(
            SeedStream(int(config["run"]["seed"])).derive("draw", args.draw_index)
        )
        with records_path.open("a", encoding="utf-8", buffering=1) as sink:
            for ordinal, problem in enumerate(pending, 1):
                prompt = _prompt_tokens(backend, problem, config)
                before = backend.snapshot()
                proposal_before = proposal_backend.snapshot() if proposal_backend else None
                (tokens, diagnostics), elapsed = _timed(
                    lambda backend=backend, problem=problem, prompt=prompt, proposal_backend=proposal_backend: (
                        _run_method(
                            args.method,
                            backend,
                            problem,
                            prompt,
                            config,
                            seeds,
                            proposal_backend,
                        )
                    )
                )
                after = backend.snapshot()
                proposal_after = proposal_backend.snapshot() if proposal_backend else None
                text = backend.decode(tokens)
                segments = diagnostics["output_segments"]
                prediction = extract_numeric_answer(segments["content_text"])
                record = {
                    "schema_version": 2,
                    "method": args.method,
                    "tag": args.tag,
                    "draw_index": args.draw_index,
                    "problem_index": problem.index,
                    "question_sha256": hashlib.sha256(
                        problem.question.encode()
                    ).hexdigest(),
                    "gold_answer": _fraction_text(problem.gold_answer),
                    "prediction": _fraction_text(prediction),
                    "correct": prediction == problem.gold_answer,
                    "output": text,
                    "thinking": segments["thinking_text"],
                    "content": segments["content_text"],
                    "output_tokens": len(tokens),
                    "prompt_tokens": len(prompt),
                    "elapsed_seconds": elapsed,
                    "backend_delta": _snapshot_delta(before, after),
                    "diagnostics": diagnostics,
                }
                if proposal_before is not None and proposal_after is not None:
                    record["proposal_backend_delta"] = _snapshot_delta(
                        proposal_before, proposal_after
                    )
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                fallback = segments.get("sampling_fallback_reason") or segments.get("reward_fallback_reason")
                if fallback is not None:
                    print(
                        f"gsm8k_index={problem.index} sampling_scope={segments['sampling_scope']} "
                        f"reward_scope={segments.get('reward_scope', 'unchanged')} full_fallback={fallback}",
                        flush=True,
                    )
                print(
                    f"[{ordinal}/{len(pending)}] method={args.method} "
                    f"gsm8k_index={problem.index} correct={record['correct']} "
                    f"seconds={elapsed:.3f}",
                    flush=True,
                )

        records = _load_records(records_path)
        selected = [
            record
            for record in records
            if int(record["problem_index"]) in {p.index for p in problems}
        ]
        summary = _summary(selected, manifest)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        try:
            close_backend(proposal_backend)
        finally:
            close_backend(backend)
            backend = proposal_backend = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
