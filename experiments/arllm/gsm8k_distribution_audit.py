"""Empirical, resumable answer-distribution audit for GRPO and same-target MH/IS."""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import copy
import gc
import json
import statistics
import time
import tomllib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.arllm.runtime import set_rl_adapter_override, validate_model_artifacts
from experiments.shared.methods import AR_DISTRIBUTION_METHODS

from experiments.arllm.gsm8k_reproduction import (
    IMPLEMENTATION_FILES,
    _file_sha256,
    _fingerprint,
    _implementation_hashes,
    _load_backend,
    _prompt_tokens,
    _run_method,
    _sample_one,
    _snapshot_delta,
)
from inference_scaling.shared.evaluation import extract_numeric_answer, load_gsm8k, select_problems
from inference_scaling.arllm.backends import (
    BACKEND_CHOICES,
    close_backend,
    set_backend_override,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import replace_verifier_from_file
from experiments.shared.statistics import (
    bootstrap_answer_distance,
    jensen_shannon_bits,
    probability_distribution,
    quantile,
    total_variation_distance,
)
from experiments.shared.artifacts import load_jsonl as _load_jsonl

DEFAULT_METHODS = AR_DISTRIBUTION_METHODS
AUDIT_IMPLEMENTATION_FILES = (
    *IMPLEMENTATION_FILES,
    "experiments/arllm/gsm8k_distribution_audit.py",
)


def _answer_key(answer: Fraction | None) -> str:
    if answer is None:
        return "[invalid]"
    if answer.denominator == 1:
        return str(answer.numerator)
    return f"{answer.numerator}/{answer.denominator}"


_distribution = probability_distribution
_tv = total_variation_distance
_js = jensen_shannon_bits
_quantile = quantile


def _bootstrap_answer_distance(
    left: Mapping[int, list[str]],
    right: Mapping[int, list[str]],
    problem_indices: list[int],
    *,
    samples: int = 2_000,
) -> dict[str, list[float]]:
    return bootstrap_answer_distance(
        left,
        right,
        problem_indices,
        replicates=samples,
    )


def _split_half_noise_floor(
    answers: Mapping[int, list[str]], problem_indices: list[int]
) -> dict[str, float] | None:
    if min(len(answers[index]) for index in problem_indices) < 2:
        return None
    televisions: list[float] = []
    divergences: list[float] = []
    for problem_index in problem_indices:
        values = answers[problem_index]
        midpoint = len(values) // 2
        left = _distribution(Counter(values[:midpoint]))
        right = _distribution(Counter(values[midpoint:]))
        televisions.append(_tv(left, right))
        divergences.append(_js(left, right))
    return {
        "mean_total_variation": statistics.fmean(televisions),
        "mean_jensen_shannon_bits": statistics.fmean(divergences),
    }


def _method_backend(config: dict[str, Any], method: str):
    if method.startswith("rl_"):
        adapter_base = (
            str(config["models"]["rl_base"])
            if config["models"].get("rl_kind") == "peft_adapter"
            else None
        )
        return _load_backend(str(config["models"]["rl"]), config, adapter_base=adapter_base, role="rl")
    return _load_backend(str(config["models"]["base"]), config, role="base")


def _prepare_manifest(
    *,
    config: dict[str, Any],
    methods: tuple[str, ...],
    problem_indices: list[int],
    draws: int,
    input_weight_hashes: dict[str, str],
    implementation_hashes: dict[str, str],
    raw_path: Path,
    input_metadata_hashes: dict[str, Any] | None = None,
    input_adapter_hashes: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, Path]:
    effective = {
        "config": config,
        "problem_indices": problem_indices,
        "draws_per_problem": draws,
        "methods": list(methods),
        "input_weight_sha256": input_weight_hashes,
        "input_metadata_sha256": input_metadata_hashes or {},
        "input_adapter_sha256": input_adapter_hashes or {},
        "implementation_sha256": implementation_hashes,
    }
    fingerprint = _fingerprint(effective)
    manifest = {"schema_version": 1, "fingerprint": fingerprint, "effective": effective}
    manifest_path = raw_path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"{raw_path} belongs to a different distribution audit; choose new paths"
            )
    elif raw_path.is_file() and raw_path.stat().st_size:
        raise ValueError("distribution audit records exist without their manifest")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, fingerprint, manifest_path


def _validate_existing_records(
    records: list[dict[str, Any]],
    fingerprint: str,
    expected_keys: set[tuple[str, int, int]],
) -> set[tuple[str, int, int]]:
    completed: set[tuple[str, int, int]] = set()
    for item in records:
        if item.get("manifest_fingerprint") != fingerprint:
            raise ValueError("a distribution-audit record has the wrong fingerprint")
        key = (str(item["method"]), int(item["draw"]), int(item["problem_index"]))
        if key not in expected_keys:
            raise ValueError("a distribution-audit record is outside the requested grid")
        if key in completed:
            raise ValueError("duplicate distribution-audit sample")
        completed.add(key)
    return completed


def _run_pending_samples(
    *,
    raw_path: Path,
    fingerprint: str,
    config: dict[str, Any],
    methods: tuple[str, ...],
    problems,
    draws: int,
    completed: set[tuple[str, int, int]],
    expected_count: int,
) -> None:
    with raw_path.open("a", encoding="utf-8", buffering=1) as sink:
        for method in methods:
            pending = [
                (draw, problem)
                for draw in range(draws)
                for problem in problems
                if (method, draw, problem.index) not in completed
            ]
            if not pending:
                continue
            backend = _method_backend(config, method)
            proposal_backend = None
            if method.endswith("small_proposal"):
                proposal_backend = _load_backend(str(config["models"]["proposal"]), config, role="proposal")
                if backend.tokenizer.get_vocab() != proposal_backend.tokenizer.get_vocab():
                    raise ValueError(
                        "base and proposal tokenizers do not have identical vocabularies"
                    )
            warm_prompt = _prompt_tokens(backend, problems[0], config)
            _sample_one(
                backend,
                warm_prompt,
                max_new_tokens=2,
                temperature=1.0,
                seed=int(config["run"]["seed"]),
                request_id=f"distribution-audit-warmup:{method}",
            )
            if proposal_backend is not None:
                _sample_one(
                    proposal_backend,
                    warm_prompt,
                    max_new_tokens=2,
                    temperature=1.0,
                    seed=int(config["run"]["seed"]),
                    request_id=f"distribution-audit-proposal-warmup:{method}",
                )

            for draw, problem in pending:
                draw_config = copy.deepcopy(config)
                draw_config["run"]["seed"] = int(config["run"]["seed"]) + 1_000_003 * draw
                seeds = SeedStream(int(draw_config["run"]["seed"]))
                prompt = _prompt_tokens(backend, problem, config)
                backend_before = backend.snapshot()
                proposal_before = proposal_backend.snapshot() if proposal_backend else None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                tokens, diagnostics = _run_method(
                    method,
                    backend,
                    problem,
                    prompt,
                    draw_config,
                    seeds,
                    proposal_backend,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                backend_after = backend.snapshot()
                proposal_after = proposal_backend.snapshot() if proposal_backend else None
                answer = extract_numeric_answer(diagnostics["output_segments"]["content_text"])
                record = {
                    "schema_version": 1,
                    "manifest_fingerprint": fingerprint,
                    "method": method,
                    "draw": draw,
                    "problem_index": problem.index,
                    "answer": _answer_key(answer),
                    "correct": answer == problem.gold_answer,
                    "seconds": elapsed,
                    "base_delta": _snapshot_delta(backend_before, backend_after),
                    "proposal_delta": (
                        _snapshot_delta(proposal_before, proposal_after)
                        if proposal_before is not None and proposal_after is not None
                        else None
                    ),
                }
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                completed.add((method, draw, problem.index))
                print(
                    f"[{len(completed)}/{expected_count}] method={method} "
                    f"draw={draw + 1}/{draws} gsm8k_index={problem.index} "
                    f"seconds={elapsed:.3f}",
                    flush=True,
                )
            close_backend(proposal_backend)
            close_backend(backend)
            del proposal_backend
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _aggregate_records(
    raw_records: list[dict[str, Any]], methods: tuple[str, ...], problem_indices: list[int]
) -> tuple[
    dict[str, dict[int, Counter[str]]],
    dict[str, dict[int, list[str]]],
    dict[str, float],
    dict[str, int],
    dict[str, dict[str, int | float]],
]:
    counts = {
        method: {problem_index: Counter() for problem_index in problem_indices}
        for method in methods
    }
    answer_draws: dict[str, dict[int, list[tuple[int, str]]]] = {
        method: {problem_index: [] for problem_index in problem_indices}
        for method in methods
    }
    runtimes = {method: 0.0 for method in methods}
    correctness = {method: 0 for method in methods}
    compute: dict[str, dict[str, int | float]] = {
        method: {"forward_token_slots": 0, "estimated_dense_forward_flops": 0}
        for method in methods
    }
    for item in raw_records:
        method = str(item["method"])
        problem_index = int(item["problem_index"])
        answer = str(item["answer"])
        counts[method][problem_index][answer] += 1
        answer_draws[method][problem_index].append((int(item["draw"]), answer))
        runtimes[method] += float(item["seconds"])
        correctness[method] += int(bool(item["correct"]))
        for delta in (item["base_delta"], item["proposal_delta"]):
            if delta is None:
                continue
            compute[method]["forward_token_slots"] = int(
                compute[method]["forward_token_slots"]
            ) + int(delta["generation_forward_token_slots"]) + int(
                delta["score_forward_token_slots"]
            )
            compute[method]["estimated_dense_forward_flops"] = int(
                compute[method]["estimated_dense_forward_flops"]
            ) + int(delta["estimated_dense_forward_flops"])
    ordered_answers = {
        method: {
            problem_index: [
                answer for _, answer in sorted(answer_draws[method][problem_index])
            ]
            for problem_index in problem_indices
        }
        for method in methods
    }
    for method in methods:
        compute[method]["estimated_dense_forward_petaflops"] = (
            int(compute[method]["estimated_dense_forward_flops"]) / 1e15
        )
    return counts, ordered_answers, runtimes, correctness, compute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--problem-count", type=int, default=4)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--rl-adapter", type=Path)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/gsm8k_distribution_audit.json")
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="append-only per-sample JSONL; defaults beside --output",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.problem_count <= 0 or args.draws <= 0:
        raise ValueError("problem-count and draws must be positive")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    apply_model_output_overrides(config, args)
    replace_verifier_from_file(config, args.verifier_config)
    set_backend_override(config, args.backend)
    set_rl_adapter_override(config, args.rl_adapter)
    problems = select_problems(
        load_gsm8k(args.data),
        args.problem_count,
        seed=int(config["run"]["subset_seed"]),
    )
    problem_indices = [problem.index for problem in problems]
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = set(methods) - set(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"unknown distribution-audit methods: {sorted(unknown)}")
    if "rl_sample" not in methods:
        raise ValueError("the audit requires rl_sample as its GRPO reference")

    roles = {"base", "rl"}
    if any(method.endswith("small_proposal") for method in methods):
        roles.add("proposal")
    input_artifacts = validate_model_artifacts(config, roles)
    input_weight_hashes = input_artifacts["weight_sha256"]
    implementation_hashes = _implementation_hashes(
        Path(__file__).resolve().parents[2],
        entrypoints=AUDIT_IMPLEMENTATION_FILES,
    )
    raw_path = args.raw_output or args.output.with_suffix(".records.jsonl")
    _, manifest_fingerprint, _ = _prepare_manifest(
        config=config,
        methods=methods,
        problem_indices=problem_indices,
        draws=args.draws,
        input_weight_hashes=input_weight_hashes,
        implementation_hashes=implementation_hashes,
        raw_path=raw_path,
        input_metadata_hashes=input_artifacts["metadata_sha256"],
        input_adapter_hashes=input_artifacts["adapter_sha256"],
    )
    expected_keys = {
        (method, draw, problem_index)
        for method in methods
        for draw in range(args.draws)
        for problem_index in problem_indices
    }
    completed = _validate_existing_records(
        _load_jsonl(raw_path), manifest_fingerprint, expected_keys
    )
    _run_pending_samples(
        raw_path=raw_path,
        fingerprint=manifest_fingerprint,
        config=config,
        methods=methods,
        problems=problems,
        draws=args.draws,
        completed=completed,
        expected_count=len(expected_keys),
    )

    raw_records = _load_jsonl(raw_path)
    if len(raw_records) != len(expected_keys):
        raise RuntimeError("distribution audit did not complete the requested sample grid")
    records, answer_draws, runtimes, correctness, compute = _aggregate_records(
        raw_records, methods, problem_indices
    )
    reference = "rl_sample"
    comparisons: dict[str, Any] = {}
    for method in methods:
        if method == reference:
            continue
        per_problem = []
        for problem_index in problem_indices:
            left = _distribution(records[method][problem_index])
            right = _distribution(records[reference][problem_index])
            per_problem.append(
                {
                    "problem_index": problem_index,
                    "jensen_shannon_bits": _js(left, right),
                    "total_variation": _tv(left, right),
                }
            )
        comparisons[f"{method}_vs_{reference}"] = {
            "mean_jensen_shannon_bits": statistics.fmean(
                item["jensen_shannon_bits"] for item in per_problem
            ),
            "mean_total_variation": statistics.fmean(
                item["total_variation"] for item in per_problem
            ),
            "per_problem": per_problem,
            **_bootstrap_answer_distance(
                answer_draws[method], answer_draws[reference], problem_indices
            ),
        }

    sample_count = args.problem_count * args.draws
    report = {
        "schema_version": 2,
        "manifest_fingerprint": manifest_fingerprint,
        "raw_records_sha256": _file_sha256(raw_path),
        "public_dataset": "OpenAI GSM8K official test split",
        "problem_indices": problem_indices,
        "input_weight_sha256": input_weight_hashes,
        "input_metadata_sha256": input_artifacts["metadata_sha256"],
        "input_adapter_sha256": input_artifacts["adapter_sha256"],
        "implementation_sha256": implementation_hashes,
        "draws_per_problem": args.draws,
        "distribution_level": "parsed final answer, not full token sequence",
        "target_by_method": {
            "base": "base_probability",
            "rl_sample": (
                "finite-step GRPO approximation to "
                "base_probability_times_exp_numeric_training_reward_over_beta"
            ),
            "verifier_mh": (
                "base_probability_times_exp_configured_verifier_over_beta"
            ),
            "verifier_conditional_is": (
                "finite-rollout approximation to "
                "base_probability_times_exp_configured_verifier_over_beta"
            ),
            "verifier_conditional_is_small_proposal": (
                "clipped off-policy finite-rollout approximation to "
                "base_probability_times_exp_configured_verifier_over_beta"
            ),
        },
        "configured_verifier": config["verifier"],
        "reward_temperature_beta": float(config["matched_target"]["reward_temperature"]),
        "methods": {
            method: {
                "accuracy_over_draws": correctness[method] / sample_count,
                "correct": correctness[method],
                "samples": sample_count,
                "seconds_excluding_model_load": runtimes[method],
                **compute[method],
                "answer_counts": {
                    str(index): dict(sorted(counts.items()))
                    for index, counts in records[method].items()
                },
            }
            for method in methods
        },
        "comparisons": comparisons,
        "rl_sample_split_half_sampling_noise_floor": _split_half_noise_floor(
            answer_draws[reference], problem_indices
        ),
        "compute_definition": (
            "2 * each model's parameter count * observed padded forward token slots; "
            "the small proposal is accounted separately before summation"
        ),
        "limitation": (
            "Finite draws estimate answer-level distributions only. Bootstrap intervals "
            "include finite-draw variability but not model, prompt, or hyperparameter "
            "uncertainty. The GRPO split-half distance is a sampling-noise reference, not "
            "a correction. None of these quantities can establish equality of full "
            "token-sequence distributions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
