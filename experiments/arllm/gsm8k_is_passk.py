"""Run resumable multi-draw GSM8K comparisons for conditional IS variants."""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import copy
import gc
import hashlib
import json
import random
import statistics
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.arllm.runtime import validate_model_artifacts
from experiments.shared.methods import AR_IS_PASSK_METHODS

from experiments.arllm.gsm8k_passk import (
    PASSK_IMPLEMENTATION_FILES,
    _MethodBackend,
    _chunk_plan,
    _estimated_pass_at_k,
    _load_jsonl,
    _prepare_manifest,
    _summarize_method,
    _validate_chunks,
)
from experiments.arllm.gsm8k_reproduction import (
    _file_sha256,
    _implementation_hashes,
    _load_backend,
    _prompt_tokens,
    _run_method,
    _sample_one,
    _snapshot_delta,
    _timed,
)
from inference_scaling.arllm.backends import (
    BACKEND_CHOICES,
    ContinuousBatchingBackend,
    close_backend,
    set_backend_override,
)
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import replace_verifier_from_file


IS_PASSK_METHODS = AR_IS_PASSK_METHODS
IS_PASSK_IMPLEMENTATION_FILES = tuple(
    dict.fromkeys((*PASSK_IMPLEMENTATION_FILES, "experiments/arllm/gsm8k_is_passk.py"))
)
_BATCHING_SUM_FIELDS = (
    "sample_batches",
    "score_batches",
    "sample_requests",
    "score_sequences",
)
_BATCHING_MAX_FIELDS = ("maximum_sample_batch", "maximum_score_batch")


def _uses_small_proposal(method: str) -> bool:
    return method.startswith("conditional_is_small_proposal")


def _execution_method_and_config(
    method: str, config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    if method not in {
        "conditional_is_small_proposal_unclipped",
        "conditional_is_small_proposal_uncorrected",
    }:
        return method, config
    effective = copy.deepcopy(config)
    effective["conditional_is"]["importance_log_ratio_clip"] = None
    effective["conditional_is"]["apply_importance_correction"] = (
        method != "conditional_is_small_proposal_uncorrected"
    )
    return "conditional_is_small_proposal", effective


def _combine_numeric_deltas(
    base: dict[str, int | float],
    proposal: dict[str, int | float] | None,
) -> dict[str, int | float]:
    """Sum physical counters while preserving model-specific deltas separately."""

    proposal = proposal or {}
    if set(proposal) - set(base):
        raise ValueError("proposal backend delta contains fields absent from base delta")
    return {name: value + proposal.get(name, 0) for name, value in base.items()}


def _combine_batching_snapshots(
    base: dict[str, int | float],
    proposal: dict[str, int | float] | None,
) -> dict[str, int | float]:
    proposal = proposal or {}
    return {
        **{
            name: int(base[name]) + int(proposal.get(name, 0))
            for name in _BATCHING_SUM_FIELDS
        },
        **{
            name: max(int(base[name]), int(proposal.get(name, 0)))
            for name in _BATCHING_MAX_FIELDS
        },
    }


def _summarize_model_compute(
    chunks: Sequence[dict[str, Any]], key: str
) -> dict[str, int | float] | None:
    deltas = [chunk.get(key) for chunk in chunks if chunk.get(key) is not None]
    if not deltas:
        return None
    field_names = set(deltas[0])
    if any(set(delta) != field_names for delta in deltas):
        raise ValueError(f"inconsistent {key} fields across IS pass@k chunks")
    totals = {
        name: sum(delta[name] for delta in deltas)
        for name in sorted(field_names)
    }
    total_slots = int(totals["generation_forward_token_slots"]) + int(
        totals["score_forward_token_slots"]
    )
    totals["total_forward_token_slots"] = total_slots
    totals["estimated_dense_forward_petaflops"] = (
        int(totals["estimated_dense_forward_flops"]) / 1e15
    )
    return totals


def _summarize_batching_by_model(
    chunks: Sequence[dict[str, Any]], model: str
) -> dict[str, int] | None:
    snapshots = [
        chunk["continuous_batching_by_model"].get(model)
        for chunk in chunks
        if chunk["continuous_batching_by_model"].get(model) is not None
    ]
    if not snapshots:
        return None
    return {
        **{
            name: sum(int(snapshot[name]) for snapshot in snapshots)
            for name in _BATCHING_SUM_FIELDS
        },
        **{
            name: max(int(snapshot[name]) for snapshot in snapshots)
            for name in _BATCHING_MAX_FIELDS
        },
    }


def _batching_backend(stack: ExitStack, raw_backend: Any, config: dict[str, Any]):
    return stack.enter_context(
        ContinuousBatchingBackend(
            raw_backend,
            max_batch_size=int(config["runtime"]["max_batch_size"]),
            max_batch_tokens=int(config["runtime"]["max_batch_tokens"]),
            batch_wait_seconds=0.01,
        )
    )


def _run_chunk(
    *,
    method: str,
    chunk_index: int,
    task_keys: Sequence[tuple[int, int]],
    problems_by_index: dict[int, Any],
    prompts_by_index: dict[int, tuple[int, ...]],
    raw_base: Any,
    raw_proposal: Any | None,
    config: dict[str, Any],
    workers: int,
    fingerprint: str,
) -> dict[str, Any]:
    if _uses_small_proposal(method) != (raw_proposal is not None):
        raise ValueError("proposal backend presence does not match the IS method")

    base_before = raw_base.snapshot()
    proposal_before = raw_proposal.snapshot() if raw_proposal is not None else None
    with ExitStack() as stack:
        base_batching = _batching_backend(stack, raw_base, config)
        proposal_batching = (
            _batching_backend(stack, raw_proposal, config)
            if raw_proposal is not None
            else None
        )
        base = _MethodBackend(base_batching, raw_base)
        proposal = (
            _MethodBackend(proposal_batching, raw_proposal)
            if proposal_batching is not None and raw_proposal is not None
            else None
        )

        def run_one(task_key: tuple[int, int]):
            draw, problem_index = task_key
            seeds = SeedStream(
                SeedStream(int(config["run"]["seed"])).derive("draw", draw)
            )
            execution_method, execution_config = _execution_method_and_config(
                method, config
            )
            tokens, diagnostics = _run_method(
                execution_method,
                base,
                problems_by_index[problem_index],
                prompts_by_index[problem_index],
                execution_config,
                seeds,
                proposal,
            )
            diagnostics = dict(diagnostics)
            diagnostics["reported_method"] = method
            diagnostics["execution_method"] = execution_method
            return tokens, diagnostics

        def run_parallel():
            with ThreadPoolExecutor(
                max_workers=min(workers, len(task_keys))
            ) as executor:
                return list(executor.map(run_one, task_keys))

        outputs, elapsed = _timed(run_parallel)
        base_batching_snapshot = asdict(base_batching.snapshot())
        proposal_batching_snapshot = (
            asdict(proposal_batching.snapshot())
            if proposal_batching is not None
            else None
        )

    base_after = raw_base.snapshot()
    proposal_after = raw_proposal.snapshot() if raw_proposal is not None else None
    base_delta = _snapshot_delta(base_before, base_after)
    proposal_delta = (
        _snapshot_delta(proposal_before, proposal_after)
        if proposal_before is not None and proposal_after is not None
        else None
    )

    records = []
    for (draw, problem_index), (tokens, diagnostics) in zip(
        task_keys, outputs, strict=True
    ):
        problem = problems_by_index[problem_index]
        text = raw_base.decode(tokens)
        prediction = extract_numeric_answer(diagnostics["output_segments"]["content_text"])
        records.append(
            {
                "draw_index": draw,
                "problem_index": problem_index,
                "question_sha256": hashlib.sha256(
                    problem.question.encode("utf-8")
                ).hexdigest(),
                "prediction": str(prediction) if prediction is not None else None,
                "correct": prediction == problem.gold_answer,
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "output_tokens": len(tokens),
                "diagnostics": diagnostics,
            }
        )

    return {
        "schema_version": 1,
        "manifest_fingerprint": fingerprint,
        "method": method,
        "chunk_index": chunk_index,
        "records": records,
        "seconds_excluding_model_load": elapsed,
        "backend_delta": _combine_numeric_deltas(base_delta, proposal_delta),
        "base_backend_delta": base_delta,
        "proposal_backend_delta": proposal_delta,
        "continuous_batching": _combine_batching_snapshots(
            base_batching_snapshot, proposal_batching_snapshot
        ),
        "continuous_batching_by_model": {
            "base": base_batching_snapshot,
            "proposal": proposal_batching_snapshot,
        },
    }


def _run_pending_chunks(
    *,
    raw_path: Path,
    fingerprint: str,
    plan: dict[tuple[str, int], tuple[tuple[int, int], ...]],
    completed: dict[tuple[str, int], dict[str, Any]],
    methods: Sequence[str],
    problems_by_index: dict[int, Any],
    config: dict[str, Any],
    workers: int,
) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:
        pending = [key for key in plan if key[0] == method and key not in completed]
        if not pending:
            continue
        raw_base = _load_backend(str(config["models"]["base"]), config, role="base")
        raw_proposal = (
            _load_backend(str(config["models"]["proposal"]), config, role="proposal")
            if _uses_small_proposal(method)
            else None
        )
        if (
            raw_proposal is not None
            and raw_base.tokenizer.get_vocab() != raw_proposal.tokenizer.get_vocab()
        ):
            raise ValueError("base and proposal tokenizers do not have identical vocabularies")
        prompts_by_index = {
            index: _prompt_tokens(raw_base, problem, config)
            for index, problem in problems_by_index.items()
        }
        if raw_proposal is not None:
            proposal_prompts = {
                index: _prompt_tokens(raw_proposal, problem, config)
                for index, problem in problems_by_index.items()
            }
            if proposal_prompts != prompts_by_index:
                raise ValueError("base and proposal tokenizers render different prompts")
        first_prompt = prompts_by_index[next(iter(problems_by_index))]
        _sample_one(
            raw_base,
            first_prompt,
            max_new_tokens=2,
            temperature=1.0,
            seed=int(config["run"]["seed"]),
            request_id=f"is-passk-warmup:{method}:base",
        )
        if raw_proposal is not None:
            _sample_one(
                raw_proposal,
                first_prompt,
                max_new_tokens=2,
                temperature=1.0,
                seed=int(config["run"]["seed"]),
                request_id=f"is-passk-warmup:{method}:proposal",
            )

        with raw_path.open("a", encoding="utf-8", buffering=1) as sink:
            for key in pending:
                chunk = _run_chunk(
                    method=method,
                    chunk_index=key[1],
                    task_keys=plan[key],
                    problems_by_index=problems_by_index,
                    prompts_by_index=prompts_by_index,
                    raw_base=raw_base,
                    raw_proposal=raw_proposal,
                    config=config,
                    workers=workers,
                    fingerprint=fingerprint,
                )
                sink.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
                completed[key] = chunk
                print(
                    f"[{len(completed)}/{len(plan)}] method={method} "
                    f"chunk={key[1]} tasks={len(plan[key])} "
                    f"seconds={chunk['seconds_excluding_model_load']:.3f}",
                    flush=True,
                )

        close_backend(raw_proposal)
        close_backend(raw_base)
        del raw_proposal
        del raw_base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _paired_pass_at_k_difference(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    draws: int,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    reference_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in reference["per_problem"]
    }
    candidate_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in candidate["per_problem"]
    }
    if reference_by_problem.keys() != candidate_by_problem.keys():
        raise ValueError("paired IS comparison requires the same problem indices")
    problem_indices = tuple(reference_by_problem)
    rng = random.Random(seed)
    result: dict[str, Any] = {}
    for k_text in reference["estimated_pass_at_k"]:
        k = int(k_text)
        differences = [
            _estimated_pass_at_k(candidate_by_problem[index], draws, k)
            - _estimated_pass_at_k(reference_by_problem[index], draws, k)
            for index in problem_indices
        ]
        bootstrap = sorted(
            statistics.fmean(
                differences[rng.randrange(len(differences))] for _ in differences
            )
            for _ in range(replicates)
        )
        result[k_text] = {
            "candidate_minus_reference": statistics.fmean(differences),
            "paired_problem_bootstrap_95": [
                bootstrap[int(0.025 * replicates)],
                bootstrap[int(0.975 * replicates)],
            ],
        }
    return result


def _cost_ratio(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_over_candidate_wall_time": (
            float(reference["seconds_excluding_model_load"])
            / float(candidate["seconds_excluding_model_load"])
        ),
        "reference_over_candidate_flops": (
            int(reference["estimated_dense_forward_flops"])
            / int(candidate["estimated_dense_forward_flops"])
        ),
        "interpretation": (
            "The named reference method is the numerator and the named candidate "
            "method is the denominator. A ratio above one means the candidate used "
            "less wall time or FLOPs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--tag", default="is-passk")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", default=",".join(IS_PASSK_METHODS))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--output", type=Path)
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.draws <= 0 or args.limit <= 0 or args.workers <= 0:
        raise ValueError("draws, limit, and workers must be positive")

    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods) - set(IS_PASSK_METHODS))
    if unknown:
        raise ValueError(f"unsupported IS pass@k methods: {', '.join(unknown)}")
    if len(methods) != len(set(methods)):
        raise ValueError("IS pass@k methods must not contain duplicates")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    apply_model_output_overrides(config, args)
    replace_verifier_from_file(config, args.verifier_config)
    set_backend_override(config, args.backend)
    config["run"]["sample_count"] = args.limit
    config["is_passk"] = {
        "method_definitions": {
            "conditional_is": {
                "candidate_model": "base",
                "rollout_model": "base",
                "importance_log_ratio_clip": None,
            },
            "conditional_is_small_proposal": {
                "candidate_model": "base",
                "rollout_model": "proposal",
                "importance_log_ratio_clip": config["conditional_is"].get(
                    "importance_log_ratio_clip"
                ),
            },
            "conditional_is_small_proposal_unclipped": {
                "candidate_model": "base",
                "rollout_model": "proposal",
                "importance_log_ratio_clip": None,
                "apply_importance_correction": True,
            },
            "conditional_is_small_proposal_uncorrected": {
                "candidate_model": "base",
                "rollout_model": "proposal",
                "importance_log_ratio_clip": None,
                "apply_importance_correction": False,
            },
        }
    }
    if str(config["runtime"]["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    problems = select_problems(
        load_gsm8k(args.data), args.limit, seed=int(config["run"]["subset_seed"])
    )
    problem_indices = [problem.index for problem in problems]
    problems_by_index = {problem.index: problem for problem in problems}
    profile = str(config["run"]["name"])
    output = args.output or Path(f"results/{profile}_{args.tag}.json")
    raw_path = args.raw_output or output.with_suffix(".chunks.jsonl")
    implementation_sha256 = _implementation_hashes(
        Path(__file__).resolve().parents[2],
        entrypoints=IS_PASSK_IMPLEMENTATION_FILES,
    )
    roles = {"base"}
    if any(_uses_small_proposal(method) for method in methods):
        roles.add("proposal")
    input_artifacts = validate_model_artifacts(config, roles)
    input_weight_sha256 = input_artifacts["weight_sha256"]
    _, fingerprint, _ = _prepare_manifest(
        config=config,
        data_path=args.data,
        methods=methods,
        draws=args.draws,
        workers=args.workers,
        problem_indices=problem_indices,
        input_weight_sha256=input_weight_sha256,
        implementation_sha256=implementation_sha256,
        raw_path=raw_path,
        input_metadata_sha256=input_artifacts["metadata_sha256"],
        input_adapter_sha256=input_artifacts["adapter_sha256"],
    )
    plan = _chunk_plan(methods, args.draws, problem_indices, args.workers)
    completed = _validate_chunks(_load_jsonl(raw_path), fingerprint, plan)
    if not args.summarize_only:
        _run_pending_chunks(
            raw_path=raw_path,
            fingerprint=fingerprint,
            plan=plan,
            completed=completed,
            methods=methods,
            problems_by_index=problems_by_index,
            config=config,
            workers=args.workers,
        )

    chunks = _load_jsonl(raw_path)
    completed = _validate_chunks(chunks, fingerprint, plan)
    if len(completed) != len(plan):
        raise RuntimeError(f"IS pass@k grid is incomplete: {len(completed)}/{len(plan)} chunks")

    ks = tuple(k for k in (1, 2, 4, 8, 16, 32) if k <= args.draws)
    table: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_chunks = [completed[key] for key in plan if key[0] == method]
        method_records = [
            record for chunk in method_chunks for record in chunk["records"]
        ]
        summary = _summarize_method(
            method_records,
            method_chunks,
            problem_indices,
            args.draws,
            ks,
            bootstrap_seed=SeedStream(int(config["run"]["seed"])).derive(method),
        )
        summary["compute_by_model"] = {
            "base": _summarize_model_compute(method_chunks, "base_backend_delta"),
            "proposal": _summarize_model_compute(
                method_chunks, "proposal_backend_delta"
            ),
        }
        summary["continuous_batching_by_model"] = {
            "base": _summarize_batching_by_model(method_chunks, "base"),
            "proposal": _summarize_batching_by_model(method_chunks, "proposal"),
        }
        table[method] = summary

    comparisons: dict[str, Any] = {}
    if "conditional_is" in table:
        standard = table["conditional_is"]
        if "conditional_is_small_proposal" in table:
            clipped = table["conditional_is_small_proposal"]
            comparisons["clipped_small_proposal_minus_standard_pass_at_k"] = (
                _paired_pass_at_k_difference(
                    standard,
                    clipped,
                    draws=args.draws,
                    seed=SeedStream(int(config["run"]["seed"])).derive(
                        "is-passk-paired-clipped"
                    ),
                )
            )
            comparisons["standard_over_clipped_small_proposal_cost"] = _cost_ratio(
                standard, clipped
            )
        if "conditional_is_small_proposal_unclipped" in table:
            unclipped = table["conditional_is_small_proposal_unclipped"]
            comparisons["unclipped_small_proposal_minus_standard_pass_at_k"] = (
                _paired_pass_at_k_difference(
                    standard,
                    unclipped,
                    draws=args.draws,
                    seed=SeedStream(int(config["run"]["seed"])).derive(
                        "is-passk-paired-unclipped"
                    ),
                )
            )
            comparisons["standard_over_unclipped_small_proposal_cost"] = _cost_ratio(
                standard,
                unclipped,
            )
        if "conditional_is_small_proposal_uncorrected" in table:
            uncorrected = table["conditional_is_small_proposal_uncorrected"]
            comparisons["uncorrected_small_proposal_minus_standard_pass_at_k"] = (
                _paired_pass_at_k_difference(
                    standard,
                    uncorrected,
                    draws=args.draws,
                    seed=SeedStream(int(config["run"]["seed"])).derive(
                        "is-passk-paired-uncorrected-standard"
                    ),
                )
            )
            comparisons["standard_over_uncorrected_small_proposal_cost"] = _cost_ratio(
                standard,
                uncorrected,
            )
    if {
        "conditional_is_small_proposal",
        "conditional_is_small_proposal_unclipped",
    }.issubset(table):
        clipped = table["conditional_is_small_proposal"]
        unclipped = table["conditional_is_small_proposal_unclipped"]
        comparisons["clipped_minus_unclipped_pass_at_k"] = (
            _paired_pass_at_k_difference(
                unclipped,
                clipped,
                draws=args.draws,
                seed=SeedStream(int(config["run"]["seed"])).derive(
                    "is-passk-paired-clipping"
                ),
            )
        )
        comparisons["unclipped_over_clipped_cost"] = _cost_ratio(
            unclipped, clipped
        )
    if {
        "conditional_is_small_proposal",
        "conditional_is_small_proposal_uncorrected",
    }.issubset(table):
        clipped = table["conditional_is_small_proposal"]
        uncorrected = table["conditional_is_small_proposal_uncorrected"]
        comparisons["uncorrected_minus_clipped_pass_at_k"] = (
            _paired_pass_at_k_difference(
                clipped,
                uncorrected,
                draws=args.draws,
                seed=SeedStream(int(config["run"]["seed"])).derive(
                    "is-passk-paired-uncorrected-clipped"
                ),
            )
        )
        comparisons["clipped_over_uncorrected_cost"] = _cost_ratio(
            clipped, uncorrected
        )

    report = {
        "schema_version": 1,
        "benchmark": "OpenAI GSM8K official test split",
        "profile": profile,
        "methods": table,
        "comparisons": comparisons or None,
        "problem_indices": problem_indices,
        "draws_per_problem": args.draws,
        "workers": args.workers,
        "method_definitions": config["is_passk"]["method_definitions"],
        "manifest_fingerprint": fingerprint,
        "raw_chunks_sha256": _file_sha256(raw_path),
        "input_weight_sha256": input_weight_sha256,
        "input_metadata_sha256": input_artifacts["metadata_sha256"],
        "input_adapter_sha256": input_artifacts["adapter_sha256"],
        "implementation_sha256": implementation_sha256,
        "compute_definition": (
            "2 * each model's parameter count * that model's observed padded forward "
            "token slots, summed across base and proposal models; model loading and "
            "warmup are excluded from wall time"
        ),
        "independence_definition": (
            "Each draw has a separate deterministic random stream. Continuous batching "
            "merges physical model calls but does not share candidate or rollout samples "
            "between draws."
        ),
        "limitations": (
            "Eight draws estimate pass@k only through k=8. All four methods use the "
            "same problem grid and candidate/rollout budgets. Finite importance-weight "
            "variance affects both small-proposal methods; the clipped variant adds "
            "finite bias, while the unclipped variant can have higher variance. The "
            "uncorrected variant removes weight variance but targets proposal-model "
            "rather than base-model continuation weighting."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
