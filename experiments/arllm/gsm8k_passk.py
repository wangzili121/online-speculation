"""Run resumable, continuously batched GSM8K pass@k replicates."""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import gc
import hashlib
import json
import random
import statistics
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.arllm.runtime import set_rl_adapter_override, validate_model_artifacts
from experiments.shared.methods import AR_PASSK_METHODS

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
    _timed,
    _trim_eos,
)
from inference_scaling.arllm.algorithms import run_mh_chains_batched
from inference_scaling.arllm.backends import (
    BACKEND_CHOICES,
    AbsorbingEOSBackend,
    ContinuousBatchingBackend,
    ScoreCachingBackend,
    close_backend,
    set_backend_override,
)
from inference_scaling.arllm.config import MHConfig, SamplingConfig
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import replace_verifier_from_file
from experiments.shared.statistics import estimated_pass_at_k
from experiments.shared.artifacts import load_jsonl as _load_jsonl

PASSK_METHODS = AR_PASSK_METHODS
PASSK_IMPLEMENTATION_FILES = (
    *IMPLEMENTATION_FILES,
    "experiments/arllm/gsm8k_passk.py",
    "src/inference_scaling/arllm/backends/batching.py",
)


from inference_scaling.arllm.backends.execution import ExecutionBackend as _MethodBackend
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.shared.generation import generation_config_for_prompt


_estimated_pass_at_k = estimated_pass_at_k


def _chunk_plan(
    methods: Sequence[str],
    draws: int,
    problem_indices: Sequence[int],
    chunk_size: int,
) -> dict[tuple[str, int], tuple[tuple[int, int], ...]]:
    if draws <= 0 or chunk_size <= 0 or not problem_indices:
        raise ValueError("draws, chunk size, and problem indices must be non-empty")
    plan: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {}
    for method in methods:
        chunk_index = 0
        for problem_index in problem_indices:
            tasks = [(draw, problem_index) for draw in range(draws)]
            for offset in range(0, len(tasks), chunk_size):
                plan[(method, chunk_index)] = tuple(
                    tasks[offset : offset + chunk_size]
                )
                chunk_index += 1
    return plan


def _prepare_manifest(
    *,
    config: dict[str, Any],
    data_path: Path,
    methods: Sequence[str],
    draws: int,
    workers: int,
    problem_indices: Sequence[int],
    input_weight_sha256: dict[str, str],
    implementation_sha256: dict[str, str],
    raw_path: Path,
    input_metadata_sha256: dict[str, Any] | None = None,
    input_adapter_sha256: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, Path]:
    effective = {
        "config": config,
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "methods": list(methods),
        "draws": draws,
        "workers": workers,
        "problem_indices": list(problem_indices),
        "input_weight_sha256": input_weight_sha256,
        "input_metadata_sha256": input_metadata_sha256 or {},
        "input_adapter_sha256": input_adapter_sha256 or {},
        "implementation_sha256": implementation_sha256,
    }
    fingerprint = _fingerprint(effective)
    manifest = {"schema_version": 1, "fingerprint": fingerprint, "effective": effective}
    manifest_path = raw_path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"{raw_path} belongs to a different pass@k grid; choose a new output"
            )
    elif raw_path.is_file():
        raise ValueError("pass@k chunks exist without their manifest")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest, fingerprint, manifest_path


def _validate_chunks(
    chunks: Sequence[dict[str, Any]],
    fingerprint: str,
    plan: dict[tuple[str, int], tuple[tuple[int, int], ...]],
) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in chunks:
        if chunk.get("manifest_fingerprint") != fingerprint:
            raise ValueError("pass@k chunk has the wrong manifest fingerprint")
        key = (str(chunk.get("method")), int(chunk.get("chunk_index", -1)))
        if key not in plan:
            raise ValueError(f"unexpected pass@k chunk: {key}")
        if key in completed:
            raise ValueError(f"duplicate pass@k chunk: {key}")
        task_keys = tuple(
            (int(record["draw_index"]), int(record["problem_index"]))
            for record in chunk.get("records", [])
        )
        if task_keys != plan[key]:
            raise ValueError(f"pass@k chunk {key} contains the wrong task grid")
        completed[key] = chunk
    return completed


def _run_chunk(
    *,
    method: str,
    chunk_index: int,
    task_keys: Sequence[tuple[int, int]],
    problems_by_index: dict[int, Any],
    prompts_by_index: dict[int, tuple[int, ...]],
    raw_backend: Any,
    config: dict[str, Any],
    workers: int,
    fingerprint: str,
) -> dict[str, Any]:
    before = raw_backend.snapshot()
    with ContinuousBatchingBackend(
        raw_backend,
        max_batch_size=int(config["runtime"]["max_batch_size"]),
        max_batch_tokens=int(config["runtime"]["max_batch_tokens"]),
        batch_wait_seconds=0.01,
    ) as batching:
        backend = _MethodBackend(batching, raw_backend)

        def run_one(task_key: tuple[int, int]):
            draw, problem_index = task_key
            problem = problems_by_index[problem_index]
            seeds = SeedStream(
                SeedStream(int(config["run"]["seed"])).derive("draw", draw)
            )
            tokens, diagnostics = _run_method(
                method,
                backend,
                problem,
                prompts_by_index[problem_index],
                config,
                seeds,
                None,
            )
            return tokens, diagnostics

        def run_parallel():
            if method != "mh" or config.get("output", {}).get("sampling_scope") == "thinking" or float(config.get("sampling", {}).get("temperature", 1.0)) != 1.0:
                with ThreadPoolExecutor(
                    max_workers=min(workers, len(task_keys))
                ) as executor:
                    return list(executor.map(run_one, task_keys))

            problem_indices = {problem_index for _, problem_index in task_keys}
            if len(problem_indices) != 1:
                raise ValueError("a vectorized MH chunk must contain one problem")
            problem_index = next(iter(problem_indices))
            prompt = prompts_by_index[problem_index]
            bounded_config, length_budget = generation_config_for_prompt(config, len(prompt), [raw_backend])
            section = bounded_config["mh"]
            alpha = float(section["alpha"])
            absorbing = AbsorbingEOSBackend(
                ScoreCachingBackend(backend),
                raw_backend.tokenizer.eos_token_id,
                absorbing_after=len(prompt),
            )
            chain_seeds = tuple(
                SeedStream(
                    SeedStream(int(config["run"]["seed"])).derive("draw", draw)
                )
                for draw, _ in task_keys
            )
            chain_seeds = tuple(
                SeedStream(stream.derive("mh", problem_index))
                for stream in chain_seeds
            )
            results = run_mh_chains_batched(
                absorbing,
                prompt,
                MHConfig(
                    alpha=alpha,
                    total_length=int(bounded_config["generation"]["max_new_tokens"]),
                    block_size=int(section["block_size"]),
                    steps_per_block=int(section["steps_per_block"]),
                    iterations=section.get("iterations"),
                    suffix_schedule=str(section.get("suffix_schedule", "uniform")),
                ),
                SamplingConfig(temperature=1.0 / alpha),
                chain_seeds,
            )
            return [
                (
                    _trim_eos(result.token_ids, raw_backend.tokenizer.eos_token_id),
                    {
                        "alpha": alpha,
                        "block_size": int(section["block_size"]),
                        "steps_per_block": int(section["steps_per_block"]),
                        "suffix_schedule": str(
                            section.get("suffix_schedule", "uniform")
                        ),
                        "attempts": result.attempts,
                        "accepted": result.accepted,
                        "acceptance_rate": result.acceptance_rate,
                        "execution": "lockstep_vectorized_independent_chains",
                        "generation_budget": length_budget,
                        "output_segments": SamplingScope.from_config(raw_backend, config, active=False).describe_output(raw_backend, prompt, _trim_eos(result.token_ids, raw_backend.tokenizer.eos_token_id)),
                    },
                )
                for result in results
            ]

        outputs, elapsed = _timed(run_parallel)
        batching_snapshot = asdict(batching.snapshot())
    after = raw_backend.snapshot()

    records = []
    for (draw, problem_index), (tokens, diagnostics) in zip(
        task_keys, outputs, strict=True
    ):
        problem = problems_by_index[problem_index]
        text = raw_backend.decode(tokens)
        prediction = extract_numeric_answer(diagnostics.get("output_segments", {}).get("content_text", text))
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
        "backend_delta": _snapshot_delta(before, after),
        "continuous_batching": batching_snapshot,
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
        model_key = "rl" if method == "rl_sample" else "base"
        adapter_base = None
        if model_key == "rl" and config["models"].get("rl_kind") == "peft_adapter":
            adapter_base = str(config["models"]["rl_base"])
        raw_backend = _load_backend(
            str(config["models"][model_key]), config, adapter_base=adapter_base, role=model_key
        )
        prompts_by_index = {
            index: _prompt_tokens(raw_backend, problem, config)
            for index, problem in problems_by_index.items()
        }
        first_prompt = prompts_by_index[next(iter(problems_by_index))]
        _sample_one(
            raw_backend,
            first_prompt,
            max_new_tokens=2,
            temperature=1.0,
            seed=int(config["run"]["seed"]),
            request_id=f"passk-warmup:{method}",
        )
        with raw_path.open("a", encoding="utf-8", buffering=1) as sink:
            for key in pending:
                chunk = _run_chunk(
                    method=method,
                    chunk_index=key[1],
                    task_keys=plan[key],
                    problems_by_index=problems_by_index,
                    prompts_by_index=prompts_by_index,
                    raw_backend=raw_backend,
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
        close_backend(raw_backend)
        del raw_backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _bootstrap_pass_at_k(
    per_problem: Sequence[dict[str, Any]],
    k: int,
    draws: int,
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float]:
    rng = random.Random(seed)
    values = [
        _estimated_pass_at_k(int(item["correct_draws"]), draws, k)
        for item in per_problem
    ]
    estimates = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    )
    return estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates)]


def _summarize_method(
    records: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    problem_indices: Sequence[int],
    draws: int,
    ks: Sequence[int],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (int(record["draw_index"]), int(record["problem_index"]))
        if key in by_key:
            raise ValueError(f"duplicate pass@k sample: {key}")
        by_key[key] = record
    expected = {
        (draw, problem_index)
        for draw in range(draws)
        for problem_index in problem_indices
    }
    if set(by_key) != expected:
        raise ValueError("pass@k samples do not cover the requested draw/problem grid")

    per_problem = []
    for problem_index in problem_indices:
        problem_records = [by_key[(draw, problem_index)] for draw in range(draws)]
        parsed = {
            record["prediction"]
            for record in problem_records
            if record["prediction"] is not None
        }
        per_problem.append(
            {
                "problem_index": problem_index,
                "correct_draws": sum(bool(record["correct"]) for record in problem_records),
                "unique_parsed_answers": len(parsed),
                "unique_full_outputs": len(
                    {record["output_sha256"] for record in problem_records}
                ),
                "unparseable_draws": sum(
                    record["prediction"] is None for record in problem_records
                ),
            }
        )

    pass_at_k = {}
    pass_at_k_bootstrap = {}
    for k in ks:
        values = [
            _estimated_pass_at_k(int(item["correct_draws"]), draws, k)
            for item in per_problem
        ]
        pass_at_k[str(k)] = statistics.fmean(values)
        pass_at_k_bootstrap[str(k)] = _bootstrap_pass_at_k(
            per_problem,
            k,
            draws,
            seed=SeedStream(bootstrap_seed).derive("pass-at-k", k),
        )

    total_samples = len(problem_indices) * draws
    base_slots = sum(
        int(chunk["backend_delta"]["generation_forward_token_slots"])
        + int(chunk["backend_delta"]["score_forward_token_slots"])
        for chunk in chunks
    )
    total_flops = sum(
        int(chunk["backend_delta"]["estimated_dense_forward_flops"])
        for chunk in chunks
    )
    seconds = sum(float(chunk["seconds_excluding_model_load"]) for chunk in chunks)
    return {
        "examples": len(problem_indices),
        "draws_per_example": draws,
        "generated_answers": total_samples,
        "single_draw_accuracy": sum(
            bool(record["correct"]) for record in records
        )
        / total_samples,
        "estimated_pass_at_k": pass_at_k,
        "estimated_pass_at_k_problem_bootstrap_95": pass_at_k_bootstrap,
        "mean_unique_parsed_answers_across_all_draws": statistics.fmean(
            int(item["unique_parsed_answers"]) for item in per_problem
        ),
        "mean_unique_full_outputs_across_all_draws": statistics.fmean(
            int(item["unique_full_outputs"]) for item in per_problem
        ),
        "unparseable_fraction": sum(
            int(item["unparseable_draws"]) for item in per_problem
        )
        / total_samples,
        "total_forward_token_slots": base_slots,
        "estimated_dense_forward_flops": total_flops,
        "estimated_dense_forward_petaflops": total_flops / 1e15,
        "estimated_dense_flops_per_generated_answer": total_flops / total_samples,
        "seconds_excluding_model_load": seconds,
        "seconds_per_generated_answer": seconds / total_samples,
        "continuous_batching": {
            "sample_batches": sum(
                int(chunk["continuous_batching"]["sample_batches"])
                for chunk in chunks
            ),
            "score_batches": sum(
                int(chunk["continuous_batching"]["score_batches"])
                for chunk in chunks
            ),
            "sample_requests": sum(
                int(chunk["continuous_batching"]["sample_requests"])
                for chunk in chunks
            ),
            "score_sequences": sum(
                int(chunk["continuous_batching"]["score_sequences"])
                for chunk in chunks
            ),
            "maximum_sample_batch": max(
                int(chunk["continuous_batching"]["maximum_sample_batch"])
                for chunk in chunks
            ),
            "maximum_score_batch": max(
                int(chunk["continuous_batching"]["maximum_score_batch"])
                for chunk in chunks
            ),
        },
        "per_problem": per_problem,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--tag", default="passk")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", default=",".join(PASSK_METHODS))
    parser.add_argument("--rl-adapter", type=Path)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument(
        "--mh-suffix-schedule",
        choices=("uniform", "inverse_length", "multiscale"),
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--output", type=Path)
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.draws <= 0 or args.limit <= 0 or args.workers <= 0:
        raise ValueError("draws, limit, and workers must be positive")

    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods) - set(PASSK_METHODS))
    if unknown:
        raise ValueError(f"unsupported pass@k methods: {', '.join(unknown)}")
    if len(methods) != len(set(methods)):
        raise ValueError("pass@k methods must not contain duplicates")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    apply_model_output_overrides(config, args)
    replace_verifier_from_file(config, args.verifier_config)
    if args.mh_suffix_schedule is not None:
        config["mh"]["suffix_schedule"] = args.mh_suffix_schedule
    set_backend_override(config, args.backend)
    set_rl_adapter_override(config, args.rl_adapter)
    config["run"]["sample_count"] = args.limit
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
        entrypoints=PASSK_IMPLEMENTATION_FILES,
    )
    roles = {"base"}
    if "rl_sample" in methods:
        roles.add("rl")
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
        raise RuntimeError(
            f"pass@k grid is incomplete: {len(completed)}/{len(plan)} chunks"
        )

    ks = tuple(k for k in (1, 2, 4, 8, 16, 32) if k <= args.draws)
    table = {}
    for method in methods:
        method_chunks = [
            completed[key] for key in plan if key[0] == method
        ]
        method_records = [
            record for chunk in method_chunks for record in chunk["records"]
        ]
        table[method] = _summarize_method(
            method_records,
            method_chunks,
            problem_indices,
            args.draws,
            ks,
            bootstrap_seed=SeedStream(int(config["run"]["seed"])).derive(method),
        )

    report = {
        "schema_version": 2,
        "benchmark": "OpenAI GSM8K official test split",
        "profile": profile,
        "methods": table,
        "problem_indices": problem_indices,
        "draws_per_problem": args.draws,
        "workers": args.workers,
        "manifest_fingerprint": fingerprint,
        "raw_chunks_sha256": _file_sha256(raw_path),
        "input_weight_sha256": input_weight_sha256,
        "input_metadata_sha256": input_artifacts["metadata_sha256"],
        "input_adapter_sha256": input_artifacts["adapter_sha256"],
        "implementation_sha256": implementation_sha256,
        "pass_at_k_definition": (
            "for n independent draws with c correct answers, each problem contributes "
            "1 - choose(n-c,k)/choose(n,k); the report averages this estimator over "
            "the same fixed public rows"
        ),
        "diversity_scope": (
            "pass@k is the primary diversity diagnostic; distinct parsed numeric "
            "answers and full-output hashes are supplemental and do not identify "
            "distinct reasoning paths"
        ),
        "compute_definition": (
            "2 * model parameter count * observed padded forward token slots; each "
            "method uses the same continuous-batching worker count, and model loading "
            "and warmup are excluded from wall time"
        ),
        "limitations": (
            "Eight draws estimate pass@k only through k=8. Problem bootstrap intervals "
            "do not include model, prompt, hyperparameter, or finite-chain mixing "
            "uncertainty. Batched CUDA execution can differ numerically from sequential "
            "execution even with request-local seeds."
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
