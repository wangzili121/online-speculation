"""Measure rollout-infrastructure ablations on one fixed public workload.

The benchmark keeps model, dtype, prompts, random seeds, output length, and
algorithm budgets fixed within each comparison.  Cache construction, online
evaluation, and background drain are timed and accounted separately so an
offline cost cannot disappear from the reported speedup.
"""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides, require_full_scope

import argparse
import copy
import gc
import json
import platform
import subprocess
import threading
import time
import tomllib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from inference_scaling.arllm.acceleration import (
    ActiveBatchSpeculationConfig,
    LowPriorityRunAheadBackend,
    RolloutTokenTree,
    SpeculationTier,
)
from inference_scaling.arllm.algorithms import run_conditional_is
from inference_scaling.experimental.arllm.progressive_is import (
    run_progressive_conditional_is,
)
from inference_scaling.experimental.arllm.smc_forest import run_smc_rollout_forest
from inference_scaling.arllm.backends import (
    TransformersBackend,
    close_backend,
    load_backend_from_config,
)
from inference_scaling.arllm.config import (
    ConditionalISConfig,
    ProgressiveISConfig,
    SMCForestConfig,
    SamplingConfig,
)
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import replace_verifier_from_file
from experiments.arllm.gsm8k_reproduction import _configured_verifier_reward
from inference_scaling.arllm.types import GenerationRequest, SequenceSample, TokenSequence


ROOT = Path(__file__).resolve().parents[2]


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _NvidiaMonitor:
    """Low-rate hardware telemetry; values include other desktop GPU users."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._rows: list[tuple[float, float, float]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read() -> tuple[float, float, float] | None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            first = completed.stdout.strip().splitlines()[0]
            utilization, memory, power = first.split(",")[:3]
            return float(utilization), float(memory), float(power)
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            row = self._read()
            if row is not None:
                self._rows.append(row)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "_NvidiaMonitor":
        self._thread.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, float | int | None]:
        if not self._rows:
            return {
                "samples": 0,
                "mean_gpu_utilization_percent": None,
                "maximum_gpu_memory_used_mib": None,
                "mean_gpu_power_watts": None,
            }
        return {
            "samples": len(self._rows),
            "mean_gpu_utilization_percent": sum(row[0] for row in self._rows)
            / len(self._rows),
            "maximum_gpu_memory_used_mib": max(row[1] for row in self._rows),
            "mean_gpu_power_watts": sum(row[2] for row in self._rows)
            / len(self._rows),
        }


def _measure(call: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _cuda_sync()
    with _NvidiaMonitor() as monitor:
        started = time.perf_counter()
        result = call()
        _cuda_sync()
        wall = time.perf_counter() - started
    telemetry = monitor.summary()
    telemetry.update(
        {
            "wall_seconds": wall,
            "torch_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else None
            ),
            "torch_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20
                if torch.cuda.is_available()
                else None
            ),
        }
    )
    return result, telemetry


def _numeric_snapshot(value: Any) -> dict[str, int | float]:
    if value is None:
        return {}
    raw = asdict(value) if is_dataclass(value) else dict(value)
    return {
        name: number
        for name, number in raw.items()
        if isinstance(number, (int, float)) and not isinstance(number, bool)
    }


def _delta(before: Any, after: Any) -> dict[str, int | float]:
    left = _numeric_snapshot(before)
    right = _numeric_snapshot(after)
    return {name: right[name] - left.get(name, 0) for name in right}


def _snapshot(backend: Any) -> Any:
    raw = backend.backend if isinstance(backend, LowPriorityRunAheadBackend) else backend
    return raw.snapshot()


def _draft_snapshot(backend: Any) -> Any:
    raw = backend.backend if isinstance(backend, LowPriorityRunAheadBackend) else backend
    callback = getattr(raw, "draft_cache_snapshot", None)
    return callback() if callable(callback) else None


def _prompt_tokens(tokenizer: Any, question: str, config: dict[str, Any] | None = None) -> TokenSequence:
    from inference_scaling.shared.prompting import render_prompt
    rendered = render_prompt(
        tokenizer,
        [{"role": "user", "content": gsm8k_prompt(question)}],
        config or {},
    )
    return tuple(
        int(token)
        for token in tokenizer.encode(str(rendered), add_special_tokens=False)
    )


def _speculation(
    *,
    maximum_batch: int,
    maximum_draft_tokens: int,
    dynamic: bool,
) -> ActiveBatchSpeculationConfig:
    if dynamic:
        # The manual Transformers verifier is most useful in the final single-
        # request tail; repeated-prefix batching remains better at BS >= 2.
        # vLLM consumes the same conservative schedule through its native
        # dynamic-speculation table.
        tiers = (
            SpeculationTier(1, maximum_draft_tokens),
            *(
                (SpeculationTier(maximum_batch, 0),)
                if maximum_batch > 1
                else ()
            ),
        )
    else:
        tiers = (SpeculationTier(maximum_batch, maximum_draft_tokens),)
    # Remove duplicate thresholds for small custom workloads.
    deduplicated: list[SpeculationTier] = []
    for tier in tiers:
        if deduplicated and tier.max_batch == deduplicated[-1].max_batch:
            deduplicated[-1] = tier
        else:
            deduplicated.append(tier)
    return ActiveBatchSpeculationConfig(
        tiers=tuple(deduplicated),
        min_context_tokens=4,
        min_token_probability=0.1,
        tree_max_context_tokens=48,
        tree_max_contexts=200_000,
        vllm_max_cached_requests=10_000,
    )


class _BackendFactory:
    def __init__(self, config: dict[str, Any], backend_kind: str, dtype: str) -> None:
        self.config = copy.deepcopy(config)
        self.backend_kind = backend_kind
        self.config.setdefault("runtime", {})["backend"] = backend_kind
        self.config["runtime"]["dtype"] = dtype
        self._shared: TransformersBackend | None = None
        if backend_kind == "transformers":
            self._shared = load_backend_from_config(
                str(self.config["models"]["base"]), self.config
            )
            self.tokenizer = self._shared.tokenizer
        else:
            from transformers import AutoTokenizer

            path = str(self.config["models"]["base"])
            from inference_scaling.shared.model_loading import model_loading_options
            options = model_loading_options(self.config, "base")
            self.tokenizer = AutoTokenizer.from_pretrained(
                options.get("tokenizer_name_or_path", path),
                revision=options.get("tokenizer_revision", options.get("revision") if "tokenizer_name_or_path" not in options else None),
                cache_dir=options.get("cache_dir"), local_files_only=options["local_files_only"],
                trust_remote_code=options["trust_remote_code"], **options.get("tokenizer_kwargs", {}),
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def create(
        self,
        speculation: ActiveBatchSpeculationConfig | None,
        *,
        dynamic_vllm: bool = False,
    ) -> Any:
        if self._shared is not None:
            tree = RolloutTokenTree.from_config(speculation) if speculation else None
            return TransformersBackend(
                self._shared.model,
                self._shared.tokenizer,
                model_id=self._shared.model_id,
                device=self._shared.device,
                max_score_batch_size=self._shared.max_score_batch_size,
                score_chunk_size=self._shared.score_chunk_size,
                draft_tree=tree,
                speculation=speculation,
            )
        arm = copy.deepcopy(self.config)
        if speculation is not None:
            arm["acceleration"] = {
                "speculation": {
                    "enabled": True,
                    "dynamic_vllm": dynamic_vllm,
                    "tiers": [
                        [tier.max_batch, tier.draft_tokens]
                        for tier in speculation.tiers
                    ],
                    "min_context_tokens": speculation.min_context_tokens,
                    "min_token_probability": speculation.min_token_probability,
                    "tree_max_context_tokens": speculation.tree_max_context_tokens,
                    "tree_max_contexts": speculation.tree_max_contexts,
                    "vllm_max_cached_requests": speculation.vllm_max_cached_requests,
                }
            }
        return load_backend_from_config(str(arm["models"]["base"]), arm)

    def release(self, backend: Any) -> None:
        if isinstance(backend, LowPriorityRunAheadBackend):
            backend.close()
            backend = backend.backend
        if self.backend_kind != "transformers":
            close_backend(backend)
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> None:
        if self._shared is not None:
            close_backend(self._shared)


def _warmup(backend: Any, tokenizer: Any, eos: int | None) -> None:
    prefix = tuple(
        int(token)
        for token in tokenizer.encode(
            "Kernel warmup: unrelated fixed text 91357.", add_special_tokens=True
        )
    )
    backend.sample_batch(
        [
            GenerationRequest(
                prefix,
                4,
                SamplingConfig(eos_token_id=eos),
                731,
                "infra:warmup",
            )
        ]
    )


def _history_batches(
    prompts: Sequence[TokenSequence],
    *,
    count: int,
    length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
) -> list[list[GenerationRequest]]:
    return [
        [
            GenerationRequest(
                prompt,
                length,
                sampling,
                seeds.derive("infra", "history", prompt_index, draw),
                f"infra:history:prompt:{prompt_index}:draw:{draw}",
            )
            for draw in range(count)
        ]
        for prompt_index, prompt in enumerate(prompts)
    ]


def _evaluation_batches(
    prompts: Sequence[TokenSequence],
    *,
    batch_sizes: Sequence[int],
    length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
) -> list[list[GenerationRequest]]:
    batches: list[list[GenerationRequest]] = []
    for prompt_index, prompt in enumerate(prompts):
        for stage, batch_size in enumerate(batch_sizes):
            batches.append(
                [
                    GenerationRequest(
                        prompt,
                        length,
                        sampling,
                        seeds.derive(
                            "infra", "evaluation", prompt_index, stage, draw
                        ),
                        (
                            f"infra:evaluation:prompt:{prompt_index}:"
                            f"stage:{stage}:draw:{draw}"
                        ),
                    )
                    for draw in range(batch_size)
                ]
            )
    return batches


def _run_batches(
    backend: Any, batches: Sequence[Sequence[GenerationRequest]]
) -> list[SequenceSample]:
    outputs: list[SequenceSample] = []
    for batch in batches:
        outputs.extend(backend.sample_batch(batch))
    return outputs


def _quality(
    backend: Any,
    samples: Sequence[SequenceSample],
    gold_by_prompt: dict[TokenSequence, Any],
) -> dict[str, Any]:
    predictions = [extract_numeric_answer(backend.decode(sample.token_ids)) for sample in samples]
    correct = [
        prediction == gold_by_prompt[sample.prefix]
        for prediction, sample in zip(predictions, samples, strict=True)
    ]
    return {
        "sequences": len(samples),
        "output_tokens": sum(len(sample.token_ids) for sample in samples),
        "parseable_fraction": (
            sum(value is not None for value in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
    }


def _decode_arm(
    factory: _BackendFactory,
    *,
    name: str,
    speculation: ActiveBatchSpeculationConfig | None,
    dynamic_vllm: bool,
    history_batches: Sequence[Sequence[GenerationRequest]],
    evaluation_batches: Sequence[Sequence[GenerationRequest]],
    gold_by_prompt: dict[TokenSequence, Any],
) -> tuple[dict[str, Any], dict[str, TokenSequence]]:
    backend = factory.create(speculation, dynamic_vllm=dynamic_vllm)
    try:
        eos = backend.tokenizer.eos_token_id
        _warmup(backend, backend.tokenizer, eos)
        cache_before = _snapshot(backend)
        if speculation is None:
            cache_samples: list[SequenceSample] = []
            cache_telemetry = {"wall_seconds": 0.0}
        else:
            cache_samples, cache_telemetry = _measure(
                lambda: _run_batches(backend, history_batches)
            )
        cache_after = _snapshot(backend)
        draft_before = _draft_snapshot(backend)
        online_before = _snapshot(backend)
        samples, online_telemetry = _measure(
            lambda: _run_batches(backend, evaluation_batches)
        )
        online_after = _snapshot(backend)
        draft_after = _draft_snapshot(backend)
        quality = _quality(backend, samples, gold_by_prompt)
        quality["output_tokens_per_second"] = (
            quality["output_tokens"] / online_telemetry["wall_seconds"]
            if online_telemetry["wall_seconds"]
            else 0.0
        )
        return (
            {
                "name": name,
                "backend": factory.backend_kind,
                "speculation_tiers": (
                    []
                    if speculation is None
                    else [asdict(tier) for tier in speculation.tiers]
                ),
                "cache_build": {
                    "telemetry": cache_telemetry,
                    "main_model": _delta(cache_before, cache_after),
                    "sequences": len(cache_samples),
                    "tokens": sum(len(sample.token_ids) for sample in cache_samples),
                },
                "online": {
                    "telemetry": online_telemetry,
                    "main_model": _delta(online_before, online_after),
                    "draft_cache": _delta(draft_before, draft_after),
                    "quality": quality,
                },
            },
            {sample.request_id: sample.token_ids for sample in samples},
        )
    finally:
        factory.release(backend)


def _algorithm_arm(
    factory: _BackendFactory,
    *,
    name: str,
    speculation: ActiveBatchSpeculationConfig,
    problems: Sequence[Any],
    prompts: Sequence[TokenSequence],
    history_batches: Sequence[Sequence[GenerationRequest]],
    maximum: int,
    block_size: int,
    candidate_count: int,
    rollout_count: int,
    particle_count: int,
    branch_factor: int,
    seed: int,
) -> dict[str, Any]:
    # Keep algorithm-level comparisons independent of vLLM's experimental
    # dynamic batch table.  The decode section measures that variable directly.
    raw = factory.create(speculation, dynamic_vllm=False)
    run_ahead: LowPriorityRunAheadBackend | None = None
    backend: Any = raw
    if name == "progressive_streaming_runahead":
        run_ahead = LowPriorityRunAheadBackend(
            raw,
            None,
            chunk_tokens=block_size,
            outputs_already_observed=True,
        )
        backend = run_ahead
    try:
        _warmup(backend, raw.tokenizer, raw.tokenizer.eos_token_id)
        cache_before = _snapshot(backend)
        cache_samples, cache_telemetry = _measure(
            lambda: _run_batches(backend, history_batches)
        )
        cache_after = _snapshot(backend)
        online_before = _snapshot(backend)
        draft_before = _draft_snapshot(backend)
        sampling = SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
        diagnostics: list[dict[str, Any]] = []

        def run_all() -> list[TokenSequence]:
            outputs: list[TokenSequence] = []
            for problem, prompt in zip(problems, prompts, strict=True):
                verifier_reward = _configured_verifier_reward(
                    raw, problem, factory.config
                )

                problem_seed = SeedStream(seed).derive(
                    "infra", "algorithm", name, problem.index
                )
                if name == "conditional_fixed":
                    result = run_conditional_is(
                        backend,
                        prompt,
                        ConditionalISConfig(
                            candidate_count=candidate_count,
                            rollout_count=rollout_count,
                            block_size=block_size,
                            total_length=maximum,
                            reward_temperature=0.1,
                        ),
                        verifier_reward,
                        SeedStream(problem_seed),
                        base_sampling=sampling,
                    )
                    diagnostics.append(
                        {
                            "steps": len(result.steps),
                            "evaluation_rollouts": sum(
                                len(candidate.rollouts)
                                for step in result.steps
                                for candidate in step.candidates
                            ),
                        }
                    )
                elif name in {"progressive", "progressive_streaming_runahead"}:
                    result = run_progressive_conditional_is(
                        backend,
                        prompt,
                        ProgressiveISConfig(
                            candidate_count=candidate_count,
                            pilot_rollouts_per_candidate=1,
                            evaluation_cost_budget=1.0,
                            evaluation_reference_rollouts_per_candidate=max(
                                1, rollout_count - 1
                            ),
                            minimum_evaluation_per_candidate=1,
                            block_size=block_size,
                            total_length=maximum,
                            reward_temperature=0.1,
                            reward_workers=4,
                            run_ahead_rollouts_per_candidate=(
                                1 if run_ahead is not None else 0
                            ),
                        ),
                        verifier_reward,
                        SeedStream(problem_seed),
                        base_sampling=sampling,
                        streaming_rewards=(name == "progressive_streaming_runahead"),
                    )
                    diagnostics.append(
                        {
                            "steps": len(result.steps),
                            "pilot_rollouts": sum(
                                candidate.pilot.rollout_count
                                for step in result.steps
                                for candidate in step.candidates
                            ),
                            "evaluation_rollouts": sum(
                                candidate.evaluation_count
                                for step in result.steps
                                for candidate in step.candidates
                            ),
                            "frozen_evaluation_cost": sum(
                                step.frozen_evaluation_cost for step in result.steps
                            ),
                            "reward_tail_seconds": sum(
                                candidate.pilot.streaming.reward_tail_seconds
                                for step in result.steps
                                for candidate in step.candidates[:1]
                                if candidate.pilot.streaming is not None
                            ),
                        }
                    )
                elif name in {"smc_no_reuse", "smc_reuse"}:
                    result = run_smc_rollout_forest(
                        backend,
                        prompt,
                        SMCForestConfig(
                            particle_count=particle_count,
                            branch_factor=branch_factor,
                            rollout_count=rollout_count,
                            block_size=block_size,
                            total_length=maximum,
                            reward_temperature=0.1,
                            reuse_rollout_forest=name == "smc_reuse",
                        ),
                        verifier_reward,
                        SeedStream(problem_seed),
                        base_sampling=sampling,
                        streaming_rewards=True,
                    )
                    diagnostics.append(
                        {
                            "steps": len(result.steps),
                            "fresh_rollouts": result.fresh_rollouts,
                            "reused_rollouts": result.reused_rollouts,
                            "mean_effective_sample_size": (
                                sum(step.effective_sample_size for step in result.steps)
                                / len(result.steps)
                            ),
                        }
                    )
                else:
                    raise ValueError(f"unknown algorithm arm {name!r}")
                outputs.append(result.token_ids)
            return outputs

        outputs, online_telemetry = _measure(run_all)
        online_after = _snapshot(backend)
        draft_after = _draft_snapshot(backend)
        drain_telemetry: dict[str, Any] = {"wall_seconds": 0.0}
        drain_main_model: dict[str, int | float] = {}
        run_ahead_snapshot = None
        if run_ahead is not None:
            drain_before = _snapshot(backend)
            _, drain_telemetry = _measure(run_ahead.wait_for_run_ahead)
            drain_after = _snapshot(backend)
            drain_main_model = _delta(drain_before, drain_after)
            run_ahead_snapshot = asdict(run_ahead.snapshot())
        predictions = [extract_numeric_answer(raw.decode(tokens)) for tokens in outputs]
        correct = [
            prediction == problem.gold_answer
            for prediction, problem in zip(predictions, problems, strict=True)
        ]
        output_tokens = sum(len(tokens) for tokens in outputs)
        return {
            "name": name,
            "backend": factory.backend_kind,
            "cache_build": {
                "telemetry": cache_telemetry,
                "main_model": _delta(cache_before, cache_after),
                "sequences": len(cache_samples),
                "tokens": sum(len(sample.token_ids) for sample in cache_samples),
            },
            "online": {
                "telemetry": online_telemetry,
                "main_model": _delta(online_before, online_after),
                "draft_cache": _delta(draft_before, draft_after),
                "problems": len(problems),
                "output_tokens": output_tokens,
                "output_tokens_per_second": (
                    output_tokens / online_telemetry["wall_seconds"]
                    if online_telemetry["wall_seconds"]
                    else 0.0
                ),
                "parseable_fraction": (
                    sum(value is not None for value in predictions) / len(predictions)
                    if predictions
                    else 0.0
                ),
                "accuracy": sum(correct) / len(correct) if correct else 0.0,
                "diagnostics": diagnostics,
            },
            "background_drain": {
                "telemetry": drain_telemetry,
                "main_model": drain_main_model,
                "run_ahead": run_ahead_snapshot,
            },
        }
    finally:
        if run_ahead is not None:
            run_ahead.close()
        factory.release(raw)


def _machine() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_total_memory_mib": (
            torch.cuda.get_device_properties(0).total_memory / 2**20 if cuda else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/gsm8k_3090_aligned.toml"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("transformers", "vllm", "vllm-sync"), default="transformers")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--section", choices=("decode", "algorithm", "all"), default="all")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--history-rollouts", type=int, default=2)
    parser.add_argument("--batch-sizes", default="4,2,1")
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--rollout-count", type=int, default=2)
    parser.add_argument("--particle-count", type=int, default=3)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--verifier-config", type=Path)
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.rollout_count < 2:
        raise ValueError("rollout-count must be at least two for a cost-matched pilot split")
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch-sizes must contain positive integers")
    if args.max_new_tokens % args.block_size:
        raise ValueError("max-new-tokens must be divisible by block-size")

    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    apply_model_output_overrides(config, args)
    require_full_scope(config, "benchmark_rollout_infra")
    replace_verifier_from_file(config, args.verifier_config)
    problems = select_problems(
        load_gsm8k(args.data), args.limit, seed=int(config["run"]["subset_seed"])
    )
    factory = _BackendFactory(config, args.backend, args.dtype)
    try:
        prompts = tuple(
            _prompt_tokens(factory.tokenizer, problem.question, config) for problem in problems
        )
        gold_by_prompt = {
            prompt: problem.gold_answer
            for prompt, problem in zip(prompts, problems, strict=True)
        }
        sampling = SamplingConfig(eos_token_id=factory.tokenizer.eos_token_id)
        histories = _history_batches(
            prompts,
            count=args.history_rollouts,
            length=args.max_new_tokens,
            sampling=sampling,
            seeds=SeedStream(args.seed),
        )
        evaluations = _evaluation_batches(
            prompts,
            batch_sizes=batch_sizes,
            length=args.max_new_tokens,
            sampling=sampling,
            seeds=SeedStream(args.seed),
        )
        static_speculation = _speculation(
            maximum_batch=max(batch_sizes),
            maximum_draft_tokens=args.draft_tokens,
            dynamic=False,
        )
        dynamic_speculation = _speculation(
            maximum_batch=max(batch_sizes),
            maximum_draft_tokens=args.draft_tokens,
            dynamic=True,
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "created_at_unix": time.time(),
            "machine": _machine(),
            "setting": {
                "backend": args.backend,
                "dtype": args.dtype,
                "dataset": "pinned OpenAI GSM8K test split",
                "problem_indices": [problem.index for problem in problems],
                "model": str(config["models"]["base"]),
                "max_new_tokens": args.max_new_tokens,
                "block_size": args.block_size,
                "history_rollouts_per_prompt": args.history_rollouts,
                "active_batch_sequence": list(batch_sizes),
                "candidate_count": args.candidate_count,
                "rollout_count": args.rollout_count,
                "particle_count": args.particle_count,
                "branch_factor": args.branch_factor,
                "seed": args.seed,
                "verifier": factory.config["verifier"],
            },
            "decode_arms": [],
            "algorithm_arms": [],
            "accounting": {
                "main_model_flops": "2 * parameter_count * measured target forward token slots",
                "vllm_speculation": (
                    "rejected verified draft tokens are added from native vLLM counters"
                ),
                "exclusions": [
                    "attention sequence-length term",
                    "draft-tree CPU work",
                    "sampling and reward CPU work",
                    "model loading",
                ],
            },
        }
        if args.section in {"decode", "all"}:
            traces: dict[str, dict[str, TokenSequence]] = {}
            for name, spec, dynamic_vllm in (
                ("baseline", None, False),
                ("history_tree_static", static_speculation, False),
                ("history_tree_load_aware", dynamic_speculation, True),
            ):
                arm, trace = _decode_arm(
                    factory,
                    name=name,
                    speculation=spec,
                    dynamic_vllm=dynamic_vllm,
                    history_batches=histories,
                    evaluation_batches=evaluations,
                    gold_by_prompt=gold_by_prompt,
                )
                traces[name] = trace
                report["decode_arms"].append(arm)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            baseline = traces["baseline"]
            for arm in report["decode_arms"]:
                trace = traces[arm["name"]]
                common = sorted(set(baseline).intersection(trace))
                arm["online"]["exact_token_trace_match_fraction_vs_baseline"] = (
                    sum(trace[key] == baseline[key] for key in common) / len(common)
                    if common
                    else None
                )
        if args.section in {"algorithm", "all"}:
            for name in (
                "conditional_fixed",
                "progressive",
                "progressive_streaming_runahead",
                "smc_no_reuse",
                "smc_reuse",
            ):
                report["algorithm_arms"].append(
                    _algorithm_arm(
                        factory,
                        name=name,
                        speculation=dynamic_speculation,
                        problems=problems,
                        prompts=prompts,
                        history_batches=histories,
                        maximum=args.max_new_tokens,
                        block_size=args.block_size,
                        candidate_count=args.candidate_count,
                        rollout_count=args.rollout_count,
                        particle_count=args.particle_count,
                        branch_factor=args.branch_factor,
                        seed=args.seed,
                    )
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        factory.close()


if __name__ == "__main__":
    main()
