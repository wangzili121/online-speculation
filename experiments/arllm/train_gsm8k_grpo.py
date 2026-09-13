"""Train a resumable GRPO LoRA on the pinned public GSM8K training split.

The script deliberately uses the same prompt builder and numeric parser as the
inference benchmarks.  It also records wall time, generated rollout tokens,
peak CUDA memory, and sampled GPU power integral so training can be compared with the
per-query cost of MH and conditional importance sampling.
"""

from __future__ import annotations

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides

import argparse
import json
import os
import platform
import subprocess
import threading
import time
import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from inference_scaling.shared.model_loading import model_loading_options, resolve_checkpoint_path
from trl import GRPOConfig, GRPOTrainer

from inference_scaling.shared.compute import (
    estimate_grpo_compute,
    estimate_grpo_compute_from_logs,
)
from inference_scaling.shared.evaluation import (
    GSM8K_TEST_SHA256,
    GSM8K_TRAIN_SHA256,
    gsm8k_prompt,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.verifier import (
    ConfiguredTrainingVerifierReward,
    replace_verifier_from_file,
    verifier_spec_from_config,
)

from experiments.shared.artifacts import (
    checkpoint_metadata_hashes,
    checkpoint_weight_hashes,
    weight_manifest_digest,
    implementation_hashes,
    json_fingerprint,
)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass
class NvidiaPowerMonitor:
    interval_seconds: float
    samples: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _started: float | None = field(default=None, init=False)

    def _sample(self) -> None:
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ]
            output = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(1.0, self.interval_seconds),
            ).stdout.strip().splitlines()[0]
            power, utilization, memory, temperature = (
                float(value.strip()) for value in output.split(",")
            )
            assert self._started is not None
            self.samples.append(
                {
                    "seconds": time.perf_counter() - self._started,
                    "power_watts": power,
                    "utilization_percent": utilization,
                    "memory_mib": memory,
                    "temperature_c": temperature,
                }
            )
        except Exception as caught:  # monitoring must never abort training
            self.error = f"{type(caught).__name__}: {caught}"

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2.0)
        self._sample()
        power_integral_wh = 0.0
        for left, right in zip(self.samples, self.samples[1:]):
            duration_hours = (right["seconds"] - left["seconds"]) / 3600.0
            power_integral_wh += duration_hours * (
                left["power_watts"] + right["power_watts"]
            ) / 2.0
        return {
            "sample_interval_seconds": self.interval_seconds,
            "samples": len(self.samples),
            "gpu_power_integral_wh": power_integral_wh,
            "mean_power_watts": (
                sum(sample["power_watts"] for sample in self.samples) / len(self.samples)
                if self.samples
                else None
            ),
            "peak_power_watts": (
                max(sample["power_watts"] for sample in self.samples)
                if self.samples
                else None
            ),
            "mean_utilization_percent": (
                sum(sample["utilization_percent"] for sample in self.samples)
                / len(self.samples)
                if self.samples
                else None
            ),
            "peak_nvidia_smi_memory_mib": (
                max(sample["memory_mib"] for sample in self.samples)
                if self.samples
                else None
            ),
            "peak_temperature_c": (
                max(sample["temperature_c"] for sample in self.samples)
                if self.samples
                else None
            ),
            "monitor_error": self.error,
        }


def _latest_checkpoint(output: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in output.glob("checkpoint-*"):
        if path.is_dir():
            try:
                checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
            except ValueError:
                continue
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _resume_fingerprint(effective: dict[str, Any]) -> str:
    stable = json.loads(json.dumps(effective))
    for key in ("max_steps", "save_steps", "logging_steps"):
        stable["training"].pop(key, None)
    return json_fingerprint(stable)


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    replace_verifier_from_file(config, getattr(args, "verifier_config", None))
    if args.output_dir is not None:
        config["model"]["output"] = str(args.output_dir)
    if args.train_limit is not None:
        config["data"]["train_count"] = args.train_limit
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.num_generations is not None:
        config["training"]["num_generations"] = args.num_generations
        config["training"]["per_device_train_batch_size"] = args.num_generations
        config["training"]["generation_batch_size"] = args.num_generations
    if args.max_completion_length is not None:
        config["training"]["max_completion_length"] = args.max_completion_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_grpo.toml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or an explicit checkpoint directory",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    apply_model_output_overrides(config, args)
    _apply_overrides(config, args)
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires CUDA in this reproduction")

    loading = model_loading_options(config, "base")
    loading.setdefault("revision", config["model"].get("revision"))
    base_path = resolve_checkpoint_path(
        str(config["model"]["base"]), revision=loading.get("revision"),
        cache_dir=loading.get("cache_dir"), local_files_only=loading["local_files_only"],
    )
    weight_files = checkpoint_weight_hashes(
        base_path, cache_directory=Path(__file__).resolve().parents[2] / ".cache" / "artifact_hashes",
    )
    actual_weight_hash = weight_manifest_digest(weight_files)
    expected_weight_hash = config["model"].get("weight_sha256")
    if expected_weight_hash is not None and expected_weight_hash != actual_weight_hash:
        raise ValueError("training checkpoint weight hash mismatch")

    train_path = Path(config["data"]["train"])
    test_path = Path(config["data"]["test"])
    train_problems = load_gsm8k(train_path, split="train")
    test_problems = load_gsm8k(test_path, split="test")
    train_questions = {problem.question for problem in train_problems}
    test_questions = {problem.question for problem in test_problems}
    overlap = train_questions & test_questions
    if overlap:
        raise ValueError(f"GSM8K train/test leakage: {len(overlap)} exact questions overlap")

    train_count = int(config["data"]["train_count"])
    selected = select_problems(
        train_problems,
        train_count,
        seed=int(config["run"]["seed"]),
    )
    rows = [
        {
            "prompt": [{"role": "user", "content": gsm8k_prompt(problem.question)}],
            "gold_answer": _fraction_text(problem.gold_answer),
            "problem_index": problem.index,
        }
        for problem in selected
    ]
    dataset = Dataset.from_list(rows)

    training = config["training"]
    output = Path(config["model"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    effective = {
        "run": config["run"],
        "data": {**config["data"], "selected_indices": [problem.index for problem in selected]},
        "model": config["model"],
        "model_loading": loading,
        "input_shard_sha256": weight_files,
        "training": training,
        "lora": config["lora"],
        "verifier": verifier_spec_from_config(config).as_dict(),
        "input_weight_sha256": actual_weight_hash,
        "input_metadata_sha256": checkpoint_metadata_hashes(base_path),
        "implementation_sha256": implementation_hashes(
            Path(__file__).resolve().parents[2],
            entrypoints=(
                "experiments/arllm/train_gsm8k_grpo.py",
                "src/inference_scaling/shared/verifier.py",
                "src/inference_scaling/shared/evaluation/numeric.py",
            ),
        ),
    }
    fingerprint = _resume_fingerprint(effective)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["resume_fingerprint"] != fingerprint:
            raise ValueError(
                f"{output} was created with different data or hyperparameters; "
                "choose a new --output-dir"
            )

    manifest = {
        "schema_version": 1,
        "status": "initializing",
        "resume_fingerprint": fingerprint,
        "effective": effective,
        "dataset": {
            "name": "OpenAI GSM8K",
            "train_rows": len(train_problems),
            "selected_train_rows": len(selected),
            "test_rows_reserved_for_evaluation": len(test_problems),
            "exact_question_overlap": len(overlap),
            "train_sha256": GSM8K_TRAIN_SHA256,
            "test_sha256": GSM8K_TEST_SHA256,
        },
        "base_model": {
            "source": config["model"].get("source", config["model"]["base"]),
            "revision": loading.get("revision"),
            "weight_sha256": actual_weight_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "git": _git_metadata(),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        loading.get("tokenizer_name_or_path", base_path),
        revision=loading.get("tokenizer_revision"), cache_dir=loading.get("cache_dir"),
        local_files_only=loading["local_files_only"], trust_remote_code=loading["trust_remote_code"],
        **loading.get("tokenizer_kwargs", {}),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=lora["target_modules"] if isinstance(lora["target_modules"], str) else list(lora["target_modules"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    grpo_config = GRPOConfig(
        output_dir=str(output),
        run_name=str(config["run"]["name"]),
        seed=int(config["run"]["seed"]),
        data_seed=int(config["run"]["seed"]),
        max_steps=int(training["max_steps"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        generation_batch_size=int(training["generation_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        num_generations=int(training["num_generations"]),
        max_completion_length=int(training["max_completion_length"]),
        learning_rate=float(training["learning_rate"]),
        beta=float(training["beta"]),
        epsilon=float(training["epsilon"]),
        temperature=float(training["temperature"]),
        top_p=float(training["top_p"]),
        loss_type=str(training["loss_type"]),
        scale_rewards=str(training["scale_rewards"]),
        warmup_steps=int(training["warmup_steps"]),
        max_grad_norm=float(training["max_grad_norm"]),
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        report_to="none",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=0,
        model_init_kwargs={
            **loading.get("model_kwargs", {}),
            "dtype": str(config["model"].get("dtype", "bfloat16")),
            "attn_implementation": loading.get("attn_implementation", config["model"].get("attn_implementation", "sdpa")),
            "local_files_only": True,
            "trust_remote_code": loading["trust_remote_code"],
        },
    )
    reward = ConfiguredTrainingVerifierReward(
        verifier_spec_from_config(config),
        reference_field="gold_answer",
    )
    initialization_started = time.perf_counter()
    trainer = GRPOTrainer(
        model=str(base_path),
        reward_funcs=reward,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())

    if args.resume == "none":
        checkpoint = None
    elif args.resume == "auto":
        checkpoint = _latest_checkpoint(output)
    else:
        checkpoint = Path(args.resume)
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
    previous_cost_path = output / "training_cost.json"
    previous_cost = (
        json.loads(previous_cost_path.read_text(encoding="utf-8"))
        if checkpoint is not None and previous_cost_path.is_file()
        else None
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = NvidiaPowerMonitor(float(training["power_sample_seconds"]))
    monitor.start()
    training_started = time.perf_counter()
    manifest["status"] = "training"
    manifest["resume_from_checkpoint"] = str(checkpoint) if checkpoint else None
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        train_result = trainer.train(
            resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None
        )
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["elapsed_before_interruption_seconds"] = (
            time.perf_counter() - training_started
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        power = monitor.stop()
    training_seconds = time.perf_counter() - training_started

    trainer.save_model(str(output))
    trainer.save_state()
    tokenizer.save_pretrained(output)
    state = trainer.state
    segment_rollouts = reward.snapshot(num_generations=int(training["num_generations"]))
    previous_rollouts = previous_cost.get("rollouts", {}) if previous_cost else {}
    cumulative_rollouts: dict[str, Any] = {
        key: int(segment_rollouts[key]) + int(previous_rollouts.get(key, 0))
        for key in (
            "reward_calls",
            "generated_completions",
            "generated_prompt_groups",
            "generated_completion_tokens",
        )
    }
    cumulative_rollouts["reward_sum"] = float(segment_rollouts["reward_sum"]) + float(
        previous_rollouts.get("reward_sum", 0.0)
    )
    generated_completions = int(cumulative_rollouts["generated_completions"])
    cumulative_rollouts["observed_mean_reward"] = (
        float(cumulative_rollouts["reward_sum"]) / generated_completions
        if generated_completions
        else None
    )
    for aggregate, reducer in (
        ("observed_minimum_reward", min),
        ("observed_maximum_reward", max),
    ):
        values = [
            float(value)
            for value in (
                previous_rollouts.get(aggregate),
                segment_rollouts.get(aggregate),
            )
            if value is not None
        ]
        cumulative_rollouts[aggregate] = reducer(values) if values else None
    previous_wall_seconds = (
        float(
            previous_cost.get(
                "cumulative_training_wall_seconds",
                previous_cost.get("training_wall_seconds", 0.0),
            )
        )
        if previous_cost
        else 0.0
    )
    previous_power_integral_wh = (
        float(
            previous_cost.get(
                "cumulative_gpu_power_integral_wh",
                previous_cost.get("gpu_monitor", {}).get("gpu_power_integral_wh", 0.0),
            )
        )
        if previous_cost
        else 0.0
    )
    trainer_reported_model_tokens = max(
        (
            int(entry["num_tokens"])
            for entry in state.log_history
            if "num_tokens" in entry
        ),
        default=0,
    )
    try:
        primary_compute = estimate_grpo_compute_from_logs(
            log_history=state.log_history,
            sequences_per_optimizer_step=(
                int(training["per_device_train_batch_size"])
                * int(training["gradient_accumulation_steps"])
            ),
            generated_completions=generated_completions,
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            optimizer_steps=int(state.global_step),
            gradient_checkpointing=bool(grpo_config.gradient_checkpointing),
            reference_scoring=float(training["beta"]) != 0,
        ).as_dict()
    except ValueError:
        primary_compute = (
            estimate_grpo_compute(
                model_sequence_tokens=trainer_reported_model_tokens,
                generated_completions=generated_completions,
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                optimizer_steps=int(state.global_step),
                gradient_checkpointing=bool(grpo_config.gradient_checkpointing),
                reference_scoring=float(training["beta"]) != 0,
            ).as_dict()
            if trainer_reported_model_tokens
            >= generated_completions
            else None
        )
    cost = {
        "schema_version": 3,
        "status": "complete",
        "resume_fingerprint": fingerprint,
        "initialization_seconds_excluded_from_training_cost": initialization_seconds,
        "training_wall_seconds": training_seconds,
        "cumulative_training_wall_seconds": previous_wall_seconds + training_seconds,
        "cumulative_gpu_power_integral_wh": previous_power_integral_wh
        + float(power.get("gpu_power_integral_wh", 0.0)),
        "global_step": state.global_step,
        "epoch": state.epoch,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "trainer_metrics": _jsonable(train_result.metrics),
        "trainer_reported_model_tokens": trainer_reported_model_tokens,
        "primary_compute": primary_compute,
        "segment_rollouts": segment_rollouts,
        "rollouts": cumulative_rollouts,
        "gpu_monitor": power,
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "effective": effective,
        "environment": manifest["environment"],
    }
    (output / "training_cost.json").write_text(
        json.dumps(cost, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["status"] = "complete"
    manifest["global_step"] = state.global_step
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(cost, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
