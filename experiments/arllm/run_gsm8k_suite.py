"""Resumable orchestrator for the aligned GSM8K experiment matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides, model_output_cli_arguments
from inference_scaling.arllm.backends import BACKEND_CHOICES
from experiments.shared.methods import AR_DEFAULT_METHODS, AR_METHODS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = AR_DEFAULT_METHODS
SUPPORTED_METHODS = AR_METHODS
_CLI_OVERRIDES: list[str] = []


def _run(command: list[str], environment: dict[str, str]) -> None:
    script = Path(command[1]).name
    if not script.startswith(("summarize_", "plot_", "render_")):
        command[2:2] = _CLI_OVERRIDES
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--rl-adapter", type=Path)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument(
        "--mh-suffix-schedule",
        choices=("uniform", "inverse_length", "multiscale"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--with-replay", action="store_true")
    parser.add_argument("--with-dynamic-is", action="store_true")
    parser.add_argument("--with-async", action="store_true")
    parser.add_argument("--with-ablations", action="store_true")
    parser.add_argument("--with-budget-curve", action="store_true")
    parser.add_argument("--with-length-ablation", action="store_true")
    parser.add_argument("--with-matched-target", action="store_true")
    parser.add_argument("--with-passk", action="store_true")
    parser.add_argument("--ablation-limit", type=int, default=32)
    parser.add_argument("--passk-limit", type=int, default=32)
    parser.add_argument("--passk-draws", type=int, default=8)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("results"),
        help="directory for aggregate reports",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()
    global _CLI_OVERRIDES
    _CLI_OVERRIDES = model_output_cli_arguments(args)

    with args.config.open("rb") as source:
        loaded_config = tomllib.load(source)
    apply_model_output_overrides(loaded_config, args)
    configured_length = int(loaded_config["generation"]["max_new_tokens"])
    configured_beams = int(loaded_config["beam"]["num_beams"])
    configured_best_of_n = int(loaded_config["best_of_n"]["samples"])
    configured_candidates = int(loaded_config["conditional_is"]["candidate_count"])
    configured_rollouts = int(loaded_config["conditional_is"]["rollout_count"])
    configured_conditional_block = int(
        loaded_config["conditional_is"]["block_size"]
    )
    if args.profile == "smoke":
        mh_alphas = (2.0,)
        mh_steps = (1,)
        candidate_counts = (3,)
        rollout_counts = (1,)
        guidance_steps_values = (2,)
        reward_sources = (
            "log_probability",
            "sequence_log_probability",
            "negative_entropy",
            "self_certainty",
            "consilience",
            "verifier",
        )
        reward_methods = ("conditional_is",)
        temperatures = (0.7,)
        budget_beams = (4,)
        budget_samples = (4,)
        budget_candidates = (3,)
        generation_lengths = (32,)
    else:
        mh_alphas = (1.0, 2.0, 4.0, 8.0)
        mh_steps = (1, 2, 5, 10)
        candidate_counts = (3, 5, 10)
        rollout_counts = (1, 5)
        guidance_steps_values = (2, 8, 16)
        reward_sources = (
            "log_probability",
            "sequence_log_probability",
            "negative_entropy",
            "self_certainty",
            "consilience",
            "verifier",
        )
        reward_methods = (
            "best_of_n",
            "conditional_is",
            "conditional_is_small_proposal",
        )
        temperatures = (0.7, 1.5)
        budget_beams = (4, 8)
        budget_samples = (4, 8)
        budget_candidates = (3, 5, 10)
        generation_lengths = (128, 256, 512)

    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT), existing or "")
    ).rstrip(os.pathsep)
    common = ["--config", str(args.config), "--tag", args.tag]
    if args.verifier_config is not None:
        common.extend(("--verifier-config", str(args.verifier_config)))
    summary_common = ["--config", str(args.config), "--tag", args.tag]
    backend_args = [] if args.backend is None else ["--backend", args.backend]
    verifier_args = (
        []
        if args.verifier_config is None
        else ["--verifier-config", str(args.verifier_config)]
    )
    rl_args = [] if args.rl_adapter is None else ["--rl-adapter", str(args.rl_adapter)]
    mh_args = (
        []
        if args.mh_suffix_schedule is None
        else ["--mh-suffix-schedule", args.mh_suffix_schedule]
    )
    if args.backend is not None:
        common.extend(backend_args)
    if args.limit is not None:
        common.extend(["--limit", str(args.limit)])
        summary_common.extend(["--limit", str(args.limit)])
    method_common = [*common, *rl_args, *mh_args]

    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    for method in methods:
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_reproduction.py",
                *method_common,
                "--method",
                method,
            ],
            environment,
        )

    if args.with_matched_target:
        for method in (
            "verifier_mh",
            "verifier_conditional_is",
            "verifier_conditional_is_small_proposal",
        ):
            if method in methods:
                continue
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *method_common,
                    "--method",
                    method,
                ],
                environment,
            )

    if args.with_replay:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        replay_output = str(
            args.summary_root / f"{args.config.stem}_replay_{args.tag}.json"
        )
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_replay_benchmark.py",
                *common,
                "--aggregate-output",
                replay_output,
            ],
            environment,
        )
        _run(
            [
                sys.executable,
                "experiments/arllm/summarize_gsm8k_replay.py",
                *summary_common,
                "--output",
                replay_output,
            ],
            environment,
        )
    if args.with_dynamic_is:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_dynamic_is_benchmark.py",
                *common,
                "--aggregate-output",
                str(
                    args.summary_root
                    / f"{args.config.stem}_dynamic_is_{args.tag}.json"
                ),
            ],
            environment,
        )
    if args.with_async:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        async_limit = args.limit if args.limit is not None else min(args.ablation_limit, 32)
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_async_benchmark.py",
                "--config",
                str(args.config),
                *backend_args,
                "--limit",
                str(async_limit),
                "--output",
                str(
                    args.summary_root
                    / f"{args.config.stem}_async_grouped_{args.tag}.json"
                ),
            ],
            environment,
        )

    if args.with_passk:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_passk.py",
                "--config",
                str(args.config),
                "--tag",
                f"{args.tag}-passk",
                "--limit",
                str(args.passk_limit),
                "--draws",
                str(args.passk_draws),
                "--output",
                str(args.summary_root / f"{args.config.stem}_passk_{args.tag}.json"),
                *backend_args,
                *rl_args,
                *mh_args,
                *verifier_args,
            ],
            environment,
        )
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_is_passk.py",
                "--config",
                str(args.config),
                "--tag",
                f"{args.tag}-is-passk",
                "--limit",
                str(args.passk_limit),
                "--draws",
                str(args.passk_draws),
                "--workers",
                str(min(args.passk_draws, 8)),
                "--output",
                str(args.summary_root / f"{args.config.stem}_is_passk_{args.tag}.json"),
                *backend_args,
                *verifier_args,
            ],
            environment,
        )

    if args.with_ablations:
        ablation_common = [
            "--config",
            str(args.config),
            *backend_args,
            *rl_args,
            *mh_args,
            *verifier_args,
            "--limit",
            str(args.ablation_limit),
        ]
        for alpha in mh_alphas:
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "mh",
                    "--tag",
                    f"{args.tag}-alpha-{alpha:g}",
                    "--mh-alpha",
                    str(alpha),
                ],
                environment,
            )
        for steps in mh_steps:
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "mh",
                    "--tag",
                    f"{args.tag}-steps-{steps}",
                    "--mh-steps",
                    str(steps),
                ],
                environment,
            )
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_reproduction.py",
                *ablation_common,
                "--method",
                "conditional_is",
                "--tag",
                f"{args.tag}-conditional-reference",
                "--candidate-count",
                str(configured_candidates),
                "--rollout-count",
                str(configured_rollouts),
                "--block-size",
                str(configured_conditional_block),
                "--sampling-temperature",
                "1.0",
            ],
            environment,
        )
        for component_tag, clip in (
            ("conditional-small-proposal-reference", "10"),
            ("conditional-small-proposal-unclipped", "none"),
        ):
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is_small_proposal",
                    "--tag",
                    f"{args.tag}-{component_tag}",
                    "--candidate-count",
                    str(configured_candidates),
                    "--rollout-count",
                    str(configured_rollouts),
                    "--block-size",
                    str(configured_conditional_block),
                    "--sampling-temperature",
                    "1.0",
                    "--importance-log-ratio-clip",
                    clip,
                ],
                environment,
            )
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_reproduction.py",
                *ablation_common,
                "--method",
                "best_of_n",
                "--tag",
                f"{args.tag}-best-of-n-reference",
                "--best-of-n-samples",
                str(configured_best_of_n),
                "--sampling-temperature",
                "1.0",
            ],
            environment,
        )
        for candidates in candidate_counts:
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is",
                    "--tag",
                    f"{args.tag}-candidates-{candidates}-rollouts-3",
                    "--candidate-count",
                    str(candidates),
                    "--rollout-count",
                    "3",
                ],
                environment,
            )
        for rollouts in rollout_counts:
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is",
                    "--tag",
                    f"{args.tag}-candidates-10-rollouts-{rollouts}",
                    "--candidate-count",
                    "10",
                    "--rollout-count",
                    str(rollouts),
                ],
                environment,
            )
        for guidance_steps in guidance_steps_values:
            block_size = configured_length // guidance_steps
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is",
                    "--tag",
                    f"{args.tag}-guidance-steps-{guidance_steps}",
                    "--max-new-tokens",
                    str(configured_length),
                    "--block-size",
                    str(block_size),
                    "--candidate-count",
                    str(configured_candidates),
                    "--rollout-count",
                    str(configured_rollouts),
                ],
                environment,
            )
        for reward_source in reward_sources:
            for method in reward_methods:
                command = [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    method,
                    "--tag",
                    f"{args.tag}-{method}-reward-{reward_source}",
                    "--conditional-reward",
                    reward_source,
                ]
                if method == "best_of_n":
                    command.extend(
                        [
                            "--best-of-n-samples",
                            str(configured_best_of_n),
                            "--sampling-temperature",
                            "1.0",
                        ]
                    )
                else:
                    command.extend(
                        [
                            "--candidate-count",
                            str(configured_candidates),
                            "--rollout-count",
                            str(configured_rollouts),
                            "--block-size",
                            str(configured_conditional_block),
                            "--sampling-temperature",
                            "1.0",
                        ]
                    )
                _run(command, environment)
        for temperature in temperatures:
            for method in (
                "beam",
                "best_of_n",
                "conditional_is",
                "conditional_is_small_proposal",
            ):
                command = [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    method,
                    "--tag",
                    f"{args.tag}-{method}-temperature-{temperature:g}",
                    "--sampling-temperature",
                    str(temperature),
                ]
                if method == "beam":
                    command.extend(["--num-beams", str(configured_beams)])
                elif method == "best_of_n":
                    command.extend(
                        ["--best-of-n-samples", str(configured_best_of_n)]
                    )
                else:
                    command.extend(
                        [
                            "--candidate-count",
                            str(configured_candidates),
                            "--rollout-count",
                            str(configured_rollouts),
                            "--block-size",
                            str(configured_conditional_block),
                        ]
                    )
                _run(command, environment)

    if args.with_budget_curve:
        curve_common = [
            "--config",
            str(args.config),
            *backend_args,
            *rl_args,
            *verifier_args,
            "--limit",
            str(args.ablation_limit),
        ]
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_reproduction.py",
                *curve_common,
                "--method",
                "beam",
                "--tag",
                f"{args.tag}-beam-reference",
                "--num-beams",
                str(configured_beams),
            ],
            environment,
        )
        _run(
            [
                sys.executable,
                "experiments/arllm/gsm8k_reproduction.py",
                *curve_common,
                "--method",
                "best_of_n",
                "--tag",
                f"{args.tag}-best-of-n-reference",
                "--best-of-n-samples",
                str(configured_best_of_n),
                "--sampling-temperature",
                "1.0",
            ],
            environment,
        )
        for method, component_tag in (
            ("conditional_is", "conditional-reference"),
            (
                "conditional_is_small_proposal",
                "conditional-small-proposal-reference",
            ),
        ):
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *curve_common,
                    "--method",
                    method,
                    "--tag",
                    f"{args.tag}-{component_tag}",
                    "--candidate-count",
                    str(configured_candidates),
                    "--rollout-count",
                    str(configured_rollouts),
                    "--block-size",
                    str(configured_conditional_block),
                    "--sampling-temperature",
                    "1.0",
                ],
                environment,
            )
        for beams in budget_beams:
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *curve_common,
                    "--method",
                    "beam",
                    "--tag",
                    f"{args.tag}-budget-beam-{beams}",
                    "--num-beams",
                    str(beams),
                ],
                environment,
            )
        for samples in budget_samples:
            _run(
                [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *curve_common,
                    "--method",
                    "best_of_n",
                    "--tag",
                    f"{args.tag}-budget-best-of-n-{samples}",
                    "--best-of-n-samples",
                    str(samples),
                ],
                environment,
            )
        for candidates in budget_candidates:
            for method in ("conditional_is", "conditional_is_small_proposal"):
                _run(
                    [
                        sys.executable,
                        "experiments/arllm/gsm8k_reproduction.py",
                        *curve_common,
                        "--method",
                        method,
                        "--tag",
                        f"{args.tag}-budget-{method}-m{candidates}-k3",
                        "--candidate-count",
                        str(candidates),
                        "--rollout-count",
                        "3",
                    ],
                    environment,
                )

    if args.with_length_ablation:
        length_common = [
            "--config",
            str(args.config),
            *backend_args,
            *rl_args,
            *verifier_args,
            "--limit",
            str(args.ablation_limit),
        ]
        for length in generation_lengths:
            block_size = max(1, length // 4)
            for method in (
                "base",
                "best_of_n",
                "beam",
                "conditional_is",
                "conditional_is_small_proposal",
                "rl_greedy",
            ):
                command = [
                    sys.executable,
                    "experiments/arllm/gsm8k_reproduction.py",
                    *length_common,
                    "--method",
                    method,
                    "--tag",
                    f"{args.tag}-length-{length}",
                    "--max-new-tokens",
                    str(length),
                ]
                if method.startswith("conditional_is"):
                    command.extend(["--block-size", str(block_size)])
                _run(command, environment)


if __name__ == "__main__":
    main()
