"""Single CLI entry point for AR-LLM preparation, GRPO, and inference suites."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.shared.model_cli import add_model_output_arguments, model_output_cli_arguments
from experiments.shared.components import COMPONENTS, FULL_COMPONENTS
from experiments.shared.environment import validate_environment
from experiments.shared.methods import AR_DEFAULT_METHODS, AR_METHODS
from experiments.shared.suite_runner import run_manifested_commands


def build_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    scripts = root / "experiments" / "arllm"
    adapter_args = (
        []
        if args.training_output is None
        else ["--rl-adapter", str(args.training_output)]
    )
    verifier_args = (
        []
        if getattr(args, "verifier_config", None) is None
        else ["--verifier-config", str(args.verifier_config)]
    )
    commands: list[list[str]] = []
    include_training = args.stage in {"train", "all"}
    include_inference = args.stage in {"inference", "all"}
    if args.stage == "prepare" or include_training:
        commands.append(
            [
                sys.executable,
                str(scripts / "prepare_gsm8k.py"),
                "--config",
                str(args.config),
            ]
        )
    if include_training:
        command = [
            sys.executable,
            str(scripts / "train_gsm8k_grpo.py"),
            "--config",
            str(args.training_config),
            "--resume",
            args.resume,
            *verifier_args,
        ]
        if args.training_output is not None:
            command.extend(("--output-dir", str(args.training_output)))
        overrides = (
            ("--train-limit", args.train_limit),
            ("--max-steps", args.max_train_steps),
            ("--num-generations", args.num_generations),
            ("--max-completion-length", args.max_completion_length),
        )
        for flag, value in overrides:
            if value is not None:
                command.extend((flag, str(value)))
        commands.append(command)
    if not include_inference:
        for command in commands:
            command[2:2] = model_output_cli_arguments(args)
        return commands

    components = set(args.components)
    suite_components = {
        "quality",
        "matched_target",
        "replay",
        "dynamic_is",
        "async",
        "passk",
        "ablations",
        "budget_curve",
        "length_ablation",
    }
    if components & suite_components:
        methods = args.methods if "quality" in components else ()
        command = [
            sys.executable,
            str(scripts / "run_gsm8k_suite.py"),
            "--config",
            str(args.config),
            "--tag",
            args.tag,
            "--profile",
            args.profile,
            "--methods",
            ",".join(methods),
            "--summary-root",
            str(args.summary_root),
            "--ablation-limit",
            str(args.ablation_limit),
            "--passk-limit",
            str(args.passk_limit),
            "--passk-draws",
            str(args.passk_draws),
            *adapter_args,
            *verifier_args,
        ]
        mh_suffix_schedule = getattr(args, "mh_suffix_schedule", "multiscale")
        if mh_suffix_schedule is not None:
            command.extend(("--mh-suffix-schedule", mh_suffix_schedule))
        if args.backend is not None:
            command.extend(("--backend", args.backend))
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        for component, flag in (
            ("matched_target", "--with-matched-target"),
            ("replay", "--with-replay"),
            ("dynamic_is", "--with-dynamic-is"),
            ("async", "--with-async"),
            ("passk", "--with-passk"),
            ("ablations", "--with-ablations"),
            ("budget_curve", "--with-budget-curve"),
            ("length_ablation", "--with-length-ablation"),
        ):
            if component in components:
                command.append(flag)
        commands.append(command)

    args.summary_root.mkdir(parents=True, exist_ok=True)
    backend = args.backend or "transformers"
    if "distribution" in components:
        command = [
            sys.executable,
            str(scripts / "gsm8k_distribution_audit.py"),
            "--config",
            str(args.config),
            "--problem-count",
            str(args.distribution_problems),
            "--draws",
            str(args.distribution_draws),
            "--output",
            str(args.summary_root / f"arllm_distribution_{args.tag}.json"),
            *adapter_args,
            *verifier_args,
        ]
        if args.backend is not None:
            command.extend(("--backend", args.backend))
        commands.append(command)
    if "infra" in components:
        commands.extend(
            (
                [
                    sys.executable,
                    str(scripts / "benchmark_rollout_infra.py"),
                    "--config",
                    str(args.config),
                    "--backend",
                    backend,
                    "--dtype",
                    args.dtype,
                    "--section",
                    "all",
                    "--limit",
                    str(args.infra_limit),
                    "--output",
                    str(args.summary_root / f"arllm_rollout_infra_{args.tag}.json"),
                    *verifier_args,
                ],
                [
                    sys.executable,
                    str(scripts / "benchmark_is_mh_reuse.py"),
                    "--config",
                    str(args.config),
                    "--backend",
                    backend,
                    "--dtype",
                    args.dtype,
                    "--section",
                    "all",
                    "--output",
                    str(args.summary_root / f"arllm_is_mh_infra_{args.tag}.json"),
                ],
            )
        )
    if "vllm" in components:
        commands.append(
            [
                sys.executable,
                str(scripts / "run_vllm_backend_benchmark.py"),
                "--config",
                str(args.config),
                "--limit",
                str(args.vllm_limit),
                "--workers",
                str(args.vllm_workers),
                "--tag",
                args.tag,
            ]
        )
    extra = model_output_cli_arguments(args)
    for command in commands:
        command[2:2] = extra
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "train", "inference", "all"), default="inference")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--training-config", type=Path, default=Path("configs/gsm8k_grpo.toml"))
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument("--tag", default="arllm-reproduction")
    parser.add_argument("--methods", nargs="+", choices=AR_METHODS)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS)
    parser.add_argument("--backend", choices=("transformers", "vllm", "vllm-sync"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--mh-suffix-schedule",
        choices=("uniform", "inverse_length", "multiscale"),
        default="multiscale",
        help="MH suffix-length proposal schedule",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--ablation-limit", type=int)
    parser.add_argument("--passk-limit", type=int)
    parser.add_argument("--passk-draws", type=int)
    parser.add_argument("--distribution-problems", type=int)
    parser.add_argument("--distribution-draws", type=int)
    parser.add_argument("--infra-limit", type=int, default=1)
    parser.add_argument("--vllm-limit", type=int)
    parser.add_argument("--vllm-workers", type=int)
    parser.add_argument("--summary-root", type=Path, default=Path("results/arllm"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="replace an existing suite manifest and execute the full command plan",
    )
    parser.add_argument(
        "--no-environment-check",
        action="store_true",
        help="skip the role-specific dependency preflight",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(
            "configs/gsm8k_quick.toml"
            if args.profile == "smoke"
            else "configs/gsm8k_3090_aligned.toml"
        )
    args.methods = tuple(args.methods or AR_DEFAULT_METHODS)
    args.components = tuple(
        args.components
        or (("quality",) if args.profile == "smoke" else FULL_COMPONENTS)
    )
    if args.profile == "smoke":
        if args.training_output is None and args.stage in {"train", "all"}:
            args.training_output = Path(
                f"models/arllm-grpo-smoke-{args.tag}"
            )
        args.limit = args.limit or 1
        args.train_limit = args.train_limit or 4
        args.max_train_steps = args.max_train_steps or 1
        args.num_generations = args.num_generations or 2
        args.max_completion_length = args.max_completion_length or 96
        args.ablation_limit = args.ablation_limit or 1
        args.passk_limit = args.passk_limit or 1
        args.passk_draws = args.passk_draws or 2
        args.distribution_problems = args.distribution_problems or 1
        args.distribution_draws = args.distribution_draws or 2
        args.vllm_limit = args.vllm_limit or 1
        args.vllm_workers = args.vllm_workers or 1
    else:
        args.ablation_limit = args.ablation_limit or 32
        args.passk_limit = args.passk_limit or 32
        args.passk_draws = args.passk_draws or 8
        args.distribution_problems = args.distribution_problems or 4
        args.distribution_draws = args.distribution_draws or 8
        args.vllm_limit = args.vllm_limit or 32
        args.vllm_workers = args.vllm_workers or 8

    if not args.dry_run and not args.no_environment_check:
        validate_environment(
            "arllm",
            stage=args.stage,
            components=args.components,
        )

    root = REPOSITORY_ROOT
    commands = build_commands(args, root)
    run_manifested_commands(
        commands=commands,
        root=root,
        manifest_path=args.summary_root / args.tag / "arllm_suite_manifest.json",
        metadata={
            "family": "arllm",
            "stage": args.stage,
            "profile": args.profile,
            "tag": args.tag,
            "methods": args.methods,
            "components": args.components,
            "verifier_config": (
                str(args.verifier_config) if args.verifier_config is not None else None
            ),
        },
        dry_run=args.dry_run,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
