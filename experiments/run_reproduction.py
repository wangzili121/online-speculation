"""Unified CLI for AR-LLM and diffusion-LLM reproduction pipelines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.shared.methods import (
    AR_DEFAULT_METHODS,
    AR_METHODS,
    DLLM_DEFAULT_METHODS,
    DLLM_METHODS,
)
from experiments.shared.model_cli import add_model_output_arguments, model_output_cli_arguments
from experiments.shared.components import COMPONENTS, DLLM_COMPONENTS, FULL_COMPONENTS
from experiments.shared.suite_runner import run_manifested_commands


def _default_python(environment_variable: str) -> str:
    """Select a family interpreter from the environment or current process."""

    return os.environ.get(environment_variable) or sys.executable


def build_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    if args.family == "dllm" and args.backend is not None:
        raise ValueError("--ar-backend/--backend applies only to an AR-LLM run")
    commands: list[list[str]] = []
    if args.family in {"arllm", "both"}:
        command = [
            args.ar_python,
            str(root / "experiments" / "arllm" / "run_arllm_suite.py"),
            "--stage",
            args.stage,
            "--profile",
            args.profile,
            "--tag",
            f"{args.tag}-arllm",
            "--training-config",
            str(args.ar_training_config),
            "--summary-root",
            str(args.output_root / "arllm"),
            "--methods",
            *args.ar_methods,
            "--components",
            *args.components,
        ]
        if args.ar_config is not None:
            command.extend(("--config", str(args.ar_config)))
        if getattr(args, "verifier_config", None) is not None:
            command.extend(("--verifier-config", str(args.verifier_config)))
        if args.backend is not None:
            command.extend(("--backend", args.backend))
        ar_mh_suffix_schedule = getattr(args, "ar_mh_suffix_schedule", "multiscale")
        if ar_mh_suffix_schedule is not None:
            command.extend(("--mh-suffix-schedule", ar_mh_suffix_schedule))
        for flag, value in (
            ("--limit", args.limit),
            ("--train-limit", args.train_limit),
            ("--training-output", args.ar_training_output),
            ("--max-train-steps", args.max_train_steps),
            ("--num-generations", args.num_generations),
            ("--max-completion-length", args.max_completion_length),
            ("--ablation-limit", args.ablation_limit),
            ("--passk-limit", args.passk_limit),
            ("--passk-draws", args.passk_draws),
            ("--distribution-problems", args.distribution_problems),
            ("--distribution-draws", args.distribution_draws),
            ("--infra-limit", args.infra_limit),
        ):
            if value is not None:
                command.extend((flag, str(value)))
        if args.dry_run:
            command.append("--dry-run")
        if getattr(args, "restart", False):
            command.append("--restart")
        command[2:2] = model_output_cli_arguments(args)
        commands.append(command)

    if args.family in {"dllm", "both"}:
        dllm_components = tuple(
            component for component in args.components if component in DLLM_COMPONENTS
        )
        unsupported = tuple(
            component
            for component in args.components
            if component not in DLLM_COMPONENTS
        )
        if unsupported and getattr(args, "components_explicit", False):
            raise ValueError(
                "components unsupported by the dLLM suite: " + ", ".join(unsupported)
            )
        dllm_config = str(args.dllm_config)
        if args.stage == "prepare":
            commands.append(
                [
                    args.dllm_python,
                    str(root / "experiments" / "dllm" / "download_llada.py"),
                    "--config",
                    dllm_config,
                ]
            )
        elif args.stage == "train":
            if args.profile == "smoke":
                commands.append(
                    [
                        args.dllm_python,
                        str(root / "experiments" / "dllm" / "train_gsm8k_vrpo.py"),
                        "--config",
                        dllm_config,
                        "--preflight",
                    ]
                )
            else:
                commands.extend(
                    (
                        [
                            args.dllm_python,
                            str(
                                root / "experiments" / "dllm" / "prepare_gsm8k_vrpo.py"
                            ),
                            "--config",
                            dllm_config,
                            "--profile",
                            "full",
                            *(
                                []
                                if getattr(args, "verifier_config", None) is None
                                else [
                                    "--verifier-config",
                                    str(args.verifier_config),
                                ]
                            ),
                        ],
                        [
                            args.dllm_python,
                            str(root / "experiments" / "dllm" / "train_gsm8k_vrpo.py"),
                            "--config",
                            dllm_config,
                            "--resume",
                            "auto",
                        ],
                    )
                )
        else:
            if args.stage == "all":
                commands.append(
                    [
                        args.dllm_python,
                        str(root / "experiments" / "dllm" / "download_llada.py"),
                        "--config",
                        dllm_config,
                    ]
                )
            vrpo = "skip"
            if args.stage == "all":
                vrpo = "preflight" if args.profile == "smoke" else "train"
            command = [
                args.dllm_python,
                str(root / "experiments" / "dllm" / "run_llada_suite.py"),
                "--config",
                dllm_config,
                "--profile",
                args.profile,
                "--tag",
                f"{args.tag}-dllm",
                "--output-root",
                str(args.output_root / "dllm"),
                "--vrpo",
                vrpo,
                "--methods",
                *args.dllm_methods,
                "--components",
                *dllm_components,
            ]
            if getattr(args, "verifier_config", None) is not None:
                command.extend(("--verifier-config", str(args.verifier_config)))
            for flag, value in (
                ("--limit", args.limit),
                ("--ablation-limit", args.ablation_limit),
                ("--passk-limit", args.passk_limit),
                ("--passk-draws", args.passk_draws),
                ("--distribution-problems", args.distribution_problems),
                ("--distribution-draws", args.distribution_draws),
                ("--infra-limit", args.infra_limit),
            ):
                if value is not None:
                    command.extend((flag, str(value)))
            if (
                any(method.startswith("vrpo_") for method in args.dllm_methods)
                and vrpo != "train"
            ):
                command.append("--with-aligned")
            if args.dry_run:
                command.append("--dry-run")
            if getattr(args, "restart", False):
                command.append("--restart")
            commands.append(command)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("arllm", "dllm", "both"), default="arllm")
    parser.add_argument(
        "--stage", choices=("prepare", "train", "inference", "all"), default="inference"
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--tag", default="reproduction")
    parser.add_argument("--ar-config", type=Path)
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument(
        "--ar-training-config", type=Path, default=Path("configs/gsm8k_grpo.toml")
    )
    parser.add_argument(
        "--dllm-config",
        type=Path,
        default=Path("configs/gsm8k_llada_moe_3090.toml"),
    )
    parser.add_argument("--ar-methods", nargs="+", choices=AR_METHODS)
    parser.add_argument("--dllm-methods", nargs="+", choices=DLLM_METHODS)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS)
    parser.add_argument(
        "--ar-backend",
        "--backend",
        dest="backend",
        choices=("transformers", "vllm", "vllm-sync"),
        help="AR-LLM inference backend; ignored values are rejected for --family dllm",
    )
    parser.add_argument(
        "--ar-mh-suffix-schedule",
        choices=("uniform", "inverse_length", "multiscale"),
        default="multiscale",
        help="AR-LLM MH suffix schedule; multiscale is the production default",
    )
    parser.add_argument(
        "--ar-python",
        default=_default_python("AR_PYTHON"),
        help="AR executable name/path; defaults to AR_PYTHON, then current Python",
    )
    parser.add_argument(
        "--dllm-python",
        default=_default_python("DLLM_PYTHON"),
        help="dLLM executable name/path; defaults to DLLM_PYTHON, then current Python",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--ar-training-output", type=Path)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--ablation-limit", type=int)
    parser.add_argument("--passk-limit", type=int)
    parser.add_argument("--passk-draws", type=int)
    parser.add_argument("--distribution-problems", type=int)
    parser.add_argument("--distribution-draws", type=int)
    parser.add_argument("--infra-limit", type=int, default=1)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/reproduction")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="replace an existing suite manifest and execute the full command plan",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.family == "dllm" and model_output_cli_arguments(args):
        parser.error("model/output override flags in this entry point apply to arllm; configure dllm through --dllm-config")

    # Keep research-only methods available through an explicit CLI selection,
    # but do not place rejected or unmatched methods in the default Qwen run.
    args.ar_methods = tuple(args.ar_methods or AR_DEFAULT_METHODS)
    args.dllm_methods = tuple(args.dllm_methods or DLLM_DEFAULT_METHODS)
    args.components_explicit = args.components is not None
    args.components = tuple(
        args.components
        or (("quality", "replay") if args.profile == "smoke" else FULL_COMPONENTS)
    )
    root = REPOSITORY_ROOT
    commands = build_commands(args, root)
    run_manifested_commands(
        commands=commands,
        root=root,
        manifest_path=args.output_root / args.tag / "reproduction_manifest.json",
        metadata={
            "family": args.family,
            "stage": args.stage,
            "profile": args.profile,
            "tag": args.tag,
            "ar_methods": args.ar_methods,
            "dllm_methods": args.dllm_methods,
            "components": args.components,
            "ar_mh_suffix_schedule": args.ar_mh_suffix_schedule,
            "verifier_config": (
                str(args.verifier_config) if args.verifier_config is not None else None
            ),
            "python_executables": {
                "controller": sys.executable,
                "arllm": args.ar_python,
                "dllm": args.dllm_python,
            },
        },
        dry_run=args.dry_run,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
