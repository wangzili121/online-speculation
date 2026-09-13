"""Single resumable entry point for the paired LLaDA experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dllm.profiles import apply_execution_profile
from experiments.shared.components import DLLM_COMPONENTS, FULL_COMPONENTS
from experiments.shared.environment import validate_environment
from experiments.shared.methods import (
    DLLM_ALIGNED_METHODS,
    DLLM_DEFAULT_METHODS,
    DLLM_DYNAMIC_METHODS,
    DLLM_METHODS,
)
from experiments.shared.paired_protocol import MethodPair, load_pairing
from experiments.shared.suite_runner import run_manifested_commands

METHODS = DLLM_METHODS
DYNAMIC_METHODS = DLLM_DYNAMIC_METHODS
DEFAULT_METHODS = DLLM_DEFAULT_METHODS
ALIGNED_METHODS = DLLM_ALIGNED_METHODS
IMPLEMENTED_COMPONENTS = DLLM_COMPONENTS


def _paired_methods(
    pairing: dict[str, tuple[MethodPair, ...]], section: str
) -> tuple[str, ...]:
    return tuple(pair.dllm for pair in pairing[section])


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _component_tag(tag: str, component: str, suffix: str | None = None) -> str:
    parts = (tag, "components", component, suffix)
    return "/".join(part for part in parts if part)


def build_commands(
    args: argparse.Namespace,
    root: Path,
    config: dict[str, Any],
    pairing: dict[str, tuple[MethodPair, ...]],
    methods: Sequence[str],
    tag: str,
) -> list[list[str]]:
    runner = root / "experiments" / "dllm" / "gsm8k_reproduction.py"
    replay_runner = root / "experiments" / "dllm" / "gsm8k_replay_benchmark.py"
    infra_runner = root / "experiments" / "dllm" / "benchmark_infra.py"
    analyzer = root / "experiments" / "dllm" / "gsm8k_analysis.py"
    prepare_vrpo = root / "experiments" / "dllm" / "prepare_gsm8k_vrpo.py"
    train_vrpo = root / "experiments" / "dllm" / "train_gsm8k_vrpo.py"
    commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(command: list[str]) -> None:
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            commands.append(command)

    if args.vrpo == "preflight":
        add(
            [
                sys.executable,
                str(train_vrpo),
                "--config",
                str(args.config),
                "--preflight",
            ]
        )
    elif args.vrpo == "train":
        add(
            [
                sys.executable,
                str(prepare_vrpo),
                "--config",
                str(args.config),
                "--profile",
                "full",
                *(
                    []
                    if getattr(args, "verifier_config", None) is None
                    else ["--verifier-config", str(args.verifier_config)]
                ),
            ]
        )
        add(
            [
                sys.executable,
                str(train_vrpo),
                "--config",
                str(args.config),
                "--resume",
                "auto",
            ]
        )

    base_common = [
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--output-root",
        str(args.output_root),
        "--profile",
        args.profile,
    ]
    if getattr(args, "verifier_config", None) is not None:
        base_common.extend(("--verifier-config", str(args.verifier_config)))

    def add_run(
        method: str,
        *,
        run_tag: str,
        limit: int | None,
        draw_index: int | None = 0,
        draws: int | None = None,
        overrides: Sequence[str] = (),
    ) -> None:
        command = [
            sys.executable,
            str(runner),
            *base_common,
            "--tag",
            run_tag,
            "--method",
            method,
        ]
        if limit is not None:
            command.extend(("--limit", str(limit)))
        if draws is not None:
            command.extend(("--draws", str(draws)))
        elif draw_index is not None:
            command.extend(("--draw-index", str(draw_index)))
        for override in overrides:
            command.extend(("--set", override))
        add(command)

    components = set(args.components)
    quality_methods: list[str] = []
    if "quality" in components:
        quality_methods.extend(methods)
    if "matched_target" in components:
        quality_methods.extend(
            (
                "verifier_mh",
                "verifier_conditional_is",
                "verifier_conditional_is_reduced_layer_proposal",
            )
        )
    for method in _unique(quality_methods):
        add_run(
            method,
            run_tag=tag,
            limit=args.limit,
            draw_index=args.draw_index,
        )

    if "replay" in components:
        command = [
            sys.executable,
            str(replay_runner),
            *base_common,
            "--tag",
            tag,
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        add(command)

    if "dynamic_is" in components:
        dynamic_tag = _component_tag(tag, "dynamic_is")
        for method in DYNAMIC_METHODS:
            add_run(
                method,
                run_tag=f"{dynamic_tag}/{method}",
                limit=args.ablation_limit or args.limit,
            )
        add(
            [
                sys.executable,
                str(analyzer),
                "--kind",
                "sweep",
                "--run-root",
                str(args.output_root / dynamic_tag),
            ]
        )

    if "infra" in components or "async" in components:
        section = "all" if "infra" in components else "async"
        command = [
            sys.executable,
            str(infra_runner),
            *base_common,
            "--tag",
            tag,
            "--section",
            section,
        ]
        infra_limit = args.infra_limit or args.limit
        if infra_limit is not None:
            command.extend(("--limit", str(infra_limit)))
        add(command)

    bootstrap_replicates = 100 if args.profile == "smoke" else 2_000
    if "passk" in components:
        passk_methods = _paired_methods(pairing, "passk_pairs")
        passk_draws = args.passk_draws or int(config["passk"]["draws"])
        passk_limit = (
            args.passk_limit or args.limit or int(config["run"]["sample_count"])
        )
        passk_tag = _component_tag(tag, "passk")
        for method in passk_methods:
            add_run(
                method,
                run_tag=passk_tag,
                limit=passk_limit,
                draw_index=None,
                draws=passk_draws,
            )
        passk_root = args.output_root / passk_tag
        add(
            [
                sys.executable,
                str(analyzer),
                "--kind",
                "passk",
                "--run-root",
                str(passk_root),
                "--methods",
                *passk_methods,
                "--draws",
                str(passk_draws),
                "--k",
                *(str(k) for k in config["passk"]["k"] if int(k) <= passk_draws),
                "--bootstrap-replicates",
                str(bootstrap_replicates),
            ]
        )

    if "distribution" in components:
        distribution_methods = _paired_methods(pairing, "distribution_pairs")
        distribution_draws = args.distribution_draws or (
            2 if args.profile == "smoke" else int(config["passk"]["draws"])
        )
        distribution_limit = args.distribution_problems or args.limit or (
            1 if args.profile == "smoke" else 4
        )
        distribution_tag = _component_tag(tag, "distribution")
        for method in distribution_methods:
            add_run(
                method,
                run_tag=distribution_tag,
                limit=distribution_limit,
                draw_index=None,
                draws=distribution_draws,
            )
        distribution_root = args.output_root / distribution_tag
        add(
            [
                sys.executable,
                str(analyzer),
                "--kind",
                "distribution",
                "--run-root",
                str(distribution_root),
                "--methods",
                *distribution_methods,
                "--draws",
                str(distribution_draws),
                "--reference",
                "vrpo_sample",
                "--bootstrap-replicates",
                str(bootstrap_replicates),
            ]
        )

    sweep_limit = args.ablation_limit or args.limit or (
        1 if args.profile == "smoke" else int(config["run"]["sample_count"])
    )
    generation_length = int(config["generation"]["max_new_tokens"])
    diffusion_block = int(config["generation"]["block_length"])
    conditional = config["conditional_is"]

    def add_sweep(
        component: str,
        axis: str,
        label: str,
        method: str,
        overrides: Sequence[str],
    ) -> None:
        add_run(
            method,
            run_tag=_component_tag(tag, component, f"{axis}/{label}"),
            limit=sweep_limit,
            overrides=overrides,
        )

    if "ablations" in components:
        alpha_values = (
            (4.0, 8.0)
            if args.profile == "smoke"
            else (1.0, 2.0, 4.0, 8.0)
        )
        update_values = (
            (1, 2) if args.profile == "smoke" else (1, 2, 5, 10)
        )
        candidate_values = (
            (3, 8) if args.profile == "smoke" else (3, 5, 10)
        )
        rollout_values = (
            (1, 3) if args.profile == "smoke" else (1, 3, 5)
        )
        temperature_values = (
            (1.0, 1.5)
            if args.profile == "smoke"
            else (0.7, 1.0, 1.5)
        )
        clip_values: tuple[float | None, ...] = (
            (10.0, None) if args.profile == "smoke" else (5.0, 10.0, None)
        )
        for value in alpha_values:
            add_sweep(
                "ablations",
                "trajectory_mh_alpha",
                f"alpha-{value:g}",
                "trajectory_power_mh",
                (f"mh.alpha={value}",),
            )
        for value in update_values:
            add_sweep(
                "ablations",
                "trajectory_mh_updates",
                f"updates-{value}",
                "trajectory_power_mh",
                (f"mh.updates_per_stage={value}",),
            )
        for value in candidate_values:
            add_sweep(
                "ablations",
                "candidate_count",
                f"candidates-{value}",
                "conditional_is",
                (f"conditional_is.candidate_count={value}",),
            )
        for value in rollout_values:
            add_sweep(
                "ablations",
                "rollout_count",
                f"rollouts-{value}",
                "conditional_is",
                (f"conditional_is.rollout_count={value}",),
            )
        possible_stages = tuple(
            stages
            for stages in (1, 2, 4)
            if generation_length % stages == 0
            and (generation_length // stages) % diffusion_block == 0
        )
        for stages in possible_stages:
            add_sweep(
                "ablations",
                "decision_stages",
                f"stages-{stages}",
                "conditional_is",
                (f"conditional_is.decision_block_size={generation_length // stages}",),
            )
        for value in clip_values:
            encoded = "none" if value is None else f"{value:g}"
            add_sweep(
                "ablations",
                "trajectory_importance_clip",
                f"clip-{encoded}",
                "conditional_is_reduced_layer_proposal",
                (f"conditional_is.importance_log_ratio_clip={encoded}",),
            )
        add_sweep(
            "ablations", "reward_source", "self-consistency", "conditional_is", ()
        )
        add_sweep(
            "ablations",
            "reward_source",
            "configured-verifier",
            "verifier_conditional_is",
            (),
        )
        for value in temperature_values:
            add_sweep(
                "ablations",
                "temperature",
                f"temperature-{value:g}",
                "conditional_is",
                (f"generation.temperature={value}",),
            )
        ablation_root = args.output_root / _component_tag(tag, "ablations")
        add(
            [
                sys.executable,
                str(analyzer),
                "--kind",
                "sweep",
                "--run-root",
                str(ablation_root),
            ]
        )

    if "budget_curve" in components:
        budget_values = (
            (4, 8) if args.profile == "smoke" else (3, 4, 5, 8, 10)
        )
        for width in (4, 8):
            add_sweep(
                "budget_curve",
                "block_beam",
                f"width-{width}",
                "block_beam",
                (f"search.width={width}",),
            )
        for samples in (4, 8):
            add_sweep(
                "budget_curve",
                "best_of_n",
                f"samples-{samples}",
                "best_of_n",
                (f"best_of_n.samples={samples}",),
            )
        for candidates in budget_values:
            for method in (
                "conditional_is",
                "conditional_is_reduced_layer_proposal",
            ):
                add_sweep(
                    "budget_curve",
                    method,
                    f"candidates-{candidates}",
                    method,
                    (
                        f"conditional_is.candidate_count={candidates}",
                        "conditional_is.rollout_count=3",
                    ),
                )
        budget_root = args.output_root / _component_tag(tag, "budget_curve")
        add(
            [
                sys.executable,
                str(analyzer),
                "--kind",
                "sweep",
                "--run-root",
                str(budget_root),
            ]
        )

    if "length_ablation" in components:
        maximum_blocks = generation_length // diffusion_block
        requested_blocks = (
            (2, maximum_blocks) if args.profile == "smoke" else (2, 3, 4)
        )
        lengths = tuple(
            dict.fromkeys(
                diffusion_block * blocks
                for blocks in requested_blocks
                if 1 <= blocks <= maximum_blocks
            )
        )
        length_methods = (
            "base",
            "best_of_n",
            "block_beam",
            "conditional_is",
            "conditional_is_reduced_layer_proposal",
            "vrpo_greedy",
        )
        for length in lengths:
            for method in length_methods:
                overrides = [f"generation.max_new_tokens={length}"]
                if method == "conditional_is" or "reduced_layer_proposal" in method:
                    overrides.append(
                        "conditional_is.decision_block_size="
                        f"{min(int(conditional['decision_block_size']), length)}"
                    )
                if method == "block_beam":
                    overrides.append(
                        "search.decision_block_size="
                        f"{min(int(config['search']['decision_block_size']), length)}"
                    )
                add_sweep(
                    "length_ablation",
                    f"length-{length}",
                    method,
                    method,
                    overrides,
                )
        length_root = args.output_root / _component_tag(tag, "length_ablation")
        add(
            [
                sys.executable,
                str(analyzer),
                "--kind",
                "sweep",
                "--run-root",
                str(length_root),
            ]
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--verifier-config", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dllm/gsm8k")
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--components", nargs="+", choices=DLLM_COMPONENTS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--draw-index", type=int, default=0)
    parser.add_argument("--ablation-limit", type=int)
    parser.add_argument("--passk-limit", type=int)
    parser.add_argument("--passk-draws", type=int)
    parser.add_argument("--distribution-problems", type=int)
    parser.add_argument("--distribution-draws", type=int)
    parser.add_argument("--infra-limit", type=int)
    parser.add_argument("--with-aligned", action="store_true")
    parser.add_argument(
        "--vrpo",
        choices=("skip", "preflight", "train"),
        default="preflight",
        help=(
            "skip VRPO, run a CPU-only implementation preflight, or prepare "
            "preferences and train the resumable adapter before evaluation"
        ),
    )
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
    args = parser.parse_args()

    config, pairing = load_pairing(args.config)
    config = apply_execution_profile(config, args.profile)
    components = list(
        args.components
        or (("quality", "replay") if args.profile == "smoke" else FULL_COMPONENTS)
    )
    unsupported = sorted(set(components) - set(IMPLEMENTED_COMPONENTS))
    if unsupported:
        raise ValueError(
            "components unsupported by the dLLM suite: " + ", ".join(unsupported)
        )
    args.components = tuple(components)
    if not args.dry_run and not args.no_environment_check:
        validate_environment(
            "dllm",
            stage="all" if args.vrpo in {"preflight", "train"} else "inference",
            components=args.components,
        )

    methods = list(args.methods or DEFAULT_METHODS)
    required_methods = set(methods if "quality" in components else ())
    if "matched_target" in components:
        required_methods.update(
            (
                "verifier_mh",
                "verifier_conditional_is",
                "verifier_conditional_is_reduced_layer_proposal",
            )
        )
    for component, section in (
        ("passk", "passk_pairs"),
        ("distribution", "distribution_pairs"),
    ):
        if component in components:
            required_methods.update(_paired_methods(pairing, section))
    if "length_ablation" in components:
        required_methods.add("vrpo_greedy")

    include_aligned = (
        args.with_aligned
        or args.vrpo == "train"
        or any(method in ALIGNED_METHODS for method in required_methods)
    )
    if include_aligned and args.vrpo != "train":
        adapter = Path(str(config["alignment"]["adapter"]))
        if not adapter.is_dir():
            raise FileNotFoundError(
                f"aligned LLaDA adapter is absent: {adapter}; run the VRPO stage first"
            )
    if args.with_aligned or args.vrpo == "train":
        for method in ALIGNED_METHODS:
            if method not in methods:
                methods.append(method)
    elif any(method in ALIGNED_METHODS for method in methods) and not include_aligned:
        raise ValueError("aligned methods require --with-aligned")

    tag = args.tag or f"llada-{args.profile}"
    commands = build_commands(args, REPOSITORY_ROOT, config, pairing, methods, tag)
    run_manifested_commands(
        commands=commands,
        root=REPOSITORY_ROOT,
        manifest_path=args.output_root / tag / "suite_manifest.json",
        metadata={
            "family": "dllm",
            "profile": args.profile,
            "tag": tag,
            "methods": methods,
            "components": args.components,
            "with_aligned": include_aligned,
            "vrpo": args.vrpo,
            "verifier_config": (
                str(args.verifier_config) if args.verifier_config is not None else None
            ),
        },
        dry_run=args.dry_run,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
