"""Validate and display the frozen ARLLM--dLLM experiment pairing."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from experiments.shared.methods import (
    AR_DISTRIBUTION_METHODS as REGISTERED_AR_DISTRIBUTION_METHODS,
    AR_IS_PASSK_METHODS as REGISTERED_AR_IS_PASSK_METHODS,
    AR_PAIRED_METHODS as REGISTERED_AR_METHODS,
    AR_PASSK_METHODS as REGISTERED_AR_PASSK_METHODS,
)

AR_MAIN_METHODS = set(REGISTERED_AR_METHODS)
AR_PASSK_METHODS = set(
    (*REGISTERED_AR_PASSK_METHODS, *REGISTERED_AR_IS_PASSK_METHODS)
)
AR_DISTRIBUTION_METHODS = set(REGISTERED_AR_DISTRIBUTION_METHODS)
AR_REPLAY_ARMS = {"fresh_only", "warm_replay"}
AR_DYNAMIC_ARMS = {
    "base_candidate_fixed",
    "replay_aware_fixed",
    "replay_aware_optimal",
}
AR_ABLATION_AXES = {
    "candidate_count",
    "rollout_count",
    "guidance_steps",
    "mh_alpha",
    "mh_updates",
    "importance_clip",
    "generation_length",
    "reward_source",
    "temperature",
}
AR_INFRA_FAMILIES = {
    "continuous_batching",
    "resume_partial_rollouts",
    "streaming_reward",
    "history_token_tree",
    "mh_proposal_tree_prefetch",
    "delayed_acceptance_mh",
    "replay_mixture_mh",
    "progressive_is",
    "smc_rollout_forest",
    "vllm_backend",
}


@dataclass(frozen=True, slots=True)
class MethodPair:
    ar: str
    dllm: str
    relation: str | None = None
    comparison: str | None = None


EXPECTED_SETS = {
    "main_pairs": AR_MAIN_METHODS,
    "passk_pairs": AR_PASSK_METHODS,
    "distribution_pairs": AR_DISTRIBUTION_METHODS,
    "replay_pairs": AR_REPLAY_ARMS,
    "dynamic_pairs": AR_DYNAMIC_ARMS,
    "ablation_pairs": AR_ABLATION_AXES,
    "infra_pairs": AR_INFRA_FAMILIES,
    "training_pairs": {"grpo"},
}


def _pairs(config: dict[str, Any], section: str) -> tuple[MethodPair, ...]:
    return tuple(MethodPair(**item) for item in config.get(section, ()))


def validate_pairing(config: dict[str, Any]) -> dict[str, tuple[MethodPair, ...]]:
    parsed: dict[str, tuple[MethodPair, ...]] = {}
    for section, expected in EXPECTED_SETS.items():
        pairs = _pairs(config, section)
        ar_names = [pair.ar for pair in pairs]
        if len(ar_names) != len(set(ar_names)):
            raise ValueError(f"{section} contains duplicate ARLLM entries")
        missing = sorted(expected - set(ar_names))
        extra = sorted(set(ar_names) - expected)
        if missing or extra:
            raise ValueError(f"{section} mismatch: missing={missing}, extra={extra}")
        if any(not pair.dllm for pair in pairs):
            raise ValueError(f"{section} contains an empty dLLM counterpart")
        parsed[section] = pairs

    main_relations = {pair.relation for pair in parsed["main_pairs"]}
    if not main_relations <= {"exact_rule", "matched_role", "adapted"}:
        raise ValueError("main_pairs contains an unknown relation")
    if None in main_relations:
        raise ValueError("every main pair must state its relation")

    run = config.get("run", {})
    if int(run.get("sample_count", 0)) != 32:
        raise ValueError("paired quality protocol must retain the 32-problem main split")
    if int(config.get("passk", {}).get("draws", 0)) != 8:
        raise ValueError("paired pass@k protocol must retain eight independent draws")
    if config.get("model", {}).get("architecture") != "masked_diffusion_moe":
        raise ValueError("the dLLM counterpart must identify its masked-diffusion MoE architecture")
    return parsed


def load_pairing(path: Path) -> tuple[dict[str, Any], dict[str, tuple[MethodPair, ...]]]:
    with path.open("rb") as source:
        config = tomllib.load(source)
    return config, validate_pairing(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gsm8k_llada_moe_3090.toml"),
    )
    args = parser.parse_args()
    _, sections = load_pairing(args.config)
    for section, pairs in sections.items():
        print(f"[{section}]")
        for pair in pairs:
            relation = f" ({pair.relation})" if pair.relation else ""
            print(f"{pair.ar} -> {pair.dllm}{relation}")


if __name__ == "__main__":
    main()
