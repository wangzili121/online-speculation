from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pytest

import experiments.arllm.run_gsm8k_suite as ar_matrix
import experiments.run_reproduction as reproduction_entry
from experiments.arllm.run_arllm_suite import build_commands as build_ar_commands
from experiments.run_reproduction import (
    _default_python,
    build_commands as build_paired_commands,
)
from experiments.arllm.run_gsm8k_suite import SUPPORTED_METHODS
from experiments.shared.methods import AR_DEFAULT_METHODS


def _ar_args(**overrides):
    values = {
        "stage": "all",
        "profile": "full",
        "config": Path("configs/gsm8k_3090_aligned.toml"),
        "training_config": Path("configs/gsm8k_grpo.toml"),
        "resume": "auto",
        "train_limit": None,
        "training_output": None,
        "max_train_steps": None,
        "num_generations": None,
        "max_completion_length": None,
        "components": (
            "quality",
            "matched_target",
            "replay",
            "dynamic_is",
            "async",
            "passk",
            "ablations",
            "budget_curve",
            "length_ablation",
            "distribution",
            "infra",
            "vllm",
        ),
        "methods": ("base", "mh", "conditional_is", "rl_sample"),
        "tag": "test",
        "summary_root": Path("results/test"),
        "ablation_limit": 3,
        "passk_limit": 4,
        "passk_draws": 2,
        "backend": "transformers",
        "ar_python": "python-ar",
        "dllm_python": "python-dllm",
        "limit": 5,
        "distribution_problems": 2,
        "distribution_draws": 3,
        "dtype": "bfloat16",
        "infra_limit": 1,
        "vllm_limit": 6,
        "vllm_workers": 2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_ar_full_entry_preserves_training_and_every_suite_family(tmp_path):
    commands = build_ar_commands(_ar_args(summary_root=tmp_path), Path.cwd())
    scripts = [Path(command[1]).name for command in commands]

    assert scripts == [
        "prepare_gsm8k.py",
        "train_gsm8k_grpo.py",
        "run_gsm8k_suite.py",
        "gsm8k_distribution_audit.py",
        "benchmark_rollout_infra.py",
        "benchmark_is_mh_reuse.py",
        "run_vllm_backend_benchmark.py",
    ]
    suite = commands[2]
    for flag in (
        "--with-matched-target",
        "--with-replay",
        "--with-dynamic-is",
        "--with-async",
        "--with-passk",
        "--with-ablations",
        "--with-budget-curve",
        "--with-length-ablation",
    ):
        assert flag in suite
    assert suite[suite.index("--methods") + 1] == "base,mh,conditional_is,rl_sample"
    assert suite[suite.index("--profile") + 1] == "full"
    assert suite[suite.index("--mh-suffix-schedule") + 1] == "multiscale"


def test_shared_model_and_scope_flags_reach_every_ar_command(tmp_path):
    args = _ar_args(summary_root=tmp_path, model="org/model", model_revision="fixed",
                    tokenizer="org/tokenizer", sampling_scope="thinking", max_new_tokens=32768,
                    mh_iterations=5, allow_download=False, thinking_mode="enabled")
    for command in build_ar_commands(args, Path.cwd()):
        for flag, value in (("--model", "org/model"), ("--model-revision", "fixed"),
                            ("--tokenizer", "org/tokenizer"), ("--sampling-scope", "thinking"),
                            ("--max-new-tokens", "32768"), ("--mh-iterations", "5")):
            assert command[command.index(flag) + 1] == value
        assert "--no-allow-download" in command


def test_model_override_removes_stale_identity_and_honors_offline_override():
    from experiments.shared.model_cli import apply_model_output_overrides
    config = {"models": {"base": "old", "base_weight_sha256": "old-hash", "base_revision": "old-ref", "rl_base": "old"},
              "model_loading": {"base": {"local_files_only": True}}}
    apply_model_output_overrides(config, argparse.Namespace(model="org/new", model_revision="commit", allow_download=True))
    assert config["models"]["base"] == config["models"]["rl_base"] == "org/new"
    assert "base_weight_sha256" not in config["models"]
    assert config["model_loading"]["base"]["revision"] == "commit"
    assert config["model_loading"]["base"]["local_files_only"] is False


def test_ar_component_without_quality_does_not_run_main_methods(tmp_path):
    commands = build_ar_commands(
        _ar_args(
            stage="inference",
            components=("replay",),
            summary_root=tmp_path,
        ),
        Path.cwd(),
    )

    assert len(commands) == 1
    assert commands[0][commands[0].index("--methods") + 1] == ""
    assert "--with-replay" in commands[0]


def test_ar_entry_accepts_existing_verifier_methods(tmp_path):
    commands = build_ar_commands(
        _ar_args(
            stage="inference",
            components=("quality",),
            methods=("verifier_mh", "verifier_conditional_is"),
            summary_root=tmp_path,
        ),
        Path.cwd(),
    )

    methods = commands[0][commands[0].index("--methods") + 1]
    assert methods == "verifier_mh,verifier_conditional_is"
    assert {
        "verifier_mh",
        "verifier_conditional_is",
        "verifier_conditional_is_small_proposal",
    } <= set(SUPPORTED_METHODS)


def test_ar_training_output_is_reused_by_quality_passk_and_distribution(tmp_path):
    adapter = tmp_path / "adapter"
    commands = build_ar_commands(
        _ar_args(training_output=adapter, summary_root=tmp_path),
        Path.cwd(),
    )

    suite = commands[2]
    distribution = commands[3]
    assert suite[suite.index("--rl-adapter") + 1] == str(adapter)
    assert distribution[distribution.index("--rl-adapter") + 1] == str(adapter)


def test_verifier_configuration_is_routed_to_all_reward_consumers(tmp_path):
    verifier = tmp_path / "verifier.toml"
    ar_commands = build_ar_commands(
        _ar_args(verifier_config=verifier, summary_root=tmp_path), Path.cwd()
    )
    by_script = {Path(command[1]).name: command for command in ar_commands}
    for script in (
        "run_gsm8k_suite.py",
        "gsm8k_distribution_audit.py",
        "benchmark_rollout_infra.py",
    ):
        command = by_script[script]
        assert command[command.index("--verifier-config") + 1] == str(verifier)

    paired = build_paired_commands(
        _paired_args(verifier_config=verifier), Path.cwd()
    )
    ar_command = next(
        command for command in paired if Path(command[1]).name == "run_arllm_suite.py"
    )
    dllm_command = next(
        command for command in paired if Path(command[1]).name == "run_llada_suite.py"
    )
    assert ar_command[ar_command.index("--verifier-config") + 1] == str(verifier)
    assert dllm_command[dllm_command.index("--verifier-config") + 1] == str(verifier)


def test_ar_passk_component_runs_general_and_is_variant_grids(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        ar_matrix,
        "_run",
        lambda command, _environment: commands.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gsm8k_suite.py",
            "--config",
            "configs/gsm8k_quick.toml",
            "--methods",
            "",
            "--profile",
            "smoke",
            "--with-passk",
            "--mh-suffix-schedule",
            "multiscale",
            "--passk-limit",
            "1",
            "--passk-draws",
            "2",
            "--summary-root",
            str(tmp_path),
        ],
    )

    ar_matrix.main()

    assert [Path(command[1]).name for command in commands] == [
        "gsm8k_passk.py",
        "gsm8k_is_passk.py",
    ]
    general_grid = commands[0]
    assert Path(general_grid[general_grid.index("--output") + 1]).parent == tmp_path
    assert (
        general_grid[general_grid.index("--mh-suffix-schedule") + 1]
        == "multiscale"
    )
    is_grid = commands[1]
    assert is_grid[is_grid.index("--workers") + 1] == "2"
    assert Path(is_grid[is_grid.index("--output") + 1]).parent == tmp_path


def test_ar_smoke_profile_exercises_each_sweep_with_bounded_lengths(
    monkeypatch, tmp_path
):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        ar_matrix,
        "_run",
        lambda command, _environment: commands.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gsm8k_suite.py",
            "--config",
            "configs/gsm8k_quick.toml",
            "--methods",
            "",
            "--profile",
            "smoke",
            "--with-ablations",
            "--with-budget-curve",
            "--with-length-ablation",
            "--ablation-limit",
            "1",
            "--summary-root",
            str(tmp_path),
        ],
    )

    ar_matrix.main()

    tags = {
        command[command.index("--tag") + 1]
        for command in commands
        if "--tag" in command
    }
    assert {
        "default-alpha-2",
        "default-steps-1",
        "default-candidates-3-rollouts-3",
        "default-candidates-10-rollouts-1",
        "default-guidance-steps-2",
        "default-conditional_is-reward-verifier",
        "default-conditional_is-reward-sequence_log_probability",
        "default-conditional_is-reward-consilience",
        "default-beam-temperature-0.7",
        "default-budget-beam-4",
        "default-budget-best-of-n-4",
        "default-budget-conditional_is-m3-k3",
        "default-length-32",
    } <= tags
    length_commands = [
        command for command in commands if "default-length-32" in command
    ]
    assert len(length_commands) == 6
    assert all(
        command[command.index("--max-new-tokens") + 1] == "32"
        for command in length_commands
    )


def test_ar_adapter_is_not_forwarded_to_replay_or_dynamic_scripts(
    monkeypatch, tmp_path
):
    commands: list[list[str]] = []
    adapter = tmp_path / "adapter"
    monkeypatch.setattr(
        ar_matrix,
        "_run",
        lambda command, _environment: commands.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gsm8k_suite.py",
            "--config",
            "configs/gsm8k_quick.toml",
            "--methods",
            "rl_sample",
            "--rl-adapter",
            str(adapter),
            "--with-replay",
            "--with-dynamic-is",
            "--summary-root",
            str(tmp_path),
        ],
    )

    ar_matrix.main()

    by_script = {Path(command[1]).name: command for command in commands}
    assert "--rl-adapter" in by_script["gsm8k_reproduction.py"]
    assert "--rl-adapter" not in by_script["gsm8k_replay_benchmark.py"]
    assert "--rl-adapter" not in by_script["gsm8k_dynamic_is_benchmark.py"]


def _paired_args(**overrides):
    values = {
        "family": "both",
        "stage": "all",
        "profile": "full",
        "tag": "paired",
        "ar_training_config": Path("configs/gsm8k_grpo.toml"),
        "ar_config": Path("configs/gsm8k_3090_aligned.toml"),
        "dllm_config": Path("configs/gsm8k_llada_moe_3090.toml"),
        "output_root": Path("results/reproduction"),
        "ar_methods": ("base", "rl_sample"),
        "dllm_methods": ("base", "conditional_is"),
        "components": ("quality", "replay"),
        "backend": None,
        "ar_python": "python-ar",
        "dllm_python": "python-dllm",
        "limit": 3,
        "train_limit": None,
        "ar_training_output": None,
        "max_train_steps": None,
        "num_generations": None,
        "max_completion_length": None,
        "ablation_limit": None,
        "passk_limit": None,
        "passk_draws": None,
        "distribution_problems": None,
        "distribution_draws": None,
        "infra_limit": 1,
        "dry_run": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_paired_full_entry_routes_grpo_and_vrpo_training():
    commands = build_paired_commands(_paired_args(), Path.cwd())

    assert len(commands) == 3
    assert commands[0][0] == "python-ar"
    assert commands[1][0] == "python-dllm"
    assert Path(commands[0][1]).name == "run_arllm_suite.py"
    assert "--stage" in commands[0] and "all" in commands[0]
    assert (
        commands[0][commands[0].index("--mh-suffix-schedule") + 1]
        == "multiscale"
    )
    assert Path(commands[1][1]).name == "download_llada.py"
    assert Path(commands[2][1]).name == "run_llada_suite.py"
    assert commands[2][commands[2].index("--vrpo") + 1] == "train"
    assert "--methods" in commands[2]
    assert "--no-with-replay" not in commands[2]


def test_paired_smoke_uses_cpu_vrpo_preflight_instead_of_full_training():
    commands = build_paired_commands(
        _paired_args(family="dllm", profile="smoke"), Path.cwd()
    )

    assert len(commands) == 2
    assert Path(commands[0][1]).name == "download_llada.py"
    assert Path(commands[1][1]).name == "run_llada_suite.py"
    assert commands[1][commands[1].index("--vrpo") + 1] == "preflight"


def test_dllm_inference_with_aligned_method_loads_existing_adapter():
    commands = build_paired_commands(
        _paired_args(
            family="dllm",
            stage="inference",
            dllm_methods=("vrpo_sample", "vrpo_greedy"),
        ),
        Path.cwd(),
    )

    assert len(commands) == 1
    assert "--with-aligned" in commands[0]
    assert commands[0][commands[0].index("--vrpo") + 1] == "skip"


def test_dllm_smoke_training_stage_is_preflight_only():
    commands = build_paired_commands(
        _paired_args(family="dllm", stage="train", profile="smoke"), Path.cwd()
    )

    assert len(commands) == 1
    assert Path(commands[0][1]).name == "train_gsm8k_vrpo.py"
    assert "--preflight" in commands[0]


def test_dllm_prepare_stage_downloads_or_validates_model():
    commands = build_paired_commands(
        _paired_args(family="dllm", stage="prepare", profile="full"), Path.cwd()
    )

    assert len(commands) == 1
    assert Path(commands[0][1]).name == "download_llada.py"
    assert "--config" in commands[0]
    assert "--validate-only" not in commands[0]


def test_paired_entry_routes_supported_components_to_dllm_suite():
    commands = build_paired_commands(
        _paired_args(
            family="dllm",
            stage="inference",
            components=("passk", "distribution"),
            passk_limit=4,
            passk_draws=3,
            distribution_problems=2,
            distribution_draws=5,
        ),
        Path.cwd(),
    )

    command = commands[0]
    component_index = command.index("--components")
    assert command[component_index + 1 : component_index + 3] == [
        "passk",
        "distribution",
    ]
    assert command[command.index("--passk-draws") + 1] == "3"
    assert command[command.index("--distribution-draws") + 1] == "5"


def test_dllm_only_entry_rejects_ar_backend_option():
    with pytest.raises(ValueError, match="applies only"):
        build_paired_commands(
            _paired_args(family="dllm", backend="transformers"),
            Path.cwd(),
        )


def test_interpreter_default_supports_environment_and_current_python(monkeypatch):
    monkeypatch.delenv("AR_PYTHON", raising=False)
    assert _default_python("AR_PYTHON") == sys.executable

    monkeypatch.setenv("AR_PYTHON", "custom-ar-python")
    assert _default_python("AR_PYTHON") == "custom-ar-python"

    monkeypatch.setenv("AR_PYTHON", "")
    assert _default_python("AR_PYTHON") == sys.executable


def test_unified_entry_defaults_to_the_registered_qwen_methods(monkeypatch, tmp_path):
    captured = {}

    def record_plan(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(reproduction_entry, "run_manifested_commands", record_plan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_reproduction.py",
            "--family",
            "arllm",
            "--profile",
            "smoke",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
    )

    reproduction_entry.main()

    assert captured["metadata"]["family"] == "arllm"
    assert tuple(captured["metadata"]["ar_methods"]) == AR_DEFAULT_METHODS
    assert len(captured["commands"]) == 1
    assert "iterated_conditional_is" not in captured["commands"][0]


def test_public_entrypoints_do_not_require_pythonpath(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for script in (
        root / "experiments" / "run_reproduction.py",
        root / "experiments" / "arllm" / "run_arllm_suite.py",
        root / "experiments" / "dllm" / "run_llada_suite.py",
        root / "experiments" / "dllm" / "gsm8k_reproduction.py",
        root / "experiments" / "dllm" / "gsm8k_analysis.py",
        root / "experiments" / "dllm" / "gsm8k_replay_benchmark.py",
        root / "experiments" / "dllm" / "benchmark_infra.py",
        root / "experiments" / "dllm" / "prepare_gsm8k_vrpo.py",
        root / "experiments" / "dllm" / "train_gsm8k_vrpo.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
