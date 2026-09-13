from __future__ import annotations

from experiments.dllm.profiles import apply_execution_profile


def _config():
    return {
        "run": {"sample_count": 32},
        "generation": {"max_new_tokens": 192, "block_length": 48, "denoising_steps": 12},
        "exact_policy": {"block_length": 48, "denoising_steps": 12},
        "search": {"width": 8, "branching_factor": 2, "decision_block_size": 48},
        "best_of_n": {"samples": 8},
        "mh": {"decision_block_size": 48, "updates_per_stage": 12, "updates": 48},
        "conditional_is": {"candidate_count": 8, "rollout_count": 3, "decision_block_size": 48},
        "replay": {"history_rollouts": 2, "fresh_rollouts": 1},
        "passk": {"draws": 8, "k": [1, 2, 4, 8]},
    }


def test_smoke_profile_retains_a_nonterminal_rollout_stage():
    original = _config()
    smoke = apply_execution_profile(original, "smoke")

    assert smoke["run"]["sample_count"] == 1
    assert smoke["generation"]["max_new_tokens"] == 96
    assert smoke["generation"]["denoising_steps"] == 4
    assert smoke["conditional_is"]["candidate_count"] == 2
    assert smoke["conditional_is"]["rollout_count"] == 1
    assert smoke["generation"]["max_new_tokens"] > smoke["conditional_is"]["decision_block_size"]
    assert original["run"]["sample_count"] == 32
    assert original["generation"]["denoising_steps"] == 12


def test_full_profile_is_an_isolated_copy():
    original = _config()
    full = apply_execution_profile(original, "full")

    assert full == original
    assert full is not original
    full["run"]["sample_count"] = 1
    assert original["run"]["sample_count"] == 32
