from __future__ import annotations

from collections import Counter
from math import log

from inference_scaling.experimental.arllm.smc_forest import run_smc_rollout_forest
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SMCForestConfig
from inference_scaling.shared.metrics import total_variation
from inference_scaling.shared.rng import SeedStream


def test_smc_terminal_target_converges_to_base_times_reward() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5], model_id="base")
    expected = {0: 0.2, 1: 0.8}
    counts: Counter[int] = Counter()
    trials = 350
    config = SMCForestConfig(
        particle_count=64,
        branch_factor=1,
        rollout_count=1,
        block_size=1,
        total_length=1,
    )
    for trial in range(trials):
        result = run_smc_rollout_forest(
            backend,
            (),
            config,
            lambda _prompt, generated: log(4.0) if generated[0] == 1 else 0.0,
            SeedStream(20_000 + trial),
            streaming_rewards=False,
        )
        counts[result.token_ids[0]] += 1
    empirical = {token: value / trials for token, value in counts.items()}
    assert total_variation(empirical, expected) < 0.06


def test_rollout_forest_reuses_suffixes_and_reduces_fresh_generation() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.4], model_id="base")
    common = dict(
        particle_count=6,
        branch_factor=2,
        rollout_count=3,
        block_size=1,
        total_length=3,
    )
    reused = run_smc_rollout_forest(
        backend,
        (),
        SMCForestConfig(**common, reuse_rollout_forest=True),
        lambda _prompt, generated: float(sum(generated)),
        SeedStream(71),
        streaming_rewards=False,
    )
    fresh = run_smc_rollout_forest(
        backend,
        (),
        SMCForestConfig(**common, reuse_rollout_forest=False),
        lambda _prompt, generated: float(sum(generated)),
        SeedStream(71),
        streaming_rewards=False,
    )

    assert reused.reused_rollouts > 0
    assert reused.fresh_rollouts < fresh.fresh_rollouts
    assert len(reused.token_ids) == 3
