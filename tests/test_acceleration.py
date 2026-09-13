from __future__ import annotations

from collections import Counter

import pytest

from inference_scaling.arllm.acceleration import (
    ActiveBatchSpeculationConfig,
    LowPriorityRunAheadBackend,
    RolloutTokenTree,
    SpeculationTier,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest


def test_rollout_tree_uses_longest_suffix_and_tracks_verification() -> None:
    tree = RolloutTokenTree(
        max_context_tokens=4,
        max_contexts=100,
        min_context_tokens=1,
        min_token_probability=0.5,
    )
    tree.observe((9, 1, 2, 3, 4))
    tree.observe((8, 1, 2, 3, 5))
    tree.observe((7, 2, 3, 4, 6))

    proposal = tree.draft((0, 1, 2, 3), 3)

    assert proposal.token_ids[:1] == (4,)
    assert proposal.matched_context_tokens == 3
    tree.record_verification(proposed=len(proposal.token_ids), accepted=1)
    snapshot = tree.snapshot()
    assert snapshot.hits == 1
    assert snapshot.proposed_tokens == len(proposal.token_ids)
    assert snapshot.accepted_tokens == 1


def test_stochastic_rollout_tree_returns_its_full_empirical_proposal() -> None:
    tree = RolloutTokenTree(
        max_context_tokens=4,
        max_contexts=100,
        min_context_tokens=1,
        min_token_probability=0.0,
    )
    for _ in range(8):
        tree.observe((7, 8, 0))
    for _ in range(2):
        tree.observe((7, 8, 1))

    draws = [
        tree.draft((7, 8), 1, stochastic=True, seed=seed)
        for seed in range(2000)
    ]
    counts = Counter(proposal.token_ids[0] for proposal in draws)

    assert counts[0] / len(draws) == pytest.approx(0.8, abs=0.035)
    assert draws[0].token_distributions[0] == ((0, 0.8), (1, 0.2))
    assert all(
        proposal.token_probabilities[0]
        == dict(proposal.token_distributions[0])[proposal.token_ids[0]]
        for proposal in draws
    )


def test_active_batch_schedule_is_shared_with_vllm_schema() -> None:
    config = ActiveBatchSpeculationConfig(
        tiers=(SpeculationTier(2, 8), SpeculationTier(5, 3), SpeculationTier(12, 0))
    )
    assert config.draft_tokens(1) == 8
    assert config.draft_tokens(4) == 3
    assert config.draft_tokens(8) == 0
    native = config.vllm_suffix_config(dynamic=True)
    assert native["method"] == "custom_class"
    assert native["model"].endswith("DynamicSuffixDecodingProposer")
    assert native["num_speculative_tokens_per_batch_size"] == [
        [1, 2, 8],
        [3, 5, 3],
        [6, 12, 0],
    ]
    static = config.vllm_suffix_config()
    assert static["method"] == "suffix"
    assert "model" not in static
    assert "num_speculative_tokens_per_batch_size" not in static


def test_vllm_config_rejects_the_transformers_only_stochastic_tree() -> None:
    config = ActiveBatchSpeculationConfig(stochastic_tree=True)
    with pytest.raises(ValueError, match="Transformers verifier"):
        config.vllm_suffix_config()


def test_low_priority_run_ahead_populates_only_the_draft_tree() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2], model_id="base")
    tree = RolloutTokenTree(
        max_context_tokens=4,
        max_contexts=100,
        min_context_tokens=1,
    )
    wrapper = LowPriorityRunAheadBackend(backend, tree, chunk_tokens=2)
    try:
        accepted = wrapper.submit_run_ahead(
            [GenerationRequest((1,), 4, SamplingConfig(), 7, "idle")]
        )
        wrapper.wait_for_run_ahead()
        snapshot = wrapper.snapshot()
        assert accepted == 1
        assert snapshot.completed_requests == 1
        assert snapshot.completed_tokens == 4
        assert tree.snapshot().observed_sequences >= 2

        foreground = wrapper.sample_batch(
            [GenerationRequest((1,), 1, SamplingConfig(), 9, "foreground")]
        )
        assert len(foreground) == 1
        assert wrapper.snapshot().critical_calls == 1
    finally:
        wrapper.close()


def test_run_ahead_can_reuse_a_backend_managed_draft_store() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2], model_id="base")
    with pytest.raises(ValueError, match="draft tree"):
        LowPriorityRunAheadBackend(backend, None)

    wrapper = LowPriorityRunAheadBackend(
        backend,
        None,
        chunk_tokens=2,
        outputs_already_observed=True,
    )
    try:
        wrapper.submit_run_ahead(
            [GenerationRequest((1,), 2, SamplingConfig(), 7, "managed")]
        )
        wrapper.wait_for_run_ahead()
        assert wrapper.snapshot().completed_tokens == 2
    finally:
        wrapper.close()
