from experiments.arllm.gsm8k_passk import _estimated_pass_at_k
from experiments.arllm.summarize_gsm8k_compute import _comparison
from experiments.arllm.summarize_gsm8k_batching import _compare_method
from experiments.arllm.summarize_gsm8k_replay import (
    _minimum_warm_online_uses,
    _paired_quality,
)
from inference_scaling.shared.compute import (
    estimate_grpo_compute,
    estimate_grpo_compute_from_logs,
)


def test_grpo_compute_is_split_into_observed_forward_equivalents() -> None:
    compute = estimate_grpo_compute(
        model_sequence_tokens=1_000,
        generated_completions=10,
        total_parameters=100,
        trainable_parameters=5,
        optimizer_steps=2,
        gradient_checkpointing=True,
        reference_scoring=True,
    )

    assert compute.rollout_generation_forward_token_slots == 990
    assert compute.reference_scoring_forward_token_slots == 1_000
    assert compute.policy_forward_backward_equivalent_token_slots == 3_000
    assert compute.total_forward_equivalent_token_slots == 4_990
    assert compute.estimated_dense_model_flops == 998_000
    assert compute.estimated_optimizer_flops == 100
    assert compute.estimated_total_flops == 998_100


def test_grpo_logs_reconstruct_padded_generation_and_training_slots() -> None:
    compute = estimate_grpo_compute_from_logs(
        log_history=[
            {
                "step": 1,
                "num_tokens": 40,
                "completions/mean_length": 5,
                "completions/max_length": 6,
            },
            {
                "step": 2,
                "num_tokens": 90,
                "completions/mean_length": 6,
                "completions/max_length": 8,
            },
        ],
        sequences_per_optimizer_step=4,
        generated_completions=8,
        total_parameters=100,
        trainable_parameters=5,
        optimizer_steps=2,
        gradient_checkpointing=True,
        reference_scoring=True,
    )

    assert compute.rollout_generation_forward_token_slots == 94
    assert compute.reference_scoring_forward_token_slots == 102
    assert compute.policy_forward_backward_equivalent_token_slots == 306
    assert compute.total_forward_equivalent_token_slots == 502
    assert compute.estimated_dense_model_flops == 100_400
    assert compute.estimated_optimizer_flops == 100
    assert compute.estimated_total_flops == 100_500
    assert compute.accounting_basis.startswith("padded forward token slots")


def test_standard_pass_at_k_estimator() -> None:
    assert _estimated_pass_at_k(correct=1, draws=4, k=1) == 0.25
    assert _estimated_pass_at_k(correct=1, draws=4, k=2) == 0.5
    assert _estimated_pass_at_k(correct=1, draws=4, k=4) == 1.0


def test_compute_break_even_does_not_claim_distribution_match_without_audit() -> None:
    method = {
        "method": "verifier_mh",
        "examples": 2,
        "accuracy": 0.5,
        "total_forward_token_slots": 60,
        "estimated_dense_forward_flops": 600,
        "sum_example_seconds": 6.0,
    }
    rl = {
        "method": "rl_sample",
        "examples": 2,
        "accuracy": 0.5,
        "total_forward_token_slots": 20,
        "estimated_dense_forward_flops": 200,
        "sum_example_seconds": 2.0,
    }
    report = _comparison(
        method,
        rl,
        training_forward_token_slots=100,
        training_flops=1_000,
        training_seconds=10.0,
        accuracy_tolerance=0.05,
        answer_distribution=None,
        answer_tv_tolerance=0.25,
        answer_js_tolerance=0.2,
    )

    assert report["accuracy_matched"] is True
    assert report["accuracy_matched_flop_break_even_queries"] == 5
    assert report["answer_distribution_matched"] is None
    assert report["joint_empirical_match_flop_break_even_queries"] is None


def test_replay_cache_break_even_counts_repeated_warm_uses() -> None:
    assert _minimum_warm_online_uses(100.0, 30.0, 10.0) == 5
    assert _minimum_warm_online_uses(100.0, 10.0, 10.0) is None
    assert _minimum_warm_online_uses(100.0, 9.0, 10.0) is None


def test_replay_quality_is_paired_and_keeps_answer_agreement() -> None:
    records = [
        {
            "fresh": {"correct": True, "prediction": "1"},
            "warm_replay": {"correct": True, "prediction": "1"},
        },
        {
            "fresh": {"correct": True, "prediction": "2"},
            "warm_replay": {"correct": False, "prediction": "3"},
        },
        {
            "fresh": {"correct": False, "prediction": "4"},
            "warm_replay": {"correct": True, "prediction": "4"},
        },
        {
            "fresh": {"correct": True, "prediction": "5"},
            "warm_replay": {"correct": False, "prediction": "6"},
        },
    ]
    report = _paired_quality(records, bootstrap_samples=1_000)

    assert report["fresh_accuracy"] == 0.75
    assert report["warm_accuracy"] == 0.5
    assert report["warm_minus_fresh_accuracy"] == -0.25
    assert report["same_numeric_prediction_rate"] == 0.5


def test_batching_comparison_names_both_speedup_denominators() -> None:
    synchronous = {
        "estimated_dense_forward_flops": 100,
        "base_backend": {"shared_prefill_tokens_saved": 10},
        "proposal_backend": None,
    }
    baseline = {
        "asynchronous_continuous_batching_seconds": 12.0,
        "asynchronous_compute": {
            "estimated_dense_forward_flops": 200,
            "base_backend": {"shared_prefill_tokens_saved": 2},
            "proposal_backend": None,
        },
    }
    grouped = {
        "asynchronous_continuous_batching_seconds": 8.0,
        "wall_time_speedup_synchronous_over_asynchronous": 1.25,
        "synchronous_compute": synchronous,
        "asynchronous_compute": {
            "estimated_dense_forward_flops": 110,
            "base_backend": {"shared_prefill_tokens_saved": 10},
            "proposal_backend": None,
        },
        "output_exact_match_count": 4,
        "answer_match_count": 4,
        "synchronous_accuracy": 0.5,
        "asynchronous_accuracy": 0.5,
    }

    report = _compare_method(baseline, grouped)

    assert report["baseline_over_grouped_async_wall_time_factor"] == 1.5
    assert report["grouped_sync_over_async_wall_time_factor"] == 1.25
    assert report["baseline_over_grouped_async_flop_factor"] == 200 / 110
    assert report["grouped_async_over_sync_flop_factor"] == 1.1


def test_compute_break_even_requires_both_answer_distribution_thresholds() -> None:
    summary = {
        "method": "method",
        "examples": 1,
        "accuracy": 1.0,
        "total_forward_token_slots": 2,
        "estimated_dense_forward_flops": 20,
        "sum_example_seconds": 2.0,
    }
    rl = {**summary, "method": "rl", "total_forward_token_slots": 1,
          "estimated_dense_forward_flops": 10, "sum_example_seconds": 1.0}
    report = _comparison(
        summary,
        rl,
        training_forward_token_slots=1,
        training_flops=10,
        training_seconds=1.0,
        accuracy_tolerance=0.0,
        answer_distribution={
            "mean_total_variation": 0.1,
            "mean_jensen_shannon_bits": 0.3,
        },
        answer_tv_tolerance=0.25,
        answer_js_tolerance=0.2,
    )

    assert report["accuracy_matched"] is True
    assert report["answer_distribution_matched"] is False
    assert report["joint_accuracy_and_answer_distribution_match"] is False
