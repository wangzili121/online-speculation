from experiments.arllm.summarize_gsm8k_dynamic_is import METHODS, build_summary


def _method(correct: bool, *, hit: int, history: int, fresh: int, flops: int):
    return {
        "correct": correct,
        "steps": 1,
        "history_generated": 4,
        "history_used": history,
        "fresh_used": fresh,
        "candidate_cache_hits": hit,
        "nonterminal_candidates": 2,
        "auxiliary_candidates": hit,
        "candidate_count": 2,
        "outer_weight_ess_sum": 1.5,
        "final_weight_ess_sum": 1.25,
        "proxy_budget": 6.0,
        "proxy_cost_used": 6.0,
        "cache_build_seconds": 2.0,
        "design_seconds": 0.0,
        "steady_online_seconds": flops / 10.0,
        "online_total_seconds": flops / 10.0,
        "one_shot_seconds": 2.0 + flops / 10.0,
        "cache_build_forward_token_slots": 20,
        "design_forward_token_slots": 0,
        "steady_online_forward_token_slots": flops,
        "online_total_forward_token_slots": flops,
        "cache_build_estimated_dense_forward_flops": 200,
        "design_estimated_dense_forward_flops": 0,
        "steady_online_estimated_dense_forward_flops": flops,
        "online_total_estimated_dense_forward_flops": flops,
        "one_shot_estimated_dense_forward_flops": 200 + flops,
        "candidate_reproduction_all": True,
    }


def test_dynamic_summary_keeps_factor_direction_and_paired_quality() -> None:
    manifest = {
        "fingerprint": "fingerprint",
        "effective": {
            "config": {"run": {"name": "test", "seed": 9}},
            "problem_indices": [3, 7],
            "settings": {},
            "input_weight_sha256": {},
            "implementation_sha256": {},
        },
    }
    records = [
        {
            "problem_index": 3,
            "manifest_fingerprint": "fingerprint",
            "methods": {
                METHODS[0]: _method(False, hit=0, history=0, fresh=6, flops=100),
                METHODS[1]: _method(True, hit=1, history=2, fresh=4, flops=50),
                METHODS[2]: _method(True, hit=1, history=2, fresh=4, flops=40),
            },
        },
        {
            "problem_index": 7,
            "manifest_fingerprint": "fingerprint",
            "methods": {
                METHODS[0]: _method(True, hit=0, history=0, fresh=6, flops=100),
                METHODS[1]: _method(True, hit=1, history=2, fresh=4, flops=50),
                METHODS[2]: _method(False, hit=1, history=2, fresh=4, flops=40),
            },
        },
    ]

    summary = build_summary(manifest, records, bootstrap_replicates=1_000)

    fixed = summary["comparisons"]["replay_aware_fixed_vs_base_candidate_fixed"]
    optimal = summary["comparisons"][
        "replay_aware_optimal_vs_replay_aware_fixed"
    ]
    assert fixed["candidate_minus_reference_accuracy"] == 0.5
    assert fixed["base_over_replay_aware_steady_online_flop_factor"] == 2.0
    assert optimal["candidate_minus_reference_accuracy"] == -0.5
    assert optimal["fixed_over_optimal_steady_online_flop_factor"] == 1.25
    assert summary["methods"][METHODS[1]]["candidate_replay_hit_rate"] == 0.5


def test_dynamic_summary_rejects_incomplete_problem_grid() -> None:
    manifest = {
        "fingerprint": "fingerprint",
        "effective": {
            "config": {"run": {"name": "test", "seed": 9}},
            "problem_indices": [3],
            "settings": {},
            "input_weight_sha256": {},
            "implementation_sha256": {},
        },
    }
    try:
        build_summary(manifest, [], bootstrap_replicates=100)
    except ValueError as error:
        assert "problem grid" in str(error)
    else:
        raise AssertionError("incomplete records must not produce a formal summary")
