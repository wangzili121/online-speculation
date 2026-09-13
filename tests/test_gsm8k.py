import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from experiments.arllm.gsm8k_reproduction import (
    _answer_counts,
    _apply_overrides,
    _minmax_rewards,
)
from inference_scaling.shared.evaluation import (
    CumulativeConsensusReward,
    GSM8KProblem,
    consensus_index,
    extract_numeric_answer,
    gsm8k_prompt,
    modal_answer,
    select_problems,
)


def test_extract_numeric_answer_prefers_explicit_final_markers() -> None:
    assert extract_numeric_answer("2 + 3 = 5\n#### 5") == Fraction(5)
    assert extract_numeric_answer(r"work 100 then \boxed{3/4}") == Fraction(3, 4)
    assert extract_numeric_answer("The final answer is $1,250.50") == Fraction(2501, 2)
    assert extract_numeric_answer("intermediate 7, last 9") == Fraction(9)


def test_training_and_evaluation_share_one_prompt_contract() -> None:
    prompt = gsm8k_prompt("What is 2 + 3?")
    assert prompt.startswith("What is 2 + 3?")
    assert "#### <number>" in prompt


def test_consensus_is_deterministic_and_uses_likelihood_for_representative() -> None:
    texts = ("#### 2", "#### 3", "#### 2")
    assert modal_answer([extract_numeric_answer(text) for text in texts]) == Fraction(2)
    assert consensus_index(texts, (-2.0, -0.1, -1.0)) == 2


def test_cumulative_consensus_reward_carries_counts_across_steps() -> None:
    decoded = {1: "#### 2", 2: "#### 3", 3: "#### 2"}
    reward = CumulativeConsensusReward(lambda tokens: decoded[tokens[0]])
    assert reward((), ((1,), (2,), (3,))) == (1.0, 0.0, 1.0)
    assert reward((), ((2,),)) == (0.0,)


def test_select_problems_is_seeded_and_retains_public_order() -> None:
    problems = tuple(
        GSM8KProblem(index, str(index), "#### 0", Fraction(0)) for index in range(20)
    )
    first = select_problems(problems, 6, seed=17)
    second = select_problems(problems, 6, seed=17)
    assert first == second
    assert [problem.index for problem in first] == sorted(problem.index for problem in first)


def test_confidence_reward_normalization_is_decision_local_and_stable() -> None:
    assert _minmax_rewards((-4.0, -2.0, -3.0)) == pytest.approx((0.0, 1.0, 0.5))
    assert _minmax_rewards((7.0, 7.0)) == (0.0, 0.0)


def test_best_of_n_answer_counts_are_json_stable_with_unparseable_outputs() -> None:
    counts = _answer_counts((Fraction(3), None, Fraction(3, 2), None))

    assert counts == {"3": 1, "3/2": 1, "<unparseable>": 2}
    assert json.loads(json.dumps({"answer_counts": counts}, sort_keys=True)) == {
        "answer_counts": counts
    }


def test_cli_can_disable_small_proposal_importance_correction() -> None:
    args = SimpleNamespace(
        backend=None,
        limit=None,
        max_new_tokens=None,
        sampling_temperature=None,
        num_beams=None,
        best_of_n_samples=None,
        conditional_reward=None,
        reward_temperature=None,
        importance_log_ratio_clip=None,
        disable_importance_correction=True,
        method="verifier_conditional_is_small_proposal",
        mh_alpha=None,
        mh_steps=None,
        candidate_count=None,
        rollout_count=None,
        block_size=None,
    )
    config = {
        "conditional_is": {
            "apply_importance_correction": True,
            "importance_log_ratio_clip": 10.0,
        }
    }

    _apply_overrides(config, args)

    assert config["conditional_is"]["apply_importance_correction"] is False
    assert config["conditional_is"]["importance_log_ratio_clip"] is None

    args.method = "verifier_conditional_is"
    with pytest.raises(ValueError, match="requires a small-proposal method"):
        _apply_overrides(config, args)


def test_cli_enables_fused_vllm_accounting_only_for_power_mh() -> None:
    args = SimpleNamespace(
        backend="vllm-sync",
        vllm_mh_fused_logprobs=True,
        limit=None,
        max_new_tokens=None,
        sampling_temperature=None,
        num_beams=None,
        best_of_n_samples=None,
        conditional_reward=None,
        reward_temperature=None,
        importance_log_ratio_clip=None,
        disable_importance_correction=False,
        method="mh",
        mh_alpha=None,
        mh_steps=None,
        candidate_count=None,
        rollout_count=None,
        block_size=None,
    )
    config = {"runtime": {}, "conditional_is": {}, "mh": {}}

    _apply_overrides(config, args)

    assert config["vllm"]["base"]["mh_fused_logprobs"] is True
    args.method = "base"
    with pytest.raises(ValueError, match="requires --method mh"):
        _apply_overrides(config, args)


def test_async_output_agreement_reports_token_and_answer_divergence() -> None:
    from experiments.arllm.gsm8k_async_benchmark import _output_agreement

    class Backend:
        @staticmethod
        def decode(tokens):
            return f"#### {tokens[-1]}"

    synchronous = [(1, 2, 3), (4, 5), (6,)]
    asynchronous = [(1, 2, 3), (4, 9), (6, 7)]
    problems = [SimpleNamespace(index=index) for index in (10, 20, 30)]

    result = _output_agreement(Backend(), synchronous, asynchronous, problems)

    assert result["outputs_bitwise_equal"] is False
    assert result["output_exact_match_count"] == 1
    assert result["output_exact_match_fraction"] == pytest.approx(1 / 3)
    assert result["answer_match_count"] == 1
    assert result["answer_match_fraction"] == pytest.approx(1 / 3)
    assert result["both_answers_parseable_count"] == 3
    assert result["mean_common_prefix_fraction"] == pytest.approx(2 / 3)
    assert [item["gsm8k_index"] for item in result["mismatches"]] == [20, 30]
