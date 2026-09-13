from inference_scaling.dllm.preferences import select_scored_preference_pair


def test_highest_scored_rollout_is_preferred_when_group_has_both_outcomes():
    pair = select_scored_preference_pair(
        candidate_texts=("work\n#### 12", "wrong\n#### 7"),
        candidate_rewards=(1.0, 0.0),
        reference_text="gold\n#### 12",
        reference_reward=1.0,
    )

    assert pair is not None
    assert pair.chosen == "work\n#### 12"
    assert pair.chosen_source == "candidate:0"
    assert pair.rejected_source == "candidate:1"


def test_scored_reference_completion_fills_missing_positive():
    pair = select_scored_preference_pair(
        candidate_texts=("wrong\n#### 7", "also wrong\n#### 8"),
        candidate_rewards=(0.0, 0.0),
        reference_text="gold\n#### 12",
        reference_reward=1.0,
    )

    assert pair is not None
    assert pair.chosen == "gold\n#### 12"
    assert pair.chosen_source == "dataset_reference_completion"


def test_equal_reward_group_is_omitted():
    pair = select_scored_preference_pair(
        candidate_texts=("first\n#### 12", "second\n#### 12"),
        candidate_rewards=(1.0, 1.0),
        reference_text="gold\n#### 12",
        reference_reward=1.0,
    )

    assert pair is None
