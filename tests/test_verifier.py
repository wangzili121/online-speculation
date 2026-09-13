from pathlib import Path

import pytest

from inference_scaling.shared.verifier import (
    ConfiguredTrainingVerifierReward,
    ConfiguredVerifier,
    VerifierContext,
    VerifierInput,
    VerifierSpec,
    build_token_verifier_reward,
    build_verifier,
    load_verifier_table,
    replace_verifier_from_file,
)


def test_numeric_reference_verifier_is_selected_only_by_configuration() -> None:
    config = {
        "verifier": {
            "provider": "python",
            "name": "test_numeric_reference",
            "factory": (
                "inference_scaling.shared.evaluation.numeric:"
                "build_numeric_reference_verifier"
            ),
            "requires_reference": True,
            "options": {
                "correct_reward": 3.0,
                "incorrect_reward": -1.0,
                "unparseable_reward": -2.0,
            },
        }
    }
    decoded = {0: "question", 1: "work\n#### 5", 2: "#### 4", 3: "unknown"}
    reward = build_token_verifier_reward(
        config,
        context=VerifierContext(prompt="question", reference="5"),
        decoder=lambda tokens: decoded[tokens[0]],
    )

    assert reward((0,), (1,)) == 3.0
    assert reward.batch((0,), ((1,), (2,), (3,))) == (3.0, -1.0, -2.0)
    assert reward.describe()["factory"] == config["verifier"]["factory"]


def test_constant_verifier_needs_no_dataset_or_reference() -> None:
    spec = VerifierSpec.from_mapping(
        {
            "provider": "constant",
            "name": "control",
            "value": 0.25,
        }
    )
    verifier = build_verifier(spec, context=VerifierContext(prompt="any task"))

    assert verifier.score("prompt", "completion") == 0.25
    assert verifier.score_batch(
        (VerifierInput("a", "b"), VerifierInput("c", "d"))
    ) == (0.25, 0.25)


def test_reference_requirement_fails_before_scoring() -> None:
    spec = VerifierSpec.from_mapping(
        {
            "provider": "python",
            "name": "reference",
            "factory": (
                "inference_scaling.shared.evaluation.numeric:"
                "build_numeric_reference_verifier"
            ),
            "requires_reference": True,
        }
    )
    with pytest.raises(ValueError, match="requires a reference"):
        build_verifier(spec, context=VerifierContext(prompt="question"))


def test_verifier_configuration_rejects_unknown_fields_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="unknown verifier"):
        VerifierSpec.from_mapping(
            {"provider": "constant", "value": 0.0, "typo": True}
        )
    spec = VerifierSpec.from_mapping(
        {"provider": "constant", "name": "bad", "value": "nan"}
    )
    with pytest.raises(ValueError, match="non-finite"):
        build_verifier(spec, context=VerifierContext(prompt="question"))
    with pytest.raises(TypeError, match="must be a boolean"):
        VerifierSpec.from_mapping(
            {
                "provider": "constant",
                "value": 0.0,
                "requires_reference": "false",
            }
        )


def test_verifier_batch_must_preserve_input_cardinality() -> None:
    spec = VerifierSpec.from_mapping(
        {"provider": "constant", "name": "shape-check", "value": 0.0}
    )
    verifier = ConfiguredVerifier(spec, lambda _prompt, _completion: 0.0, lambda _: ())

    with pytest.raises(ValueError, match="returned 0 rewards for 1 inputs"):
        verifier.score_batch((VerifierInput("prompt", "completion"),))


def test_training_adapter_uses_the_same_configured_verifier_and_tracks_cost() -> None:
    spec = VerifierSpec.from_mapping(
        {
            "provider": "python",
            "name": "training_numeric_reference",
            "factory": (
                "inference_scaling.shared.evaluation.numeric:"
                "build_numeric_reference_verifier"
            ),
            "requires_reference": True,
        }
    )
    reward = ConfiguredTrainingVerifierReward(spec, reference_field="gold_answer")
    prompts = (
        ({"role": "user", "content": "first"},),
        ({"role": "user", "content": "first"},),
    )
    completions = (
        ({"role": "assistant", "content": "work\n#### 5"},),
        ({"role": "assistant", "content": "work\n#### 4"},),
    )

    values = reward(
        prompts,
        completions,
        ((1, 2, 3), (4, 5)),
        gold_answer=("5", "5"),
    )

    assert values == [1.0, 0.0]
    assert reward.snapshot(num_generations=2) == {
        "reward_calls": 1,
        "generated_completions": 2,
        "generated_prompt_groups": 1,
        "generated_completion_tokens": 5,
        "reward_sum": 1.0,
        "observed_mean_reward": 0.5,
        "observed_minimum_reward": 0.0,
        "observed_maximum_reward": 1.0,
    }


def test_training_adapter_does_not_require_a_reference_column_when_disabled() -> None:
    spec = VerifierSpec.from_mapping(
        {"provider": "constant", "name": "training-control", "value": 0.25}
    )
    reward = ConfiguredTrainingVerifierReward(spec)

    assert reward(prompts=("prompt",), completions=("completion",)) == [0.25]


def test_standalone_verifier_file_replaces_only_that_component(tmp_path: Path) -> None:
    source = tmp_path / "verifier.toml"
    source.write_text(
        """
[verifier]
provider = "constant"
name = "replacement"
value = 0.75
requires_reference = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert load_verifier_table(source)["name"] == "replacement"
    config = {"run": {"seed": 7}, "verifier": {"old": True}}

    replace_verifier_from_file(config, source)

    assert config["run"] == {"seed": 7}
    assert config["verifier"]["provider"] == "constant"
    assert config["verifier"]["value"] == 0.75
