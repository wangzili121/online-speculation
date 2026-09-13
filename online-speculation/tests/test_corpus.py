import pytest
import json

from blockspec.corpus import assert_disjoint, convert_row, question_hash, split_for_question
from blockspec.data import assert_split_files_disjoint


class StubTokenizer:
    def render_chat(self, messages):
        return "".join(x["role"] + ":" + x["content"] for x in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(x) for x in text]


def test_same_question_answers_always_share_split():
    assert question_hash("What is 2 + 2?") == question_hash("  What  is 2 + 2?\n")
    rows = [{"row_idx": i, "row": {"conversations": [{"from": "human", "value": "What is 2 + 2?"},
                                                    {"from": "gpt", "value": answer}]}}
            for i, answer in enumerate(["First answer" * 30, "Another answer" * 30])]
    converted = [convert_row(r, StubTokenizer())[0] for r in rows]
    assert converted[0]["text_sha256"] != converted[1]["text_sha256"]
    assert converted[0]["group_sha256"] == converted[1]["group_sha256"]
    assert converted[0]["split"] == converted[1]["split"]
    assert_disjoint(converted)


def test_group_leakage_rejected():
    with pytest.raises(ValueError, match="multiple splits"):
        assert_disjoint([{"group_sha256": "x", "split": "train"},
                         {"group_sha256": "x", "split": "validation"}])


def test_truncated_viewer_cells_not_silently_trained():
    assert convert_row({"truncated_cells": ["conversations"]}, StubTokenizer()) == (None, "viewer_truncated")


def test_all_splits_populated_and_assignment_is_deterministic():
    assignments = [split_for_question(f"question {i}") for i in range(100)]
    assert set(assignments) == {"train", "validation", "test"}
    assert assignments == [split_for_question(f"question {i}") for i in range(100)]


@pytest.mark.parametrize("key", ["group_sha256", "input_ids"])
def test_training_entry_rejects_overlapping_files(tmp_path, key):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    first = {"group_sha256": "a", "input_ids": [1, 2, 3]}
    second = {"group_sha256": "b", "input_ids": [4, 5, 6]}
    second[key] = first[key]
    train.write_text(json.dumps(first), encoding="utf-8")
    validation.write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        assert_split_files_disjoint(train, validation)
