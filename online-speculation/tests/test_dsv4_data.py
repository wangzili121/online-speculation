"""Packed DSV4 data keeps draft blocks inside intended supervision spans."""

import json

import pytest

from blockspec.dsv4_data import (
    PackedTokenWriter,
    paper_sample,
    online_sample,
    split_for_group,
)
from blockspec.parallel.fitting import BatchStream, TokenDataset


def test_packed_samples_never_offer_cross_sample_anchors(tmp_path):
    path = tmp_path / "train.tokens.jsonl"
    writer = PackedTokenWriter(path, sequence_length=12, block_size=4)
    writer.add([1, 2, 3, 4, 5, 6])
    writer.add([11, 12, 13, 14, 15, 16])
    details = writer.close()
    assert details["records"] == 1
    row = json.loads(path.read_text())
    assert row["anchor_ranges"] == [[0, 6], [6, 12]]

    dataset = TokenDataset(path, vocab_size=32, sequence_length=12)
    tokens, candidates = BatchStream(dataset, seed=1).batch_with_anchor_candidates(1, 4)
    assert tokens.shape == (1, 12)
    assert candidates[0].tolist() == [0, 1, 2, 6, 7, 8]


def test_online_tokens_supervise_only_the_completed_response():
    tokens, ranges, group = online_sample(
        {
            "request_id": "request-7",
            "status": "completed",
            "prompt_token_ids": [1, 2, 3],
            "output_token_ids": [4, 5, 6, 7],
        }
    )
    assert tokens == [1, 2, 3, 4, 5, 6, 7]
    assert ranges == [(2, 7)]
    assert group == "request-7"


def test_online_data_rejects_unfinished_requests():
    with pytest.raises(ValueError, match="not complete"):
        online_sample(
            {
                "status": "aborted",
                "prompt_token_ids": [1],
                "output_token_ids": [2, 3, 4],
            }
        )


def test_split_is_group_stable():
    first = split_for_group(
        "same-request", seed=42, validation_fraction=0.1, test_fraction=0.1
    )
    second = split_for_group(
        "same-request", seed=42, validation_fraction=0.1, test_fraction=0.1
    )
    assert first == second


def test_writer_drops_an_incomplete_tail(tmp_path):
    path = tmp_path / "tail.jsonl"
    writer = PackedTokenWriter(path, sequence_length=8, block_size=4)
    writer.add([1, 2, 3, 4, 5])
    details = writer.close()
    assert details["records"] == 0
    assert details["trailing_tokens_dropped"] == 5
    assert path.read_text() == ""


def test_paper_data_uses_stable_assistant_turn_boundaries():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize
            result = []
            for message in messages:
                if message["role"] == "user":
                    result.extend([10, len(message["content"]), 11])
                else:
                    result.extend([20, len(message["content"]), 21])
            if add_generation_prompt:
                result.append(20)
            return result

    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "followup"},
        {"role": "assistant", "content": "second"},
    ]
    tokens, ranges, _ = paper_sample({"messages": messages}, Tokenizer())
    assert tokens == [10, 8, 11, 20, 6, 21, 10, 8, 11, 20, 6, 21]
    assert ranges == [(4, 6), (10, 12)]
