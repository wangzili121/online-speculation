import threading

import pytest

from inference_scaling.arllm.vllm_suffix_proposer import DynamicSuffixDecodingProposer


class _Delegate:
    def __init__(self) -> None:
        self.num_speculative_tokens = 8
        self.seen = []

    def propose(self, count, input_batch, sampled_token_ids, slot_mappings):
        assert self.num_speculative_tokens == count
        self.seen.append((count, input_batch, sampled_token_ids, slot_mappings))
        return [[count] for _ in sampled_token_ids]

    def load_model(self, *args, **kwargs):
        return args, kwargs


def _proposer() -> DynamicSuffixDecodingProposer:
    proposer = DynamicSuffixDecodingProposer.__new__(DynamicSuffixDecodingProposer)
    proposer._delegate = _Delegate()
    proposer._lock = threading.Lock()
    return proposer


def test_dynamic_suffix_proposer_applies_and_restores_runtime_k() -> None:
    proposer = _proposer()

    assert proposer.propose(2, "batch", [[1], [2]], "slots") == [[2], [2]]
    assert proposer._delegate.num_speculative_tokens == 8
    assert proposer._delegate.seen == [(2, "batch", [[1], [2]], "slots")]


def test_dynamic_suffix_proposer_supports_zero_and_rejects_negative_k() -> None:
    proposer = _proposer()

    assert proposer.propose(0, None, [[]], None) == [[0]]
    with pytest.raises(ValueError, match="non-negative"):
        proposer.propose(-1, None, [[]], None)
