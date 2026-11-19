"""Check lazy row boundaries against the existing exact rejection verifier."""

import pytest
import torch

pytest.importorskip("serving.runtime")
from blockspec.sampling import SamplingConfig
from blockspec.pypto_qwen38.lazy_proposals import LazyProposals
from blockspec.pypto_qwen38.oneserving_tp2 import check_sampled_chunk


@pytest.mark.parametrize("budget", [1, 3, 4, 7, 31])
@pytest.mark.parametrize("reject", [None, 0, 2, 4, 30])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_eager_and_lazy_reach_the_same_rejection_or_bonus(budget, reject, chunk_size):
    reads = []
    def read(first, count):
        reads.append((first, count))
        return torch.tensor([[-10000., 10000.]]).repeat(count, 1)
    rng = torch.Generator().manual_seed(743)
    lazy = LazyProposals(budget, read, SamplingConfig(1), rng)
    point = torch.tensor([0., 1.])
    eager = [point] * budget

    def verify(proposal, delayed):
        sequence = [99] if delayed else [99] + [1] * budget
        offset, result = 0, []
        while True:
            if delayed:
                sequence.extend(proposal.ensure(offset + chunk_size))
            count = min(chunk_size, len(sequence) - offset)
            def target(row):
                return (1 - point if offset + row == budget or offset + row == reject else point)
            emitted, reached, accepted, done = check_sampled_chunk(
                sequence, offset, count, target, proposal, rng)
            result.extend(emitted)
            offset += reached
            if done:
                return result, offset

    assert verify(lazy, True) == verify(eager, False)
    if reject == 0:
        assert reads == [(0, min(chunk_size, budget))]
    assert sum(count for _, count in reads) <= budget


def test_an_unread_distribution_is_not_available_to_verification():
    lazy = LazyProposals(31, lambda first, count: torch.zeros(count, 4), SamplingConfig(1),
                         torch.Generator().manual_seed(743))
    lazy.ensure(4)
    with pytest.raises(IndexError):
        lazy[4]
    assert lazy.ensure(4) == []
    assert len(lazy.ensure(8)) == 4


def test_reader_cannot_silently_return_a_different_number_of_rows():
    lazy = LazyProposals(7, lambda first, count: torch.zeros(1, 4), SamplingConfig(1),
                         torch.Generator().manual_seed(743))
    with pytest.raises(ValueError, match="requested draft rows"):
        lazy.ensure(4)


def test_several_draft_groups_cannot_overwrite_each_others_logits():
    source = torch.tensor([[-10000., 10000.]]).repeat(7, 1)
    lazy = LazyProposals(7, lambda first, count: source[first:first + count], SamplingConfig(1),
                         torch.Generator().manual_seed(743), retain_logits=True)
    source.copy_(source.flip(-1))
    assert lazy.ensure(4) == [1, 1, 1, 1]


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_randomized_lazy_correction_keeps_the_target_first_token_law(chunk_size):
    # Unequal p/q exercise both acceptance and residual replacement. This test
    # checks the observable output distribution, not equality of RNG traces.
    p = torch.tensor([.7, .3])
    q_logits = torch.tensor([[.2, .8]]).log()
    rng = torch.Generator().manual_seed(94743)
    zeros = 0
    for _ in range(2000):
        proposal = LazyProposals(7, lambda first, count: q_logits.repeat(count, 1), SamplingConfig(1), rng)
        tokens = proposal.ensure(chunk_size)
        emitted, *_ = check_sampled_chunk([99, *tokens], 0, chunk_size, lambda row: p, proposal, rng)
        zeros += emitted[0] == 0
    assert .66 < zeros / 2000 < .74
