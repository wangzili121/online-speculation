"""Native paged KV and speculative GDN accepted-boundary selection."""

from types import SimpleNamespace as NS

import pytest
import torch

from blockspec.vllm_ascend.qwen38_history import gdn_history, paged_history


@pytest.mark.parametrize("length", [1, 3, 4, 5, 11, 12])
def test_paged_history_crosses_permuted_pages_without_writes(length):
    cache = torch.arange(4 * 3 * 2 * 5.).reshape(4, 3, 2, 5)
    before = cache.clone()
    table = torch.tensor([[2, 0, 3, 1]])
    history = paged_history(cache, table, length)
    expected = torch.cat([cache[i] for i in table[0]], dim=0)[:length].transpose(0, 1)[None]
    torch.testing.assert_close(history, expected, rtol=0, atol=0)
    history.zero_()
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [1, 2, 7, 16, 17, 31, 32])
def test_gdn_selects_current_acceptance_not_previous_round_or_rejected_tail(accepted):
    conv = torch.arange(40 * 34 * 6.).reshape(40, 34, 6)
    ssm = torch.arange(40 * 2 * 5 * 3.).reshape(40, 2, 5, 3)
    before = (conv.clone(), ssm.clone())
    indices = torch.randperm(40)[:32][None]
    metadata = NS(spec_sequence_masks=torch.tensor([True]), spec_state_indices_tensor=indices,
                  spec_decode_metadata=NS(spec_causal_conv1d=NS(cache_indices=indices,
                                                                num_accepted_tokens=torch.tensor([4]))))
    c, s = gdn_history((conv, ssm), metadata, accepted=accepted, query_length=32, kernel_width=4)
    torch.testing.assert_close(c[0, :, 1:], conv[indices[0, 0], accepted - 1:accepted + 2].T)
    assert not c[0, :, 0].any()
    torch.testing.assert_close(s[0], ssm[indices[0, accepted - 1]].transpose(-1, -2))
    c.zero_()
    s.zero_()
    for value, old in zip((conv, ssm), before, strict=True):
        torch.testing.assert_close(value, old, rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [0, 33])
def test_gdn_rejects_invalid_acceptance(accepted):
    with pytest.raises(ValueError, match="valid accepted prefix"):
        gdn_history((torch.empty(40, 34, 6), torch.empty(40, 2, 5, 3)),
                    NS(spec_sequence_masks=torch.tensor([True])), accepted=accepted,
                    query_length=32, kernel_width=4)


def test_paged_history_rejects_short_table():
    with pytest.raises(ValueError, match="exceeds native block table"):
        paged_history(torch.empty(4, 3, 2, 5), torch.tensor([[0]]), 4)
