"""GLM latent-basis, KPool boundary and immutable-state contracts."""

import sys
from types import SimpleNamespace as NS

import pytest
import torch
from torch.nn import functional as F

from blockspec.vllm_ascend.glm53_ops import (
    Glm53Architecture, kda_native, kda_reference, select_history, sparse_draft_attention,
)


@pytest.mark.parametrize("prefix", [0, 1, 3, 4, 5, 11, 16, 37])
def test_selection_excludes_future_and_keeps_incomplete_tail(prefix):
    query = torch.randn(8, 3, 5)
    keys = torch.randn(prefix // 4, 5)
    weights = torch.randn(8, 3)
    before = keys.clone()
    indices = select_history(query, weights, keys, prefix, topk=8)
    assert indices.shape == (8, min(prefix // 4, 2) * 4 + prefix % 4)
    for row in indices.tolist():
        assert len(set(row)) == len(row)
        assert all(0 <= i < prefix for i in row)
        assert set(range(prefix // 4 * 4, prefix)) <= set(row)
    torch.testing.assert_close(keys, before, rtol=0, atol=0)


def test_selection_requires_exact_committed_pool_boundary():
    with pytest.raises(ValueError, match="committed prefix"):
        select_history(torch.ones(4, 2, 3), torch.ones(4, 2), torch.ones(2, 3), 7, topk=4)


def explicit_attention(query, history, block, indices, ar_key, ar_value, draft_key, draft_value):
    key_ar = torch.einsum("qsr,hdr->qhsd", history[indices], ar_key)
    value_ar = torch.einsum("qsr,hvr->qhsv", history[indices], ar_value)
    key_d = torch.einsum("sr,hdr->hsd", block, draft_key)
    value_d = torch.einsum("sr,hvr->hsv", block, draft_value)
    keys = torch.cat((key_ar, key_d.expand(query.shape[0], -1, -1, -1)), 2)
    values = torch.cat((value_ar, value_d.expand(query.shape[0], -1, -1, -1)), 2)
    return F.scaled_dot_product_attention(query.unsqueeze(2), keys, values).squeeze(2)


@pytest.mark.parametrize("prefix", [0, 3, 17])
@pytest.mark.parametrize("width", [2, 8, 32])
def test_two_latent_bases_match_expanded_sdpa(prefix, width):
    torch.manual_seed(171)
    q = torch.randn(width, 2, 4) * .2
    history, block = torch.randn(prefix, 7), torch.randn(width, 7)
    indices = torch.arange(prefix).expand(width, -1)
    ar_key, draft_key = torch.randn(2, 4, 7), torch.randn(2, 4, 7)
    ar_value, draft_value = torch.randn(2, 5, 7), torch.randn(2, 5, 7)
    before = history.clone()
    actual = sparse_draft_attention(q, history, block, indices, ar_key=ar_key,
                                   ar_value=ar_value, draft_key=draft_key, draft_value=draft_value)
    expected = explicit_attention(q, history, block, indices, ar_key, ar_value, draft_key, draft_value)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(history, before, rtol=0, atol=0)


def test_final_mask_position_can_change_first_output():
    q = torch.ones(8, 1, 2)
    empty = torch.empty(0, 2)
    indices = torch.empty(8, 0, dtype=torch.long)
    block = torch.zeros(8, 2)
    weight = torch.eye(2)[None]
    def run():
        return sparse_draft_attention(q, empty, block, indices, ar_key=weight,
                                      ar_value=weight, draft_key=weight, draft_value=weight)
    previous = run()
    block[-1] = 3
    assert not torch.equal(run()[0], previous[0])


def test_kda_acceptance_boundary_is_not_rejected_final_state():
    torch.manual_seed(37)
    q, k, v = [torch.randn(8, 2, 4) for _ in range(3)]
    g, b = torch.randn_like(q), torch.randn(8, 2)
    initial = torch.randn(2, 4, 4)
    old = initial.clone()
    a, bias = torch.zeros(2), torch.randn(2 * 4)
    out, states = kda_reference(q, k, v, g, b, initial, a, bias)
    for accepted in range(1, 9):
        _, prefix_states = kda_reference(q[:accepted], k[:accepted], v[:accepted],
                                        g[:accepted], b[:accepted], initial, a, bias)
        torch.testing.assert_close(states[accepted - 1], prefix_states[-1], rtol=0, atol=0)
    torch.testing.assert_close(initial, old, rtol=0, atol=0)
    assert out.shape == (8, 2, 4)


@pytest.mark.parametrize("width", [1, 2, 8])
def test_native_kda_uses_shared_backend_with_owned_state(monkeypatch, width):
    initial = torch.randn(2, 128, 128)
    before = initial.clone()
    called = []

    def run(q, k, v, gate, beta, states, cu_seqlens, indices, a_log, bias, **kwargs):
        assert states.data_ptr() != initial.data_ptr()
        torch.testing.assert_close(states[0], initial, rtol=0, atol=0)
        assert cu_seqlens.tolist() == [0, width]
        assert indices.tolist() == [list(range(width))]
        assert kwargs == dict(beta_is_preprocessed=False, lower_bound=-5.)
        called.append(True)
        states.fill_(7)
        return torch.ones_like(v)

    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.kda", NS(run_recurrent_kda=run))
    q, k, v, gate = [torch.randn(width, 2, 128) for _ in range(4)]
    out, states = kda_native(q, k, v, gate, torch.zeros(width, 2), initial,
                             torch.zeros(2), torch.zeros(256))
    assert called == [True]
    assert out.shape == (width, 2, 128)
    assert torch.all(states == 7)
    torch.testing.assert_close(initial, before, rtol=0, atol=0)


def test_config_does_not_accept_full_glm_or_dsv4():
    for name in ("glm_moe_dsa", "deepseek_v4"):
        with pytest.raises(ValueError):
            Glm53Architecture.from_config({"model_type": name})
