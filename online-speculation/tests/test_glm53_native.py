"""Pure CPU checks of the GLM native adapter's ownership and dispatch."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from blockspec.vllm_ascend.glm53_native import (
    GLMHistory, SparseDraft, clone_projection, read_pages, replace_attention,
)


def test_projection_copy_does_not_alias_target_or_keep_loader_parameter():
    class LoaderParameter(nn.Parameter):
        def __new__(cls, data, *, loader):
            return super().__new__(cls, data, False)
    module = nn.Linear(3, 4, bias=False)
    module.weight = LoaderParameter(module.weight.detach(), loader=object())
    copied = clone_projection(module)
    assert type(copied.weight) is nn.Parameter
    assert copied.weight.data_ptr() != module.weight.data_ptr()
    torch.testing.assert_close(copied.weight, module.weight, rtol=0, atol=0)
    copied.weight.zero_()
    assert module.weight.abs().sum() > 0


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 12])
def test_glm_paged_prefix_owns_only_committed_rows(length):
    cache = torch.arange(4 * 3 * 5.).reshape(4, 3, 1, 5)
    before = cache.clone()
    table = torch.tensor([[2, 0, 3, 1]])
    actual = read_pages(cache, table, length)
    expected = torch.cat([cache[i, :, 0] for i in table[0]])[:length]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.zero_()
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_attention_override_binds_each_layer_and_restores_native_forward(fail):
    class Attention(nn.Module):
        def forward(self, hidden_states, positions):
            return hidden_states - 1
    layers = [NS(self_attn=Attention()), NS(self_attn=Attention())]
    branches = [lambda hidden, state: hidden + state, lambda hidden, state: hidden * state]
    history = GLMHistory((3, 5), 7)
    try:
        with replace_attention(layers, branches, history):
            assert layers[0].self_attn(2, 8) == 5
            assert layers[1].self_attn(2, 8) == 10
            if fail:
                raise RuntimeError("injected failure")
    except RuntimeError:
        assert fail
    for layer in layers:
        assert "forward" not in layer.self_attn.__dict__
        assert layer.self_attn(2, 8) == 1


def test_glm_override_is_recognized_by_common_runner_adapter():
    from blockspec.vllm_ascend.qwen38_runtime import _ours
    assert _ours(NS(drafter=NS(glm53_native_proposer=True)))
    assert not _ours(NS(drafter=NS()))


def test_sparse_draft_survives_disposed_ar_kv_projection(monkeypatch):
    import blockspec.vllm_ascend.glm53_native as native
    from blockspec.vllm_ascend.glm53_ops import select_history, sparse_draft_attention

    class Linear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None
    torch.manual_seed(57)
    ar_key, ar_value = torch.randn(2, 4, 7), torch.randn(2, 5, 7)
    ar_before = ar_key.clone()
    original = NS(num_local_heads=2, qk_head_dim=4, v_head_dim=5,
                  kv_lora_rank=7, q_lora_rank=6,
                  fused_qkv_a_proj=Linear(9, 13, bias=False),
                  q_a_layernorm=nn.RMSNorm(6), kv_a_layernorm=nn.RMSNorm(7),
                  q_b_proj=Linear(6, 8, bias=False), o_proj=Linear(10, 9, bias=False),
                  indexer=NS(n_head=3, head_dim=5, index_kpool=4, topk_tokens=8,
                             wq_b=Linear(6, 15, bias=False),
                             wk_weights_proj=Linear(9, 8, bias=False)),
                  mla_attn=NS(mla_attn=NS(impl=NS(W_UK_T=ar_key, W_UV=ar_value.transpose(-1, -2)))))
    # 原生加载已释放 kv_b_proj：构造 draft 不得读取这个旧参数。
    branch = SparseDraft(original)
    assert branch.ar_key is ar_key
    assert branch.draft_key.data_ptr() != ar_key.data_ptr()
    hidden, history, pools = torch.randn(8, 9), torch.randn(13, 7), torch.randn(3, 5)
    before = history.clone()
    def torch_attention(*args, **kwargs):
        kwargs["backend"] = "torch"
        return sparse_draft_attention(*args, **kwargs)
    monkeypatch.setattr(native, "sparse_draft_attention", torch_attention)
    actual = branch(hidden, (history, pools))
    q, block = original.fused_qkv_a_proj(hidden)[0].split((6, 7), -1)
    q = original.q_a_layernorm(q)
    query = original.q_b_proj(q)[0].reshape(8, 2, 4)
    iq = original.indexer.wq_b(q)[0].reshape(8, 3, 5)
    iw = torch.nn.functional.linear(hidden, original.indexer.wk_weights_proj.weight[5:]) / (15**.5)
    indices = select_history(iq, iw, pools, 13, topk=8)
    from test_glm53_ops import explicit_attention
    heads = explicit_attention(query, history, original.kv_a_layernorm(block), indices,
                               ar_key, ar_value, ar_key, ar_value)
    expected = original.o_proj(heads.flatten(1))[0]
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(history, before, rtol=0, atol=0)
    with torch.no_grad():
        branch.draft_key.add_(.2)
    assert not torch.equal(branch(hidden, (history, pools)), actual)
    torch.testing.assert_close(original.mla_attn.mla_attn.impl.W_UK_T, ar_before, rtol=0, atol=0)
