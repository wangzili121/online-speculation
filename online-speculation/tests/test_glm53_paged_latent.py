"""同步 draft 的按需读取：逻辑边界、租用生命周期及与快照的等价性。"""

from contextlib import nullcontext
import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from blockspec.vllm_ascend.glm53_history import PagedLatent, gather_history, read_pages
from blockspec.vllm_ascend.glm53_native import GLMHistory, NativeGLM53History, SharedGLM53Draft
from blockspec.vllm_ascend.glm53_ops import sparse_draft_attention


@pytest.mark.parametrize("mode", ["auto", "snapshot", "paged"])
@pytest.mark.parametrize("length", [0, 1, 47, 48, 49, 191])
def test_selected_pages_match_owned_prefix_with_padding_and_subpages(mode, length):
    arena = torch.arange(34 * 15 * 10.).reshape(34, 15, 1, 10)
    cache = arena[1:33, :12, :, :7]
    table = torch.arange(127, -1, -1).reshape(1, -1)
    before = arena.clone()
    expected = read_pages(cache, table, length, block_size=3)
    counters = dict(paged=0, snapshot=0)
    history = PagedLatent(cache, table, length, block_size=3, mode=mode, counters=counters)
    table.fill_(-1)
    indices = torch.tensor([[length - 1, 0, length - 1], [0, length - 1, 0]]) if length else torch.empty(2, 0, dtype=torch.long)
    actual = gather_history(history, indices)
    torch.testing.assert_close(actual, expected[indices], atol=0, rtol=0)
    actual.zero_()
    assert torch.equal(arena, before)
    assert sum(counters.values()) == bool(length)
    history.release()
    history.release()
    with pytest.raises(RuntimeError, match="released"):
        history.gather(indices)


@pytest.mark.parametrize("index", [-1, 10, 11, 12, 200])
def test_paged_reads_cannot_expose_uncommitted_rows_in_last_page(index):
    history = PagedLatent(torch.randn(8, 12, 1, 7), torch.tensor([[4, 5, 0, 1]]),
                          10, block_size=3, mode="paged")
    with pytest.raises((IndexError, RuntimeError)):
        history.gather(torch.tensor([[index]]))


def test_unsupported_layout_falls_back_without_materializing_arena():
    cache = torch.randn(8, 12, 1, 14)[..., ::2]
    table = torch.arange(32).reshape(1, -1)
    history = PagedLatent(cache, table, 40, block_size=3)
    indices = torch.tensor([[39, 0, 39]])
    torch.testing.assert_close(history.gather(indices), read_pages(cache, table, 40, block_size=3)[indices])
    assert history.last_path == "snapshot"
    with pytest.raises(ValueError, match="row pitch"):
        PagedLatent(cache, table, 40, block_size=3, mode="paged")


def test_paged_gather_never_selects_all_valid_prefix_rows(monkeypatch):
    cache = torch.randn(32, 12, 1, 10)[..., :7]
    table = torch.arange(128).reshape(1, -1)
    history = PagedLatent(cache, table, 191, block_size=3, mode="paged")
    indices = torch.tensor([[190, 0, 180], [3, 3, 3]])
    expected = read_pages(cache, table, 191, block_size=3)[indices]
    original = torch.Tensor.index_select
    selected_counts = []
    def checked(tensor, dim, rows):
        if tensor.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr():
            assert tensor.is_contiguous()
            selected_counts.append(rows.numel())
        return original(tensor, dim, rows)
    monkeypatch.setattr(torch.Tensor, "index_select", checked)
    torch.testing.assert_close(history.gather(indices), expected, atol=0, rtol=0)
    assert selected_counts == [indices.numel()]


def test_paged_history_matches_two_basis_attention():
    torch.manual_seed(5313)
    cache = torch.randn(8, 12, 1, 10)[..., :7]
    table = torch.arange(31, -1, -1).reshape(1, -1)
    history = PagedLatent(cache, table, 43, block_size=3, mode="paged")
    dense = read_pages(cache, table, 43, block_size=3)
    q, block = torch.randn(2, 3, 4), torch.randn(2, 7)
    indices = torch.tensor([[42, 0, 7], [1, 3, 1]])
    weights = dict(ar_key=torch.randn(3, 4, 7), ar_value=torch.randn(3, 5, 7),
                   draft_key=torch.randn(3, 4, 7), draft_value=torch.randn(3, 5, 7))
    actual = sparse_draft_attention(q, history, block, indices, **weights)
    expected = sparse_draft_attention(q, dense, block, indices, **weights)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_new_capture_invalidates_a_previous_borrowed_history():
    layer = NS(kv_cache=torch.zeros(8, 12, 1, 7, dtype=torch.bfloat16))
    indexer = NS(index_kpool=4, k_cache=NS(prefix="index", kv_cache=torch.zeros(8, 4, 1, 3, dtype=torch.bfloat16)))
    target = NS(model=NS(layers=[NS(layer_kind="mla", self_attn=NS(mla_attn=NS(mla_attn=layer), indexer=indexer))]))
    runner = NS(compilation_config=NS(static_forward_context={"mla": layer}))
    reader = NativeGLM53History(target, runner)
    metadata = dict(mla=NS(seq_lens=torch.tensor([10]), query_start_loc=torch.tensor([0, 1]),
                           block_table=torch.tensor([[0, 1, 2, 3]]), block_size=3),
                    index=NS(block_table=torch.tensor([[0]])))
    reader.capture(metadata, "first")
    previous = reader.read("first", 10, sampled_count=1, draft_count=0)
    reader.capture(metadata, "second")
    with pytest.raises(RuntimeError, match="released"):
        previous.layers[0][0].gather(torch.tensor([[0]]))
    with pytest.raises(RuntimeError, match="current request"):
        reader.read("first", 10, sampled_count=1, draft_count=0)


@pytest.mark.parametrize("alias,valid", [(True, True), (False, True), (False, False)])
def test_shared_metadata_skips_only_redundant_equality_checks(monkeypatch, alias, valid):
    cache = NS(kv_cache=torch.zeros(8, 12, 1, 7, dtype=torch.bfloat16))
    indexer = NS(index_kpool=4, k_cache=NS(prefix="index", kv_cache=torch.zeros(8, 4, 1, 3, dtype=torch.bfloat16)))
    attention = NS(mla_attn=NS(mla_attn=cache), indexer=indexer)
    target = NS(model=NS(layers=[NS(layer_kind="mla", self_attn=attention)]))
    runner = NS(compilation_config=NS(static_forward_context={"mla": cache}))
    reader = NativeGLM53History(target, runner)
    meta = NS(seq_lens=torch.tensor([10]), query_start_loc=torch.tensor([0, 1]),
              block_table=torch.tensor([[0, 1, 2, 3]]), block_size=3)
    other = NS(seq_lens=meta.seq_lens if alias else torch.tensor([10 if valid else 11]),
               query_start_loc=meta.query_start_loc if alias else torch.tensor([0, 1]),
               block_table=meta.block_table, block_size=3)
    reader.entries.append(("mla", attention, cache, "other"))
    reader.capture(dict(mla=meta, other=other, index=NS(block_table=torch.tensor([[0]]))), "request")
    equal = torch.equal
    calls = []
    def checked(left, right):
        calls.append((left, right))
        return equal(left, right)
    monkeypatch.setattr(torch, "equal", checked)
    if valid:
        reader.read("request", 10, sampled_count=1, draft_count=0).release()
        assert len(calls) == (0 if alias else 2)
    else:
        with pytest.raises(RuntimeError, match="disagree"):
            reader.read("request", 10, sampled_count=1, draft_count=0)
        assert len(calls) == 1


@pytest.mark.parametrize("fail", [False, "model", "auditor"])
def test_shared_forward_always_releases_target_references(monkeypatch, fail):
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_forward_context",
                        NS(set_ascend_forward_context=lambda *args, **kwargs: nullcontext()))
    history = PagedLatent(torch.ones(2, 4, 1, 7), torch.tensor([[0]]), 3, mode="paged")
    class Target:
        model = NS(layers=[NS(self_attn=nn.Identity())])
        def __call__(self, tokens, positions):
            assert history.gather(torch.tensor([[0]])).shape == (1, 1, 7)
            if fail == "model":
                raise RuntimeError("injected model failure")
            return torch.ones(2, 4)
        def compute_logits(self, hidden):
            return hidden
    model = SharedGLM53Draft.__new__(SharedGLM53Draft)
    nn.Module.__init__(model)
    model.target = Target()
    model.layers = nn.ModuleList([nn.Identity()])
    model.vllm_config, model.cache_auditor = NS(), None
    if fail == "auditor":
        def audit():
            raise RuntimeError("injected auditor failure")
        model.cache_auditor = audit
    try:
        model(torch.tensor([[1, 0]]), view="draft", cache=GLMHistory(((history, torch.zeros(0)),), 3))
    except RuntimeError as error:
        assert fail and "injected" in str(error)
    with pytest.raises(RuntimeError, match="released"):
        history.gather(torch.tensor([[0]]))
    assert "forward" not in model.target.model.layers[0].self_attn.__dict__


@pytest.mark.parametrize("length,selected,path", [(8195, 4, "snapshot"), (32768, 4, "paged"),
                                                  (32768, 32769, "snapshot")])
def test_auto_history_strategy_uses_measured_long_prefix_boundary(length, selected, path):
    cache = torch.zeros(4096, 16, 1, 1)
    history = PagedLatent(cache, torch.arange(4096).reshape(1, -1), length)
    history.gather(torch.arange(selected).remainder(length).reshape(1, -1))
    assert history.last_path == path


@pytest.mark.skipif(os.environ.get("ORTHRUS_RUN_NPU_TESTS") != "1", reason="allocated NPU required")
@pytest.mark.parametrize("pitch", [512, 576])
@pytest.mark.parametrize("width,length", [(2, 7), (8, 2055), (8, 32771)])
def test_npu_selected_rows_are_exact_without_arena_or_prefix_copy(pitch, width, length):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(5313)
    cache = torch.randn(1024, 128, 1, pitch, device="npu", dtype=torch.bfloat16)[1:1023, :96, :, :512]
    table = torch.randperm(1022 * 6).to(device="npu", dtype=torch.int32)[None]
    dense = read_pages(cache, table, length, block_size=16)
    indices = torch.randint(length, (width, min(length, 2051)), device="npu")
    expected = gather_history(dense, indices)
    history = PagedLatent(cache, table, length, block_size=16, mode="paged")
    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    actual = history.gather(indices)
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - allocated
    assert torch.equal(actual, expected)
    assert peak < 16 * 1024**2 + indices.numel() * (pitch + 512) * 2
    actual.zero_()
    assert torch.equal(gather_history(read_pages(cache, table, length, block_size=16), indices), expected)
