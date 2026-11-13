from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest
import torch
from safetensors.torch import save_file

from blockspec.vllm_ascend.dsv41_attention import (
    DraftSelection,
    PagedRows,
    attend_layer,
    draft_attention,
    select_history,
)
from blockspec.vllm_ascend.dsv41_contract import (
    V41Topology,
    head_manifest,
    head_shapes,
    validate_head,
)
from blockspec.vllm_ascend.head_contract import ContractError, write_json
from blockspec.vllm_ascend.dsv41_precise_attention import precise_draft_attention
from blockspec.vllm_ascend import dsv41_native_attention as native_attention
from blockspec.vllm_ascend import dsv41_runtime as runtime


@pytest.fixture
def config():
    return {
        "model_type": "deepseek_v41",
        "bos_token_id": 0,
        "eos_token_id": [1],
        "text_config": {
            "model_type": "deepseek_v41_text",
            "num_hidden_layers": 40,
            "num_nextn_predict_layers": 3,
            "compress_ratios": [0] * 2 + [2] * 18 + [1] * 20 + [0] * 3,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "sliding_window": 128,
            "index_topk": 512,
            "engram_layer_ids": [1, 14],
            "vocab_size": 100,
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "q_lora_rank": 4,
            "o_lora_rank": 2,
            "o_groups": 2,
            "index_n_heads": 2,
            "index_head_dim": 4,
        },
    }


def test_1017_topology(config):
    roles = V41Topology.from_config(config).layers
    assert len(roles) == 40
    assert [r.layer for r in roles if r.mode == "swa"] == [0, 1]
    assert [r.layer for r in roles if r.mode == "full"] == [2, 8, 14, 20]
    assert [r.layer for r in roles if r.mode == "reindex"] == [24, 28, 32, 36]
    assert (roles[19].kv_source, roles[19].index_source) == (14, 14)
    assert (roles[39].kv_source, roles[39].index_source, roles[39].candidate_source) == (20, 36, 20)
    assert len(head_shapes(config, V41Topology.from_config(config))) == 336
    config["text_config"]["index_source_layer_ids"].remove(2)
    with pytest.raises(ContractError, match="KV source"):
        V41Topology.from_config(config)


def test_head_rejects_incompatible_artifacts(tmp_path, config):
    model, head = tmp_path / "base", tmp_path / "head"
    model.mkdir()
    head.mkdir()
    write_json(model / "config.json", config)
    write_json(model / "tokenizer.json", {"fixture": 1})
    index = "model.safetensors.index.json"
    write_json(model / index, {"weight_map": {"embedding": "base.safetensors"}})
    with pytest.raises(ContractError, match="overlaps eos"):
        head_manifest(model, mask_token_id=1, artifact_role="trained")
    manifest = head_manifest(model, mask_token_id=99, artifact_role="initialized-plumbing")
    write_json(head / "manifest.json", manifest)
    shapes = head_shapes(config, V41Topology.from_config(config))
    tensors = {
        key: torch.zeros(shape, dtype=torch.float32 if key.endswith(".attn_sink") else torch.bfloat16)
        for key, shape in shapes.items()
    }
    filename = "head.safetensors"
    save_file(tensors, head / filename)
    mapping = dict.fromkeys(tensors, filename)
    write_json(head / index, {"weight_map": mapping})
    assert validate_head(model, head, mask_token_id=99) == manifest
    with pytest.raises(ContractError, match="trained"):
        validate_head(model, head, mask_token_id=99, require_trained=True)
    with pytest.raises(ContractError, match="disagrees"):
        validate_head(model, head, mask_token_id=98)
    write_json(model / "tokenizer.json", {"fixture": 2})
    with pytest.raises(ContractError, match="disagrees"):
        validate_head(model, head, mask_token_id=99)
    write_json(model / "tokenizer.json", {"fixture": 1})
    name = "layers.20.attn.indexer.wq_b.weight"
    write_json(head / index, {"weight_map": {key: shard for key, shard in mapping.items() if key != name}})
    with pytest.raises(ContractError, match="tensor names"):
        validate_head(model, head, mask_token_id=99)
    write_json(head / index, {"weight_map": mapping})
    tensors[name] = tensors[name].float()
    save_file(tensors, head / filename)
    with pytest.raises(ContractError, match="wrong shape or dtype"):
        validate_head(model, head, mask_token_id=99)


def test_runtime_cache_plan_uses_shared_v41_sources(config):
    topology = V41Topology.from_config(config)
    layer2 = runtime.cache_names(topology, 2)
    layer7 = runtime.cache_names(topology, 7)
    layer8 = runtime.cache_names(topology, 8)
    assert layer2.long_kv == "model.layers.2.self_attn.long_kv_cache"
    assert layer7.long_kv == layer2.long_kv
    assert layer7.index_k == layer2.index_k
    assert layer8.long_kv == "model.layers.8.self_attn.long_kv_cache"
    names = runtime.target_cache_names(topology)
    assert len(names) == 40 + 4 + 4
    assert len(names) == len(set(names))


def test_runtime_metadata_is_noncausal_and_write_protected():
    calls = []

    class Builder:
        def build(self, prefix, common, **kwargs):
            calls.append((prefix, common, kwargs))
            return SimpleNamespace(causal=True)

    groups = [
        SimpleNamespace(
            kv_cache_group_id=3,
            layer_names=["model.layers.0.self_attn.swa_cache"],
            get_metadata_builder=lambda: Builder(),
        ),
        SimpleNamespace(
            kv_cache_group_id=7,
            layer_names=["model.layers.2.self_attn.long_kv_cache"],
            get_metadata_builder=lambda: Builder(),
        ),
    ]
    common = SimpleNamespace(
        num_reqs=2,
        slot_mapping=torch.tensor([9, 10, 11, 12], dtype=torch.int32),
        causal=True,
    )
    tables = {
        3: torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),
        7: torch.tensor([[8, 9], [10, 11]], dtype=torch.int32),
    }
    result = runtime.build_readonly_metadata(
        groups, common, tables, num_query_heads=8
    )
    assert set(result) == {
        "model.layers.0.self_attn.swa_cache",
        "model.layers.2.self_attn.long_kv_cache",
    }
    assert all(not metadata.causal for metadata in result.values())
    assert common.causal and torch.equal(common.slot_mapping, torch.tensor([9, 10, 11, 12]))
    for (_, built, kwargs), expected in zip(calls, tables.values(), strict=True):
        assert not built.causal
        assert built.slot_mapping.eq(-1).all()
        assert torch.equal(built.block_table_tensor, expected)
        assert kwargs["update_cache"] is False
        assert kwargs["skip_ring_state_update"] is True
        assert kwargs["build_compressor_metadata"] is False
        assert kwargs["num_query_heads"] == 8


def test_runtime_binds_readonly_target_pages(config):
    topology = V41Topology.from_config(config)
    anchors = torch.tensor([3, 4], dtype=torch.int32)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    swa_value = torch.randn(4, 2, 1, 8)
    long_value = torch.randn(4, 2, 1, 8)
    index_key = torch.randint(-8, 8, (4, 2, 1, 4), dtype=torch.int8)
    index_scale = torch.rand(4, 2, 1, dtype=torch.float16)

    swa_meta = SimpleNamespace(block_table=table, cache_seq_lens=anchors)
    compressed = anchors // 2
    long_meta = SimpleNamespace(block_table=table, cache_seq_lens=compressed)
    index_meta = SimpleNamespace(block_table=table, cache_seq_lens=compressed)
    layer_meta = SimpleNamespace(
        swa=swa_meta,
        attention=long_meta,
        indexer=SimpleNamespace(cache=index_meta),
    )

    layers = []
    for _ in topology.layers:
        layers.append(SimpleNamespace(self_attn=SimpleNamespace()))
    source = layers[2].self_attn
    source.long_kv_cache = SimpleNamespace(kv_cache=[long_value])
    source.indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=[(index_key, index_scale)])
    )
    consumer = layers[3].self_attn
    consumer.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=[swa_value])
    )
    consumer.v41_impl = SimpleNamespace(
        _get_layer_metadata=lambda _: layer_meta
    )
    target = SimpleNamespace(model=SimpleNamespace(layers=layers))

    bound = runtime.bind_target_history(target, {}, topology, 3, anchors)
    assert bound.swa.values.data_ptr() == swa_value.data_ptr()
    assert bound.long_kv.values.data_ptr() == long_value.data_ptr()
    assert bound.index_k.key is index_key
    assert bound.index_k.scale is index_scale
    assert bound.swa.table is table and bound.long_kv.table is table
    assert torch.equal(bound.global_lengths, compressed)
    assert torch.equal(swa_value, consumer.dsa_attn.swa_cache_layer.kv_cache[0])
    assert torch.equal(long_value, source.long_kv_cache.kv_cache[0])

    bad = SimpleNamespace(**vars(layer_meta))
    bad.attention = SimpleNamespace(block_table=table, cache_seq_lens=compressed + 1)
    consumer.v41_impl = SimpleNamespace(_get_layer_metadata=lambda _: bad)
    with pytest.raises(ValueError, match="incomplete or future"):
        runtime.bind_target_history(target, {}, topology, 3, anchors)


def test_runtime_drives_index_sources_and_reuses_private_topk(monkeypatch, config):
    topology = V41Topology.from_config(config)
    anchors = torch.tensor([8], dtype=torch.int32)
    pages = PagedRows(torch.zeros(1, 8, 8), torch.zeros(1, 1, dtype=torch.int32))
    index = runtime.IndexPages(
        torch.zeros(1, 8, 1, 4, dtype=torch.int8),
        torch.ones(1, 8, 1, dtype=torch.float16),
        torch.zeros(1, 1, dtype=torch.int32),
    )
    history = runtime.V41TargetHistory(pages, anchors, pages, index, anchors // 2)
    selected = torch.arange(3, dtype=torch.int32).view(1, 1, 3).expand(1, 2, -1)
    calls = []

    def fake_select(query, weights, runtime_anchors, runtime_pages, **kwargs):
        calls.append(kwargs["candidate_mode"])
        assert runtime_anchors is anchors and runtime_pages is index
        return selected, kwargs.get("candidates")

    seen = []

    def fake_attention(query, temporary, swa, runtime_anchors, sink, **kwargs):
        seen.append(kwargs["global_indices"])
        return query.float(), query.new_zeros((*query.shape[:-1], 1), dtype=torch.float32)

    monkeypatch.setattr(runtime, "select_projected_history", fake_select)
    monkeypatch.setattr(runtime, "native_draft_attention", fake_attention)
    query = torch.zeros(1, 2, 2, 8)
    temporary = torch.zeros(1, 2, 8)
    index_query = torch.zeros(1, 2, 2, 4)
    index_weights = torch.zeros(1, 2, 2)
    state = runtime.V41NativeDraftState()

    _, _, state = runtime.run_native_layer(
        topology, 2, query, temporary, torch.zeros(2), history, state,
        index_query=index_query, index_weights=index_weights,
    )
    assert calls == ["plain"] and state.topk[2] is selected and seen[-1] is selected
    _, _, reused = runtime.run_native_layer(
        topology, 3, query, temporary, torch.zeros(2), history, state
    )
    assert calls == ["plain"] and reused is not state and seen[-1] is selected


def _pages(rows, order):
    batch, width, dim = rows.shape
    return PagedRows(
        rows.reshape(-1, 2, dim)[order],
        order.argsort().view(batch, width // 2),
    )


def _direct_attention(query, local, lengths, window, temp, sink, sparse=None, selected=None):
    outputs, lses = [], []
    for b, length in enumerate(lengths.tolist()):
        batch_output, batch_lse = [], []
        for i, q in enumerate(query[b]):
            rows = [local[b, max(0, length - window):length], temp[b]]
            if sparse is not None:
                rows.append(sparse[b, selected[b, i][selected[b, i] >= 0]])
            kv = torch.cat(rows).float()
            logits = torch.cat((q.float() @ kv.T * q.shape[-1] ** -0.5, sink[:, None]), -1)
            batch_output.append(logits.softmax(-1)[:, :-1] @ kv)
            batch_lse.append(logits.logsumexp(-1, keepdim=True))
        outputs.append(torch.stack(batch_output))
        lses.append(torch.stack(batch_lse))
    return torch.stack(outputs), torch.stack(lses)


def test_native_attention_segments_preserve_joint_softmax(monkeypatch):
    generator = torch.Generator().manual_seed(41)
    batch, block, heads, dim = 2, 3, 2, 4
    query = torch.randn(batch, block, heads, dim, generator=generator)
    temporary = torch.randn(batch, block, dim, generator=generator)
    local_rows = torch.randn(batch, 6, dim, generator=generator)
    global_rows = torch.randn(batch, 4, dim, generator=generator)
    local = _pages(local_rows, torch.arange(6))
    global_kv = _pages(global_rows, torch.arange(4))
    anchors = torch.tensor([0, 5])
    sink = torch.tensor([-1.0, 0.5])
    selected = torch.tensor(
        [[[-1, -1], [-1, -1], [-1, -1]], [[0, 2], [1, 2], [2, -1]]]
    )
    global_lengths = torch.tensor([0, 3])
    calls = []

    def fake_plane(query, pages, indices, lengths, plane_sink, **kwargs):
        calls.append((indices.clone(), lengths.clone(), kwargs["maximum_length"]))
        if indices.ndim == 2:
            indices = indices[:, None].expand(-1, block, -1)
        kv = pages.gather(indices).float()
        valid = indices >= 0
        logits = torch.einsum("bqhd,bqrd->bqhr", query.float(), kv) * dim**-0.5
        logits = logits.masked_fill(~valid[:, :, None], -torch.inf)
        sink_logits = plane_sink.view(1, 1, heads, 1).expand(batch, block, -1, -1)
        lse = torch.cat((logits, sink_logits), -1).logsumexp(-1, keepdim=True)
        output = torch.einsum("bqhr,bqrd->bqhd", (logits - lse).exp(), kv)
        has_rows = valid.any(-1)[:, :, None, None]
        return torch.where(has_rows, output, torch.zeros_like(output)), lse

    def fake_block(query, temporary):
        logits = torch.einsum("bqhd,brd->bqhr", query, temporary) * dim**-0.5
        lse = logits.logsumexp(-1, keepdim=True)
        return torch.einsum("bqhr,brd->bqhd", (logits - lse).exp(), temporary), lse

    monkeypatch.setattr(native_attention, "_native_sparse_plane", fake_plane)
    monkeypatch.setattr(native_attention, "_temporary_block", fake_block)
    actual = native_attention.native_draft_attention(
        query,
        temporary,
        local,
        anchors,
        sink,
        window=4,
        global_kv=global_kv,
        global_indices=selected,
        global_lengths=global_lengths,
        max_swa_history=5,
        max_global_history=3,
    )
    expected = draft_attention(
        query,
        temporary,
        local,
        anchors,
        sink,
        window=4,
        global_kv=global_kv,
        global_indices=selected,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert [call[2] for call in calls] == [5, 3]


def test_native_attention_rejects_pre_post7_sparse_abi(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.post4"))
    assert not native_attention._runtime_topk_length_supported()
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="2.10.0.post7"))
    assert native_attention._runtime_topk_length_supported()


@pytest.mark.parametrize("width", [1, 32])
def test_c2a_full_reuse_odd_even_and_paging(config, width):
    topology = replace(V41Topology.from_config(config), window=4, topk=3)
    anchors = torch.tensor([0, 5, 6])
    key = torch.tensor([[9, 1, 7, 6], [1, 4, 999, 999], [1, 3, 9, 999]]).float()[..., None]
    q_index = torch.ones(3, width, 1, 1)
    weights = torch.ones_like(q_index[..., 0])
    weights[2, 0] = -1
    torch.manual_seed(42)
    swa_rows = torch.randn(3, 8, 8)
    global_rows = torch.randn(3, 4, 8)
    swa_rows[0] = float("nan")
    swa_rows[1, 5:] = float("nan")
    global_rows[1, 2:] = float("nan")
    query = torch.randn(3, width, 2, 8)
    temp = torch.randn(3, width, 8)
    sink = torch.tensor([-0.1, 1.2])
    previous = None
    for order in (torch.arange(6), torch.tensor([3, 1, 5, 0, 4, 2])):
        index_pages = _pages(key, order)
        global_pages = _pages(global_rows, order)
        swa = _pages(swa_rows, torch.arange(12).flip(0))
        planes = (swa, global_pages, index_pages)
        saved = [(p.values.clone(), p.table.clone()) for p in planes]
        selection, state = select_history(
            topology, 2, anchors, DraftSelection(), query=q_index, weights=weights,
            index_keys={2: index_pages},
        )
        reused, same_state = select_history(topology, 3, anchors, state)
        assert reused is selection and same_state is state
        assert selection[:, 0].tolist() == [[-1, -1, -1], [0, 1, -1], [0, 1, 2]]
        if previous is not None:
            assert torch.equal(previous, selection)
        previous = selection
        for layer in (2, 3):
            actual = attend_layer(
                topology, layer, query, temp, swa, anchors, sink,
                global_pages={2: global_pages}, selection=selection,
            )
            expected = _direct_attention(query, swa_rows, anchors, 4, temp, sink, global_rows, selection)
            torch.testing.assert_close(actual, expected)
            precise = precise_draft_attention(
                query, temp, swa, anchors, sink, window=4, global_kv=global_pages,
                global_indices=selection,
            )
            torch.testing.assert_close(precise, expected)
        bad = selection.clone()
        bad[1, 0, 0] = 2
        with pytest.raises(ValueError, match="outside"):
            attend_layer(topology, 3, query, temp, swa, anchors, sink, global_pages={2: global_pages}, selection=bad)
        for p, (values, table) in zip(planes, saved, strict=True):
            torch.testing.assert_close(p.values, values, equal_nan=True)
            assert torch.equal(p.table, table)


def test_c1a_candidate_pool_is_per_query_and_draft_private(config):
    topology = replace(V41Topology.from_config(config), candidate_block_size=2, candidate_blocks=2, topk=2)
    anchors = torch.tensor([5, 0])
    keys = torch.tensor([[[8., 1.], [7., 1.], [1., 7.], [1., 8.], [2., 2.], [999., 999.]]] * 2)
    index_pages = {20: _pages(keys, torch.arange(6))}
    query = torch.eye(2).view(1, 2, 1, 2).expand(2, -1, -1, -1)
    weights = torch.ones(2, 2, 1)
    full, state = select_history(
        topology, 20, anchors, DraftSelection(), query=query, weights=weights, index_keys=index_pages,
    )
    assert full[0].tolist() == [[0, 1], [2, 3]]
    assert state.candidates[0, :, :5].tolist() == [[True, True, False, False, True], [False, False, True, True, True]]
    reindexed, updated = select_history(
        topology, 24, anchors, state, query=query.flip(1), weights=weights, index_keys=index_pages,
    )
    assert reindexed[0].tolist() == [[0, 4], [2, 4]]
    assert reindexed[1].eq(-1).all()
    assert 24 not in state.topk
    assert select_history(topology, 25, anchors, updated)[0] is reindexed
    assert select_history(topology, 23, anchors, updated)[0] is full
    torch.manual_seed(41)
    local = torch.randn(2, 6, 2)
    long = torch.randn_like(local)
    temp = torch.randn(2, 2, 2)
    attn_query = query.expand(-1, -1, 2, -1)
    sink = torch.tensor([0.25, -1.])
    global_pages = {20: _pages(long, torch.tensor([2, 1, 5, 3, 0, 4]))}
    for layer in (24, 25):
        torch.testing.assert_close(
            attend_layer(
                topology, layer, attn_query, temp, _pages(local, torch.arange(6)), anchors, sink,
                global_pages=global_pages, selection=reindexed,
            ),
            _direct_attention(attn_query, local, anchors, topology.window, temp, sink, long, reindexed),
        )
    with pytest.raises(ValueError, match="candidate source"):
        select_history(topology, 24, anchors, DraftSelection(), query=query, weights=weights, index_keys=index_pages)
    with pytest.raises(ValueError, match="has not run"):
        select_history(topology, 25, anchors, DraftSelection())
    empty = PagedRows(torch.empty(0, 2, 2), torch.empty(2, 0, dtype=torch.long))
    result, _ = select_history(
        topology, 20, torch.zeros_like(anchors), DraftSelection(),
        query=query, weights=weights, index_keys={20: empty},
    )
    assert result.eq(-1).all()


def test_swa_future_draft_rows_and_sink_participate(config):
    topology = V41Topology.from_config(config)
    anchors = torch.tensor([0])
    swa = PagedRows(torch.empty(0, 2, 1), torch.empty(1, 0, dtype=torch.long))
    q = torch.zeros(1, 3, 1, 1)
    temp = torch.tensor([[[2.], [4.], [8.]]], requires_grad=True)
    sink = torch.tensor([0.])
    output, lse = draft_attention(q, temp, swa, anchors, sink, window=topology.window)
    torch.testing.assert_close(output, torch.full_like(output, 3.5))
    torch.testing.assert_close(lse, torch.full_like(lse, 4.).log())
    torch.testing.assert_close(
        precise_draft_attention(q, temp, swa, anchors, sink, window=topology.window), (output, lse)
    )
    output[0, 0].sum().backward()
    torch.testing.assert_close(temp.grad, torch.full_like(temp, 0.25))
