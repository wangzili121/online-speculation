"""CPU checks of rank-local loading and head-parallel Attention/GDN math."""

import hashlib
import json

import pytest
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from blockspec.pypto_qwen38.head import HeadSource, merge_parameter
from blockspec.pypto_qwen38.tp import TPLayout, reshard_parameter
from blockspec.pypto_qwen38.tp_programs import validate_linear_tiles


@pytest.fixture
def small_weights(monkeypatch):
    import blockspec.pypto_qwen38.head as head

    def schema(layer, tp=1):
        if layer % 4 == 3:
            return {"q_proj.weight": (32 // tp, 12), "k_proj.weight": (8 // tp, 12),
                    "v_proj.weight": (8 // tp, 12), "o_proj.weight": (12, 16 // tp),
                    "q_norm.weight": (256,), "k_norm.weight": (256,)}
        return {"in_proj_qkv.weight": (40 // tp, 12), "in_proj_z.weight": (24 // tp, 12),
                "in_proj_a.weight": (12 // tp, 12), "in_proj_b.weight": (12 // tp, 12),
                "out_proj.weight": (12, 24 // tp), "conv1d.weight": (40 // tp, 1, 4),
                "A_log": (12 // tp,), "dt_bias": (12 // tp,), "norm.weight": (128,)}

    monkeypatch.setattr(head, "tensor_schema", schema)
    monkeypatch.setattr(head, "KEY_WIDTH", 8)
    monkeypatch.setattr(head, "VALUE_WIDTH", 24)
    generator = torch.Generator().manual_seed(98)
    return {layer: {name: (torch.randn(shape, generator=generator) * .1).to(
        torch.float32 if name == "A_log" else torch.bfloat16)
        for name, shape in schema(layer).items()} for layer in (0, 3)}


def reference_shard(name, value, tp, rank):
    if name in {"q_norm.weight", "k_norm.weight", "norm.weight"}:
        return value.clone()
    if name in {"in_proj_qkv.weight", "conv1d.weight"}:
        return torch.cat([section.chunk(tp, dim=0)[rank] for section in value.split((8, 8, 24))])
    return value.chunk(tp, dim=1 if name in {"o_proj.weight", "out_proj.weight"} else 0)[rank].contiguous()


@pytest.mark.parametrize("source_tp", (1, 2, 4))
@pytest.mark.parametrize("target_tp", (1, 2, 4))
@pytest.mark.parametrize("layer", (0, 3))
def test_resharding_reads_only_owned_sections_and_preserves_values(small_weights, source_tp, target_tp, layer):
    for name, full in small_weights[layer].items():
        source = [reference_shard(name, full, source_tp, rank) for rank in range(source_tp)]
        assert torch.equal(merge_parameter(layer, name, source), full)
        for rank in range(target_tp):
            read_elements = []

            def read(index, selection, source=source, read_elements=read_elements):
                value = source[index][selection].clone()
                read_elements.append(value.numel())
                return value

            actual = reshard_parameter(layer, name, source_tp=source_tp,
                                       layout=TPLayout(target_tp, rank), read=read)
            expected = reference_shard(name, full, target_tp, rank)
            assert torch.equal(actual, expected) and actual.dtype == full.dtype
            assert sum(read_elements) == expected.numel() * (
                source_tp if name.endswith("norm.weight") else 1)


@pytest.mark.parametrize("tp", (1, 2, 4))
def test_local_heads_and_history_cover_the_target_once(tp):
    layouts = [TPLayout(tp, rank) for rank in range(tp)]
    for name, count in (("full_query", 24), ("full_kv", 4), ("gdn_key", 16), ("gdn_value", 48)):
        assert [i for layout in layouts for i in range(*layout.head_ranges[name])] == list(range(count))
    for layout in layouts:
        shapes = layout.history_shapes(slots=2, pages=5)
        layout.validate_history_shapes(shapes, slots=2, pages=5)
        wrong = dict(shapes, recurrent=(2, 48, 48, 128, 128))
        if tp > 1:
            with pytest.raises(ValueError, match="history"):
                layout.validate_history_shapes(wrong, slots=2, pages=5)


@pytest.mark.parametrize("size,rank", [(0, 0), (3, 0), (True, 0), (2, -1), (2, 2), (2, False)])
def test_invalid_tp_is_rejected(size, rank):
    with pytest.raises(ValueError):
        TPLayout(size, rank)


@pytest.mark.parametrize("tp", (1, 2, 4))
def test_native_linear_tile_requirements_do_not_silently_drop_mlp_tail(tp):
    validate_linear_tiles(6144 // tp, 5120)
    validate_linear_tiles(5120, 17408 // tp)
    if tp == 4:
        with pytest.raises(ValueError, match="tail-aware"):
            validate_linear_tiles(17408 // tp, 5120)
        validate_linear_tiles(4608, 5120)  # 需要显式补齐 256 列，不是直接截掉尾部。
    else:
        validate_linear_tiles(17408 // tp, 5120)


def test_replicated_norm_and_nonfinite_local_shards_are_rejected(small_weights):
    norm = small_weights[0]["norm.weight"]
    with pytest.raises(ValueError, match="differs across ranks"):
        reshard_parameter(0, "norm.weight", source_tp=2, layout=TPLayout(2, 0),
                          read=lambda rank, selection: norm + rank)
    with pytest.raises(ValueError, match="invalid rank-local"):
        reshard_parameter(0, "A_log", source_tp=1, layout=TPLayout(),
                          read=lambda rank, selection: torch.full((12,), float("nan")))


def make_checkpoint(tmp_path, weights, source_tp):
    model, head = tmp_path / "model", tmp_path / "head"
    model.mkdir()
    head.mkdir()
    config = dict(hidden_size=5120, num_hidden_layers=64, num_attention_heads=24,
                  num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
                  linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
                  vocab_size=248320, layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16)
    (model / "config.json").write_text(json.dumps(dict(text_config=config)))
    base, shards = {}, [{} for _ in range(source_tp)]
    for layer in range(64):
        for name, value in weights[3 if layer % 4 == 3 else 0].items():
            kind = "self_attn" if layer % 4 == 3 else "linear_attn"
            base[f"model.language_model.layers.{layer}.{kind}.{name}"] = value.clone()
            for rank in range(source_tp):
                shards[rank][f"layers.{layer}.attention.draft.{name}"] = reference_shard(
                    name, value, source_tp, rank).clone()
    save_file(base, model / "model.safetensors")
    (model / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={
        name: "model.safetensors" for name in base})))
    hashes = [hashlib.sha256((model / name).read_bytes()).hexdigest()
              for name in ("config.json", "model.safetensors.index.json")]
    (head / "manifest.json").write_text(json.dumps(dict(format="orthrus-qwen38-native-tp-v1",
        base_config_sha256=hashes[0], base_index_sha256=hashes[1], tp_size=source_tp,
        block_size=32, mask_token_id=1, hybrid_policy="orthrus_qwen35_reference")))
    for rank, shard in enumerate(shards):
        save_file(shard, head / f"rank{rank}.safetensors")
    return model, head


@pytest.mark.parametrize("source_tp", (1, 2, 4))
def test_real_file_slices_and_prepared_banks_are_rank_local(tmp_path, small_weights, source_tp):
    model, directory = make_checkpoint(tmp_path, small_weights, source_tp)
    for target_tp in (1, 2, 4):
        for rank in range(target_tp):
            head = HeadSource(model, directory, runtime_tp=target_tp, runtime_rank=rank)
            assert head.tp == source_tp and head.layout == TPLayout(target_tp, rank)
            for layer in (0, 3):
                assert all(torch.equal(value, reference_shard(name, small_weights[layer][name], target_tp, rank))
                           for name, value in head.layer(layer).items())
            head.prepare()
            assert head.banks(0)[2].shape == (128, 12)
            assert torch.count_nonzero(head.banks(0)[2][12 // target_tp:]) == 0
            assert torch.count_nonzero(head.banks(0)[5][0, 12 // target_tp:]) == 0
            assert head.banks(0)[5].dtype == torch.float32
    # AR 复制初始化同样支持局部读取，不要求先生成一个完整 draft checkpoint。
    initialized = HeadSource(model, runtime_tp=4, runtime_rank=2)
    for layer in (0, 3):
        assert all(torch.equal(value, reference_shard(name, small_weights[layer][name], 4, 2))
                   for name, value in initialized.layer(layer).items())


@pytest.mark.parametrize("name", ("q_proj.weight", "A_log"))
def test_checkpoint_header_rejects_wrong_or_mixed_rank_dtypes(tmp_path, small_weights, name):
    model, directory = make_checkpoint(tmp_path, small_weights, 2)
    path = directory / "rank1.safetensors"
    values = load_file(path)
    layer = 3 if name == "q_proj.weight" else 0
    key = f"layers.{layer}.attention.draft.{name}"
    values[key] = values[key].float() if name == "q_proj.weight" else values[key].bfloat16()
    save_file(values, path)
    with pytest.raises(ValueError, match="incompatible dtype"):
        HeadSource(model, directory, runtime_tp=2, runtime_rank=0)


def full_attention(hidden, weights, history):
    qg = F.linear(hidden, weights["q_proj.weight"].float()).reshape(32, -1, 4)
    q, gate = qg.chunk(2, dim=-1)
    k = F.linear(hidden, weights["k_proj.weight"].float()).reshape(32, -1, 2)
    v = F.linear(hidden, weights["v_proj.weight"].float()).reshape(32, -1, 2)
    # 所有 K32 query 读取相同历史与整个临时块，不沿 32 个位置切分。
    k = torch.cat((history[0], k.transpose(0, 1)), dim=1)
    v = torch.cat((history[1], v.transpose(0, 1)), dim=1)
    groups = q.shape[1] // k.shape[0]
    result = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
        k.repeat_interleave(groups, 0).unsqueeze(0), v.repeat_interleave(groups, 0).unsqueeze(0))
    gated = result[0].transpose(0, 1) * gate.sigmoid()
    return F.linear(gated.flatten(1), weights["o_proj.weight"].float())


@pytest.mark.parametrize("tp", (2, 4))
def test_head_parallel_bidirectional_attention_reduces_to_the_unsplit_result(small_weights, tp):
    torch.manual_seed(12)
    hidden = torch.randn(32, 12)
    history = (torch.randn(4, 5, 2), torch.randn(4, 5, 2))
    full = full_attention(hidden, small_weights[3], history)
    local = []
    for rank in range(tp):
        weights = {name: reference_shard(name, value, tp, rank) for name, value in small_weights[3].items()}
        local.append(full_attention(hidden, weights, tuple(value.chunk(tp)[rank] for value in history)))
    torch.testing.assert_close(torch.stack(local).sum(0), full, rtol=1e-5, atol=1e-6)


def gdn(hidden, weights, conv_history, recurrent):
    # 缩小 head_dim 的参考计算：检查 Q/K/V 分段、3:1 分组及已有 AR 状态的分片。
    project = lambda name: F.linear(hidden, weights[name].float())
    heads = weights["A_log"].numel()
    convolved = F.conv1d(torch.cat((conv_history, project("in_proj_qkv.weight"))).T.unsqueeze(0),
                        weights["conv1d.weight"].float(), groups=10 * heads // 3)[0].T
    q, k, v = F.silu(convolved).split((2 * heads // 3, 2 * heads // 3, 2 * heads), dim=-1)
    q = F.normalize(q.reshape(32, -1, 2), dim=-1).repeat_interleave(3, 1) / 2 ** .5
    k = F.normalize(k.reshape(32, -1, 2), dim=-1).repeat_interleave(3, 1)
    v = v.reshape(32, heads, 2)
    decay = (-weights["A_log"].float().exp() * F.softplus(
        project("in_proj_a.weight") + weights["dt_bias"].float())).exp()
    beta = project("in_proj_b.weight").sigmoid()
    state, outputs = recurrent.clone(), []
    for row in range(32):
        state = state * decay[row, :, None, None]
        delta = beta[row, :, None] * (v[row] - torch.einsum("hk,hkv->hv", k[row], state))
        state = state + torch.einsum("hk,hv->hkv", k[row], delta)
        outputs.append(torch.einsum("hk,hkv->hv", q[row], state))
    output = torch.stack(outputs)
    output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-6)
    output = output * weights["norm.weight"][:2].float()
    output = output * F.silu(project("in_proj_z.weight").reshape(32, heads, 2))
    return F.linear(output.flatten(1), weights["out_proj.weight"].float())


@pytest.mark.parametrize("tp", (2, 4))
def test_head_parallel_gdn_reduces_to_the_unsplit_result_without_mutating_history(small_weights, tp):
    torch.manual_seed(35)
    hidden, conv, recurrent = torch.randn(32, 12), torch.randn(3, 40), torch.randn(12, 2, 2)
    original = (conv.clone(), recurrent.clone())
    full, local = gdn(hidden, small_weights[0], conv, recurrent), []
    for rank in range(tp):
        weights = {name: reference_shard(name, value, tp, rank) for name, value in small_weights[0].items()}
        local_conv = torch.cat([part.chunk(tp, dim=1)[rank] for part in conv.split((8, 8, 24), dim=1)], dim=1)
        local.append(gdn(hidden, weights, local_conv, recurrent.chunk(tp)[rank]))
    torch.testing.assert_close(torch.stack(local).sum(0), full, rtol=1e-5, atol=1e-6)
    assert torch.equal(conv, original[0]) and torch.equal(recurrent, original[1])


def test_tp_head_cannot_enter_the_existing_tp1_graphs(small_weights, tmp_path):
    from blockspec.pypto_qwen38.draft import NeuralDraft
    model, directory = make_checkpoint(tmp_path, small_weights, 2)
    with pytest.raises(ValueError, match="full-model TP executor"):
        NeuralDraft(None, HeadSource(model, directory, runtime_tp=2, runtime_rank=0))
