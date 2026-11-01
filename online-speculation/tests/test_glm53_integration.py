import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

import blockspec.vllm_ascend.glm53_native as native
from blockspec.vllm_ascend.glm53_ops import kda_reference
from blockspec.vllm_ascend.glm53_proposer import GLM53OrthrusProposer


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Linear(nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs), None


class GatedNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

    def forward(self, inputs, gate):
        return self.norm(inputs) * gate.sigmoid()


def original_kda():
    heads, dim, width = 2, 128, 16
    size = heads * dim
    return NS(local_num_heads=heads, head_dim=dim, conv_size=4, kda_lower_bound=-5.,
              in_proj_qkvbfg_a=Linear(width, 3 * size + heads + 2 * dim, bias=False),
              f_b_proj=Linear(dim, size, bias=False), g_b_proj=Linear(dim, size, bias=False),
              o_norm=GatedNorm(dim), o_proj=Linear(size, width, bias=False),
              q_conv1d=nn.Conv1d(size, size, 4, groups=size, bias=False),
              k_conv1d=nn.Conv1d(size, size, 4, groups=size, bias=False),
              v_conv1d=nn.Conv1d(size, size, 4, groups=size, bias=False),
              A_log=nn.Parameter(torch.zeros(heads)), dt_bias=nn.Parameter(torch.full((size,), -3.)))


@pytest.mark.parametrize("weight", [0., .25])
def test_native_bidirectional_layer_uses_owned_reverse_state(monkeypatch, weight):
    torch.manual_seed(31)
    original = original_kda()
    causal = native.KDADraft(original)
    bidir = native.KDADraft(original, bidirectional=True, backward_weight=weight)
    history = (torch.randn(3, 768), torch.randn(2, 128, 128))
    before = tuple(x.clone() for x in history)
    hidden = torch.randn(8, 16)
    calls = []

    def core(*args, **kwargs):
        calls.append(args)
        return kda_reference(*args, **kwargs)

    monkeypatch.setattr(native, "kda_native", core)
    left = causal(hidden, history)
    calls.clear()
    right = bidir(hidden, history)
    assert len(calls) == 2 and bidir.reverse_calls == 1
    assert torch.count_nonzero(calls[1][5]) == 0
    torch.testing.assert_close(calls[1][3], calls[0][3].flip(0), rtol=0, atol=0)
    torch.testing.assert_close(calls[1][4], calls[0][4].flip(0), rtol=0, atol=0)
    if weight == 0:
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        assert not torch.equal(left, right)
    for x, y in zip(history, before, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def make_config(tmp_path, *, bidirectional=True):
    text = dict(model_type="glm5_next_text", num_hidden_layers=2, hidden_size=16,
                layer_types=["linear_attention", "deepseek_sparse_attention"],
                qk_rope_head_dim=0, num_attention_heads=2, qk_nope_head_dim=128,
                v_head_dim=128, kv_lora_rank=16, index_n_heads=2, index_head_dim=16,
                index_topk=8, index_kpool=4, linear_attn_config=dict(num_heads=2, head_dim=128),
                vocab_size=32)
    (tmp_path / "config.json").write_text(json.dumps(dict(text_config=text)))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text("{}")
    (tmp_path / "quant_model_description.json").write_text("{}")
    options = dict(mask_token_id=0, initialize_from_target=True,
                   hybrid_policy=native.BIDIRECTIONAL_POLICY if bidirectional else native.HYBRID_POLICY,
                   audit_cache=True)
    return NS(additional_config={},
              speculative_config=NS(method="custom_class", num_speculative_tokens=7,
                                    rejection_sample_method="standard", draft_sample_method="probabilistic"),
              model_config=NS(model=str(tmp_path), quantization="ascend", enforce_eager=True,
                              dtype=torch.bfloat16, hf_config=NS(orthrus_glm53=options)),
              quant_config=NS(quant_description={"kind": "W8A8_DYNAMIC"}),
              scheduler_config=NS(max_num_seqs=1),
              parallel_config=NS(pipeline_parallel_size=1, data_parallel_size=1),
              cache_config=NS(cache_dtype="auto", enable_prefix_caching=False))


@pytest.mark.parametrize("bidirectional", [False, True])
def test_head_roundtrip_and_policy_mismatch(tmp_path, monkeypatch, bidirectional):
    import blockspec.vllm_ascend.glm53_proposer as proposer

    monkeypatch.setattr(proposer, "install_runner_hooks", lambda: None)
    config = make_config(tmp_path, bidirectional=bidirectional)
    draft = GLM53OrthrusProposer(config)
    draft.tp = NS(size=1, rank=0)
    draft.model = NS(layers=nn.ModuleList([native.KDADraft(
        original_kda(), bidirectional=bidirectional, backward_weight=.25)]))
    original = draft.head_digest()
    directory = tmp_path / "head"
    draft._save_head(directory)
    with torch.no_grad():
        for parameter in draft.model.layers.parameters():
            parameter.zero_()
    assert draft.head_digest() != original
    draft._load_head(directory)
    assert draft.head_digest() == original
    with pytest.raises(ValueError, match="overwrite"):
        draft._save_head(directory)
    draft.options["hybrid_policy"] = (native.HYBRID_POLICY if bidirectional else native.BIDIRECTIONAL_POLICY)
    with pytest.raises(ValueError, match="does not match"):
        draft._load_head(directory)


@pytest.mark.parametrize("weight", [True, 1., -1., float("nan"), float("inf"), "0.25"])
def test_invalid_backward_initializer_fails_closed(tmp_path, monkeypatch, weight):
    import blockspec.vllm_ascend.glm53_proposer as proposer

    monkeypatch.setattr(proposer, "install_runner_hooks", lambda: None)
    config = make_config(tmp_path)
    config.model_config.hf_config.orthrus_glm53["kda_backward_weight"] = weight
    with pytest.raises(ValueError, match="backward weight"):
        GLM53OrthrusProposer(config)


def test_smoke_crop_preserves_architecture_and_filters_all_layer_lists():
    crop = load_script("prepare_glm53_smoke_model")
    source = dict(text_config=dict(model_type="glm5_next_text", num_hidden_layers=8,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"] + ["linear_attention"] * 4,
        indexer_types=["full"] * 8, mlp_layer_types=["dense"] * 3 + ["sparse"] * 5,
        num_nextn_predict_layers=1, hidden_size=4096,
        linear_attn_config=dict(kda_layers=[0, 1, 2, 4, 5, 6, 7], full_attn_layers=[3])))
    result = crop.truncated_config(source, 5)["text_config"]
    assert result["num_hidden_layers"] == 5 and result["hidden_size"] == 4096
    assert result["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4]
    assert all(len(result[key]) == 5 for key in ("layer_types", "indexer_types", "mlp_layer_types"))
    assert crop.keep_weight("model.language_model.layers.4.self_attn.q_proj.weight", 5)
    assert not crop.keep_weight("model.language_model.layers.5.self_attn.q_proj.weight", 5)
    assert not crop.keep_weight("model.mtp.layers.0.weight", 5)
    assert source["text_config"]["num_hidden_layers"] == 8


def test_cache_audit_owns_valid_target_rows():
    reader = native.NativeGLM53History.__new__(native.NativeGLM53History)
    conv = torch.randn(4, 768, 11)
    recurrent = torch.randn(8, 2, 128, 128)
    latent = torch.randn(3, 4, 1, 16, dtype=torch.bfloat16)
    pools = torch.randn(3, 4, 1, 8, dtype=torch.bfloat16)
    tail = torch.randn(3, 11, 16)
    tail[0, 0, 0] = float("nan")
    indexer = NS(index_kpool=4, k_cache=NS(prefix="index", kv_cache=pools),
                 tail_cache=NS(prefix="tail", kv_cache=[tail]))
    reader.entries = [("kda", None, NS(kv_cache=(conv, recurrent)), "kda"),
                      ("mla", NS(indexer=indexer), NS(kv_cache=latent), "mla")]
    table = torch.tensor([[0, 1, 2]])
    reader.metadata = dict(kda=NS(spec_sequence_masks=torch.ones(1), num_spec_decodes=1,
        spec_state_indices_tensor=torch.tensor([[1, 2, 3, 4]]),
        spec_decode_metadata=NS(spec_causal_conv1d=NS(cache_indices=torch.tensor([[2]])))),
        mla=NS(seq_lens=torch.tensor([7]), query_start_loc=torch.tensor([0, 2]), block_table=table),
        index=NS(block_table=table), tail=NS(block_table=table))
    before = reader.snapshot_target_cache()
    assert before["kda.state"].shape[0] == 2
    assert before["mla.latent"].shape[0] == 7
    assert torch.equal(before["mla.indexer_tail"], reader.snapshot_target_cache()["mla.indexer_tail"])
    latent[0, 0, 0] += 1
    after = reader.snapshot_target_cache()
    assert not torch.equal(before["mla.latent"], after["mla.latent"])
    assert torch.equal(before["kda.state"], after["kda.state"])
    tail[0, 1, 0] += 1
    assert not torch.equal(before["mla.indexer_tail"], reader.snapshot_target_cache()["mla.indexer_tail"])


@pytest.mark.parametrize("accepted", [1, 3, 8])
def test_kda_history_uses_only_selected_padded_rows(monkeypatch, accepted):
    conv = torch.arange(5 * 768 * 16).reshape(5, 768, 16).float()[:, :, :11]
    state = torch.arange(12 * 2 * 128 * 160).reshape(12, 2, 128, 160).float()[:, :, :, :128]
    original = NS(kv_cache=(conv, state), _conv_state_dim_first=True, conv_size=4)
    metadata = NS(spec_sequence_masks=torch.ones(1), num_spec_decodes=1,
        spec_state_indices_tensor=torch.arange(2, 10).reshape(1, 8),
        spec_decode_metadata=NS(spec_causal_conv1d=NS(cache_indices=torch.tensor([[3]]))))
    def forbidden(*args, **kwargs):
        raise AssertionError("must not gather from the complete padded arena")
    monkeypatch.setattr(torch.Tensor, "index_select", forbidden)
    tail, matrix = native.kda_history(original, metadata, accepted=accepted, query_length=8)
    torch.testing.assert_close(tail, conv[3, :, accepted - 1:accepted + 2].T)
    torch.testing.assert_close(matrix, state[accepted + 1])
    matrix.zero_()
    assert torch.count_nonzero(state[accepted + 1]) > 0


def test_kda_prefill_uses_native_prefill_slot():
    original = NS(kv_cache=(torch.randn(5, 11, 768), torch.randn(6, 2, 128, 128)),
                  _conv_state_dim_first=False, conv_size=4)
    metadata = NS(spec_sequence_masks=torch.zeros(1), num_spec_decodes=0, num_prefills=1,
                  prefill_state_indices=torch.tensor([4]), non_spec_state_indices_tensor=torch.tensor([1]),
                  non_spec_prefill_metadata=NS(causal_conv1d=NS(cache_indices=torch.tensor([2]))))
    tail, matrix = native.kda_history(original, metadata, accepted=7, query_length=7)
    torch.testing.assert_close(tail, original.kv_cache[0][2, :3])
    torch.testing.assert_close(matrix, original.kv_cache[1][4])


def test_expanded_operator_table_reads_subpages_of_padded_physical_cache(monkeypatch):
    arena = torch.arange(4 * 12 * 7.).reshape(4, 12, 1, 7)
    cache = arena[:, :8, :, :5]
    assert not cache.is_contiguous()
    # 每个物理页含两个四行子页；4、5 对应物理页 2，0 对应物理页 0。
    table = torch.tensor([[4, 5, 0]])
    def forbidden(*args, **kwargs):
        raise AssertionError("must not materialize the whole padded cache")
    monkeypatch.setattr(torch.Tensor, "index_select", forbidden)
    actual = native.read_pages(cache, table, 10, block_size=4)
    torch.testing.assert_close(actual, torch.cat((cache[2, :, 0], cache[0, :2, 0])))
    actual.zero_()
    assert cache[2].count_nonzero() > 0


def test_nope_cache_accepts_only_an_empty_rope_placeholder():
    latent = torch.zeros(2, 4, 1, 16, dtype=torch.bfloat16)
    assert native.cache_tensor([(latent, torch.empty(2, 4, 1, 0))]) is latent
    with pytest.raises(ValueError, match="BF16"):
        native.cache_tensor((latent, torch.ones(2, 4, 1, 2)))
    with pytest.raises(ValueError, match="BF16"):
        native.cache_tensor(latent.to(torch.int8))


def test_runtime_bootstrap_preserves_image_pythonpath(monkeypatch):
    runtime = load_script("run_in_runtime")
    monkeypatch.setenv("PYTHONPATH", "/image/cann:/image/tbe")
    monkeypatch.setattr(runtime.sys, "argv", ["run_in_runtime.py", "target.py", "--flag"])
    monkeypatch.setattr(runtime.sys, "path", list(runtime.sys.path))
    calls = []
    monkeypatch.setattr(runtime.runpy, "run_path", lambda path, **kwargs: calls.append((path, kwargs)))
    runtime.main()
    assert runtime.os.environ["PYTHONPATH"].endswith(":/image/cann:/image/tbe")
    assert runtime.sys.argv == ["target.py", "--flag"]
    assert calls == [("target.py", {"run_name": "__main__"})]


def test_cli_selects_bidirectional_policy_and_audit():
    cli = load_script("run_glm53_vllm")
    args = NS(model="/model", tp=1, mode="orthrus", max_model_len=2048,
              max_num_batched_tokens=256, experimental_hybrid_policy=True, head_dir=None,
              initialize_from_target=True, mask_token_id=0, block_size=8, kda_mode="bidirectional",
              kda_backward_weight=.25, audit_cache=True)
    options = cli.settings(args)["hf_overrides"]["orthrus_glm53"]
    assert options["hybrid_policy"] == native.BIDIRECTIONAL_POLICY
    assert options["kda_backward_weight"] == .25 and options["audit_cache"]
    args.head_dir, args.initialize_from_target = "/head", False
    with pytest.raises(ValueError, match="backward weight"):
        cli.settings(args)


def test_ngram_control_does_not_install_neural_proposer_or_worker_extension():
    cli = load_script("run_glm53_vllm")
    args = NS(model="/model", tp=4, mode="ngram", max_model_len=18432,
              max_num_batched_tokens=256, block_size=8)
    options = cli.settings(args)
    assert options["speculative_config"]["method"] == "ngram"
    assert options["speculative_config"]["num_speculative_tokens"] == 7
    assert "worker_extension_cls" not in options and "hf_overrides" not in options
    assert not options["disable_log_stats"]


@pytest.mark.parametrize("alias", [False, True])
def test_head_replacement_probe_restores_parameters_and_detects_aliases(monkeypatch, alias):
    import copy
    from contextlib import nullcontext
    import sys
    from blockspec.vllm_ascend.glm53_validation import probe_head_replacement

    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_forward_context",
                        NS(set_ascend_forward_context=lambda *a, **kw: nullcontext()))

    def projection():
        layer = Linear(4, 4, bias=False)
        layer.input_size = 4
        return layer

    ar_kda = NS(in_proj_qkvbfg_a=projection(), o_proj=projection())
    ar_mla = NS(fused_qkv_a_proj=projection(), o_proj=projection(), indexer=NS(wq_b=projection()))
    draft_kda = copy.deepcopy(ar_kda)
    draft_mla = NS(fused_qkv_a_proj=copy.deepcopy(ar_mla.fused_qkv_a_proj),
                   o_proj=copy.deepcopy(ar_mla.o_proj), index_query=copy.deepcopy(ar_mla.indexer.wq_b))
    if alias:
        draft_kda.in_proj_qkvbfg_a = ar_kda.in_proj_qkvbfg_a
    target = NS(model=NS(layers=[NS(layer_kind="kda", self_attn=ar_kda),
                                NS(layer_kind="mla", self_attn=ar_mla)]))
    model = NS(target=target, layers=[draft_kda, draft_mla], embedding=nn.Embedding(8, 4), vllm_config=NS())
    worker = NS(model_runner=NS(drafter=NS(model=model, tp=NS(rank=0))))
    original = ar_kda.in_proj_qkvbfg_a.weight.detach().clone()
    if alias:
        with pytest.raises(AssertionError, match="aliases target"):
            probe_head_replacement(worker)
    else:
        result = probe_head_replacement(worker)
        assert len(result["projections"]) == 5
        assert all(p["changed"] and p["restored"] and p["target_unchanged"] for p in result["projections"])
    torch.testing.assert_close(ar_kda.in_proj_qkvbfg_a.weight, original, rtol=0, atol=0)
