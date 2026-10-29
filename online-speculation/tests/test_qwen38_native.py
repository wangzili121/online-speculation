"""Native weight aliases, inference-mode replay, TP gradients and runner hooks."""

import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from blockspec.feedback import Feedback
from blockspec.parallel.online import SuffixConfig
from blockspec.vllm_ascend.qwen38_runtime import install_runner_hooks
from blockspec.vllm_ascend.qwen38_attention import verification_chunks
from blockspec.vllm_ascend.qwen38_proposer import Qwen38OrthrusProposer
from blockspec.vllm_ascend.qwen38_weights import (
    NativeSuffixLearner, ParallelLinear, TensorParallel, from_native_model,
)


def fixture():
    torch.manual_seed(43)
    config = Qwen3_5TextConfig(
        vocab_size=48, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "sdpa"
    return Qwen3_5ForCausalLM(config).eval().requires_grad_(False)


def native_fixture(hf, rank=0, size=1, inference=False):
    def tensor(value, axis=None):
        value = value if axis is None else value.chunk(size, dim=axis)[rank]
        with torch.inference_mode(inference):
            return nn.Parameter(value.detach().clone(), requires_grad=False)

    def packed(*values):
        return NS(weight=tensor(torch.cat([v.chunk(size)[rank] for v in values])))

    layers = []
    for original in hf.model.layers:
        layer = NS(input_layernorm=NS(weight=tensor(original.input_layernorm.weight)),
                   post_attention_layernorm=NS(weight=tensor(original.post_attention_layernorm.weight)),
                   mlp=NS(gate_up_proj=packed(original.mlp.gate_proj.weight, original.mlp.up_proj.weight),
                          down_proj=NS(weight=tensor(original.mlp.down_proj.weight, 1))))
        if hasattr(original, "self_attn"):
            a = original.self_attn
            layer.self_attn = NS(qkv_proj=packed(a.q_proj.weight, a.k_proj.weight, a.v_proj.weight),
                                 o_proj=NS(weight=tensor(a.o_proj.weight, 1)),
                                 q_norm=NS(weight=tensor(a.q_norm.weight)), k_norm=NS(weight=tensor(a.k_norm.weight)))
        else:
            a = original.linear_attn
            q, k, v = a.in_proj_qkv.weight.split((a.key_dim, a.key_dim, a.value_dim))
            cq, ck, cv = a.conv1d.weight.split((a.key_dim, a.key_dim, a.value_dim))
            layer.linear_attn = NS(
                gqa_interleaved_layout=False, disable_tp_for_ba_proj=False,
                in_proj_qkvz=packed(q, k, v, a.in_proj_z.weight),
                in_proj_ba=packed(a.in_proj_b.weight, a.in_proj_a.weight),
                out_proj=NS(weight=tensor(a.out_proj.weight, 1)),
                conv1d=NS(weight=tensor(torch.cat([x.chunk(size)[rank] for x in (cq, ck, cv)]))),
                A_log=tensor(a.A_log, 0), dt_bias=tensor(a.dt_bias, 0), norm=NS(weight=tensor(a.norm.weight)),
            )
        layers.append(layer)
    width = hf.config.vocab_size // size
    embedding = NS(weight=tensor(hf.model.embed_tokens.weight, 0), num_added_embeddings=0,
                   shard_indices=NS(org_vocab_start_index=rank * width, org_vocab_end_index=(rank + 1) * width))
    return NS(model=NS(embed_tokens=embedding, norm=NS(weight=tensor(hf.model.norm.weight)), layers=layers),
              lm_head=NS(weight=tensor(hf.lm_head.weight, 0)))


@pytest.mark.parametrize("inference", [False, True])
def test_native_aliases_ar_and_online_update(inference):
    hf = fixture()
    native = native_fixture(hf, inference=inference)
    with torch.inference_mode(False):
        model = from_native_model(native, hf.config, tp=TensorParallel(), block_size=4, mask_token_id=1)
        assert model.embedding.weight is native.model.embed_tokens.weight
        assert model.layers[-1].mlp.down_proj.weight is native.model.layers[-1].mlp.down_proj.weight
        assert not torch.is_inference(model.layers[-1].attention.draft.o_proj.weight)
        ids = torch.tensor([[3, 5, 8]])
        with torch.no_grad():
            torch.testing.assert_close(model(ids).logits, hf(ids, use_cache=False).logits, rtol=1e-4, atol=1e-5)
        learner = NativeSuffixLearner(model, SuffixConfig(stride=1, loss="tv", learning_rate=1e-3), TensorParallel())
        frozen = native.lm_head.weight.clone()
        with torch.no_grad():
            history = model(ids).cache
            block = torch.tensor([[9, 1, 1, 1]])
            draft = model(block, view="draft", cache=history, capture_layer=3)
            teacher = model(torch.tensor([[9, 11, 12, 13]]), cache=history).logits[0, :3]
        before = model.layers[-1].attention.draft.o_proj.weight.clone()
        # Simulate the native worker's enclosing inference_mode.
        with torch.inference_mode():
            with torch.inference_mode(False):
                learner.observe(Feedback(block, history, teacher, 3, draft.boundary))
        assert learner.updates == 1
        assert not torch.equal(before, model.layers[-1].attention.draft.o_proj.weight)
        torch.testing.assert_close(native.lm_head.weight, frozen, rtol=0, atol=0)


def test_inference_frozen_linear_has_exact_input_gradient():
    torch.manual_seed(2)
    with torch.inference_mode():
        weight = nn.Parameter(torch.randn(5, 7), requires_grad=False)
    value = torch.randn(3, 7, requires_grad=True)
    output = ParallelLinear(weight, TensorParallel(), "row")(value)
    output.square().sum().backward()
    reference = value.detach().clone().requires_grad_()
    torch.nn.functional.linear(reference, weight.clone()).square().sum().backward()
    torch.testing.assert_close(value.grad, reference.grad)
    assert weight.grad is None


def test_draft_copies_custom_loader_parameters_as_plain_parameters():
    class LoaderParameter(nn.Parameter):
        def __new__(cls, data, *, loader=None):
            return super().__new__(cls, data, requires_grad=False)

    hf = fixture()
    native = native_fixture(hf)
    original = native.model.layers[0].linear_attn.conv1d.weight
    native.model.layers[0].linear_attn.conv1d.weight = LoaderParameter(original.detach())
    model = from_native_model(native, hf.config, tp=TensorParallel(), block_size=4, mask_token_id=1)
    ar = model.layers[0].attention.ar.conv1d.weight
    draft = model.layers[0].attention.draft.conv1d.weight
    assert ar is native.model.layers[0].linear_attn.conv1d.weight
    assert type(draft) is nn.Parameter and draft.data_ptr() != ar.data_ptr()
    torch.testing.assert_close(ar, draft, rtol=0, atol=0)


def test_scoped_hooks_retain_custom_and_leave_others_unchanged():
    events = []

    class Runner:
        def _prepare_inputs(self, scheduler_output):
            return None, scheduler_output, None

        def _get_drafter(self):
            return "ordinary"

        def _set_up_drafter(self):
            self.drafter = None
            self.drafter = self._get_drafter()

        def propose_draft_token_ids(self, valid_sampled_token_ids, scheduler_output):
            return "ordinary draft"

        def _sample(self, logits, spec_decode_metadata):
            return "ordinary sample"

    class Proposer:
        qwen38_native_proposer = True

        def bind_runner(self, runner):
            events.append("bind")

        def propose_batch(self, sampled, scheduler):
            events.append("draft")
            return [[4, 5]]

        def teacher_rows(self, logits, metadata):
            events.append("teacher")
            return logits.clone()

        def verification_probs(self, metadata):
            return "q"

        def observe_verified(self, metadata, teacher, sampled):
            events.append("update")

    install_runner_hooks(Runner)
    install_runner_hooks(Runner)
    ordinary = Runner()
    ordinary._set_up_drafter()
    assert ordinary.drafter == "ordinary"
    assert ordinary.propose_draft_token_ids([], None) == "ordinary draft"
    assert ordinary._sample(None, None) == "ordinary sample"
    runner = Runner()
    plugin = Proposer()
    runner.drafter = plugin
    runner._set_up_drafter()
    assert runner.drafter is plugin
    counter = NS(np=np.zeros(1), copy_to_gpu=lambda: events.append("counter"))
    runner.num_decode_draft_tokens = counter
    runner._prepare_inputs(None)
    assert counter.np.tolist() == [-1]
    runner._prepare_inputs(NS())
    assert runner.propose_draft_token_ids([[2]], None) == [[4, 5]]
    runner.input_batch = NS(update_async_output_token_ids=lambda: None, sampling_metadata=None)

    def reject(meta, q, logits, sampling):
        assert q == "q"
        events.append("verify")
        logits.zero_()
        return NS(sampled_token_ids=torch.tensor([[7, -1]]))

    runner.rejection_sampler = reject
    runner._sample(torch.ones(2, 4), NS())
    assert events == ["bind", "counter", "draft", "teacher", "verify", "update"]


def test_hooks_reject_incompatible_runner():
    with pytest.raises(RuntimeError, match="unsupported Ascend"):
        install_runner_hooks(NS)


def proposer_fixture():
    hf = fixture()
    proposer = Qwen38OrthrusProposer.__new__(Qwen38OrthrusProposer)
    proposer.tp = TensorParallel()
    proposer.model = from_native_model(native_fixture(hf, inference=True), hf.config,
                                       tp=proposer.tp, block_size=4, mask_token_id=1)
    proposer.learner = NativeSuffixLearner(proposer.model, SuffixConfig(stride=1, loss="tv"), proposer.tp)
    proposer.config = NS(speculative_config=NS(draft_sample_method="probabilistic"))
    proposer.options, proposer.block_size = {"mask_token_id": 1}, 4
    proposer.generator = torch.Generator().manual_seed(7)
    proposer.prefix = proposer.history = proposer.pending = proposer.request_id = None
    proposer.proposals = proposer.verified = proposer.accepted = 0
    proposer.max_verified_tokens = proposer.full_blocks = 0
    proposer.runner = NS(input_batch=NS(num_reqs=1, req_ids=["r1"], num_tokens_no_spec=[4],
                                        token_ids_cpu=torch.tensor([[3, 5, 8, 9]])), max_model_len=128)
    return proposer


def test_native_feedback_uses_only_reached_rows_and_clears_old_request():
    proposer = proposer_fixture()
    with torch.inference_mode():
        candidates = proposer.propose_batch([[9]], NS())[0]
        metadata = NS(num_draft_tokens=[3], draft_token_ids=torch.tensor(candidates),
                      target_logits_indices=torch.arange(3))
        q = proposer.verification_probs(metadata).clone()
        # Native sampler accepts one candidate then rejects the next: 2 teacher rows.
        teacher = proposer.teacher_rows(torch.randn(4, 48), metadata)
        proposer.observe_verified(metadata, teacher, torch.tensor([[candidates[0], 17, -1, -1]]))
        assert proposer.learner.updates == 1
        assert proposer.learner.replay[0].valid == 2
        torch.testing.assert_close(q.sum(-1), torch.ones(3))
        assert proposer.history.length == 3  # Candidate/draft KV did not enter AR history.
        proposer.runner.input_batch.req_ids = ["r2"]
        proposer.propose_batch([[9]], NS())
        assert not proposer.learner.replay
        assert proposer.pending["request_id"] == "r2"


def test_native_verifier_rejects_misaligned_pending_tokens():
    proposer = proposer_fixture()
    candidates = proposer.propose_batch([[9]], NS())[0]
    wrong = torch.tensor(candidates)
    wrong[0] = (wrong[0] + 1) % 48
    with pytest.raises(RuntimeError, match="do not match"):
        proposer.verification_probs(NS(num_draft_tokens=[3], draft_token_ids=wrong))


def test_native_head_checkpoint_roundtrip_and_contract(tmp_path):
    proposer = proposer_fixture()
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen3_5"}')
    (base / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    proposer.config.model_config = NS(model=str(base))
    proposer.learner.version = 1
    before = {name: p.clone() for name, p in proposer._draft_parameters().items()}
    proposer._save_head(tmp_path / "heads")
    with torch.no_grad():
        for parameter in proposer._draft_parameters().values():
            parameter.add_(1)
        proposer._load_head(tmp_path / "heads/version1")
    for name, parameter in proposer._draft_parameters().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    proposer.block_size = 8
    with pytest.raises(ValueError, match="head must match"):
        proposer._load_head(tmp_path / "heads/version1")


def test_cann_backend_layout_and_readonly_initial_state(monkeypatch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm, torch_chunk_gated_delta_rule
    from blockspec.vllm_ascend.qwen38_gdn_backend import install_cann_gdn_prefill

    def primitive(q, k, value, *, beta, initial_state, actual_seq_lengths, scale, g):
        assert actual_seq_lengths.tolist() == [3]
        assert initial_state.dtype == torch.bfloat16
        output, final = torch_chunk_gated_delta_rule(
            q[None].repeat_interleave(3, dim=2), k[None].repeat_interleave(3, dim=2),
            value[None], g[None], beta[None], initial_state=initial_state.transpose(-1, -2).float(),
            output_final_state=True)
        initial_state.zero_()
        return output[0], final.transpose(-1, -2).bfloat16()

    module = NS(chunk_gated_delta_rule=lambda **kw: "original")
    config = NS(additional_config={"qwen38_gdn_prefill_backend": "cann"})
    monkeypatch.setitem(sys.modules, "torch_npu", NS(npu_chunk_gated_delta_rule=primitive))
    monkeypatch.setitem(sys.modules, "vllm.config", NS(get_current_vllm_config=lambda: config))
    monkeypatch.setitem(sys.modules, "vllm.third_party.flash_linear_attention.ops.l2norm", NS(l2norm_fwd=l2norm))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", NS(gdn=module))
    install_cann_gdn_prefill()
    q, k = [torch.randn(1, 3, 2, 128).bfloat16() for _ in range(2)]
    value = torch.randn(1, 3, 6, 128).bfloat16()
    state = torch.randn(1, 6, 128, 128)
    original = state.clone()
    output, final = module.chunk_gated_delta_rule(
        q, k, value, -torch.rand(1, 3, 6), torch.rand(1, 3, 6).bfloat16(), initial_state=state,
        cu_seqlens=torch.tensor([0, 3]), output_final_state=True, use_qk_l2norm_in_kernel=True)
    assert output.shape == value.shape and final.shape == state.shape
    assert torch.isfinite(output).all() and torch.isfinite(final).all()
    torch.testing.assert_close(state, original, rtol=0, atol=0)
    config.additional_config.clear()
    assert module.chunk_gated_delta_rule(q, k, value, None, None) == "original"


@pytest.mark.parametrize("length", [17, 31, 32])
def test_paged_query_slices_preserve_causal_attention(length):
    torch.manual_seed(2)
    past = 9
    query, key, value = (torch.randn(1, 2, n, 8) for n in (length, past + length, past + length))
    mask = torch.arange(past + length)[None] <= past + torch.arange(length)[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    metadata = NS(actual_seq_lengths_q=[length], seq_lens_list=[past + length], causal=True,
                  seq_lens=torch.tensor([past + length]), seq_lens_cpu=torch.tensor([past + length]),
                  query_start_loc=torch.tensor([0, length]))
    parts = []
    for rows, chunk in verification_chunks(metadata):
        end = chunk.seq_lens_list[0]
        assert chunk.actual_seq_lengths_q[0] <= 16
        parts.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, rows], key[:, :, :end], value[:, :, :end], attn_mask=mask[rows, :end]))
    torch.testing.assert_close(torch.cat(parts, dim=2), expected)
    assert metadata.seq_lens_list == [past + length]


def _tp_worker(rank, rendezvous, last_layers):
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torch_npu_available = imports.is_torch_npu_available = lambda: False
    import torch.distributed as dist
    from blockspec.parallel.qwen38 import Qwen38DualView
    from blockspec.parallel.online import SuffixLearner

    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}")
    try:
        hf = fixture()
        tp = TensorParallel(dist.group.WORLD, (0, 1))
        local = from_native_model(native_fixture(hf, rank, 2, inference=True), hf.config,
                                  tp=tp, block_size=4, mask_token_id=1)
        full = Qwen38DualView(hf, block_size=4, mask_token_id=1, hybrid_policy="orthrus_qwen35_reference")
        ids, block = torch.tensor([[3, 5, 8]]), torch.tensor([[9, 1, 1, 1]])
        with torch.no_grad():
            lh, fh = local(ids).cache, full(ids).cache
            ld = local(block, view="draft", cache=lh, capture_layer=4 - last_layers)
            fd = full(block, view="draft", cache=fh, capture_layer=4 - last_layers)
            torch.testing.assert_close(ld.logits, fd.logits, rtol=2e-4, atol=2e-5)
            teacher = full(torch.tensor([[9, 11, 12, 13]]), cache=fh).logits[0, :3]
        policy = SuffixConfig(last_layers=last_layers, stride=100, loss="tv", learning_rate=1e-3)
        ll = NativeSuffixLearner(local, policy, tp)
        fl = SuffixLearner(full, policy)
        ll.observe(Feedback(block, lh, teacher, 3, ld.boundary), may_update=False)
        fl.observe(Feedback(block, fh, teacher, 3, fd.boundary), may_update=False)
        ll.backward()
        fl.backward()

        def shard(name, value):
            if name in ll.replicated:
                return value
            if ".conv1d.weight" in name or ".in_proj_qkv.weight" in name:
                mixer = hf.model.layers[int(name.split(".")[1])].linear_attn
                parts = value.split((mixer.key_dim, mixer.key_dim, mixer.value_dim))
                return torch.cat([part.chunk(2)[rank] for part in parts])
            return value.chunk(2, dim=1 if ".o_proj." in name or ".out_proj." in name else 0)[rank]

        for name, parameter in ll.master.items():
            expected = shard(name, fl.master[name].grad)
            torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=2e-5,
                                       msg=lambda error, name=name: f"{name}: {error}")
        torch.testing.assert_close(ll.clip_gradients(), fl.clip_gradients(), rtol=3e-4, atol=2e-5)
        ll.optimizer.step()
        fl.optimizer.step()
        ll.publish()
        fl.publish()
        for name, parameter in ll.master.items():
            expected = shard(name, fl.master[name])
            torch.testing.assert_close(parameter, expected, rtol=1e-4, atol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="CPU distributed integration is run in the Linux image")
@pytest.mark.parametrize("last_layers", [1, 4])
def test_two_rank_forward_gradients_and_global_clip_match_unsharded(tmp_path, last_layers):
    torch.multiprocessing.start_processes(_tp_worker, args=(str(tmp_path / "gloo"), last_layers), nprocs=2,
                                          join=True, start_method="spawn")
