"""Hybrid reference semantics; no checkpoint or accelerator required."""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

from blockspec.feedback import Feedback
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.generation import generate, generate_ar
from blockspec.parallel.online import SuffixConfig, SuffixLearner
from blockspec.parallel.qwen38 import Qwen38Branch, Qwen38DualView, native_reference_logits
from blockspec.sampling import SamplingConfig


def make_model(block_size=4):
    torch.manual_seed(9)
    config = Qwen3_5TextConfig(
        vocab_size=48, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [2, 2, 4]},
    )
    config._attn_implementation = "sdpa"
    target = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    model = Qwen38DualView(target, block_size=block_size, mask_token_id=1,
                           hybrid_policy="orthrus_qwen35_reference")
    return target, model


def assert_history_same(left, right):
    assert left.length == right.length
    for pair, other in zip(left, right, strict=True):
        for x, y in zip(pair, other, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


@torch.no_grad()
def test_ar_matches_transformers_and_cached_chunks():
    target, model = make_model()
    ids = torch.tensor([[3, 5, 8, 9, 11]])
    reference = target(ids, use_cache=False).logits
    torch.testing.assert_close(model(ids).logits, reference, rtol=1e-4, atol=1e-5)
    prefix = model(ids[:, :2]).cache
    chunk = model(ids[:, 2:], cache=prefix)
    torch.testing.assert_close(chunk.logits, reference[:, 2:], rtol=1e-4, atol=1e-5)
    original = model(ids[:, :2]).cache
    assert_history_same(prefix, original)


@torch.no_grad()
@pytest.mark.parametrize("block_size", [2, 4, 8, 32])
def test_draft_is_readonly_and_suffix_matches(block_size):
    _, model = make_model(block_size)
    history = model(torch.tensor([[3, 5, 8]])).cache
    snapshot = model(torch.tensor([[3, 5, 8]])).cache
    ids = torch.full((1, block_size), 1)
    ids[:, 0] = 9
    draft = model(ids, view="draft", cache=history, capture_layer=3)
    suffix = model.forward_suffix(draft.boundary, cache=history[3:])
    torch.testing.assert_close(suffix.logits, draft.logits)
    assert draft.cache is None
    assert_history_same(history, snapshot)


@torch.no_grad()
def test_greedy_exact_and_rejection_state_audit():
    _, model = make_model()
    branch = Qwen38Branch(model)
    prompt = torch.tensor([[3, 5, 8]])
    ar = generate_ar(branch, prompt, 16)
    speculative = generate(branch, prompt, 16, audit_cache=True)
    assert speculative.tokens == ar.tokens
    assert speculative.draft_forwards > 0
    assert sum(speculative.accepted_per_round) < sum(speculative.proposed_per_round)


def test_last_layer_update_changes_only_draft_and_resumes():
    _, model = make_model()
    branch = Qwen38Branch(model)
    config = SuffixConfig(last_layers=1, stride=1, loss="tv", learning_rate=1e-3)
    learner = SuffixLearner(model, config)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = generate(branch, torch.tensor([[3, 5, 8]]), 12,
                      feedback=OnlineFeedback(learner=learner), audit_cache=True)
    assert result.updates > 0
    changed = [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]
    assert changed and all(name.startswith("layers.3.attention.draft.") for name in changed)
    assert all(p.grad is None for p in model.parameters())
    saved = learner.state_dict()
    learner.load_state_dict(saved)
    assert learner.version == result.updates


def test_loss_decreases_on_fixed_reached_rows():
    _, model = make_model()
    learner = SuffixLearner(model, SuffixConfig(last_layers=1, stride=100, loss="tv", learning_rate=1e-3))
    with torch.no_grad():
        history = model(torch.tensor([[3, 5, 8]])).cache
        inputs = torch.tensor([[9, 1, 1, 1]])
        draft = model(inputs, view="draft", cache=history, capture_layer=3)
        teacher = model(torch.tensor([[9, 11, 12, 13]]), cache=history).logits[0, :3]
    learner.observe(Feedback(inputs, history, teacher, 3, draft.boundary), may_update=False)
    losses = [learner.update()["loss"] for _ in range(12)]
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("last_layers", [1, 4])
def test_suffix_backward_and_sampling(last_layers):
    _, model = make_model()
    learner = SuffixLearner(model, SuffixConfig(last_layers=last_layers, stride=1, loss="tv"))
    result = generate(Qwen38Branch(model), torch.tensor([[3, 5, 8]]), 8,
                      sampling=SamplingConfig(temperature=1.),
                      generator=torch.Generator().manual_seed(5),
                      feedback=OnlineFeedback(learner=learner), audit_cache=True)
    assert result.updates > 0
    assert len(result.tokens) == 8


def test_hybrid_policy_must_be_explicit():
    target, _ = make_model()
    with pytest.raises(ValueError, match="policy"):
        Qwen38DualView(target, block_size=4, mask_token_id=1, hybrid_policy="qwen3")


@torch.no_grad()
def test_full_attention_draft_mask_exposes_whole_block():
    _, model = make_model()
    history = model(torch.tensor([[3, 5, 8]])).cache
    seen = []

    def capture(module, args, kwargs):
        seen.append(kwargs["attention_mask"].clone())

    hook = model.layers[-1].attention.draft.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model(torch.tensor([[9, 1, 1, 1]]), view="draft", cache=history)
    finally:
        hook.remove()
    assert seen[0].shape == (1, 1, 4, 7)
    assert torch.equal(seen[0], torch.zeros_like(seen[0]))


@torch.no_grad()
def test_all_accepted_commits_verifier_state_without_replay():
    _, model = make_model()
    branch = Qwen38Branch(model)
    history = model(torch.tensor([[3, 5, 8]])).cache
    proposal = branch.draft(torch.tensor([[9]]), history, 4)
    verified = model(proposal.verifier_inputs, cache=history).cache
    assert branch.commit_history(proposal, verified, verified.length) is verified


@torch.no_grad()
def test_explicit_layer_placement_preserves_logits():
    _, model = make_model()
    ids = torch.tensor([[3, 5, 8]])
    before = model(ids).logits
    model.place_layers(["cpu", "cpu"])
    torch.testing.assert_close(model(ids).logits, before, rtol=0, atol=0)


def test_rope_frequencies_remain_float32_during_weight_cast():
    _, model = make_model()
    before = model.rotary.inv_freq.clone()
    model.bfloat16()
    assert model.rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(model.rotary.inv_freq, before, rtol=0, atol=0)


@torch.no_grad()
def test_native_reference_preserves_hf_forward_and_removes_hooks():
    target, model = make_model()
    ids = torch.tensor([[3, 5, 8]])
    expected = target(ids, use_cache=False).logits
    torch.testing.assert_close(native_reference_logits(target, ids), expected, rtol=0, atol=0)
    assert not target.lm_head._forward_pre_hooks
    assert all(not layer._forward_pre_hooks for layer in target.model.layers)
    torch.testing.assert_close(model(ids).logits, expected, rtol=1e-4, atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("layer_index", [0, 3])
@pytest.mark.parametrize("component", [0, 1])
def test_matched_chunk_audit_detects_state_corruption(layer_index, component):
    _, model = make_model()
    branch = Qwen38Branch(model)
    ids = torch.tensor([[3, 5, 8, 9]])
    first = model(ids[:, :2]).cache
    history = model(ids[:, 2:], cache=first).cache
    branch.audit_history(history, ids, [2, 2])
    history[layer_index][component].add_(1.)
    with pytest.raises(AssertionError, match=f"layer {layer_index} state {component}"):
        branch.audit_history(history, ids, [2, 2])
