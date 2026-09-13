from dataclasses import replace
import json

import pytest
import torch
from safetensors.torch import save_file

from blockspec.sampling import SamplingConfig
from blockspec.parallel import (DualViewConfig, DualViewDecoder,
                                MaskedAttentionBranch, generate, generate_ar)
from blockspec.state import cache_length, trim_cache as crop_cache
from blockspec.parallel.training import (anchor_layout, distillation_loss, forward_kl,
                                         sample_anchors)
from blockspec.parallel.weights import (file_sha256, load_checkpoint, load_public,
                                        public_key_map, save_checkpoint)


def tiny(**kwargs):
    torch.manual_seed(731)
    return DualViewDecoder(replace(DualViewConfig(), **kwargs)).eval()


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_full_incremental_ar_and_clean_kv(backend):
    model = tiny().set_backend(backend)
    tokens = torch.tensor([[2, 8, 9, 3, 5]])
    full = model(tokens)
    cache, rows = None, []
    for index in range(tokens.shape[1]):
        result = model(tokens[:, index:index + 1], cache=cache)
        rows.append(result.logits)
        cache = result.cache
    torch.testing.assert_close(torch.cat(rows, 1), full.logits, atol=2e-6, rtol=2e-5)
    for pair, reference in zip(cache, full.cache, strict=True):
        for value, target in zip(pair, reference, strict=True):
            torch.testing.assert_close(value, target, atol=2e-6, rtol=2e-5)


def test_draft_cache_read_only_and_bidirectionality():
    model = tiny()
    history = model(torch.tensor([[3, 5, 7]])).cache
    snapshot = tuple((k.clone(), v.clone()) for k, v in history)
    first = model(torch.tensor([[9, 1, 1, 1]]), view="draft", cache=history)
    second = model(torch.tensor([[9, 1, 1, 6]]), view="draft", cache=history)
    assert first.cache is history and second.cache is history
    assert not torch.allclose(first.logits[:, 0], second.logits[:, 0])
    for pair, saved in zip(history, snapshot, strict=True):
        for current, old in zip(pair, saved, strict=True):
            assert torch.equal(current, old)


def test_projection_views_and_shared_parameters():
    model = tiny()
    assert model.head.weight is model.embedding.weight
    for layer in model.layers:
        ar, draft = layer.attention.ar, layer.attention.draft
        for key, weight in ar.named_parameters():
            other = dict(draft.named_parameters())[key]
            assert torch.equal(weight, other)
            assert weight is not other
    tokens = torch.tensor([[3, 6, 8]])
    before = model(tokens).logits.detach().clone()
    with torch.no_grad():
        model.layers[0].attention.draft.o.weight.add_(0.1)
        model.layers[0].attention.draft.q_norm.weight.mul_(2)
    assert torch.equal(before, model(tokens).logits)
    model.train_draft_only()
    assert all(parameter.requires_grad == (".attention.draft." in name)
               for name, parameter in model.named_parameters())


def test_rotary_precision_and_last_row_logits():
    model = tiny()
    original = model.frequencies.clone()
    model.bfloat16()
    assert model.frequencies.dtype == torch.float32
    assert torch.equal(original, model.frequencies)
    model.float()
    tokens = torch.tensor([[5, 3, 9]])
    torch.testing.assert_close(model(tokens, logits_to_keep=1).logits, model(tokens).logits[:, -1:])
    assert model(tokens, compute_logits=False).logits is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA table transfer check")
def test_rotary_table_is_identical_across_devices_and_weight_dtypes():
    model = tiny(head_dim=128)
    canonical = model.frequencies.clone()
    model.cuda().bfloat16()
    assert model.frequencies.dtype == torch.float32
    assert torch.equal(model.frequencies.cpu(), canonical)
    with torch.device("cuda"):
        initialized = tiny(head_dim=128)
    assert torch.equal(initialized.frequencies.cpu(), canonical)
    model.float().cpu()
    assert torch.equal(model.frequencies, canonical)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_random_anchor_isolation_and_single_block_equivalence(backend):
    model = tiny().set_backend(backend)
    clean = torch.tensor([[2, 8, 4, 7, 9, 6, 5, 3], [8, 2, 6, 3, 5, 4, 9, 7]])
    anchors = torch.tensor([[0, 3], [1, 4]])
    layout = anchor_layout(clean, anchors, 4, 1)
    teacher = model(clean)
    grouped = model(layout.tokens, view="draft", cache=teacher.cache,
                    positions=layout.positions, allowed=layout.allowed)
    for batch in range(2):
        for block in range(2):
            anchor = int(anchors[batch, block])
            history = None if anchor == 0 else model(clean[batch:batch + 1, :anchor]).cache
            single = model(layout.tokens[batch:batch + 1, 4 * block:4 * (block + 1)],
                           view="draft", cache=history)
            torch.testing.assert_close(grouped.logits[batch, 4 * block:4 * (block + 1)],
                                       single.logits[0], atol=2e-6, rtol=2e-5)
    altered = clean.clone()
    altered[0, 4:] = (altered[0, 4:] + 7) % model.config.vocab_size
    other_layout = anchor_layout(altered, anchors, 4, 1)
    other = model(other_layout.tokens, view="draft", cache=model(altered).cache,
                  positions=other_layout.positions, allowed=other_layout.allowed)
    torch.testing.assert_close(grouped.logits[0, 4:8], other.logits[0, 4:8])


def test_forward_kl_gradient_is_student_minus_teacher():
    torch.manual_seed(51)
    student = torch.randn(3, 7, requires_grad=True)
    teacher = torch.randn(3, 7, requires_grad=True)
    gradient, = torch.autograd.grad(forward_kl(student, teacher), student)
    torch.testing.assert_close(gradient, (student.softmax(-1) - teacher.softmax(-1)) / 3)
    assert teacher.grad is None


def test_chunked_loss_gradients_and_frozen_base():
    model = tiny().train_draft_only()
    tokens = torch.tensor([[3, 9, 4, 8, 2, 7, 6, 5]])
    anchors = torch.tensor([[0, 3]])
    loss = distillation_loss(model, tokens, anchors, chunk_rows=2)
    loss.backward()
    gradient = {name: value.grad.clone() for name, value in model.named_parameters() if value.requires_grad}
    assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
    assert all(torch.isfinite(value).all() for value in gradient.values())
    model.zero_grad(set_to_none=True)
    together = distillation_loss(model, tokens, anchors, chunk_rows=100)
    together.backward()
    torch.testing.assert_close(loss, together)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradient[name], atol=1e-7, rtol=1e-4)
    assert sum(value.abs().sum() for value in gradient.values()) > 0


def test_backbone_distillation_gradient_matches_finite_difference():
    model = tiny().train_draft_only()
    tokens, anchors = torch.tensor([[3, 6, 9, 2, 7, 4]]), torch.tensor([[0, 2]])
    loss = distillation_loss(model, tokens, anchors)
    loss.backward()
    parameter = model.layers[0].attention.draft.o.weight
    index = int(parameter.grad.abs().reshape(-1).argmax())
    analytic = float(parameter.grad.reshape(-1)[index])
    epsilon = 0.001
    with torch.no_grad():
        saved = parameter.reshape(-1)[index].clone()
        parameter.reshape(-1)[index] = saved + epsilon
    positive = float(distillation_loss(model, tokens, anchors).detach())
    with torch.no_grad():
        parameter.reshape(-1)[index] = saved - epsilon
    negative = float(distillation_loss(model, tokens, anchors).detach())
    with torch.no_grad():
        parameter.reshape(-1)[index] = saved
    assert (positive - negative) / (2 * epsilon) == pytest.approx(analytic, rel=.01, abs=2e-5)


def test_common_cache_preserves_packed_execution_storage():
    from blockspec.state import PackedCache
    packed = torch.randn(2, 2, 1, 2, 8, 4)
    trimmed = crop_cache(PackedCache(packed), 5)
    assert isinstance(trimmed, PackedCache)
    assert trimmed.packed.data_ptr() == packed.data_ptr()
    assert cache_length(trimmed) == 5


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 9, 24])
def test_masked_greedy_generation_and_cache_contract(budget):
    model = tiny()
    branch = MaskedAttentionBranch(model)
    prompt = torch.tensor([[3, 6, 9, 8]])
    reference = generate_ar(branch, prompt, budget)
    speculative = generate(branch, prompt, budget, block_size=4, audit_cache=True)
    assert speculative.tokens == reference.tokens
    if budget:
        assert speculative.prefill_output_tokens == 1
        assert speculative.prefill_forwards == 1
    assert len(speculative.tokens) == budget


def test_masked_eos_and_stochastic_output_budget():
    model = tiny()
    branch = MaskedAttentionBranch(model)
    prompt = torch.tensor([[3, 7, 8]])
    eos = int(model(prompt).logits[0, -1].argmax())
    output = generate(branch, prompt, 30, eos_id=eos)
    assert output.tokens == [eos]
    output = generate(branch, prompt, 17, sampling=SamplingConfig(temperature=1),
                      generator=torch.Generator().manual_seed(341), audit_cache=True)
    assert len(output.tokens) == 17


@pytest.mark.parametrize("prefill_output", [False, True])
@pytest.mark.parametrize("prefix", [1, 4, 8, 12])
@pytest.mark.parametrize("temperature", [0., 1.])
def test_timed_ar_prefix_matches_a_standalone_capped_call(prefill_output, prefix, temperature):
    branch = MaskedAttentionBranch(tiny())
    prompt = torch.tensor([[3, 7, 8]])
    options = dict(sampling=SamplingConfig(temperature), prefill_output=prefill_output)
    timed = generate_ar(branch, prompt, 8, prefix_tokens=prefix,
                         generator=torch.Generator().manual_seed(994), **options)
    full = generate_ar(branch, prompt, 8, generator=torch.Generator().manual_seed(994), **options)
    short = generate_ar(branch, prompt, min(8, prefix), generator=torch.Generator().manual_seed(994), **options)
    assert timed.tokens == full.tokens
    assert timed.tokens[:timed.prefix_tokens] == short.tokens
    assert timed.prefix_tokens == min(8, prefix)
    assert 0 < timed.prefix_seconds <= timed.seconds
    assert 0 <= timed.prefix_capture_seconds < timed.seconds
    assert timed.decode_forwards == full.decode_forwards
    if prefix >= 8:
        assert timed.prefix_seconds == timed.seconds and timed.prefix_capture_seconds == 0
    assert full.prefix_seconds is None and full.prefix_tokens == 0


def test_timed_ar_prefix_handles_eos_and_empty_output():
    branch = MaskedAttentionBranch(tiny())
    prompt = torch.tensor([[3, 7, 8]])
    eos = int(branch.model(prompt).logits[0, -1].argmax())
    output = generate_ar(branch, prompt, 8, prefix_tokens=4, eos_id=eos)
    assert output.tokens == [eos] and output.prefix_tokens == 1
    assert output.prefix_seconds == output.seconds and output.prefix_capture_seconds == 0
    empty = generate_ar(branch, prompt, 0, prefix_tokens=4)
    assert empty.prefix_tokens == 0 and empty.prefix_seconds is None


@pytest.mark.parametrize("prefix", [0, -1, 1.5, True])
def test_timed_ar_prefix_rejects_invalid_budgets(prefix):
    with pytest.raises(ValueError, match="prefix"):
        generate_ar(MaskedAttentionBranch(tiny()), torch.tensor([[3, 7]]), 8, prefix_tokens=prefix)


@pytest.mark.parametrize("temperature", [0., 1.])
@pytest.mark.parametrize("budget", [0, 1, 9])
def test_speculative_first_token_capture_preserves_sampling_and_forward_counts(temperature, budget):
    branch = MaskedAttentionBranch(tiny())
    prompt = torch.tensor([[3, 7, 8]])
    options = dict(sampling=SamplingConfig(temperature))
    timed = generate(branch, prompt, budget, prefix_tokens=1,
                     generator=torch.Generator().manual_seed(811), **options)
    ordinary = generate(branch, prompt, budget, generator=torch.Generator().manual_seed(811), **options)
    assert timed.tokens == ordinary.tokens and timed.decode_forwards == ordinary.decode_forwards
    assert timed.accepted_per_round == ordinary.accepted_per_round
    assert ordinary.prefix_seconds is None and ordinary.prefix_capture_seconds == 0
    if budget:
        assert timed.prefix_tokens == 1 and 0 < timed.prefix_seconds <= timed.seconds
        assert 0 <= timed.prefix_capture_seconds < timed.seconds
        ar = generate_ar(branch, prompt, 1, generator=torch.Generator().manual_seed(811), **options)
        assert timed.tokens[:1] == ar.tokens
    else:
        assert timed.prefix_tokens == 0 and timed.prefix_seconds is None


def test_speculative_first_token_capture_handles_eos():
    branch = MaskedAttentionBranch(tiny())
    prompt = torch.tensor([[3, 7, 8]])
    eos = int(branch.model(prompt).logits[0, -1].argmax())
    result = generate(branch, prompt, 9, prefix_tokens=1, eos_id=eos)
    assert result.tokens == [eos] and result.prefix_tokens == 1 and result.decode_forwards == 0
    assert 0 < result.prefix_seconds <= result.seconds


@pytest.mark.parametrize("prefix", [0, -1, 2, 1., True])
def test_speculative_prefix_capture_requires_one_integer_token(prefix):
    with pytest.raises(ValueError, match="prefix"):
        generate(MaskedAttentionBranch(tiny()), torch.tensor([[3, 7]]), 8, prefix_tokens=prefix)


def test_anchor_sampling_bounds_and_cache_prefix_checks():
    tokens = torch.zeros(2, 10, dtype=torch.long)
    anchors = sample_anchors(tokens, 4, 100, generator=torch.Generator().manual_seed(2))
    assert anchors.min() == 0 and anchors.max() == 6
    with pytest.raises(ValueError):
        anchor_layout(tokens, torch.tensor([[7], [0]]), 4, 1)
    with pytest.raises(ValueError):
        crop_cache(None, 1)
    assert cache_length(crop_cache(None, 0)) == 0


def public_fixture(directory, model):
    config = model.config.to_dict() | {"model_type": "qwen3", "hidden_act": "silu"}
    (directory / "config.json").write_text(json.dumps(config))
    state = model.state_dict()
    public = {source: state[own].clone().contiguous() for own, source in public_key_map(model.config).items()}
    save_file(public, directory / "model.safetensors")


def test_public_import_complete_mapping_and_frozen_parameters(tmp_path):
    model = tiny(attention_bias=True)
    public_fixture(tmp_path, model)
    loaded = load_public(tmp_path, expected_sha256=file_sha256(tmp_path / "model.safetensors"))
    assert all(not value.requires_grad for value in loaded.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    assert loaded.head.weight is loaded.embedding.weight
    assert loaded.frequencies.device.type == "cpu"
    assert "directory" not in loaded.source
    with pytest.raises(ValueError, match="SHA256"):
        load_public(tmp_path, expected_sha256="bad")


def test_checkpoint_drops_storage_metadata_and_preserves_content_binding(tmp_path):
    from blockspec.parallel.weights import source_identity

    model = tiny()
    model.source = {"directory": str(tmp_path), "model_id": "local-resource", "weight_sha256": "test-content"}
    path = tmp_path / "portable.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, weights_only=True)
    assert payload["source"] == {"weight_sha256": "test-content"}
    payload["source"]["directory"] = str(tmp_path)  # Legacy checkpoints may carry a location.
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    restored, _ = load_checkpoint(legacy)
    assert restored.source == source_identity(model.source)


def test_public_import_rejects_unmapped_tensor(tmp_path):
    model = tiny()
    public_fixture(tmp_path, model)
    state = {source: model.state_dict()[own].clone() for own, source in public_key_map(model.config).items()}
    state["extra.weight"] = torch.ones(1)
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="keys differ"):
        load_public(tmp_path)


def test_training_checkpoint_resumes_same_optimizer_step(tmp_path):
    model = tiny().train_draft_only()
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=0.001)
    tokens, anchors = torch.tensor([[3, 5, 8, 9, 6, 4]]), torch.tensor([[0, 2]])
    distillation_loss(model, tokens, anchors).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "dual.pt"
    save_checkpoint(path, model, optimizer=optimizer, step=1)
    loaded, saved = load_checkpoint(path)
    resumed = torch.optim.AdamW([value for value in loaded.parameters() if value.requires_grad], lr=0.001)
    resumed.load_state_dict(saved["optimizer"])
    assert saved["step"] == 1
    for instance, update in ((model, optimizer), (loaded, resumed)):
        distillation_loss(instance, tokens, anchors).backward()
        update.step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model)
