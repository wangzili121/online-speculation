"""Padded execution is equivalent and never leaks mutable workspaces to replay."""

import copy

import pytest
import torch

from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.model import Decoder, ModelConfig, PackedCache, trim_cache
from blockspec_ablation.online import Feedback, OnlineConfig, OnlineLearner
from blockspec_ablation.tree import CandidateTree, compact_tree_cache, generate_tree


def example(device="cpu"):
    torch.manual_seed(135)
    model = Decoder(ModelConfig(vocab_size=13, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device).train_adapters_only()
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.1)
    engine = FixedShapeExecutor(model, capacity=48, max_query=5, use_cuda_graph=device == "cuda")
    engine.prepare([(i, False, None) for i in range(1, 6)] +
                   [(i, True, c) for i in range(2, 5) for c in (None, 2)])
    return model, engine


def close_cache(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph hardware required"))])
@torch.no_grad()
def test_padded_execution_prefill_cache_tree_and_owned_outputs(device):
    model, engine = example(device)
    ids = torch.tensor([[1, 4, 2, 9, 8, 7]], device=device)
    _, past = engine(ids, return_cache=True)  # long prefill is eager
    _, reference_past = model(ids, return_cache=True)
    seed_noise = ids[:, :4]
    mask = torch.tensor([[False, True, True, True]], device=device)
    actual, cache, boundary = engine(seed_noise, cache=past, adapter_mask=mask,
                                      return_cache=True, capture_layer=2)
    expected, expected_cache, expected_boundary = model(seed_noise, cache=reference_past,
                                                        adapter_mask=mask, return_cache=True, capture_layer=2)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    close_cache(cache, expected_cache)
    torch.testing.assert_close(boundary.hidden, expected_boundary.hidden, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(boundary.allowed, expected_boundary.allowed)
    kept = (actual.clone(), cache.packed.clone(), boundary.hidden.clone(), boundary.allowed.clone())
    # Reuse the SAME signature with a different history length and adapter.
    for p in model.adapter_parameters():
        p.add_(.03)
    engine(seed_noise.flip(-1), cache=trim_cache(cache, 3), adapter_mask=mask,
           return_cache=True, capture_layer=2)
    torch.testing.assert_close((actual, cache.packed, boundary.hidden, boundary.allowed), kept, atol=0, rtol=0)
    # Stale workspace entries beyond a shorter valid prefix are masked out.
    shorter = trim_cache(past, 2)
    result = engine(seed_noise, cache=shorter, adapter_mask=mask)
    torch.testing.assert_close(result, model(seed_noise, cache=shorter, adapter_mask=mask), rtol=2e-5, atol=2e-6)
    # Tree siblings occupy equal positions but cannot see each other.
    tree = CandidateTree([3, 4, 7, 8, 9], [-1, 0, 0, 1, 2], [0, 1, 1, 2, 2],
                         [0.] * 5, [{4: 1, 7: 2}, {8: 3}, {9: 4}, {}, {}])
    tokens, positions, allowed = tree.layout(2, device=device)
    actual, verified = engine(tokens, positions=positions, allowed=allowed, cache=shorter, return_cache=True)
    expected, expected_cache = model(tokens, positions=positions, allowed=allowed, cache=shorter, return_cache=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    compact = compact_tree_cache(verified, 2, [0, 2, 4])
    assert isinstance(compact, PackedCache) and isinstance(trim_cache(compact, 3), PackedCache)
    close_cache(compact, compact_tree_cache(expected_cache, 2, [0, 2, 4]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph hardware required"))])
@pytest.mark.parametrize("adapted", [False, True])
@torch.no_grad()
def test_query_only_cache_packing_preserves_empty_short_and_full_prefixes(device, adapted):
    model, engine = example(device)
    ids = torch.tensor([[1, 4, 2]], device=device)
    mask = torch.tensor([[False, True, True]], device=device) if adapted else None
    capture = 2 if adapted else None
    slot = engine.slots[(3, adapted, capture)]
    c = model.config
    for length in (0, 2, engine.capacity):
        prefix_ids = (torch.arange(length, device=device) % c.vocab_size)[None]
        past = model(prefix_ids, return_cache=True)[1] if length else None
        output = engine(ids, cache=past, adapter_mask=mask, return_cache=True, capture_layer=capture)
        assert slot.new_cache.shape == (c.num_hidden_layers, 2, 1, c.num_key_value_heads, 3, c.head_dim)
        expected = model(ids, cache=past, adapter_mask=mask, return_cache=True, capture_layer=capture)
        torch.testing.assert_close(output[0], expected[0], rtol=2e-5, atol=2e-6)
        close_cache(output[1], expected[1])
        assert output[1].packed.shape[-2] == length + ids.shape[1]
        if length:
            for actual, original in zip(output[1], past):
                torch.testing.assert_close(tuple(kv[..., :length, :] for kv in actual), original, atol=0, rtol=0)
        snapshot = output[1].packed.clone()
        engine(ids.flip(-1), adapter_mask=mask, return_cache=True, capture_layer=capture)
        torch.testing.assert_close(output[1].packed, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
@pytest.mark.parametrize("last_layers", [None, 1])
def test_padded_online_is_real_continuation_with_immutable_feedback(generate, last_layers):
    model, engine = example()
    oracle = copy.deepcopy(model)
    options = dict(stride=1, replay_blocks=2, train_last_layers=last_layers, learning_rate=.001)
    learner, reference = OnlineLearner(model, OnlineConfig(**options)), OnlineLearner(oracle, OnlineConfig(**options))
    prompt = torch.tensor([[1, 3, 5]])
    kwargs = {"block_size": 4}
    if generate is generate_tree:
        kwargs.update(top_k=2, prefix_budget=5)
    expected = generate(oracle, prompt, 20, learner=reference,
                        generator=torch.Generator().manual_seed(27), **kwargs)
    actual = generate(model, prompt, 20, learner=learner, executor=engine,
                      generator=torch.Generator().manual_seed(27), **kwargs)
    assert actual.tokens == expected.tokens == generate_ar(model, prompt, 20, executor=engine).tokens
    assert actual.updates == expected.updates > 0
    for p, q in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(p, q, rtol=3e-4, atol=2e-6)
    assert not learner.replay


def test_feedback_survives_slot_reuse_before_a_later_backward():
    model, engine = example()
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1, stride=100))
    inputs = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, True, True]])
    with torch.no_grad():
        _, past = engine(inputs, return_cache=True)
        logits, _, boundary = engine(inputs, cache=past, adapter_mask=mask,
                                     return_cache=True, capture_layer=2)
        learner.observe(Feedback(inputs, past, logits[0, 1:], 3, boundary), may_update=False)
        expected = model.forward_suffix(learner.replay[0].boundary, cache=learner.replay[0].cache).clone()
        engine(inputs.flip(-1), cache=past, adapter_mask=mask, return_cache=True, capture_layer=2)
        engine(inputs.flip(-1), return_cache=True)
        actual = model.forward_suffix(learner.replay[0].boundary, cache=learner.replay[0].cache)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert learner.update()["version"] == 1


def test_executor_contracts_and_storage_invalidation():
    model, engine = example()
    ids = torch.tensor([[1, 2]])
    with pytest.raises(RuntimeError, match="inference-only"):
        engine(ids)
    with torch.no_grad(), pytest.raises(ValueError, match="unprepared"):
        engine(ids, adapter_mask=torch.ones_like(ids), return_cache=True, capture_layer=1)
    with torch.no_grad(), pytest.raises(ValueError, match="layout"):
        engine(ids, positions=torch.tensor([[1]]))
    with torch.no_grad(), pytest.raises(ValueError, match="layout"):
        engine(ids, adapter_mask=torch.full_like(ids, .5, dtype=torch.float32))
    with torch.no_grad(), pytest.raises(ValueError, match="batch-one"):
        engine(ids.float())
    with pytest.raises(ValueError, match="share a model"):
        engine.validate(copy.deepcopy(model))
    with pytest.raises(ValueError, match="CUDA model"):
        FixedShapeExecutor(model, capacity=4, max_query=2)
    p = model.adapter_parameters()[0]
    p.data = p.data.clone()
    with pytest.raises(RuntimeError, match="storage changed"):
        engine.validate(model)


def test_mutated_rotary_buffer_invalidates_captured_execution():
    model, engine = example()
    model.model.layers[0].self_attn.frequencies.add_(.001)
    with torch.no_grad(), pytest.raises(RuntimeError, match="buffers changed"):
        engine(torch.tensor([[1, 2]]))


def test_all_setup_cost_is_not_hidden_in_warm_throughput():
    from blockspec_ablation.benchmark import aggregate
    row = {"tokens": 30, "seconds": 2., "decode_forwards": 10, "rounds": 5,
           "accepted": 10, "proposed": 12, "updates": 1, "update_seconds": .1, "feedback_blocks": 5,
           "fully_covered_rounds": 2, "coverage_skips": 0}
    result = aggregate([row], setup_seconds=.5, engine_setup_seconds=1.5)
    assert result["tps"] == 15 and result["tps_including_learner_setup"] == 12
    assert result["tps_including_all_setup"] == 7.5


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="BF16 CUDA graphs required"))])
@pytest.mark.parametrize("master_weights", [False, True])
@torch.no_grad()
def test_bf16_graphs_preserve_cache_snapshots_and_observe_adapter_updates(device, master_weights):
    torch.manual_seed(627)
    model = Decoder(ModelConfig(vocab_size=17, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                head_dim=8, adapter_rank=4)).to(device=device, dtype=torch.bfloat16)
    if master_weights:
        model.train_adapters_only()
    for parameter in model.adapter_parameters():
        parameter.normal_(std=.06)
    assert all(p.dtype == (torch.float32 if master_weights else torch.bfloat16) for p in model.adapter_parameters())
    engine = FixedShapeExecutor(model, capacity=32, max_query=5, use_cuda_graph=device == "cuda")
    engine.prepare([(n, False, None) for n in range(1, 6)] + [(4, True, None), (4, True, 1)])
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], device=device)
    past = model(ids, return_cache=True)[1]
    inputs = ids[:, :4]
    mask = torch.tensor([[False, True, True, True]], device=device)

    def check(result, expected):
        torch.testing.assert_close(result[0], expected[0], rtol=.03, atol=.004)
        tv = (result[0].float().softmax(-1) - expected[0].float().softmax(-1)).abs().sum(-1) / 2
        assert float(tv.max()) < .003
        torch.testing.assert_close(tuple(result[1]), tuple(expected[1]), rtol=.03, atol=.004)

    for prefix in (past, trim_cache(past, 2), None):
        options = dict(cache=prefix, adapter_mask=mask, return_cache=True, capture_layer=1)
        result = engine(inputs, **options)
        check(result, model(inputs, **options))
        assert result[1].packed.dtype == torch.bfloat16
        snapshot = (result[0].clone(), result[1].packed.clone(), result[2].hidden.clone())
        for parameter in model.adapter_parameters():
            parameter.add_(.01)
        updated = engine(inputs, **options)
        check(updated, model(inputs, **options))
        assert torch.any(updated[0] != result[0])
        torch.testing.assert_close((result[0], result[1].packed, result[2].hidden), snapshot, atol=0, rtol=0)
    tree = CandidateTree([3, 4, 7, 8, 9], [-1, 0, 0, 1, 2], [0, 1, 1, 2, 2],
                         [0.] * 5, [{4: 1, 7: 2}, {8: 3}, {9: 4}, {}, {}])
    tokens, positions, allowed = tree.layout(7, device=device)
    result = engine(tokens, positions=positions, allowed=allowed, cache=past, return_cache=True)
    expected = model(tokens, positions=positions, allowed=allowed, cache=past, return_cache=True)
    check(result, expected)
    torch.testing.assert_close(tuple(compact_tree_cache(result[1], 7, [0, 2, 4])),
                               tuple(compact_tree_cache(expected[1], 7, [0, 2, 4])), rtol=.03, atol=.004)


def test_bf16_executor_rejects_mixed_base_and_foreign_cache_precision():
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=1, adapter_rank=2)).to(torch.bfloat16).train_adapters_only()
    engine = FixedShapeExecutor(model, capacity=16, max_query=3, use_cuda_graph=False)
    engine.prepare([(1, False, None)])
    ids = torch.tensor([[0, 1]])
    with torch.no_grad():
        past = model(ids, return_cache=True)[1]
        converted = tuple((k.float(), v.float()) for k, v in past)
        with pytest.raises(ValueError, match="prefix cache"):
            engine(ids[:, :1], cache=converted)
    model.model.norm.weight.data = model.model.norm.weight.data.float()
    with pytest.raises(ValueError, match="base dtype"):
        FixedShapeExecutor(model, capacity=16, max_query=3, use_cuda_graph=False)


def test_explicit_base_cast_preserves_adapter_masters_rotary_and_checkpoint_guard(tmp_path):
    from blockspec_ablation.checkpoint import adapter_state, base_fingerprint, load_checkpoint, save_checkpoint
    from blockspec_ablation.model import is_adapter

    torch.manual_seed(822)
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=1, adapter_rank=2)).train_adapters_only()
    checkpoint = tmp_path / "adapter.pt"
    save_checkpoint(checkpoint, model, adapter_only=True)
    initial, source = adapter_state(model), base_fingerprint(model)
    buffers = {n: p.clone() for n, p in model.named_buffers()}
    model, _ = load_checkpoint(checkpoint, model=model)
    assert model.set_base_dtype(torch.bfloat16) is model
    assert base_fingerprint(model) != source
    for name, parameter in model.named_parameters():
        assert parameter.dtype == (torch.float32 if is_adapter(name) else torch.bfloat16)
    assert all(torch.equal(value, adapter_state(model)[name]) for name, value in initial.items())
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    with pytest.raises(ValueError, match="different base"):
        load_checkpoint(checkpoint, model=model)
    with pytest.raises(ValueError, match="execution dtype"):
        model.set_base_dtype(torch.int8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 CUDA graphs required")
@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
def test_bf16_inference_graph_with_fp32_online_masters(generate):
    from blockspec_ablation.checkpoint import adapter_state, base_fingerprint
    from blockspec.sampling import SamplingConfig

    torch.manual_seed(28)
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                head_dim=8, adapter_rank=4)).cuda().to(torch.bfloat16).train_adapters_only()
    initial, frozen = adapter_state(model), base_fingerprint(model)
    engine = FixedShapeExecutor(model, capacity=32, max_query=5)
    engine.prepare([(n, False, None) for n in range(1, 6)] + [(n, True, 1) for n in (2, 3)])
    learner = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=1, train_last_layers=1))
    options = {"block_size": 3, "sampling": SamplingConfig(temperature=1), "executor": engine, "learner": learner}
    if generate is generate_tree:
        options.update(top_k=2, prefix_budget=5)
    for prompt in ([[0, 1]], [[0, 2]]):
        output = generate(model, torch.tensor(prompt, device="cuda"), 12,
                          generator=torch.Generator(device="cuda").manual_seed(71), **options)
        assert output.updates > 0
        assert len(output.tokens) == 12
    after = adapter_state(model)
    assert base_fingerprint(model) == frozen
    assert all(p.dtype == torch.float32 for p in model.adapter_parameters())
    assert any(not torch.equal(value, after[name]) for name, value in initial.items())
