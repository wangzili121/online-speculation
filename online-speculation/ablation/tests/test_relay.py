"""Conditional proposals, predictable stopping, gradients, and frozen-backbone replay."""

from itertools import product
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from blockspec_ablation.decoding import generate_ar
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.relay import (RelayConfig, RelayFeedback, RelayHead, RelayLearner, generate_relay,
                             load_relay, relay_candidates, relay_loss, save_relay)
from blockspec.sampling import SamplingConfig, probabilities, verify_linear


def tiny():
    torch.manual_seed(21)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).eval().requires_grad_(False)
    head = RelayHead(RelayConfig(7, 16, 3))
    return model, head


@pytest.mark.parametrize("sampling", [SamplingConfig(1.), SamplingConfig(.7, 3, .8), SamplingConfig()])
def test_zero_initialization_and_saved_conditional_rows(sampling):
    _, head = tiny()
    logits, hidden = torch.randn(4, 7), torch.randn(4, 16)
    draft = relay_candidates(head, logits, hidden, sampling=sampling)
    torch.testing.assert_close(draft.q, probabilities(logits[1:], sampling), rtol=0, atol=0)
    with torch.no_grad():
        head.projection.weight.normal_(0, .5)
    draft = relay_candidates(head, logits, hidden, sampling=sampling)
    expected = probabilities(head(logits[1:], draft.tokens[:-1]), sampling)
    torch.testing.assert_close(draft.q, expected)
    assert (draft.q.gather(1, draft.tokens[1:, None]) > 0).all()


def test_admission_occurs_before_current_draw_and_shrinks():
    _, head = tiny()
    logits, hidden = torch.randn(4, 7), torch.randn(4, 16)
    # Initial conditional confidence = 1/2: admit once (.5), then stop (.25 < .3).
    with patch("blockspec_ablation.relay.draw", wraps=__import__("blockspec.sampling", fromlist=["draw"]).draw) as draw:
        draft = relay_candidates(head, logits, hidden, sampling=SamplingConfig(1.), threshold=.3)
    assert len(draft.tokens) == 2 and draw.call_count == 1
    draft = relay_candidates(head, logits, hidden, threshold=1.)
    assert len(draft.tokens) == 1 and draft.q.shape == (0, 7)


@pytest.mark.parametrize("threshold", [0., .3, 1.])
@pytest.mark.parametrize("budget", [0, 1, 2, 3, 17])
def test_greedy_kv_budget_and_online_exactness(threshold, budget):
    model, head = tiny()
    with torch.no_grad():
        head.projection.weight.normal_(0, 1.)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    learner = RelayLearner(head, interval=1)
    prompt = torch.tensor([[1, 4, 3]])
    expected = generate_ar(model, prompt, budget)
    result = generate_relay(model, head, prompt, budget, block_size=4, threshold=threshold, learner=learner)
    assert result.tokens == expected.tokens
    assert not learner.pending
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, frozen[key], atol=0, rtol=0)
    if threshold == 1.:
        assert result.proposed == 0
        assert result.verified_tokens == budget // 2


def test_terminal_capture_and_owned_executor_features():
    model, head = tiny()
    engine = FixedShapeExecutor(model, capacity=30, max_query=4, use_cuda_graph=False)
    engine.prepare([(n, False, None) for n in range(1, 5)] + [(n, True, 2) for n in range(2, 5)])
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[False, True, True]])
    with torch.no_grad():
        actual = engine(inputs, adapter_mask=mask, return_cache=True, capture_layer=2)
        expected = model(inputs, adapter_mask=mask, return_cache=True, capture_layer=2)
        saved = actual[2].hidden.clone()
        torch.testing.assert_close(model._project(actual[2].hidden, None), actual[0])
        torch.testing.assert_close(actual[0], expected[0])
        engine(inputs.flip(-1), adapter_mask=mask, return_cache=True, capture_layer=2)
        torch.testing.assert_close(actual[2].hidden, saved, atol=0, rtol=0)
        result = generate_relay(model, head, inputs, 19, block_size=4, executor=engine, threshold=.3)
    assert result.tokens == generate_ar(model, inputs, 19).tokens


def test_head_gradients_teacher_detach_and_learning_transition():
    torch.manual_seed(9)
    head = RelayHead(RelayConfig(2, 3, 2)).double()
    previous = torch.tensor([0, 1] * 16)
    teacher = torch.tensor([[4., -4.], [-4., 4.]], dtype=torch.float64).repeat(16, 1).requires_grad_()
    logits = torch.zeros(32, 2, dtype=torch.float64, requires_grad=True)
    feedback = RelayFeedback(logits, torch.ones(32, 3, dtype=torch.float64), previous, teacher)
    initial = float(relay_loss(head, feedback)[1])
    learner = RelayLearner(head, lr=.05, interval=1)
    for _ in range(100):
        learner.observe(feedback)
    final = float(relay_loss(head, feedback)[1])
    assert initial > .49 and final < .02
    assert teacher.grad is None and logits.grad is None
    assert all(torch.isfinite(p).all() for p in head.parameters())
    assert learner.examples == 3200


def test_head_loss_numerical_gradient():
    _, head = tiny()
    head = head.double()
    with torch.no_grad():
        head.projection.weight.normal_(0, .02)
    previous = torch.tensor([0, 2, 4])
    feedback = RelayFeedback(torch.randn(3, 7).double(), torch.randn(3, 16).double(), previous,
                             torch.randn(3, 7).double())
    loss, _, _ = relay_loss(head, feedback, confidence_weight=0.)
    loss.backward()
    parameter = head.projection.weight
    expected = parameter.grad[0, 0].item()
    epsilon = 1e-5
    with torch.no_grad():
        original = parameter[0, 0].item()
        parameter[0, 0] = original + epsilon
        plus = relay_loss(head, feedback, confidence_weight=0.)[0].item()
        parameter[0, 0] = original - epsilon
        minus = relay_loss(head, feedback, confidence_weight=0.)[0].item()
        parameter[0, 0] = original
    assert abs(expected - (plus - minus) / (2 * epsilon)) < 1e-8


def test_base_projection_preserves_initial_law_rng_and_similar_tokens():
    model, head = tiny()
    original = model.model.embed_tokens.weight.detach().clone()
    source = original.clone()
    source[1] = source[0]
    rng_state = torch.get_rng_state().clone()
    head.initialize_from_embedding(source, seed=51)
    torch.testing.assert_close(torch.get_rng_state(), rng_state)
    torch.testing.assert_close(model.model.embed_tokens.weight, original, atol=0, rtol=0)
    torch.testing.assert_close(head.embedding.weight[0], head.embedding.weight[1], atol=0, rtol=0)
    torch.testing.assert_close(head.embedding.weight.norm(dim=-1), torch.ones(7))
    logits = torch.randn(3, 7)
    torch.testing.assert_close(head(logits, torch.tensor([1, 2, 4])), logits, atol=0, rtol=0)
    saved = head.embedding.weight.clone()
    head.initialize_from_embedding(source, seed=51)
    torch.testing.assert_close(head.embedding.weight, saved, atol=0, rtol=0)
    with torch.no_grad():
        head.projection.weight.fill_(1)
    with pytest.raises(ValueError, match="zero transition"):
        head.initialize_from_embedding(source)


def test_separate_confidence_rate_controls_high_dimensional_adam_step():
    # A zero-initialized dense head on 1536 RMS-normalized features can move
    # by roughly lr * 1536 on its first Adam step. Check the two rates directly.
    heads = [RelayHead(RelayConfig(2, 1536, 4)) for _ in range(2)]
    heads[1].load_state_dict(heads[0].state_dict())
    feedback = RelayFeedback(torch.zeros(1, 2), torch.ones(1, 1536), torch.tensor([0]),
                             torch.tensor([[.7, .3]]).log())
    losses = []
    for head, rate in zip(heads, [None, .0001]):
        head.embedding.requires_grad_(False)
        head.projection.requires_grad_(False)
        learner = RelayLearner(head, lr=.003, interval=1, confidence_lr=rate)
        before = float(relay_loss(head, feedback)[2])
        learner.observe(feedback)
        losses.append((before, float(relay_loss(head, feedback)[2])))
        if rate:
            assert [group["lr"] for group in learner.optimizer.param_groups] == [.003, .0001]
    assert losses[0][1] > losses[0][0]
    assert losses[1][1] < losses[1][0]
    for rate in [0, -1, float("nan")]:
        with pytest.raises(ValueError, match="confidence learning rate"):
            RelayLearner(heads[0], confidence_lr=rate)


@pytest.mark.parametrize("scheduled", [False, True])
def test_exact_two_token_joint_with_conditional_proposal_and_stopping(scheduled):
    # Enumerate candidate draws, accept/reject branches, and correction draws.
    # If verification stops at one output, the next AR draw completes the pair.
    p0 = torch.tensor([.7, .3], dtype=torch.float64)
    p1 = torch.tensor([[.2, .8], [.6, .4]], dtype=torch.float64)
    q0 = torch.tensor([.4, .6], dtype=torch.float64)
    q1 = torch.tensor([[.85, .15], [.1, .9]], dtype=torch.float64)
    observed = torch.zeros(2, 2, dtype=torch.float64)
    acceptance_mass = torch.zeros(3, dtype=torch.float64)
    for x in range(2):
        # Admission of position two can depend on x; position one's inclusion is fixed.
        admitted = 1 if scheduled and x == 1 else 2
        for y in range(2) if admitted == 2 else [0]:
            proposals = torch.tensor([x, y][:admitted])
            q = torch.stack([q0, q1[x]])[:admitted]
            p = torch.stack([p0, p1[x], p0])[:admitted + 1]
            mass = q0[x] * (q1[x, y] if admitted == 2 else 1.)
            alpha = torch.minimum(torch.ones(admitted, dtype=torch.float64),
                                  p[:admitted].gather(1, proposals[:, None]).flatten()
                                  / q.gather(1, proposals[:, None]).flatten())
            for flags in product([False, True], repeat=admitted):
                weight = mass.clone()
                uniforms = torch.empty(admitted, dtype=torch.float64)
                for i, accept in enumerate(flags):
                    weight *= alpha[i] if accept else 1 - alpha[i]
                    uniforms[i] = alpha[i] / 2 if accept else (1 + alpha[i]) / 2
                if weight == 0:
                    continue
                for z in range(2):
                    drawn = []
                    def categorical(distribution, generator=None, drawn=drawn, z=z):
                        drawn.append(distribution[z])
                        return torch.tensor(z)
                    with patch("blockspec.sampling.draw", side_effect=categorical):
                        result = verify_linear(proposals, q, p, acceptance_uniforms=uniforms)
                    branch_mass = weight * drawn[0]
                    acceptance_mass[result.accepted] += branch_mass
                    if len(result.tokens) >= 2:
                        observed[result.tokens[0], result.tokens[1]] += branch_mass
                    else:
                        observed[result.tokens[0]] += branch_mass * p1[result.tokens[0]]
    torch.testing.assert_close(observed, p0[:, None] * p1, rtol=0, atol=1e-14)
    first = torch.minimum(p0, q0)
    second_admitted = torch.tensor([1., 0. if scheduled else 1.], dtype=torch.float64)
    survival = first.sum() + (first * second_admitted * torch.minimum(p1, q1).sum(-1)).sum()
    actual_length = acceptance_mass @ torch.arange(3, dtype=torch.float64)
    torch.testing.assert_close(actual_length, survival, rtol=0, atol=1e-14)


def test_head_checkpoint_binding_and_exclusive_write(tmp_path):
    _, head = tiny()
    binding = {"base": "a" * 64, "adapter": "b" * 64}
    path = tmp_path / "relay.pt"
    save_relay(path, head, binding=binding, metadata={"test": True, "torch": torch.__version__})
    restored, metadata = load_relay(path, binding=binding)
    assert metadata == {"test": True, "torch": str(torch.__version__)}
    assert type(torch.load(path, weights_only=True)["metadata"]["torch"]) is str
    for a, b in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        save_relay(path, head, binding=binding)
    with pytest.raises(ValueError, match="binding"):
        load_relay(path, binding={**binding, "adapter": "c" * 64})


def test_paired_request_bootstrap_and_portable_file_sha(tmp_path):
    spec = importlib.util.spec_from_file_location("prefix_relay", Path(__file__).resolve().parents[1]
                                                 / "scripts" / "prefix_relay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = {"a": [{"seconds": t, "tokens": 10} for t in [1., 3., 2., 4.]],
            "b": [{"seconds": t / 2, "tokens": 10} for t in [1., 3., 2., 4.]]}
    interval = module.paired_bootstrap(rows, "a", "b", 2, resamples=200)
    assert interval["speed_ratio_95_interval"] == [2., 2.]
    assert interval == module.paired_bootstrap(rows, "a", "b", 2, resamples=200)
    path = tmp_path / "empty"
    path.touch()
    assert module.sha(path) == __import__("hashlib").sha256(b"").hexdigest()


def test_all_evaluated_heads_share_training_and_block_contract():
    spec = importlib.util.spec_from_file_location("prefix_relay", Path(__file__).resolve().parents[1]
                                                 / "scripts" / "prefix_relay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = {"train_sha256": "a" * 64, "config": {"block_size": 4}}
    module.validate_head_metadata(metadata, train_sha256="a" * 64, block_size=4)
    with pytest.raises(ValueError, match="training-file SHA"):
        module.validate_head_metadata(metadata, train_sha256="b" * 64, block_size=4)
    with pytest.raises(ValueError, match="block size"):
        module.validate_head_metadata(metadata, train_sha256="a" * 64, block_size=8)
    with pytest.raises(ValueError, match="block size"):
        module.validate_head_metadata({"train_sha256": "a" * 64}, train_sha256="a" * 64, block_size=4)
    contract = {"backbone": "hf_sdpa", "base_dtype": "bfloat16", "sampling": {"top_k": 50},
                "noise": {"low": 1, "high": 64256}}
    checked = {**metadata, "serving_contract": contract}
    module.validate_head_metadata(checked, train_sha256="a" * 64, block_size=4, serving_contract=contract)
    for changed in ({**contract, "backbone": "independent_graph"}, {**contract, "base_dtype": "float32"},
                    {**contract, "sampling": {"top_k": 0}}, {**contract, "noise": {"low": 0, "high": 64256}}):
        with pytest.raises(ValueError, match="serving contract"):
            module.validate_head_metadata(checked, train_sha256="a" * 64, block_size=4, serving_contract=changed)
    with pytest.raises(ValueError, match="serving contract"):
        module.validate_head_metadata(metadata, train_sha256="a" * 64, block_size=4, serving_contract=contract)


def test_published_execution_weights_are_checked_after_head_updates():
    spec = importlib.util.spec_from_file_location("prefix_relay", Path(__file__).resolve().parents[1]
                                                 / "scripts" / "prefix_relay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, _ = tiny()
    model.requires_grad_(False)
    frozen = {"base": module.base_fingerprint(model), "adapter": module.adapter_fingerprint(model)}
    assert module.assert_frozen(model, frozen) == {"frozen_base_unchanged": True, "frozen_adapter_unchanged": True}
    with torch.no_grad():
        next(p for name, p in model.named_parameters() if name.endswith("lora_B")).add_(1.)
    with pytest.raises(RuntimeError, match="frozen execution tensors"):
        module.assert_frozen(model, frozen)


def test_counterfactual_audit_keeps_the_reference_stream_fixed():
    spec = importlib.util.spec_from_file_location("prefix_relay", Path(__file__).resolve().parents[1]
                                                 / "scripts" / "prefix_relay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, reference = tiny()
    evaluated = RelayHead(reference.config)
    with torch.no_grad():
        evaluated.projection.weight.normal_(0, .5)
    audit = module.HeadAudit(reference, SamplingConfig(1.), evaluated_head=evaluated)
    prompt = torch.tensor([[1, 2, 3]])
    actual = generate_relay(model, reference, prompt, 21, block_size=4, sampling=SamplingConfig(1.), learner=audit,
                             generator=torch.Generator().manual_seed(43))
    expected = generate_relay(model, reference, prompt, 21, block_size=4, sampling=SamplingConfig(1.),
                               generator=torch.Generator().manual_seed(43))
    assert actual.tokens == expected.tokens
    assert audit.totals.shape == (6,) and audit.totals[3] > 0
    torch.testing.assert_close(audit.totals[0], audit.totals[4], atol=0, rtol=0)
    assert audit.totals[1] != audit.totals[4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph integration")
def test_cuda_graph_terminal_capture_online():
    model, head = tiny()
    model, head = model.cuda(), head.cuda()
    engine = FixedShapeExecutor(model, capacity=32, max_query=4)
    engine.prepare([(n, False, None) for n in range(1, 5)] + [(n, True, 2) for n in range(2, 5)])
    prompt = torch.tensor([[1, 2, 5]], device="cuda")
    learner = RelayLearner(head, interval=1)
    result = generate_relay(model, head, prompt, 21, block_size=4, executor=engine, learner=learner)
    assert result.tokens == generate_ar(model, prompt, 21, executor=engine).tokens
    assert result.updates > 0 and result.update_seconds > 0
