"""Coverage gates preserve update state and retain full soft replay on misses."""

import copy
from dataclasses import replace

import pytest
import torch

from blockspec_ablation.checkpoint import adapter_state, base_fingerprint
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.online import Feedback, OnlineConfig, OnlineLearner
from blockspec_ablation.tree import generate_tree


def example():
    torch.manual_seed(918)
    return Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2,
                                num_key_value_heads=1, head_dim=8, adapter_rank=2)).double()


def feedback(learner, covered, teacher):
    inputs = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[False, True, True, True]])
    with torch.no_grad():
        output = learner.model(inputs, adapter_mask=mask, return_cache=True, capture_layer=learner.capture_layer)
    boundary = output[2] if learner.capture_layer is not None else None
    return Feedback(inputs, None, teacher, 3, boundary, covered)


@pytest.mark.parametrize("last_layers", [None, 1])
def test_covered_window_holds_parameters_gradients_and_adam_then_mixed_window_matches_reference(last_layers):
    model = example()
    config = OnlineConfig(stride=2, replay_blocks=2, loss="forward_kl", train_last_layers=last_layers,
                          update_policy="coverage")
    learner = OnlineLearner(model, config)
    reference = OnlineLearner(copy.deepcopy(model), replace(config, update_policy="periodic"))
    frozen = base_fingerprint(model)
    # Three windows: mixed (train), covered (hold), mixed (train all six positions).
    for t, covered in enumerate((False, True, True, True, True, False), start=1):
        teacher = torch.randn(3, 7, dtype=torch.float64)
        actual = learner.observe(feedback(learner, covered, teacher))
        expected = reference.observe(feedback(reference, covered, teacher), may_update=t != 4)
        if t == 2:
            parameters = adapter_state(model)
            gradients = [p.grad.clone() for p in learner.parameters]
            state = copy.deepcopy(learner.optimizer.state_dict())
        if t == 4:
            assert actual is None and learner.coverage_skips == 1 and learner.version == learner.updates == 1
            torch.testing.assert_close(adapter_state(model), parameters, atol=0, rtol=0)
            torch.testing.assert_close([p.grad for p in learner.parameters], gradients, atol=0, rtol=0)
            torch.testing.assert_close(learner.optimizer.state_dict(), state, atol=0, rtol=0)
        if t in (2, 6):
            assert actual["positions"] == expected["positions"] == 6
            torch.testing.assert_close(adapter_state(model), adapter_state(reference.model), atol=0, rtol=0)
            torch.testing.assert_close(learner.optimizer.state_dict(), reference.optimizer.state_dict(), atol=0, rtol=0)
    assert learner.rounds == reference.rounds == 6
    assert learner.feedback_blocks == 6 and learner.updates == 2 and learner.coverage_skips == 1
    assert base_fingerprint(model) == frozen


@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph hardware required"))])
def test_fully_covered_decoder_skips_learning_and_preserves_request_clock(generate, device):
    model = example().to(device=device, dtype=torch.float32 if device == "cuda" else torch.float64)
    with torch.no_grad():
        model.lm_head.weight.zero_()
    initial = adapter_state(model)
    learner = OnlineLearner(model, OnlineConfig(stride=2, replay_blocks=1, update_policy="coverage"))
    engine = None
    if device == "cuda":
        engine = FixedShapeExecutor(model, capacity=64, max_query=8)
        engine.prepare([(i, False, None) for i in range(1, 9)] + [(2, True, None)])
    options = {"block_size": 2, "executor": engine}
    if generate is generate_tree:
        options.update(top_k=7, prefix_budget=8)
    for request in range(2):
        prompt = torch.tensor([[0, 1, 2]], device=device)
        result = generate(model, prompt, 18, learner=learner, **options)
        assert result.tokens == generate_ar(model, prompt, 18).tokens
        assert result.rounds == result.fully_covered_rounds == 6
        assert result.coverage_skips == 2 and result.updates == 0
        assert learner.rounds == 6 * (request + 1) and learner.coverage_skips == 2 * (request + 1)
        assert learner.version == 0 and learner.optimizer.state_dict()["state"] == {}
        assert not learner.replay
    torch.testing.assert_close(adapter_state(model), initial, atol=0, rtol=0)


@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
def test_root_eos_leaves_coverage_policy_clock_and_skip_count_unchanged(generate):
    model = example()
    prompt = torch.tensor([[0, 1, 2]])
    eos = generate_ar(model, prompt, 1).tokens[0]
    learner = OnlineLearner(model, OnlineConfig(stride=2, update_policy="coverage"))
    result = generate(model, prompt, 18, learner=learner, eos_id=eos)
    assert result.tokens == [eos]
    assert result.fully_covered_rounds == result.coverage_skips == result.updates == 0
    assert learner.rounds == learner.version == 0


def test_coverage_policy_metadata_and_configuration_validation():
    with pytest.raises(ValueError, match="update policy"):
        OnlineConfig(update_policy="unknown")
    learner = OnlineLearner(example(), OnlineConfig(update_policy="coverage"))
    item = feedback(learner, True, torch.randn(3, 7, dtype=torch.float64))
    assert item.detached().fully_covered is True
    with pytest.raises(ValueError, match="boolean coverage"):
        learner.observe(replace(item, fully_covered=1))
    with pytest.raises(ValueError, match="every noisy target"):
        learner.observe(replace(item, valid=1, teacher_logits=item.teacher_logits[:1]))
