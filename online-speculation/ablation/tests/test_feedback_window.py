"""Replay windows preserve every scheduled update across request boundaries."""

import copy
from dataclasses import replace

import pytest
import torch

from blockspec_ablation.checkpoint import adapter_state, base_fingerprint
from blockspec_ablation.decoding import generate_ar, generate_speculative
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.online import Feedback, OnlineConfig, OnlineLearner
from blockspec.sampling import SamplingConfig
from blockspec_ablation.tree import generate_tree


class RecordedLearner(OnlineLearner):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.history = []

    def update(self):
        samples = [(f.inputs.clone(), f.teacher_logits.clone(), f.valid, f.fully_covered) for f in self.replay]
        result = super().update()
        if result is not None:
            self.history.append((self.rounds, samples, [p.grad.clone() for p in self.parameters],
                                 [p.detach().clone() for p in self.parameters],
                                 copy.deepcopy(self.optimizer.state_dict())))
        return result


def model_example(device="cpu"):
    torch.manual_seed(137)
    model = Decoder(ModelConfig(vocab_size=13, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device)
    if device == "cpu":
        model.double()
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.1)
    return model


@pytest.mark.parametrize("stride,replay", [(1, 1), (4, 1), (4, 3), (4, 4), (3, 5)])
@pytest.mark.parametrize("last_layers", [None, 1])
@pytest.mark.parametrize("loss", ["l1", "forward_kl"])
@pytest.mark.parametrize("update_policy", ["periodic", "coverage"])
def test_window_matches_every_full_collection_update_and_optimizer_state(stride, replay, last_layers, loss,
                                                                         update_policy):
    model = model_example()
    config = OnlineConfig(stride=stride, replay_blocks=replay, train_last_layers=last_layers,
                          learning_rate=.001, loss=loss, update_policy=update_policy)
    learner = RecordedLearner(model, config)
    reference = RecordedLearner(copy.deepcopy(model), replace(config, feedback_execution="all"))
    frozen = base_fingerprint(model)
    inputs = torch.tensor([[0, 2, 4, 6]])
    mask = torch.tensor([[False, True, True, True]])
    collected = 0
    for t in range(1, 34):
        covered = bool(t // 8 % 2)
        valid = 3 if covered else 1 + t % 3
        teacher = torch.randn(valid, 13, dtype=torch.float64)
        terminal = t in (2, 7, 8, 13, 16, 27, 33)
        for target in (learner, reference):
            expected_collection = target is reference or (stride - (t - 1) % stride <= replay)
            assert target.needs_decoder_feedback == expected_collection
            if target.needs_decoder_feedback:
                with torch.no_grad():
                    output = target.model(inputs, adapter_mask=mask, return_cache=True,
                                           capture_layer=target.capture_layer)
                boundary = output[2] if last_layers else None
                target.observe(Feedback(inputs, None, teacher, valid, boundary, covered), may_update=not terminal)
                collected += target is learner
            else:
                target._skip_decoder_feedback(valid)
            if terminal:
                target.clear_replay()
        assert learner.rounds == reference.rounds == t
        assert learner.version == reference.version
        assert learner.coverage_skips == reference.coverage_skips
    torch.testing.assert_close(learner.history, reference.history, atol=1e-14, rtol=1e-14)
    assert learner.feedback_blocks == collected
    assert reference.feedback_blocks == 33
    assert learner.feedback_blocks <= reference.feedback_blocks
    assert base_fingerprint(model) == base_fingerprint(reference.model) == frozen
    assert (learner.coverage_skips > 0) == (update_policy == "coverage")


def prepared(model, last_layers, device):
    if device == "cpu":
        return None
    engine = FixedShapeExecutor(model, capacity=48, max_query=5)
    capture = None if last_layers is None else 1
    engine.prepare([(i, False, None) for i in range(1, 6)] +
                   [(i, True, c) for i in range(2, 5) for c in (None, capture)])
    return engine


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph hardware required"))])
@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
@pytest.mark.parametrize("temperature", [0, 1])
@pytest.mark.parametrize("last_layers", [None, 1])
@pytest.mark.parametrize("update_policy", ["periodic", "coverage"])
def test_decoder_window_matches_full_collection_across_requests(device, generate, temperature, last_layers,
                                                               update_policy):
    model = model_example(device)
    config = OnlineConfig(stride=4, replay_blocks=2, train_last_layers=last_layers,
                          learning_rate=.001, loss="forward_kl", update_policy=update_policy)
    learner = RecordedLearner(model, config)
    reference = RecordedLearner(copy.deepcopy(model), replace(config, feedback_execution="all"))
    engines = [prepared(target.model, last_layers, device) for target in (learner, reference)]
    options = {"block_size": 4, "sampling": SamplingConfig(temperature=temperature)}
    if generate is generate_tree:
        options.update(top_k=2, prefix_budget=5)
    feedback = [0, 0]
    for request, budget in enumerate((1, 3, 23, 29)):
        prompt = torch.tensor([[0, request + 1, 5]], device=device)
        outputs = [generate(target.model, prompt, budget, learner=target, executor=engine,
                            eos_id=1 if request == 3 else None,
                            generator=torch.Generator(device=device).manual_seed(43 + request), **options)
                   for target, engine in zip((learner, reference), engines)]
        left, right = outputs
        assert left.tokens == right.tokens and left.accepted_per_round == right.accepted_per_round
        assert left.rounds == right.rounds and left.updates == right.updates
        assert left.fully_covered_rounds == right.fully_covered_rounds
        assert left.coverage_skips == right.coverage_skips
        for i, output in enumerate(outputs):
            feedback[i] += output.feedback_blocks
        assert learner.rounds == reference.rounds and learner.version == reference.version
        assert not learner.replay and not reference.replay
        if budget == 1:
            assert learner.rounds == learner.feedback_blocks == learner.version == 0
    assert learner.updates + learner.coverage_skips > 0
    tolerance = 1e-14 if device == "cpu" else 2e-6
    torch.testing.assert_close(learner.history, reference.history, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(adapter_state(learner.model), adapter_state(reference.model),
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(learner.optimizer.state_dict(), reference.optimizer.state_dict(),
                               atol=tolerance, rtol=tolerance)
    assert feedback == [learner.feedback_blocks, reference.feedback_blocks]
    assert feedback[0] < feedback[1]


@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
@pytest.mark.parametrize("last_layers", [None, 1])
def test_root_eos_outside_window_preserves_feedback_clock(generate, last_layers):
    model = model_example()
    prompt = torch.tensor([[0, 1, 5]])
    eos = generate_ar(model, prompt, 1).tokens[0]
    config = OnlineConfig(stride=4, replay_blocks=1, train_last_layers=last_layers)
    for execution in ("windowed", "all"):
        learner = OnlineLearner(copy.deepcopy(model), replace(config, feedback_execution=execution))
        learner.rounds = 2
        output = generate(learner.model, prompt, 24, block_size=4, learner=learner, eos_id=eos,
                          generator=torch.Generator().manual_seed(91))
        assert output.tokens == [eos] and output.feedback_blocks == output.updates == 0
        assert learner.rounds == 2 and learner.version == 0 and not learner.replay


def test_generic_feedback_accepts_sparse_targets_and_skip_requires_decoder_window():
    learner = OnlineLearner(model_example(), OnlineConfig(stride=4, replay_blocks=1))
    with pytest.raises(ValueError, match="positive-feedback decoder"):
        learner._skip_decoder_feedback(0)
    assert learner.rounds == 0
    for _ in range(3):
        learner._skip_decoder_feedback(1)
    with pytest.raises(ValueError, match="outside the replay window"):
        learner._skip_decoder_feedback(1)
    empty = Feedback(torch.tensor([[0, 1]]), None, torch.empty(0, 13), 0)
    assert learner.observe(empty) is None and learner.rounds == 4
    full = Feedback(torch.tensor([[0, 1]]), None, torch.randn(1, 13, dtype=torch.float64), 1)
    learner.observe(full, may_update=False)
    for _ in range(3):
        learner.observe(empty)
    assert learner.updates == 1 and learner.feedback_blocks == 1


@pytest.mark.parametrize("options", [{"stride": 1.5}, {"stride": True}, {"replay_blocks": 2.5},
                                     {"feedback_execution": "unknown"}])
def test_window_configuration_validation(options):
    with pytest.raises(ValueError):
        OnlineConfig(**options)
