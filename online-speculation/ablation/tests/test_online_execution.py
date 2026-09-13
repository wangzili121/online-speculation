import copy

import pytest
import torch

from blockspec_ablation.distillation import divergence
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.online import Feedback, OnlineConfig, OnlineLearner


@pytest.mark.parametrize("last_layers", [None, 1])
@pytest.mark.parametrize("interval", [(0, 1), (1, 2), (1, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA projection hardware required"))])
def test_selected_projection_matches_full_logits_cache_and_parameter_gradients(last_layers, interval, dtype, device):
    torch.manual_seed(19)
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, adapter_rank=2)).to(device=device, dtype=dtype).train_adapters_only()
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.1)
        _, cache = model(torch.tensor([[0, 1]], device=device), return_cache=True)
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=last_layers))
    inputs = torch.tensor([[1, 2, 3, 4]], device=device)
    mask = torch.tensor([[False, True, True, True]], device=device)
    full, full_cache = model(inputs, adapter_mask=mask, cache=cache, return_cache=True)
    selected, selected_cache = model(inputs, adapter_mask=mask, cache=cache,
                                      return_cache=True, logit_range=interval)
    if last_layers is not None:
        with torch.no_grad():
            _, _, boundary = model(inputs, adapter_mask=mask, cache=cache, return_cache=True,
                                     capture_layer=learner.capture_layer)
        full = model.forward_suffix(boundary, cache=cache[learner.capture_layer:])
        selected = model.forward_suffix(boundary, cache=cache[learner.capture_layer:], logit_range=interval)
    expected = full[:, interval[0]:interval[1]]
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(selected, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(selected_cache, full_cache, atol=0, rtol=0)
    target = torch.randn_like(expected)
    original_gradients = torch.autograd.grad(divergence(expected, target, "forward_kl").sum(), learner.parameters)
    selected_gradients = torch.autograd.grad(divergence(selected, target, "forward_kl").sum(), learner.parameters)
    for actual, reference in zip(selected_gradients, original_gradients):
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("interval", [(0, 0), (-1, 1), (1, 5), (1., 2), [1, 2], (0, 1, 2)])
def test_projection_range_validation(interval):
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=1, adapter_rank=2))
    with pytest.raises(ValueError, match="logit range"):
        model(torch.tensor([[1, 2, 3, 4]]), logit_range=interval)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optimizer hardware required")
@pytest.mark.parametrize("last_layers", [None, 1])
def test_fused_optimizer_matches_standard_adam_across_updates(last_layers):
    torch.manual_seed(49)
    model = Decoder(ModelConfig(vocab_size=11, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, adapter_rank=2)).cuda().train_adapters_only()
    oracle = copy.deepcopy(model)
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=last_layers, optimizer="fused"))
    reference = OnlineLearner(oracle, OnlineConfig(train_last_layers=last_layers, optimizer="standard"))
    inputs = torch.tensor([[1, 2, 3, 4]], device="cuda")
    mask = torch.tensor([[False, True, True, True]], device="cuda")
    teacher = torch.randn(2, 11, device="cuda")
    for target in (learner, reference):
        with torch.no_grad():
            result = target.model(inputs, adapter_mask=mask, return_cache=True, capture_layer=target.capture_layer)
        boundary = result[2] if last_layers is not None else None
        target.observe(Feedback(inputs, None, teacher, 2, boundary), may_update=False)
    for _ in range(4):
        actual, expected = learner.update(), reference.update()
        assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
        for p, q in zip(model.parameters(), oracle.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
        for p, q in zip(learner.parameters, reference.parameters):
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(learner.optimizer.state[p][key], reference.optimizer.state[q][key],
                                           atol=1e-7, rtol=1e-5)


def test_optimizer_execution_selection():
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=1, adapter_rank=2))
    assert OnlineLearner(model).optimizer_backend == "standard"
    with pytest.raises(ValueError, match="requires CUDA"):
        OnlineLearner(model, OnlineConfig(optimizer="fused"))
    with pytest.raises(ValueError, match="optimizer execution"):
        OnlineConfig(optimizer="unknown")
