import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "kda_direction", Path(__file__).parents[1] / "benchmarks" / "glm53_kda_direction.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_zero_reverse_gate_preserves_initial_logits_and_shares_frozen_backbone():
    torch.manual_seed(15)
    teacher = experiment.Hybrid().eval().requires_grad_(False)
    causal = experiment.make_student(teacher, False)
    bidir = experiment.make_student(teacher, True)
    prefix = torch.randint(1, 32, (2, 4))
    with torch.no_grad():
        _, history = teacher(prefix)
    inputs = torch.zeros(2, 8, dtype=torch.long)
    inputs[:, 0] = 5
    left, _ = causal(inputs, history, causal=False)
    right, _ = bidir(inputs, history, causal=False)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert bidir.embedding is teacher.embedding
    assert bidir.layers[0].up is teacher.layers[0].up
    assert bidir.layers[0].mixer is not teacher.layers[0].mixer
    assert all(not p.requires_grad for p in teacher.parameters())


def test_future_block_features_reach_earlier_output_through_mla_and_causal_kda():
    torch.manual_seed(17)
    teacher = experiment.Hybrid().eval().requires_grad_(False)
    student = experiment.make_student(teacher, False)
    with torch.no_grad():
        _, history = teacher(torch.tensor([[3, 7, 9, 11]]))
        before = [tuple(x.clone() for x in pair) for pair in history]
        inputs = torch.tensor([[4, 0, 0, 0, 0, 0, 0, 0]])
        original, _ = student(inputs, history, causal=False)
        inputs[0, -1] = 12
        changed, _ = student(inputs, history, causal=False)
    assert (original[0, 0] - changed[0, 0]).abs().max() > 1e-6
    for previous, current in zip(before, history, strict=True):
        for x, y in zip(previous, current, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_causal_cached_forward_equals_full_forward():
    torch.manual_seed(19)
    model = experiment.Hybrid().eval()
    ids = torch.randint(1, 32, (2, 10))
    with torch.no_grad():
        full, _ = model(ids)
        _, prefix = model(ids[:, :4])
        suffix, _ = model(ids[:, 4:], prefix)
    torch.testing.assert_close(full[:, 4:], suffix, rtol=1e-4, atol=1e-5)


def test_distillation_has_reverse_gate_gradients_without_teacher_updates():
    torch.manual_seed(21)
    teacher = experiment.Hybrid().eval().requires_grad_(False)
    student = experiment.make_student(teacher, True)
    rows, _ = experiment.dataset(7)
    inputs, history, expected = experiment.prepare(teacher, rows[:2])[0]
    logits, _ = student(inputs, history, causal=False)
    loss = torch.nn.functional.kl_div(logits[:, :-1].log_softmax(-1), expected,
                                     reduction="none").sum(-1).mean()
    loss.backward()
    gates = [layer.mixer.backward_gate.grad for layer in student.layers
             if isinstance(layer.mixer, experiment.KDA)]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gates)
    assert any(gradient.abs().max() > 0 for gradient in gates)
    assert all(parameter.grad is None for parameter in teacher.parameters())
