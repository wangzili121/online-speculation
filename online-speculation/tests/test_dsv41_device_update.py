from types import SimpleNamespace

import torch

from blockspec.dsv41_device_update import DSV41DeviceSuffixLearner, SuffixGradient
from blockspec.dsv41_suffix import DSV41SuffixPlan, DSV41SuffixReplay


class Backend:
    def __init__(self, plan):
        self.plan = plan
        self.events = []
        self.values = {name: torch.zeros(1, dtype=torch.bfloat16) for name in plan.trainable_names()}

    def forward_layer(self, layer, hidden, captured):
        self.events.append(("forward", layer))
        return hidden + 1, captured

    def project_logits(self, hidden):
        self.events.append(("project", None))
        return torch.zeros(3, 11), hidden

    def loss_vjp(self, logits, teacher, valid, kind):
        self.events.append(("loss", kind, valid))
        return torch.tensor(.25), torch.ones_like(logits)

    def project_logits_vjp(self, gradient, state):
        self.events.append(("project_vjp", None))
        return SuffixGradient(torch.ones(32, 4), torch.ones(32, 4))

    def reverse_layer(self, layer, captured, state, boundary):
        self.events.append(("reverse", layer))
        prefix = f"layers.{layer}.attn."
        names = [name.removeprefix(prefix) for name in self.plan.trainable_names()
                 if name.startswith(prefix)]
        return boundary, {name: torch.ones(1) for name in names}

    def dp_all_reduce(self, gradients):
        self.events.append(("dp_all_reduce", len(gradients)))
        return gradients

    def zero_gradients(self, layers):
        self.events.append(("zero", tuple(layers)))
        return {name: torch.zeros(1) for name in self.plan.trainable_names()}

    def gradient_norm_and_clip(self, gradients, maximum):
        self.events.append(("clip", maximum))
        return torch.tensor(.5), torch.tensor(1.)

    def adamw(self, gradients, *, rate, step, clip):
        self.events.append(("adamw", rate, step))

    @staticmethod
    def read_scalar(value):
        return value.item()

    def publication_state(self, layers):
        return self.values


def test_device_suffix_update_replays_forward_reverses_and_reduces_before_adam():
    plan = DSV41SuffixPlan(2)
    backend = Backend(plan)
    learner = DSV41DeviceSuffixLearner(backend, last_layers=2)
    replay = DSV41SuffixReplay(plan, torch.zeros(32, 4), (SimpleNamespace(), SimpleNamespace()))
    assert learner.update_rows(replay, torch.randn(3, 11)) == (.25, .5, 1)
    assert backend.events == [
        ("forward", 38), ("forward", 39), ("project", None),
        ("loss", "tv", 3), ("project_vjp", None),
        ("reverse", 39), ("reverse", 38),
        ("dp_all_reduce", 16), ("clip", 1.), ("adamw", 1e-5, 1),
    ]
    assert learner.publication_state() == backend.values


def test_non_owner_replica_contributes_zero_then_applies_reduced_update():
    plan = DSV41SuffixPlan(4)
    backend = Backend(plan)
    learner = DSV41DeviceSuffixLearner(backend, last_layers=4)
    assert learner.update_zero(learning_rate=2e-6, update_number=1) == (0., .5, 1)
    assert backend.events == [
        ("zero", plan.layers), ("dp_all_reduce", 32),
        ("clip", 1.), ("adamw", 2e-6, 1),
    ]
