from types import SimpleNamespace

import torch
import pytest

from blockspec.dsv41_device_update import DSV41DeviceSuffixLearner, SuffixGradient
from blockspec.dsv41_resident_update import DSV41ResidentUpdateBackend
from blockspec.dsv41_serving_online import DSV41CapturedLayer
from blockspec.dsv41_suffix import DSV41SuffixPlan, DSV41SuffixReplay


def captured(layer, ranks=(0, 1, 2, 3)):
    attention = {
        rank: {name: object() for name in (
            "readonly_hidden", "hidden", "incoming", "pre", "post", "comb",
            "collapsed", "normed", "changed", "cos", "sin", "history", "selection",
            "qkv", "contribution", "lse",
        )} for rank in ranks
    }
    ffn = {
        rank: {name: object() for name in (
            "changed", "pre", "post", "comb", "collapsed", "normed",
            "router_input", "router_logits", "route_ids", "route_weights",
            "shared", "expert_outputs", "combined", "output_hidden", "output_pre_mix",
        )} for rank in ranks
    }
    return DSV41CapturedLayer(layer, attention, ffn)


class Primitives:
    def __init__(self, plan):
        self.plan, self.events = plan, []
        self.values = {name: torch.zeros(1, dtype=torch.bfloat16) for name in plan.trainable_names()}

    def loss_vjp(self, logits, teacher, valid, kind):
        self.events.append(("loss", valid, kind))
        return torch.tensor(.2), torch.ones_like(logits)

    def lm_head_vjp(self, gradient, rows):
        self.events.append(("lm-head", tuple(sorted(rows))))
        values = {rank: torch.ones(32, 4) for rank in rows}
        return SuffixGradient(values, values)

    def reverse_frozen_tail(self, **options):
        self.events.append(("tail", options["layer"], options["rank"], options["captured"]["route_ids"]))
        return options["output_gradient"]

    def reverse_frozen_tail_owner_batch(self, **options):
        self.events.append(("tail-owner-batch", options["layer"], options["ranks"]))
        return options["output_gradients"]

    def participate_frozen_tail_peer(self, **options):
        self.events.append(("tail-peer", options["layer"], options["synchronization"]["reverse"]["protocol"]))

    def reverse_draft_attention(self, **options):
        layer, rank = options["layer"], options["rank"]
        self.events.append(("attention", layer, rank, options["captured"]["selection"]))
        prefix = f"layers.{layer}.attn."
        names = [name.removeprefix(prefix) for name in self.plan.trainable_names()
                 if name.startswith(prefix)]
        return options["output_gradient"], {name: torch.ones(1) for name in names}

    def tp_assemble_gradients(self, layer, values):
        self.events.append(("tp-assemble", layer, tuple(sorted(values))))
        return values[min(values)]

    def dp_all_reduce(self, values):
        self.events.append(("dp", len(values)))
        return values

    def gradient_norm_and_clip(self, values, maximum):
        self.events.append(("clip", maximum))
        return torch.tensor(.5), torch.tensor(1.)

    def adamw(self, values, **options):
        self.events.append(("adam", options["step"]))

    def zero_gradients(self, layers, names):
        return {name: torch.zeros(1) for name in names}

    def publication_state(self, layers):
        return self.values

    @staticmethod
    def read_scalar(value):
        return value.item()


def test_resident_backend_consumes_exact_routes_mhc_and_attention_capture():
    plan = DSV41SuffixPlan(1)
    primitives = Primitives(plan)
    backend = DSV41ResidentUpdateBackend(primitives, last_layers=1)
    layer = captured(39)
    replay = DSV41SuffixReplay(
        plan, {rank: object() for rank in range(4)}, (layer,), torch.zeros(3, 11),
    )
    learner = DSV41DeviceSuffixLearner(backend, last_layers=1)
    loss, norm, version = learner.update_rows(replay, torch.randn(3, 11))
    assert loss == pytest.approx(.2) and norm == .5 and version == 1
    assert primitives.events[:2] == [("loss", 3, "tv"), ("lm-head", (0, 1, 2, 3))]
    assert [event[:3] for event in primitives.events if event[0] == "tail"] == [
        ("tail", 39, rank) for rank in range(4)
    ]
    assert [event[:3] for event in primitives.events if event[0] == "attention"] == [
        ("attention", 39, rank) for rank in range(4)
    ]
    assert ("tp-assemble", 39, (0, 1, 2, 3)) in primitives.events
    assert primitives.events[-3:] == [("dp", 8), ("clip", 1.0), ("adam", 1)]


def test_resident_backend_rejects_incomplete_owner_tp_capture():
    plan = DSV41SuffixPlan(1)
    backend = DSV41ResidentUpdateBackend(Primitives(plan), last_layers=1)
    layer = captured(39)
    broken = DSV41CapturedLayer(39, layer.attention, {0: layer.ffn[0]})
    state = backend.captured_output_state(layer)
    gradient = SuffixGradient(
        {rank: torch.ones(1) for rank in range(4)},
        {rank: torch.ones(1) for rank in range(4)},
    )
    try:
        backend.reverse_layer(39, broken, state, gradient)
    except ValueError as error:
        assert "different TP ranks" in str(error)
    else:
        raise AssertionError("incomplete TP capture was accepted")


def test_resident_backend_trains_only_available_corrected_prefix_rows():
    plan = DSV41SuffixPlan(1)
    primitives = Primitives(plan)
    backend = DSV41ResidentUpdateBackend(primitives, last_layers=1)
    loss, gradient = backend.loss_vjp(
        torch.randn(31, 17), torch.randn(7, 17), 7, "tv",
    )
    assert loss.item() == pytest.approx(.2)
    assert gradient.shape == (7, 17)
    assert primitives.events == [("loss", 7, "tv")]


def test_synchronized_owner_batches_tp_ranks_before_attention_reverse():
    plan = DSV41SuffixPlan(1)
    primitives = Primitives(plan)
    backend = DSV41ResidentUpdateBackend(primitives, last_layers=1)
    layer = captured(39)
    state = backend.captured_output_state(layer)
    values = {rank: torch.ones(32, 4) for rank in range(4)}
    contract = {
        "reverse": {
            "protocol": "dsv41-synchronized-reverse-v1",
            "layers": [39],
        },
    }
    backend.reverse_layer(
        39, layer, state, SuffixGradient(values, values),
        synchronization=contract,
    )
    assert primitives.events[0] == ("tail-owner-batch", 39, (0, 1, 2, 3))
    assert [event[:3] for event in primitives.events if event[0] == "attention"] == [
        ("attention", 39, rank) for rank in range(4)
    ]


def test_peer_enters_every_layer_in_reverse_order_and_requires_hook():
    plan = DSV41SuffixPlan(2)
    primitives = Primitives(plan)
    backend = DSV41ResidentUpdateBackend(primitives, last_layers=2)
    contract = {
        "reverse": {
            "protocol": "dsv41-synchronized-reverse-v1",
            "layers": list(plan.layers),
        },
    }
    backend.participate_reverse(plan.layers, contract)
    assert primitives.events == [
        ("tail-peer", 39, "dsv41-synchronized-reverse-v1"),
        ("tail-peer", 38, "dsv41-synchronized-reverse-v1"),
    ]

    primitives.participate_frozen_tail_peer = None
    with pytest.raises(RuntimeError, match="participate_frozen_tail_peer"):
        backend.participate_reverse(plan.layers, contract)
