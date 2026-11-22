from types import SimpleNamespace

import torch

from blockspec.dsv41_online import DSV41OnlineConfig, DSV41OnlineController
from blockspec.dsv41_serving_online import (
    DSV41CapturedLayer,
    DSV41ServingOnline,
    DSV41ServingReplay,
)
from blockspec.dsv41_suffix import DSV41SuffixPlan


def layer_state(layer):
    return DSV41CapturedLayer(
        layer,
        {name: object() for name in (
            "hidden", "incoming", "pre", "post", "comb", "collapsed",
            "cos", "sin", "history", "selection",
        )},
        {name: object() for name in (
            "changed", "pre", "post", "comb", "collapsed", "normed",
            "router_input", "router_logits", "route_ids", "route_weights",
            "shared", "combined",
        )},
    )


def test_serving_bridge_runs_last_n_update_from_corrected_verifier_rows():
    class Learner:
        last_layers = 2

        def __init__(self):
            self.calls = []

        def update_rows(self, replay, teacher, **options):
            self.calls.append((replay, teacher.clone(), options))
            return .2, .4, len(self.calls)

    learner = Learner()
    published = []
    config = DSV41OnlineConfig(
        last_layers=2, teacher_rows=3, warm_update_stride=1,
        warm_update_steps=1, steady_update_stride=1, publish_every=1,
    )
    bridge = DSV41ServingOnline(DSV41OnlineController(
        learner, config, publisher=lambda owner, version: published.append(version),
    ))
    plan = DSV41SuffixPlan(2)
    replay = DSV41ServingReplay(
        "request", (7, 8), (0, 0, 1), plan, torch.zeros(32, 4),
        tuple(layer_state(layer) for layer in plan.layers),
    )
    assert bridge.capture_due() and bridge.register_replay(replay)
    first = torch.randn(2, 13)
    assert bridge.verified("request", (7, 8), (9,), (9, 10), first) is None
    second = torch.randn(1, 13)
    record = bridge.continued("request", (7, 8, 9, 10), (11,), second)
    assert record["teacher_rows"] == 3 and record["published"]
    assert learner.calls[0][0].plan == plan
    torch.testing.assert_close(learner.calls[0][1], torch.cat((first, second)))
    assert published == [1] and bridge.summary()["pending_replays"] == 0


def test_serving_bridge_rejects_teacher_rows_from_rejected_candidate_branch():
    learner = SimpleNamespace(last_layers=1, update_rows=lambda *args, **kwargs: None)
    config = DSV41OnlineConfig(
        teacher_rows=1, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=1,
    )
    bridge = DSV41ServingOnline(DSV41OnlineController(learner, config, publisher=lambda *args: None))
    plan = DSV41SuffixPlan(1)
    bridge.register_replay(DSV41ServingReplay(
        "r", (4,), (0, 0, 1), plan, torch.zeros(32, 4), (layer_state(39),),
    ))
    try:
        bridge.verified("r", (4,), (99,), (5, 6), torch.randn(2, 7))
    except ValueError as error:
        assert "accepted candidate" in str(error)
    else:
        raise AssertionError("wrong-branch feedback was accepted")
