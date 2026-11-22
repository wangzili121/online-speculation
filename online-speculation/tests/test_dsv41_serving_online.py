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
            "readonly_hidden", "hidden", "incoming", "pre", "post", "comb", "collapsed",
            "normed", "changed", "cos", "sin", "history", "selection",
            "qkv", "contribution",
        )},
        {name: object() for name in (
            "changed", "pre", "post", "comb", "collapsed", "normed",
            "router_input", "router_logits", "route_ids", "route_weights",
            "shared", "combined", "output_hidden", "output_pre_mix",
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


def test_serving_bridge_can_defer_update_until_cluster_rendezvous():
    class Learner:
        last_layers = 1

        def __init__(self):
            self.calls = 0

        def update_rows(self, replay, teacher, **options):
            self.calls += 1
            return .3, .5, self.calls

    learner = Learner()
    config = DSV41OnlineConfig(
        teacher_rows=1, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=2,
    )
    bridge = DSV41ServingOnline(DSV41OnlineController(learner, config))
    replay = DSV41ServingReplay(
        "r", (4,), (0, 0, 1), DSV41SuffixPlan(1), torch.zeros(32, 4),
        (layer_state(39),),
    )
    assert bridge.register_replay(replay)
    prepared = bridge.verified(
        "r", (4,), (), (5,), torch.randn(1, 7), defer=True,
    )
    assert prepared.update_number == 1 and learner.calls == 0
    assert not bridge.capture_due()
    assert bridge.apply(prepared)["update"] == 1 and learner.calls == 1


def test_serving_bridge_synchronizes_owner_and_zero_gradient_peer():
    class Learner:
        last_layers = 1

        def __init__(self, owner):
            self.owner, self.calls = owner, []

        def update_rows(self, replay, teacher, **options):
            self.calls.append(("owner", teacher.shape[0]))
            return .4, .6, 1

        def update_zero(self, **options):
            self.calls.append(("zero", options["update_number"]))
            return 0., .6, 1

    config = DSV41OnlineConfig(
        teacher_rows=1, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=2,
    )
    owner_learner, peer_learner = Learner(True), Learner(False)
    owner = DSV41ServingOnline(DSV41OnlineController(owner_learner, config))
    peer = DSV41ServingOnline(DSV41OnlineController(peer_learner, config))
    replay = DSV41ServingReplay(
        "r", (4,), (0, 0, 1), DSV41SuffixPlan(1), torch.zeros(32, 4),
        (layer_state(39),),
    )
    assert owner.register_replay(replay)
    prepared = owner.verified(
        "r", (4,), (), (5,), torch.randn(1, 7), defer=True,
    )
    contract = prepared.contract()
    owner_record = owner.synchronize(prepared, lambda tag, payload: None)
    peer_record = peer.synchronize(None, lambda tag, payload: contract)
    assert owner_record["update"] == peer_record["update"] == 1
    assert owner_learner.calls == [("owner", 1)]
    assert peer_learner.calls == [("zero", 1)]
