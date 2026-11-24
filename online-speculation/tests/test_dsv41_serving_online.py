import json
from types import SimpleNamespace

import pytest
import torch

from blockspec.dsv41_online import DSV41OnlineConfig, DSV41OnlineController
from blockspec.dsv41_online_profile import OnlineUpdateProfile, active_profile, profile_online, timed_stage
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
            "qkv", "contribution", "lse",
        )},
        {name: object() for name in (
            "changed", "pre", "post", "comb", "collapsed", "normed",
            "router_input", "router_logits", "route_ids", "route_weights",
            "shared", "expert_outputs", "combined", "output_hidden", "output_pre_mix",
        )},
    )


def test_online_profile_measures_four_updates_with_hot_owner_reuse(tmp_path):
    def update(version, owner_dp):
        profile = active_profile()
        with timed_stage("attention_vjp"):
            if not profile.dispatch_seen(owner_dp * 4, "attention"):
                profile.current["first_dispatch"].append({"seconds": 0.5})
        return .1, .2, version

    config = DSV41OnlineConfig(
        teacher_rows=7, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=1,
    )
    bridges, profiles = [], []
    for dp in (0, 2):
        learner = SimpleNamespace(
            last_layers=1,
            update_rows=lambda replay, teacher, **kw: update(kw["synchronization"]["update_number"], replay.owner[0]),
            update_participant=lambda contract, **kw: (0., .2, contract["update_number"]),
        )
        bridges.append(DSV41ServingOnline(DSV41OnlineController(
            learner, config, publisher=lambda *_: None,
        )))
        profiles.append(OnlineUpdateProfile(tmp_path / f"node-{dp}.jsonl"))
    for index, dp in enumerate((0, 2, 0, 0)):
        request = f"orthrus-request-{index}"
        owner = dp // 2
        for bridge, profile in zip(bridges, profiles, strict=True):
            with profile_online(profile):
                profile.begin_request(request)
                assert bridge.capture_due()
        bridge = bridges[owner]
        replay = DSV41ServingReplay(
            request, (4,), (dp, 0, 1), DSV41SuffixPlan(1), torch.zeros(32, 4), (layer_state(39),),
        )
        assert bridge.register_replay(replay)
        assert bridge.verified(request, (4,), (5, 6), (5, 6, 7), torch.randn(3, 7), defer=True) is None
        prepared = bridge.verified(
            request, (4, 5, 6, 7), (8, 9, 10), (8, 9, 10, 11), torch.randn(4, 7), defer=True,
        )
        for node, (bridge, profile) in enumerate(zip(bridges, profiles, strict=True)):
            with profile_online(profile):
                if node == owner:
                    bridge.synchronize(prepared, lambda *_: None)
                else:
                    bridge.synchronize(None, lambda *_: prepared.contract())
                assert not bridge.capture_due()

    for profile in profiles:
        assert len(profile.path.read_text().splitlines()) == 4
    report = OnlineUpdateProfile.summary(*(profile.records for profile in profiles))
    assert [row["hot"] for row in report["updates"]] == [False, False, True, True]
    assert report["hot_updates"] == 2
    assert [row["teacher_rows"] for row in report["updates"]] == [7] * 4
    assert all("attention_vjp" in row["owner_stage_seconds"] for row in report["updates"])
    assert json.loads(profiles[0].path.read_text().splitlines()[1])["loss"] is None


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
    assert bridge.capture_due()
    bridge.register_replay(DSV41ServingReplay(
        "r", (4,), (0, 0, 1), plan, torch.zeros(32, 4), (layer_state(39),),
    ))
    try:
        bridge.verified("r", (4,), (99,), (5, 6), torch.randn(2, 7))
    except ValueError as error:
        assert "accepted candidate" in str(error)
    else:
        raise AssertionError("wrong-branch feedback was accepted")


def test_serving_bridge_uses_only_remaining_rows_of_the_delivered_prefix():
    config = DSV41OnlineConfig(
        teacher_rows=3, warm_update_stride=1, warm_update_steps=1, steady_update_stride=1,
    )
    learner = SimpleNamespace(last_layers=1, update_rows=lambda *args, **kwargs: None)
    bridge = DSV41ServingOnline(DSV41OnlineController(learner, config))
    plan = DSV41SuffixPlan(1)
    assert bridge.capture_due() and bridge.register_replay(DSV41ServingReplay(
        "r", (4,), (0, 0, 1), plan, torch.zeros(32, 4), (layer_state(39),),
    ))
    first = torch.arange(7, dtype=torch.float32).reshape(1, 7)
    tail = torch.arange(14, 28, dtype=torch.float32).reshape(2, 7)
    assert bridge.verified("r", (4,), (99,), (5,), first, defer=True) is None
    assert bridge.feedback_rows_needed("r") == 2
    with pytest.raises(ValueError, match="verifier-logit rows"):
        bridge.verified("r", (4, 5), (6, 7, 8), (6, 7, 8, 9), tail[:1], defer=True)
    prepared = bridge.verified("r", (4, 5), (6, 7, 8), (6, 7, 8, 9), tail, defer=True)
    assert torch.equal(prepared.teacher, torch.cat((first, tail)))


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
    assert bridge.capture_due()
    assert bridge.register_replay(replay)
    prepared = bridge.verified(
        "r", (4,), (), (5,), torch.randn(1, 7), defer=True,
    )
    assert prepared.update_number == 1 and learner.calls == 0
    assert not bridge.capture_due()
    assert bridge.apply(prepared)["update"] == 1 and learner.calls == 1


def test_serving_bridge_synchronizes_owner_and_reverse_participant_peer():
    class Learner:
        last_layers = 1

        def __init__(self, owner):
            self.owner, self.calls = owner, []

        def update_rows(self, replay, teacher, **options):
            self.calls.append((
                "owner", teacher.shape[0],
                options["synchronization"]["reverse"]["protocol"],
                tuple(options["synchronization"]["reverse"]["ep_bucket_rows"]),
            ))
            return .4, .6, 1

        def update_participant(self, contract, **options):
            self.calls.append((
                "participant", options["update_number"],
                tuple(contract["reverse"]["layers"]),
                tuple(contract["reverse"]["ep_bucket_rows"]),
            ))
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
        (layer_state(39),), ep_bucket_rows=(32,),
    )
    assert owner.capture_due() and peer.capture_due()
    assert owner.register_replay(replay)
    prepared = owner.verified(
        "r", (4,), (), (5,), torch.randn(1, 7), defer=True,
    )
    contract = prepared.contract()
    bad = dict(contract, reverse=dict(contract["reverse"], ep_bucket_rows=[48]))
    with pytest.raises(ValueError, match="invalid EP bucket widths"):
        peer.synchronize(None, lambda tag, payload: bad)
    owner_record = owner.synchronize(prepared, lambda tag, payload: None)
    peer_record = peer.synchronize(None, lambda tag, payload: contract)
    assert owner_record["update"] == peer_record["update"] == 1
    assert owner_learner.calls == [
        ("owner", 1, "dsv41-synchronized-reverse-v1", (32,)),
    ]
    assert peer_learner.calls == [("participant", 1, (39,), (32,))]


def test_serving_bridge_rejects_remote_zero_gradient_shortcut():
    class Learner:
        last_layers = 1

        def update_rows(self, *args, **kwargs):
            return .4, .6, 1

        def update_zero(self, **options):
            raise AssertionError("remote serving must not use update_zero")

    config = DSV41OnlineConfig(
        teacher_rows=1, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=2,
    )
    bridge = DSV41ServingOnline(DSV41OnlineController(Learner(), config))
    contract = {
        "request_id": "r", "owner": [0, 0, 1], "teacher_rows": 1,
        "vocabulary": 7, "update_number": 1,
        "learning_rate": config.learning_rate_for(1),
        "objective": "tv",
        "reverse": {
            "protocol": "dsv41-synchronized-reverse-v1", "layers": [39],
        },
    }
    try:
        bridge.synchronize(None, lambda tag, payload: contract)
    except RuntimeError as error:
        assert "update_participant" in str(error)
    else:
        raise AssertionError("remote zero-gradient shortcut was accepted")


def test_serving_bridge_selects_one_capture_across_hosts():
    learner = SimpleNamespace(last_layers=1, update_rows=lambda *args, **kwargs: None)
    config = DSV41OnlineConfig(
        teacher_rows=1, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=2,
    )
    owner = DSV41ServingOnline(DSV41OnlineController(learner, config))
    peer = DSV41ServingOnline(DSV41OnlineController(learner, config))
    owner_requests = {(0, 0, 1): ("owner-request", (7, 8))}
    peer_requests = {(2, 0, 1): ("peer-request", (9, 10))}
    assert owner.capture_due() and peer.capture_due()
    owner_rows = owner._capture_rows(owner_requests)
    selected = owner.select_capture(owner_requests, lambda tag, rows: peer._capture_rows(peer_requests))
    assert selected == owner_requests
    assert peer.select_capture(peer_requests, lambda tag, rows: owner_rows) == {}
    assert peer.remote_pending == {"owner": [0, 0, 1], "request_id": "owner-request"}


def test_serving_bridge_applies_short_feedback_when_request_finishes():
    class Learner:
        last_layers = 1

        def __init__(self):
            self.teacher = None

        def update_rows(self, replay, teacher, **options):
            self.teacher = teacher.clone()
            return .3, .4, 1

    learner = Learner()
    config = DSV41OnlineConfig(
        teacher_rows=7, warm_update_stride=1, warm_update_steps=1,
        steady_update_stride=1, publish_every=2,
    )
    bridge = DSV41ServingOnline(DSV41OnlineController(learner, config))
    replay = DSV41ServingReplay(
        "short", (4,), (0, 0, 1), DSV41SuffixPlan(1), torch.zeros(32, 4),
        (layer_state(39),),
    )
    assert bridge.capture_due() and bridge.register_replay(replay)
    assert bridge.verified(
        "short", (4,), (5,), (5, 6), torch.randn(2, 9), defer=True,
    ) is None
    prepared = bridge.finish("short", defer=True)
    assert prepared.teacher.shape == (2, 9)
    record = bridge.apply(prepared)
    assert record["teacher_rows"] == 2 and learner.teacher.shape == (2, 9)
