from types import SimpleNamespace

import pytest
import torch

from blockspec.dsv41_online import DSV41OnlineConfig
from blockspec.dsv41_online_bootstrap import DSV41OnlineBootstrap


class Draft:
    def __init__(self):
        self.enabled = None
        self.events = []

    def enable_online(self, last_layers):
        self.enabled = last_layers

    def stage_online(self, learner, version):
        self.events.append(("stage", version))

    def activate_online(self, version):
        self.events.append(("activate", version))

    def commit_online(self, version):
        self.events.append(("commit", version))

    def rollback_online(self, version):
        self.events.append(("rollback", version))


class Primitives:
    def __getattr__(self, name):
        if name in {
            "loss_vjp", "lm_head_vjp", "reverse_frozen_tail",
            "reverse_draft_attention", "tp_assemble_gradients", "dp_all_reduce",
            "gradient_norm_and_clip", "adamw", "zero_gradients",
            "publication_state", "read_scalar",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def test_online_bootstrap_binds_after_native_resident_initialization():
    seen = []

    def factory(**values):
        seen.append(values)
        return Primitives()

    draft = Draft()
    native = SimpleNamespace(weights=object())
    bootstrap = DSV41OnlineBootstrap(
        factory,
        DSV41OnlineConfig(last_layers=4, warm_update_stride=1),
    )
    assert bootstrap.summary() == {"bound": False, "last_layers": 4}
    bridge = bootstrap.bind(draft, native)
    assert bootstrap.bound and bootstrap.bridge is bridge and draft.enabled == 4
    assert seen == [{"draft": draft, "native": native, "cluster": None, "config": bootstrap.config}]
    with pytest.raises(RuntimeError, match="already bound"):
        bootstrap.bind(draft, native)


def test_online_bootstrap_rejects_unimplemented_non_c1a_suffix():
    with pytest.raises(ValueError, match="last-N <= 20"):
        DSV41OnlineBootstrap(lambda **kwargs: Primitives(), DSV41OnlineConfig(last_layers=21))
