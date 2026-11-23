from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_dsv41_resident_primitives import (
    DSV41ResidentPrimitives,
    ProductionFactoryIncomplete,
    create_primitives,
    production_readiness,
)


class Draft:
    def __init__(self, *, topology=None, ranks=range(8)):
        self.topology = topology or {"dp": 4, "tp": 4, "ep": 16}
        self.rank_ids = {rank: local for local, rank in enumerate(ranks)}
        self.online_bank = object()
        self.resident = SimpleNamespace(pinned={})


def config(last_layers=1):
    return SimpleNamespace(last_layers=last_layers)


def native():
    return SimpleNamespace(weights=object(), runtime=object())


def test_production_factory_fails_before_a_long_serving_run():
    report = production_readiness()
    assert report["production_ready"] is False
    assert "fixed-route mHC/MoE reverse with EP16 expert exchange" in report["blockers"]
    with pytest.raises(ProductionFactoryIncomplete, match="factory is incomplete"):
        create_primitives(draft=Draft(), native=native(), config=config())


@pytest.mark.parametrize("last_n", [1, 2, 4, 8])
def test_terminal_factory_accepts_the_declared_c1a_scan(last_n):
    value = DSV41ResidentPrimitives(
        draft=Draft(), native=native(), cluster=None, config=config(last_n),
    )
    assert value.plan.last_layers == last_n
    assert value.capabilities["implemented"][-1] == "exact K32 token-span slicing"


@pytest.mark.parametrize("last_n", [3, 20])
def test_terminal_factory_rejects_undeclared_last_n(last_n):
    with pytest.raises(ValueError, match="last-N 1/2/4/8"):
        DSV41ResidentPrimitives(
            draft=Draft(), native=native(), cluster=None, config=config(last_n),
        )


def test_loss_is_deferred_until_it_has_a_resident_tp_owner():
    value = DSV41ResidentPrimitives(
        draft=Draft(), native=native(), cluster=None, config=config(),
    )
    loss, gradient = value.loss_vjp(
        torch.randn(31, 64), torch.randn(7, 64), 7, "tv",
    )
    assert loss.valid == 7 and loss.kind == "tv"
    assert gradient.loss is loss
    assert loss.student.shape == (7, 64) and loss.student.dtype == torch.bfloat16


def test_exact_k32_span_is_mandatory():
    assert DSV41ResidentPrimitives._span({"token_begin": 12, "token_end": 44}) == (12, 44)
    with pytest.raises(ValueError, match="exact captured K32"):
        DSV41ResidentPrimitives._span({"token_begin": 12, "token_end": 43})
