from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_dsv41_resident_primitives import (
    DSV41ResidentPrimitives,
    ProductionFactoryIncomplete,
    _HostScalar,
    _RankGradient,
    create_primitives,
    production_readiness,
)
from blockspec.pypto_dsv41_runtime import MutableDraftHeadBank


class Draft:
    def __init__(self, *, topology=None, ranks=range(8)):
        self.topology = topology or {"dp": 4, "tp": 4, "ep": 16}
        self.rank_ids = {rank: local for local, rank in enumerate(ranks)}
        self.online_bank = object()
        self.resident = SimpleNamespace(pinned={})


class FakeResident:
    def __init__(self, local_count=8):
        self.pinned = {}
        self.local_count = local_count

    def pin(self, value, ranks, *, readonly, name):
        for rank in ranks:
            self.pinned[rank, value.untyped_storage()._cdata] = {
                "source": value, "name": name, "readonly": readonly,
            }

    @staticmethod
    def workspace(shape, dtype, rank, *, fill=0):
        return torch.full(shape, fill, dtype=dtype)

    @staticmethod
    def handle(value, rank, *, readonly=False):
        return value

    @staticmethod
    def tensor(value, rank):
        return value

    @staticmethod
    def copy_tensor(destination, source, rank):
        destination.copy_(source)
        return destination

    @staticmethod
    def read_output(value, rank):
        return value.clone()

    def map_ranks(self, function):
        return [function(rank) for rank in range(self.local_count)]

    def exchange(self, stage, packets, collective, group):
        assert collective == "all_reduce"
        width = len(packets)
        if group == "tp":
            groups = [range(base, base + 4) for base in range(0, width, 4)]
        elif group == "dp":
            groups = [[lane + 4 * dp for dp in range(width // 4)] for lane in range(4)]
        else:
            raise AssertionError(group)
        output = [{} for _ in packets]
        for members in groups:
            for case in packets[members[0]]:
                output_value = sum(
                    (packets[rank][case]["value"].float() for rank in members),
                    torch.zeros_like(packets[members[0]][case]["value"], dtype=torch.float32),
                )
                dtype = packets[members[0]][case]["value"].dtype
                for rank in members:
                    output[rank][case] = {"value": output_value.to(dtype)}
        return output, []


class ExecutingDraft(Draft):
    def __init__(self, last_n=1, ranks=range(8)):
        super().__init__(ranks=ranks)
        self.resident = FakeResident(len(self.rank_ids))
        layers = tuple(range(40 - last_n, 40))
        initial = {}
        for rank in self.rank_ids:
            for layer in layers:
                initial[rank, layer] = {
                    name: torch.full(
                        (64,), rank + layer / 100,
                        dtype=torch.float32 if name == "sink" else torch.bfloat16,
                    )
                    for name in ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "sink", "wo_a", "wo_b")
                }
        self.online_bank = MutableDraftHeadBank(
            self.resident, self.rank_ids, initial, layers,
        )

    def execute(self, key, rank, factory, inputs):
        operation = key[1]
        if operation == "adam":
            gradient = inputs["gradient"].float() * float(inputs["clip"])
            mean = .9 * inputs["first"] + .1 * gradient
            variance = .999 * inputs["second"] + .001 * gradient.square()
            inputs["next_first"].copy_(mean)
            inputs["next_second"].copy_(variance)
            update = inputs["rate"] * mean / inputs["correction1"] / (
                (variance / inputs["correction2"]).sqrt() + 1e-8
            )
            inputs["next_master"].copy_(inputs["master"] - update)
        elif operation == "publish":
            inputs["execution"].copy_(inputs["master"].to(inputs["execution"].dtype))
        else:
            raise AssertionError(f"unexpected fake execution {key}")


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
    assert "exact K32 token-span slicing" in value.capabilities["implemented"]


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


def test_objective_name_matches_public_online_configuration():
    value = DSV41ResidentPrimitives(
        draft=Draft(), native=native(), cluster=None, config=config(),
    )
    loss, _ = value.loss_vjp(
        torch.randn(31, 64), torch.randn(3, 64), 3, "forward_kl",
    )
    assert loss.kind == "forward_kl"
    with pytest.raises(ValueError, match="forward_kl"):
        value.loss_vjp(torch.randn(31, 64), torch.randn(3, 64), 3, "kl")


def test_terminal_vjp_exposes_only_four_real_premix_columns():
    padded = torch.empty(32, 16)
    assert padded[:, :4].shape == (32, 4)


def test_exact_k32_span_is_mandatory():
    assert DSV41ResidentPrimitives._span({"token_begin": 12, "token_end": 44}) == (12, 44)
    with pytest.raises(ValueError, match="exact captured K32"):
        DSV41ResidentPrimitives._span({"token_begin": 12, "token_end": 43})


def test_persistent_adam_and_resident_publication_use_device_descriptors():
    draft = ExecutingDraft()
    value = DSV41ResidentPrimitives(
        draft=draft, native=native(), cluster=None, config=config(),
    )
    gradients = {}
    for full_name in value.plan.trainable_names():
        layer = int(full_name.split(".")[1])
        short = full_name.split(".attn.", 1)[1]
        gradients[full_name] = _RankGradient(layer, {
            rank: torch.full_like(value._adam[rank, layer, short].master, .25)
            for rank in value.ranks
        })
    value._last_norm = 1.
    value.adamw(gradients, rate=1e-5, step=1, clip=_HostScalar(1.))
    state = value.publication_state(value.plan.layers)
    assert set(state.public_names) == set(value.plan.trainable_names())
    assert set(state.operands) == {(rank, 39) for rank in range(8)}
    assert all(set(row) == {"wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "sink", "wo_a", "wo_b"}
               for row in state.operands.values())
    assert value._last_step == 1 and len(state.digest) == 64
    receipt = draft.online_bank.stage_resident(state, 1)
    draft.online_bank.activate(1)
    draft.online_bank.commit(1)
    assert receipt.phase == "prepared" and draft.online_bank.active_slot == 1
    for rank in value.ranks:
        for name, source in state.operands[rank, 39].items():
            torch.testing.assert_close(
                draft.online_bank.host[1, rank, 39, name], source,
            )


def test_tp4_and_dp4_collectives_keep_replication_and_sharding_exact():
    draft = ExecutingDraft(ranks=range(16))
    value = DSV41ResidentPrimitives(
        draft=draft, native=native(), cluster=None, config=config(),
    )
    local = {}
    for rank in range(4):
        local[rank] = {
            name: torch.full_like(value._adam[rank, 39, name].master, rank + 1.)
            for name in value._execution_names()
        }
    assembled = value.tp_assemble_gradients(39, local)
    replicated = value._replicated_tp_parameters()
    assert all(torch.equal(assembled[name].values[rank], torch.full((64,), 10.))
               for name in replicated for rank in range(4))
    assert all(torch.equal(assembled[name].values[rank], torch.full((64,), rank + 1.))
               for name in set(value._execution_names()) - replicated for rank in range(4))
    complete = {
        f"layers.39.attn.{name}": gradient for name, gradient in assembled.items()
    }
    reduced = value.dp_all_reduce(complete)
    for full_name, gradient in reduced.items():
        short = full_name.split(".attn.", 1)[1]
        for rank in range(16):
            expected = 10. if short in replicated else rank % 4 + 1.
            torch.testing.assert_close(gradient.values[rank], torch.full((64,), expected))
