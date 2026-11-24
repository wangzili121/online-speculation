from types import SimpleNamespace

import pytest
import torch

from blockspec.dsv41_device_update import SuffixGradient
from blockspec.dsv41_serving_online import DSV41SequenceSpan
from blockspec.pypto_dsv41_resident_primitives import (
    DSV41ResidentPrimitives,
    ProductionFactoryIncomplete,
    _DeferredLogitsGradient,
    _HostScalar,
    _RankGradient,
    _FrozenMoEEnvelope,
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


class TailDraft(Draft):
    """Fake device runtime for the exact frozen-tail envelope."""

    def __init__(self):
        super().__init__(ranks=range(4))
        self.resident = FakeResident(4)
        self.config = {"routed_scaling_factor": 1.5}
        self.events = []

    def execute(self, key, rank, factory, inputs):
        del factory
        operation = key[1]
        self.events.append((operation, rank))
        if operation == "ffn-hc-post-data-vjp":
            dy = inputs["output_gradient"].view(32, 4, 5120).float()
            inputs["value_gradient"].copy_(
                torch.einsum("tjd,tj->td", dy, inputs["post"].float()).bfloat16()
            )
            inputs["residual_gradient"].copy_(
                torch.einsum("tjd,tij->tid", dy, inputs["combination"].view(32, 4, 4).float())
                .bfloat16().view(128, 5120)
            )
        elif operation == "ffn-hc-post-gate-vjp":
            dy = inputs["output_gradient"].view(32, 4, 5120).float()
            inputs["post_gradient"][:, :4].copy_(
                torch.einsum("tjd,td->tj", dy, inputs["value"].float())
            )
            inputs["combination_gradient"][:, :4].copy_(
                torch.einsum(
                    "tid,tjd->tij", inputs["residual"].view(32, 4, 5120).float(), dy,
                ).view(128, 4)
            )
        elif operation == "moe-combine-vjp":
            dy = inputs["output_gradient"].float()
            experts = inputs["expert_outputs"].view(32, 6, 5120).float()
            inputs["shared_gradient"].copy_(dy.bfloat16())
            inputs["expert_gradient"].copy_(
                (dy[:, None] * inputs["route_weights"].float()[..., None])
                .bfloat16().view(192, 5120)
            )
            inputs["weight_gradient"][:, :6].copy_(
                torch.einsum("td,trd->tr", dy, experts)
            )
        elif operation == "router-logits-vjp":
            logits = inputs["router_logits"].float()
            ids = inputs["route_ids"].long()
            logits = logits.gather(1, ids)
            scores = torch.nn.functional.softplus(logits).sqrt()
            upstream = inputs["weight_gradient"][:, :6]
            total = scores.sum(1, keepdim=True)
            selected = inputs["route_scale"] / total * (
                upstream - (upstream * scores).sum(1, keepdim=True) / total
            )
            selected = selected * logits.sigmoid() / (2 * scores)
            inputs["router_gradient"].zero_().scatter_(1, ids, selected)
        else:
            raise AssertionError(f"unexpected fake tail execution {key}")


def config(last_layers=1):
    return SimpleNamespace(last_layers=last_layers)


def native():
    return SimpleNamespace(weights=object(), runtime=object())


def test_production_factory_reports_the_complete_synchronized_path():
    report = production_readiness()
    assert report["production_ready"] is True
    assert report["blockers"] == ()
    assert "EP16 frozen routed-expert input VJP dispatch and return" in report["implemented"]


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


def test_frozen_tail_envelope_uses_captured_routes_and_exact_k32_state():
    torch.manual_seed(7)
    draft = TailDraft()
    value = DSV41ResidentPrimitives(
        draft=draft, native=native(), cluster=None, config=config(),
    )
    captured = {
        "token_begin": 0, "token_end": 32,
        "changed": torch.randn(128, 5120).bfloat16(),
        "post": torch.randn(32, 4),
        "comb": torch.randn(128, 4),
        "combined": torch.randn(32, 5120).bfloat16(),
        "expert_outputs": torch.randn(192, 5120).bfloat16(),
        "route_weights": torch.softmax(torch.randn(32, 6), -1),
        "route_ids": torch.stack([torch.randperm(384)[:6] for _ in range(32)]).int(),
        "router_logits": torch.randn(32, 384),
    }
    output = SuffixGradient(
        torch.randn(128, 5120).bfloat16(), torch.randn(32, 4),
    )
    result = value._reverse_frozen_moe_envelope(
        layer=39, rank=0, captured=captured, state=None, output_gradient=output,
    )
    assert isinstance(result, _FrozenMoEEnvelope)
    assert result.shared.shape == (32, 5120)
    assert result.expert.shape == (192, 5120)
    assert result.router.shape == (32, 384)
    assert result.residual.shape == (128, 5120)
    assert result.post.shape == (32, 4)
    assert result.combination.shape == (128, 4)
    assert result.pre is output.pre_mix
    assert draft.events == [
        ("ffn-hc-post-data-vjp", 0), ("ffn-hc-post-gate-vjp", 0),
        ("moe-combine-vjp", 0), ("router-logits-vjp", 0),
    ]


def test_public_frozen_tail_never_returns_a_partial_gradient():
    draft = TailDraft()
    value = DSV41ResidentPrimitives(
        draft=draft, native=native(), cluster=None, config=config(),
    )
    captured = {
        "token_begin": 0, "token_end": 32,
        "changed": torch.zeros(128, 5120, dtype=torch.bfloat16),
        "post": torch.zeros(32, 4), "comb": torch.zeros(128, 4),
        "combined": torch.zeros(32, 5120, dtype=torch.bfloat16),
        "expert_outputs": torch.zeros(192, 5120, dtype=torch.bfloat16),
        "route_weights": torch.full((32, 6), 1 / 6),
        "route_ids": torch.arange(6, dtype=torch.int32).repeat(32, 1),
        "router_logits": torch.zeros(32, 384),
    }
    gradient = SuffixGradient(
        torch.zeros(128, 5120, dtype=torch.bfloat16), torch.zeros(32, 4),
    )
    with pytest.raises(ProductionFactoryIncomplete, match="EP16 expert"):
        value.reverse_frozen_tail(
            layer=39, rank=0, captured=captured, state=None, output_gradient=gradient,
        )


class AttentionBatchHarness(DSV41ResidentPrimitives):
    """Exercise the TP4 orchestration while replacing only device launches."""

    def __init__(self):
        self.ranks = (0, 1, 2, 3)
        self.events = []
        self.draft = SimpleNamespace(native=lambda _name: SimpleNamespace(), rank_ids={rank: rank for rank in self.ranks})
        self.resident = FakeResident(4)

    @staticmethod
    def _workspace(_rank, shape, dtype=torch.bfloat16, *, fill=None):
        return torch.full(shape, 0 if fill is None else fill, dtype=dtype)

    def _execute(self, key, rank, factory, **inputs):
        del factory
        self.events.append((key[0], rank))
        if "scatter" in key[0]:
            inputs["output"].copy_(inputs["a"])
        elif key[0] == "attn-output-gradient-unpack":
            capacity = len(inputs["output"])
            for row in range(inputs["active"]):
                lane, local = divmod(row, inputs["physical"])
                inputs["output"][row].copy_(inputs["packet"][lane * capacity + local])
        elif key[0] == "attn-input-gradient-pack":
            inputs["packed"].zero_()
            capacity = len(inputs["value"])
            for row in range(inputs["active"]):
                lane, local = divmod(row, inputs["physical"])
                inputs["packed"][lane * capacity + local].copy_(inputs["value"][row])
        elif key[0] == "attn-input-gradient-widen":
            inputs["output"].copy_(inputs["value"].float())
        elif key[0] == "attn-input-gradient-narrow":
            inputs["output"].copy_(inputs["value"].bfloat16())
        else:
            raise AssertionError(key)

    def _reverse_attention_post(self, layer, rank, captured, output_gradient):
        del layer, captured
        self.events.append(("post", rank))
        return {
            "value": torch.full((32, 5120), rank + 1, dtype=torch.bfloat16),
            "residual": output_gradient.hidden,
            "post": output_gradient.pre_mix,
            "combination": output_gradient.pre_mix,
            "pre": output_gradient.pre_mix,
        }

    def _tp_collective(self, values, collective, *, stage):
        del stage
        self.events.append((collective, tuple(values)))
        if collective == "all_gather":
            joined = torch.cat([values[rank] for rank in sorted(values)])
            return {rank: joined.clone() for rank in values}
        if collective == "reduce_scatter":
            total = sum(value.float() for value in values.values())
            chunks = total.chunk(4)
            return {rank: chunks[index] for index, rank in enumerate(values)}
        raise AssertionError(collective)

    def _reverse_c1a_rank(self, layer, rank, captured, contribution_gradient):
        del layer, captured
        expected = torch.arange(1, 5).repeat_interleave(8)
        torch.testing.assert_close(contribution_gradient[:, 0].float(), expected.float())
        self.events.append(("c1a", rank, tuple(contribution_gradient.shape)))
        names = self._execution_names()
        return torch.full((32, 5120), rank + 1, dtype=torch.bfloat16), {
            name: torch.ones(1) for name in names
        }

    def _reverse_attention_pre(self, layer, rank, captured, shell, normalized_gradient):
        del layer, captured, shell
        assert normalized_gradient[:8].eq(10).all()
        assert not normalized_gradient[8:].any()
        self.events.append(("pre", rank, tuple(normalized_gradient.shape)))
        return SuffixGradient(
            torch.full((128, 5120), rank + 1, dtype=torch.bfloat16),
            torch.full((32, 4), rank + 1, dtype=torch.float32),
        )


def test_c1a_owner_batch_reverses_both_tp_collectives_once():
    value = AttentionBatchHarness()
    captures = {
        rank: {
            "token_begin": 0, "token_end": 32,
            "sp_span": DSV41SequenceSpan(0, rank * 8, 8),
            "record": {"SP_physical_rows": 8, "positions": list(range(32))},
            "attention_scattered": torch.zeros(32, 5120, dtype=torch.bfloat16),
        }
        for rank in value.ranks
    }
    gradients = {
        rank: SuffixGradient(
            torch.ones(128, 5120, dtype=torch.bfloat16),
            torch.ones(32, 4),
        )
        for rank in value.ranks
    }
    result = value.reverse_draft_attention_owner_batch(
        layer=39, ranks=value.ranks, captured=captures,
        output_gradients=gradients,
    )
    assert set(result) == set(value.ranks)
    assert [event[0] for event in value.events].count("all_gather") == 1
    assert [event[0] for event in value.events].count("reduce_scatter") == 1
    assert [event[:2] for event in value.events if event[0] == "c1a"] == [
        ("c1a", rank) for rank in value.ranks
    ]
    assert all(set(result[rank][1]) == set(value._execution_names()) for rank in value.ranks)


class TerminalHarness(AttentionBatchHarness):
    def __init__(self):
        super().__init__()
        self.native = SimpleNamespace(
            weights=SimpleNamespace(head_for_rank=lambda rank: torch.full(
                (5120, 64), rank + 1, dtype=torch.bfloat16,
            )),
        )

    @staticmethod
    def _pinned(rank, name, shape):
        assert name == "norm.weight"
        return torch.ones(shape, dtype=torch.bfloat16)

    def _loss_on_rank(self, rank, token):
        return token.student, torch.zeros(32, 8)

    def _tp_collective(self, values, collective, *, stage):
        assert collective == "all_reduce"
        reduced = sum(values.values())
        if stage == "lm-head-sp-collapse":
            expected = torch.arange(1, 5).repeat_interleave(8).float()
            torch.testing.assert_close(reduced[:, 0].float(), expected)
        return {rank: reduced for rank in values}

    def _execute(self, key, rank, factory, **inputs):
        assert all(value.is_contiguous() for value in inputs.values() if torch.is_tensor(value))
        op = key[0]
        if op == "final-collapse-forward":
            values = inputs["streams"].view(32, 4, 5120).float()
            inputs["collapsed"].copy_((values * inputs["premix"].unsqueeze(-1)).sum(1))
        elif op == "final-rms-forward":
            inputs["y"].copy_(inputs["x"])
        elif op == "lm-head-input-vjp":
            inputs["output"].copy_(inputs["a"].float() @ inputs["b"].float().T)
        elif op == "final-rms-input-vjp":
            # Identity VJP lets this test isolate the row layout.
            inputs["dx"].copy_(inputs["dy"])
        elif op == "final-collapse-vjp":
            inputs["stream_gradient"].copy_(
                (inputs["output_gradient"][:, None] * inputs["premix"][:, :, None]).view(128, 5120),
            )
            inputs["premix_gradient"][:, :4].copy_(inputs["output_gradient"][:, :4])
        else:
            raise AssertionError(key)


def test_lm_head_vjp_routes_all_vocab_shards_back_to_the_owning_sp_rows():
    terminal = TerminalHarness()
    rows = {}
    for rank in terminal.ranks:
        hidden = torch.zeros(32, 4, 5120, dtype=torch.bfloat16)
        hidden[:8, 0] = rank + 1
        premix = torch.zeros(32, 4)
        premix[:8, 0] = 1.
        rows[rank] = {
            "hidden": hidden, "pre_mix": premix,
            "sp_span": DSV41SequenceSpan(0, rank * 8, 8),
        }
    logits = torch.zeros(32, 256, dtype=torch.bfloat16)
    for lane in terminal.ranks:
        logits[lane * 8:(lane + 1) * 8, lane * 64] = 1.
    gradient = _DeferredLogitsGradient(SimpleNamespace(student=logits, value=None))
    result = terminal.lm_head_vjp(gradient, rows)
    for rank in terminal.ranks:
        assert result.pre_mix[rank].is_contiguous()
        assert result.pre_mix[rank][:8].eq(rank + 1).all()
        assert not result.pre_mix[rank][8:].any()
        assert result.hidden[rank].view(32, 4, 5120)[:8, 0].eq(rank + 1).all()


class SymbolicTensor:
    """Shape-only resident descriptor used to test EP16 orchestration."""

    def __init__(self, shape, dtype=torch.bfloat16, name="tensor"):
        self.shape, self.dtype, self.name = tuple(shape), dtype, name

    def __getitem__(self, item):
        first = item[0] if isinstance(item, tuple) else item
        if not isinstance(first, slice) or first.step not in (None, 1):
            raise AssertionError(f"unsupported symbolic slice {item!r}")
        begin = 0 if first.start is None else first.start
        end = self.shape[0] if first.stop is None else first.stop
        return SymbolicTensor((end - begin, *self.shape[1:]), self.dtype, self.name)


class SymbolicTransport:
    def __init__(self):
        self.events = []

    def exchange(self, stage, packets, collective, group):
        self.events.append((stage, collective, group, len(packets)))
        return packets, []


class SymbolicExpertWeights:
    def __init__(self):
        self.calls = []

    def for_layer(self, layer):
        def expert(owner, expert_id):
            self.calls.append((layer, owner, expert_id))
            return tuple(SymbolicTensor((1,), name=f"weight-{index}") for index in range(4))
        return {"expert": expert}


class SymbolicEPPrimitives(DSV41ResidentPrimitives):
    """Exercise the real owner/peer scheduler without allocating packet slabs."""

    def __init__(self, ranks):
        draft = Draft(ranks=ranks)
        draft.resident = SimpleNamespace()
        self.transport = SymbolicTransport()
        self.weights = SymbolicExpertWeights()
        super().__init__(
            draft=draft,
            native=SimpleNamespace(
                weights=self.weights, runtime=object(), transport=self.transport,
            ),
            cluster=None,
            config=config(),
        )
        self.events = []
        self.workspace_number = 0

    def _workspace(self, rank, shape, dtype=torch.bfloat16):
        self.workspace_number += 1
        return SymbolicTensor(shape, dtype, f"workspace-{rank}-{self.workspace_number}")

    def _execute(self, key, rank, factory, **inputs):
        del factory
        self.events.append((key[0], rank, inputs))

    def _reverse_frozen_moe_envelope(self, **options):
        rank = options["rank"]
        self.events.append(("envelope", rank, options))
        return _FrozenMoEEnvelope(
            SymbolicTensor((32, 5120)), SymbolicTensor((192, 5120)),
            SymbolicTensor((32, 384), torch.float32),
            SymbolicTensor((128, 5120)),
            SymbolicTensor((32, 4), torch.float32),
            SymbolicTensor((128, 4), torch.float32),
            SymbolicTensor((32, 4), torch.float32),
        )

    def _reverse_frozen_tail_finish(self, **options):
        rank = options["rank"]
        assert options["routed_input_gradient"].shape == (32, 5120)
        self.events.append(("finish", rank, options))
        return SuffixGradient(
            SymbolicTensor((128, 5120)),
            SymbolicTensor((32, 4), torch.float32),
        )


def symbolic_capture(rank):
    return {
        "token_begin": 0, "token_end": 32,
        "normed": SymbolicTensor((32, 5120), name=f"normed-{rank}"),
        "route_ids": SymbolicTensor((32, 6), torch.int32, f"routes-{rank}"),
    }


def reverse_contract():
    return {
        "update_number": 7,
        "reverse": {
            "protocol": "dsv41-synchronized-reverse-v1",
            "layers": [39],
        },
    }


def test_owner_batch_runs_fixed_ep16_packets_and_all_local_experts():
    value = SymbolicEPPrimitives(range(8))
    ranks = tuple(range(4))
    captured = {rank: symbolic_capture(rank) for rank in ranks}
    gradients = {
        rank: SuffixGradient(
            SymbolicTensor((128, 5120)), SymbolicTensor((32, 4), torch.float32),
        )
        for rank in ranks
    }
    result = value.reverse_frozen_tail_owner_batch(
        layer=39, ranks=ranks, captured=captured,
        states={rank: object() for rank in ranks},
        output_gradients=gradients, synchronization=reverse_contract(),
    )

    assert set(result) == set(ranks)
    assert value.transport.events == [
        ("online-frozen-tail-step7-layer39-dispatch-v1", "all_to_all", "ep", 8),
        ("online-frozen-tail-step7-layer39-return-v1", "all_to_all", "ep", 8),
    ]
    packs = [event for event in value.events if event[0] == "ep-vjp-dispatch-pack"]
    assert [event[2]["active"] for event in packs] == [32] * 4 + [0] * 4
    assert len([event for event in value.events if event[0] == "ep-vjp-dispatch-unpack"]) == 8
    assert len([event for event in value.events if event[0] == "routed-expert-input-vjp"]) == 8 * 24
    assert len([event for event in value.events if event[0] == "ep-vjp-return-pack"]) == 8
    assert len([event for event in value.events if event[0] == "ep-vjp-source-restore"]) == 4
    assert [(owner, expert) for _, owner, expert in value.weights.calls] == [
        (owner, owner * 24 + local) for owner in range(8) for local in range(24)
    ]


def test_peer_participates_in_both_collectives_and_all_owned_experts():
    value = SymbolicEPPrimitives(range(8, 16))
    value.participate_frozen_tail_peer(
        layer=39, synchronization=reverse_contract(),
    )

    assert value.transport.events == [
        ("online-frozen-tail-step7-layer39-dispatch-v1", "all_to_all", "ep", 8),
        ("online-frozen-tail-step7-layer39-return-v1", "all_to_all", "ep", 8),
    ]
    packs = [event for event in value.events if event[0] == "ep-vjp-dispatch-pack"]
    assert [event[2]["active"] for event in packs] == [0] * 8
    assert len([event for event in value.events if event[0] == "routed-expert-input-vjp"]) == 8 * 24
    assert not [event for event in value.events if event[0] in {"ep-vjp-source-restore", "finish"}]
    assert [(owner, expert) for _, owner, expert in value.weights.calls] == [
        (owner, owner * 24 + local) for owner in range(8, 16) for local in range(24)
    ]


def test_ep16_reverse_rejects_a_stale_or_partial_contract():
    value = SymbolicEPPrimitives(range(8))
    with pytest.raises(ValueError, match="contract is inconsistent"):
        value.participate_frozen_tail_peer(
            layer=39,
            synchronization={"reverse": {"protocol": "old", "layers": [39]}},
        )
