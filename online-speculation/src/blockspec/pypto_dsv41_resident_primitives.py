"""Resident PyPTO primitives for Orthrus-DSV4.1 online continuation.

This module is deliberately strict: serving never falls back to Torch autograd
or to host-side model math.  The first production slice implemented here owns
the complete terminal reverse path

    full-vocabulary TV/KL -> TP4 LM head -> final RMSNorm -> final mHC collapse.

All intermediate gradients are resident tensors.  CPU tensors are used only
for the verifier/student logits which the current serving patch already makes
an explicit host boundary.  The C1A layer/MoE reverse is not wired here yet;
``create_primitives`` therefore refuses production binding instead of exposing
a backend which would fail after a long dual-node model startup.
"""

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import math

import torch

from .dsv41_device_update import SuffixGradient
from .dsv41_suffix import DSV41SuffixPlan


ROWS = 32
HIDDEN = 5120
HC_COPIES = 4
SUPPORTED_LAST_N = (1, 2, 4, 8)


class ProductionFactoryIncomplete(RuntimeError):
    """Raised before serving when the exact resident graph is incomplete."""


@dataclass
class _DeferredLoss:
    student: torch.Tensor
    teacher: torch.Tensor
    valid: int
    kind: str
    value: object | None = None


@dataclass(frozen=True)
class _DeferredLogitsGradient:
    loss: _DeferredLoss


@dataclass(frozen=True)
class _ResidentScalar:
    value: torch.Tensor
    rank: int


@dataclass(frozen=True)
class _HostScalar:
    value: float


@dataclass(frozen=True)
class _RankGradient:
    layer: int
    values: dict


@dataclass(frozen=True)
class _FrozenMoEEnvelope:
    """Resident gradients at the fixed-route expert boundary.

    Every value is a device descriptor.  The envelope deliberately contains
    no host tensor math: EP16 dispatch consumes ``expert``, while the shared
    expert and router branches consume ``shared`` and ``router``.  The mHC
    fields are retained for the exact FFN hc-pre reverse after those three
    branches have been reduced to the normalized FFN input.
    """

    shared: object
    expert: object
    router: object
    residual: object
    post: object
    combination: object
    pre: object


@dataclass
class _AdamState:
    rank: int
    layer: int
    public_name: str
    execution_name: str
    execution_dtype: torch.dtype
    master: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    candidate_master: torch.Tensor
    candidate_first: torch.Tensor
    candidate_second: torch.Tensor
    execution: torch.Tensor


def production_readiness():
    """Machine-readable gate used by preflight and deployment scripts."""
    return {
        "production_ready": False,
        "implemented": (
            "full-vocabulary TV/KL VJP",
            "TP4 sharded LM-head input VJP",
            "final RMSNorm input VJP",
            "final mHC-collapse VJP",
            "exact K32 token-span slicing",
            "TP4/DP4 resident gradient collectives",
            "persistent FP32 Adam state",
            "D2D ResidentHeadState publication",
            "FFN mHC-post and fixed-route MoE combine/router VJP",
            "synchronized dual-node owner/peer reverse orchestration",
        ),
        "blockers": (
            "resident C1A draft-attention parameter/input VJP orchestration",
            "grouped W8A8 shared/routed expert input VJP with EP16 owner/peer binding",
        ),
    }


class DSV41ResidentPrimitives:
    """Exact resident terminal VJP bound to one initialized generation.

    The class intentionally implements the public backend method names even
    for unfinished stages.  Unfinished methods raise an explicit production
    blocker; they never return zeros or detach a gradient silently.
    """

    def __init__(self, *, draft, native, cluster, config):
        plan = DSV41SuffixPlan(config.last_layers)
        if config.last_layers not in SUPPORTED_LAST_N:
            raise ValueError("production online update supports last-N 1/2/4/8")
        topology = dict(getattr(draft, "topology", {}))
        if topology != {"dp": 4, "tp": 4, "ep": 16, "pp": 1, "pcp": 1, "dcp": 1}:
            # Some tests/minimal runtimes omit the unit axes.  Normalize only
            # those axes; DP/TP/EP are part of the numerical contract.
            required = {"dp": 4, "tp": 4, "ep": 16}
            if any(topology.get(name) != value for name, value in required.items()):
                raise ValueError("DSV4.1 resident online update requires DP4/TP4/EP16")
        if getattr(draft, "online_bank", None) is None:
            raise RuntimeError("enable the resident online head bank before creating primitives")
        if getattr(native, "weights", None) is None or getattr(native, "runtime", None) is None:
            raise RuntimeError("resident target weights/runtime are not initialized")
        ranks = tuple(sorted(draft.rank_ids))
        if not ranks or any(rank not in range(16) for rank in ranks):
            raise ValueError("draft rank ownership must use global EP16 ranks")
        self.draft, self.native, self.cluster, self.config = draft, native, cluster, config
        self.plan, self.ranks = plan, ranks
        self.resident = draft.resident
        self._scalar_reads = 0
        self._adam = {}
        self._last_step = 0
        self._last_norm = None
        self._publication_digest = None
        # Focused terminal-kernel tests may deliberately bind only the loss
        # and LM-head path.  A production bank always exposes both methods and
        # is initialized eagerly, so malformed resident state still fails at
        # bootstrap rather than at the first update.
        bank = self.draft.online_bank
        if hasattr(bank, "operands") and hasattr(bank, "host"):
            self._initialize_optimizer_state()

    @staticmethod
    def _execution_names():
        return {
            "wq_a.weight": "wq_a", "q_norm.weight": "q_norm",
            "wq_b.weight": "wq_b", "wkv.weight": "wkv",
            "kv_norm.weight": "kv_norm", "attn_sink": "sink",
            "wo_a.weight": "wo_a", "wo_b.weight": "wo_b",
        }

    @staticmethod
    def _replicated_tp_parameters():
        return {"wq_a.weight", "q_norm.weight", "wkv.weight", "kv_norm.weight"}

    def _pin_mutable(self, rank, name, value):
        value = value.detach().cpu().contiguous().clone()
        self.resident.pin(
            value, (self._local(rank),), readonly=False,
            name=f"online-optimizer/rank{rank}/{name}",
        )
        return value

    def _initialize_optimizer_state(self):
        if self._adam:
            return
        names = self._execution_names()
        bank = self.draft.online_bank
        for rank in self.ranks:
            for layer in self.plan.layers:
                operands = bank.operands(rank, layer)
                for public, execution_name in names.items():
                    source = operands[execution_name]
                    # Resolve the authoritative host descriptor retained by
                    # MutableDraftHeadBank.  Optimizer state must never be
                    # initialized by reading execution HBM back to the host.
                    host = bank.host[
                        bank.active_slot, rank, layer, execution_name
                    ]
                    if source.shape != host.shape or source.dtype != host.dtype:
                        raise ValueError("active draft bank changed its execution ABI")
                    prefix = f"layer{layer}/{execution_name}"
                    master = self._pin_mutable(rank, prefix + "/master", host.float())
                    first = self._pin_mutable(rank, prefix + "/first", torch.zeros_like(master))
                    second = self._pin_mutable(rank, prefix + "/second", torch.zeros_like(master))
                    candidate_master = self._pin_mutable(
                        rank, prefix + "/candidate-master", torch.zeros_like(master),
                    )
                    candidate_first = self._pin_mutable(
                        rank, prefix + "/candidate-first", torch.zeros_like(master),
                    )
                    candidate_second = self._pin_mutable(
                        rank, prefix + "/candidate-second", torch.zeros_like(master),
                    )
                    execution = self._pin_mutable(
                        rank, prefix + "/execution", torch.zeros_like(host),
                    )
                    self._adam[rank, layer, public] = _AdamState(
                        rank, layer, public, execution_name, host.dtype,
                        master, first, second, candidate_master,
                        candidate_first, candidate_second, execution,
                    )

    def _require_optimizer_state(self):
        if not self._adam:
            raise RuntimeError("resident optimizer state is not initialized")

    def _local(self, rank):
        try:
            return self.draft.rank_ids[rank]
        except KeyError as error:
            raise ValueError(f"rank {rank} belongs to the peer host") from error

    def _workspace(self, rank, shape, dtype=torch.bfloat16):
        return self.resident.workspace(tuple(shape), dtype, self._local(rank))

    def _execute(self, key, rank, factory, **inputs):
        self.draft.execute(("online", *key), rank, factory, inputs)

    @staticmethod
    def _padded_logits(value, valid, vocabulary):
        if (
            not torch.is_tensor(value) or value.device.type != "cpu"
            or value.ndim != 2 or value.shape != (valid, vocabulary)
            or value.dtype != torch.bfloat16 or not value.is_contiguous()
        ):
            value = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            if value.shape != (valid, vocabulary):
                raise ValueError("full-vocabulary logits changed shape")
        result = torch.zeros((ROWS, vocabulary), dtype=torch.bfloat16)
        result[:valid].copy_(value)
        return result

    def _pinned(self, rank, name):
        """Resolve one immutable target operand by its native pin name."""
        local = self._local(rank)
        rows = [owner for (device, _), owner in self.resident.pinned.items()
                if device == local and owner.get("name") == name]
        if len(rows) != 1:
            raise RuntimeError(f"expected one resident {name!r} on rank {rank}, found {len(rows)}")
        source = rows[0]["source"]
        return self.resident.handle(source, local, readonly=True)

    @staticmethod
    def _span(row):
        begin, end = row.get("token_begin"), row.get("token_end")
        if type(begin) is not int or type(end) is not int or end - begin != ROWS:
            raise ValueError("terminal VJP requires the exact captured K32 token span")
        return begin, end

    def loss_vjp(self, logits, teacher, valid, kind):
        if kind not in ("tv", "forward_kl") or not 1 <= valid < ROWS:
            raise ValueError("resident loss requires TV/forward_kl and 1..31 authoritative rows")
        if logits.shape[0] < valid or teacher.shape != (valid, logits.shape[1]):
            raise ValueError("student and teacher full-vocabulary rows are not aligned")
        token = _DeferredLoss(
            logits[:valid].detach().cpu().to(torch.bfloat16).contiguous(),
            teacher.detach().cpu().to(torch.bfloat16).contiguous(),
            valid, kind,
        )
        return token, _DeferredLogitsGradient(token)

    def _loss_on_rank(self, rank, token):
        from .pypto_dsv41_online_programs import make_kl, make_tv

        vocabulary = token.student.shape[1]
        student = self._padded_logits(token.student, token.valid, vocabulary)
        teacher = self._padded_logits(token.teacher, token.valid, vocabulary)
        gradient = self._workspace(rank, (ROWS, vocabulary))
        losses = self._workspace(rank, (ROWS, 8), torch.float32)
        factory = make_tv if token.kind == "tv" else make_kl
        self._execute(
            ("loss", token.kind, vocabulary), rank,
            lambda: factory(ROWS, vocabulary),
            student=student, teacher=teacher, gradient=gradient, losses=losses,
            valid=token.valid, normalizer=token.valid,
        )
        return gradient, losses

    def _tp_all_reduce(self, values, *, stage):
        """Sum one complete local TP4 group without a host tensor boundary.

        In the two-host topology every TP group is node-local.  Calling the
        cluster-wide scheduling wrapper here would incorrectly require the
        non-owning host to enter a collective that belongs only to the owner
        replay.  We therefore use the already-created local TP communicators
        directly; DP/EP collectives still use the global transport later.
        """
        ranks = tuple(sorted(values))
        if len(ranks) != 4 or ranks != tuple(range(ranks[0] // 4 * 4, ranks[0] // 4 * 4 + 4)):
            raise ValueError("TP all-reduce requires one contiguous TP4 group")
        shape, dtype = values[ranks[0]].shape, values[ranks[0]].dtype
        if any(value.shape != shape or value.dtype != dtype for value in values.values()):
            raise ValueError("TP all-reduce operands changed shape or dtype")

        transport = getattr(self.native, "transport", None)
        if transport is None:
            packets = [
                {stage: {"value": values[rank]}}
                for rank in ranks
            ]
            result, _ = self.resident.exchange(stage, packets, "all_reduce", "tp")
            return {rank: result[index][stage]["value"] for index, rank in enumerate(ranks)}

        channels = transport._channels("tp")
        local_ranks = tuple(transport.plan.local_ranks)
        if not set(ranks) <= set(local_ranks):
            raise ValueError("a TP group may not cross hosts")
        prepared = {}
        for rank in ranks:
            local = self._local(rank)
            channel = channels[local]
            output = self._workspace(rank, shape, dtype)
            source = self.resident.tensor(values[rank], local)
            destination = self.resident.tensor(output, local)
            channel._validate_device_transfer(source, destination, "all_reduce")
            prepared[rank] = output, (source, destination, f"online/{stage}/rank{rank}")
        def transfer(local):
            rank = local_ranks[local]
            if rank in prepared:
                channels[local].exchange_device_batch([prepared[rank][1]], "all_reduce")
        self.resident.map_ranks(transfer)
        return {rank: prepared[rank][0] for rank in ranks}

    def lm_head_vjp(self, gradient, rows):
        """Run the terminal reverse on every TP owner without host gradients."""
        if not isinstance(gradient, _DeferredLogitsGradient):
            raise TypeError("LM-head VJP requires the resident loss token")
        ranks = tuple(sorted(rows))
        if len(ranks) != 4 or ranks != tuple(range(ranks[0] // 4 * 4, ranks[0] // 4 * 4 + 4)):
            raise ValueError("LM-head VJP requires one complete TP4 owner partition")
        vocabulary = gradient.loss.student.shape[1]
        if vocabulary % 4 or vocabulary // 4 % 64:
            raise ValueError("vocabulary must split into four aligned TP shards")

        from .pypto_dsv41_online_programs import make_linear, make_rms
        from .pypto_dsv41_tail_vjp import make_hc_collapse_forward, make_hc_collapse_vjp

        hidden_gradients, premix_gradients, normalized_gradients = {}, {}, {}
        terminal_states = {}
        local_vocabulary = vocabulary // 4
        for rank in ranks:
            row = rows[rank]
            begin, end = self._span(row)
            streams = row["hidden"][begin * HC_COPIES:end * HC_COPIES]
            premix = row["pre_mix"][begin:end]
            if streams.shape != (ROWS * HC_COPIES, HIDDEN) or premix.shape != (ROWS, HC_COPIES):
                raise ValueError("captured final mHC state has the wrong K32 geometry")

            collapsed = self._workspace(rank, (ROWS, HIDDEN))
            self._execute(
                ("final-collapse-forward",), rank,
                lambda: make_hc_collapse_forward(ROWS, HIDDEN),
                streams=streams, premix=premix, collapsed=collapsed,
            )
            norm = self._pinned(rank, "norm.weight").view(1, HIDDEN)
            normalized = self._workspace(rank, (ROWS, HIDDEN))
            inverse = self._workspace(rank, (ROWS, 8), torch.float32)
            self._execute(
                ("final-rms-forward",), rank,
                lambda: make_rms(ROWS, HIDDEN, epsilon=1e-20),
                x=collapsed, gamma=norm, y=normalized, inv=inverse,
            )
            terminal_states[rank] = streams, premix, collapsed, norm, inverse

            dlogits, losses = self._loss_on_rank(rank, gradient.loss)
            if gradient.loss.value is None:
                gradient.loss.value = _ResidentScalar(losses, rank)
            lane = rank % 4
            dlocal = dlogits[:, lane * local_vocabulary:(lane + 1) * local_vocabulary]
            head = self.native.weights.head_for_rank(rank)
            if head.shape != (HIDDEN, local_vocabulary):
                raise ValueError("native TP vocabulary head changed shape")
            dnormalized = self._workspace(rank, (ROWS, HIDDEN))
            self._execute(
                ("lm-head-input-vjp", local_vocabulary), rank,
                lambda: make_linear(
                    (ROWS, local_vocabulary), (HIDDEN, local_vocabulary),
                    transpose_b=True,
                ),
                a=dlocal, b=self.resident.handle(head, self._local(rank), readonly=True),
                output=dnormalized,
            )
            normalized_gradients[rank] = dnormalized

        normalized_gradients = self._tp_all_reduce(
            normalized_gradients, stage="lm-head-input-vjp",
        )
        for rank in ranks:
            streams, premix, collapsed, norm, inverse = terminal_states[rank]
            dcollapsed = self._workspace(rank, (ROWS, HIDDEN))
            self._execute(
                ("final-rms-input-vjp",), rank,
                lambda: make_rms(ROWS, HIDDEN, mode="input", epsilon=1e-20),
                x=collapsed, gamma=norm, inv=inverse,
                dy=normalized_gradients[rank], dx=dcollapsed,
            )
            dstreams = self._workspace(rank, (ROWS * HC_COPIES, HIDDEN))
            dpremix = self._workspace(rank, (ROWS, 16), torch.float32)
            self._execute(
                ("final-collapse-vjp",), rank,
                lambda: make_hc_collapse_vjp(ROWS, HIDDEN),
                streams=streams, premix=premix, output_gradient=dcollapsed,
                stream_gradient=dstreams, premix_gradient=dpremix,
            )
            hidden_gradients[rank] = dstreams
            premix_gradients[rank] = dpremix[:, :HC_COPIES]
        return SuffixGradient(hidden_gradients, premix_gradients)

    @staticmethod
    def _unfinished(stage):
        blockers = production_readiness()["blockers"]
        raise ProductionFactoryIncomplete(f"{stage} is not production-wired: {'; '.join(blockers)}")

    def reverse_frozen_tail(self, **options):
        # Build the exact resident boundary first.  The remaining EP16 expert
        # VJP and FFN hc-pre reverse are intentionally still gated below; do
        # not return a partial SuffixGradient to the learner.
        self._reverse_frozen_moe_envelope(**options)
        self._unfinished("EP16 expert and FFN mHC-pre reverse")

    def _reverse_frozen_moe_envelope(
        self, *, layer, rank, captured, state, output_gradient,
    ):
        """Reverse FFN hc-post and the immutable combine/router decisions.

        This is the first exact half of ``reverse_frozen_tail``.  It consumes
        only state captured by the same K32 forward: no Top-K recomputation,
        no changed expert order and no current-version model state.
        """
        del state
        if layer not in self.plan.layers or rank not in self.ranks:
            raise ValueError("frozen-tail replay is outside the selected resident suffix")
        begin, end = self._span(captured)
        if not isinstance(output_gradient, SuffixGradient):
            raise TypeError("frozen tail requires a suffix gradient")
        hidden_gradient, pre_gradient = output_gradient.hidden, output_gradient.pre_mix
        if hidden_gradient.shape != (ROWS * HC_COPIES, HIDDEN):
            raise ValueError("FFN output gradient changed the K32 four-stream geometry")
        if pre_gradient.shape != (ROWS, HC_COPIES):
            raise ValueError("FFN pre-mix gradient changed the exact four-column ABI")

        changed = captured["changed"][begin * HC_COPIES:end * HC_COPIES]
        post = captured["post"][begin:end]
        combination = captured["comb"][begin * HC_COPIES:end * HC_COPIES]
        combined = captured["combined"][begin:end]
        experts = captured["expert_outputs"][begin * 6:end * 6]
        route_weights = captured["route_weights"][begin:end]
        route_ids = captured["route_ids"][begin:end]
        router_logits = captured["router_logits"][begin:end]
        expected = {
            "changed": (changed, (ROWS * HC_COPIES, HIDDEN), torch.bfloat16),
            "post": (post, (ROWS, HC_COPIES), torch.float32),
            "comb": (combination, (ROWS * HC_COPIES, HC_COPIES), torch.float32),
            "combined": (combined, (ROWS, HIDDEN), torch.bfloat16),
            "expert_outputs": (experts, (ROWS * 6, HIDDEN), torch.bfloat16),
            "route_weights": (route_weights, (ROWS, 6), torch.float32),
            "route_ids": (route_ids, (ROWS, 6), torch.int32),
            "router_logits": (router_logits, (ROWS, 384), torch.float32),
        }
        for label, (value, shape, dtype) in expected.items():
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"captured {label} changed the fixed K32 ABI")

        from .pypto_dsv41_moe_vjp import make_moe_combine_vjp, make_router_logits_vjp
        from .pypto_dsv41_tail_vjp import make_hc_post_data_vjp, make_hc_post_gate_vjp

        moe_gradient = self._workspace(rank, (ROWS, HIDDEN))
        residual_gradient = self._workspace(rank, (ROWS * HC_COPIES, HIDDEN))
        post_gradient = self._workspace(rank, (ROWS, 16), torch.float32)
        combination_gradient = self._workspace(
            rank, (ROWS * HC_COPIES, 16), torch.float32,
        )
        self._execute(
            ("ffn-hc-post-data-vjp", layer), rank,
            lambda: make_hc_post_data_vjp(ROWS, HIDDEN),
            output_gradient=hidden_gradient, post=post, combination=combination,
            value_gradient=moe_gradient, residual_gradient=residual_gradient,
        )
        self._execute(
            ("ffn-hc-post-gate-vjp", layer), rank,
            lambda: make_hc_post_gate_vjp(ROWS, HIDDEN),
            value=combined, residual=changed, output_gradient=hidden_gradient,
            post_gradient=post_gradient, combination_gradient=combination_gradient,
        )

        shared_gradient = self._workspace(rank, (ROWS, HIDDEN))
        expert_gradient = self._workspace(rank, (ROWS * 6, HIDDEN))
        weight_gradient = self._workspace(rank, (ROWS, 16), torch.float32)
        self._execute(
            ("moe-combine-vjp", layer), rank,
            lambda: make_moe_combine_vjp(ROWS, HIDDEN),
            output_gradient=moe_gradient, expert_outputs=experts,
            route_weights=route_weights, shared_gradient=shared_gradient,
            expert_gradient=expert_gradient, weight_gradient=weight_gradient,
        )
        router_gradient = self._workspace(rank, (ROWS, 384), torch.float32)
        self._execute(
            ("router-logits-vjp", layer), rank,
            lambda: make_router_logits_vjp(ROWS, 384),
            router_logits=router_logits, route_ids=route_ids,
            weight_gradient=weight_gradient, router_gradient=router_gradient,
            route_scale=2.5,
        )
        return _FrozenMoEEnvelope(
            shared_gradient, expert_gradient, router_gradient,
            residual_gradient, post_gradient[:, :HC_COPIES],
            combination_gradient[:, :HC_COPIES], pre_gradient,
        )

    def _reverse_frozen_tail_finish(
        self, *, layer, rank, captured, envelope, routed_input_gradient,
    ):
        """Finish one rank after EP16 returns the six routed-expert dX rows."""
        if not isinstance(envelope, _FrozenMoEEnvelope):
            raise TypeError("frozen MoE finish requires its resident envelope")
        begin, end = self._span(captured)
        normed = captured["normed"][begin:end]
        collapsed = captured["collapsed"][begin:end]
        changed = captured["changed"][begin * HC_COPIES:end * HC_COPIES]
        incoming = captured["pre"][begin:end]
        if (
            normed.shape != (ROWS, HIDDEN) or normed.dtype != torch.bfloat16
            or collapsed.shape != (ROWS, HIDDEN) or collapsed.dtype != torch.bfloat16
            or changed.shape != (ROWS * HC_COPIES, HIDDEN)
            or incoming.shape != (ROWS, HC_COPIES)
            or routed_input_gradient.shape != (ROWS, HIDDEN)
        ):
            raise ValueError("captured FFN reverse boundary changed its K32 ABI")

        from .pypto_dsv41_moe_vjp import make_frozen_w8a8_expert_input_vjp
        from .pypto_dsv41_online_programs import make_element, make_linear, make_rms
        from .pypto_dsv41_tail_vjp import (
            make_hc_pre_data_vjp,
            make_hc_pre_gate_vjp,
            make_hc_pre_projection_forward,
        )

        # Frozen shared expert.  It uses the same W8A8 boundary as each
        # routed expert, but receives the unweighted MoE output gradient.
        shared_weights = self.native.weights.for_layer(layer)["expert"](rank, None)
        shared_input_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("shared-expert-input-vjp", layer), rank,
            lambda: make_frozen_w8a8_expert_input_vjp(ROWS, HIDDEN, 2304),
            value=normed, output_gradient=envelope.shared,
            gate_up_packed=shared_weights[0], gate_up_scale=shared_weights[1],
            down_packed=shared_weights[2], down_scale=shared_weights[3],
            input_gradient=shared_input_gradient,
        )

        # Fixed Top-K router membership is not differentiated.  The selected
        # sigmoid-score VJP already lives in ``envelope.router``; only the
        # frozen full-precision gate projection remains here.
        gate = self._pinned(rank, f"layers.{layer}.ffn.gate.weight")
        router_input_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("router-input-vjp", layer), rank,
            lambda: make_linear(
                (ROWS, 384), (384, HIDDEN), transpose_b=False,
            ),
            a=envelope.router, b=gate, output=router_input_gradient,
        )
        moe_input_gradient = self._workspace(rank, (ROWS, HIDDEN))
        partial = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("moe-input-add-routed-shared", layer), rank,
            lambda: make_element((ROWS, HIDDEN), "add"),
            a=routed_input_gradient, b=shared_input_gradient, output=partial,
        )
        self._execute(
            ("moe-input-add-router", layer), rank,
            lambda: make_element((ROWS, HIDDEN), "add"),
            a=partial, b=router_input_gradient, output=moe_input_gradient,
        )

        # Reverse the frozen FFN RMSNorm with the model's actual 1e-20 eps.
        norm = self._pinned(rank, f"layers.{layer}.ffn_norm.weight").view(1, HIDDEN)
        replay_normed = self._workspace(rank, (ROWS, HIDDEN))
        inverse = self._workspace(rank, (ROWS, 8), torch.float32)
        self._execute(
            ("ffn-rms-forward", layer), rank,
            lambda: make_rms(ROWS, HIDDEN, epsilon=1e-20),
            x=collapsed, gamma=norm, y=replay_normed, inv=inverse,
        )
        collapsed_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("ffn-rms-input-vjp", layer), rank,
            lambda: make_rms(
                ROWS, HIDDEN, mode="input", epsilon=1e-20,
            ),
            x=collapsed, gamma=norm, inv=inverse,
            dy=moe_input_gradient, dx=collapsed_gradient,
        )

        # Recompute only the small frozen mHC projection.  The exact forward
        # pre/post/Sinkhorn decisions are otherwise taken from this capture.
        function = self._pinned(rank, f"layers.{layer}.hc_ffn_fn")
        scale = self._pinned(rank, f"layers.{layer}.hc_ffn_scale")
        base = self._pinned(rank, f"layers.{layer}.hc_ffn_base")
        projection = self._workspace(rank, (ROWS, 32), torch.float32)
        hc_inverse = self._workspace(rank, (ROWS, 16), torch.float32)
        self._execute(
            ("ffn-hc-pre-projection", layer), rank,
            lambda: make_hc_pre_projection_forward(ROWS, HIDDEN),
            residual=changed, function=function,
            projection=projection, inverse=hc_inverse,
        )
        projection_gradient = self._workspace(rank, (ROWS, 32), torch.float32)
        self._execute(
            ("ffn-hc-pre-gate-vjp", layer), rank,
            lambda: make_hc_pre_gate_vjp(ROWS),
            projection=projection, scale=scale, base=base,
            pre_gradient=envelope.pre, post_gradient=envelope.post,
            combination_gradient=envelope.combination,
            projection_gradient=projection_gradient,
        )
        hidden_gradient = self._workspace(rank, (ROWS * HC_COPIES, HIDDEN))
        incoming_gradient = self._workspace(rank, (ROWS, 16), torch.float32)
        self._execute(
            ("ffn-hc-pre-data-vjp", layer), rank,
            lambda: make_hc_pre_data_vjp(ROWS, HIDDEN),
            residual=changed, incoming=incoming, function=function,
            inverse=hc_inverse, projection_gradient=projection_gradient,
            collapsed_gradient=collapsed_gradient,
            residual_passthrough=envelope.residual,
            residual_gradient=hidden_gradient,
            incoming_gradient=incoming_gradient,
        )
        return SuffixGradient(hidden_gradient, incoming_gradient[:, :HC_COPIES])

    def reverse_draft_attention(self, **options):
        self._unfinished("C1A draft Attention reverse")

    def tp_assemble_gradients(self, layer, values):
        self._require_optimizer_state()
        ranks = tuple(sorted(values))
        if len(ranks) != 4 or ranks != tuple(range(ranks[0] // 4 * 4, ranks[0] // 4 * 4 + 4)):
            raise ValueError("TP gradient assembly requires one complete TP4 group")
        names = tuple(self._execution_names())
        if any(set(values[rank]) != set(names) for rank in ranks):
            raise ValueError("rank-local draft gradient changed the eight-parameter ABI")
        result = {}
        for name in names:
            shards = {rank: values[rank][name] for rank in ranks}
            for rank, value in shards.items():
                state = self._adam[rank, layer, name]
                if value.shape != state.master.shape or value.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError(f"gradient shape differs for layer {layer} {name}")
            if name in self._replicated_tp_parameters():
                shards = self._tp_all_reduce(
                    shards, stage=f"layer{layer}-{name.replace('.', '-')}-gradient",
                )
            result[name] = _RankGradient(layer, shards)
        return result

    def dp_all_reduce(self, values):
        self._require_optimizer_state()
        if set(values) != set(self.plan.trainable_names()):
            raise ValueError("DP reduction requires the complete selected suffix")
        packets = []
        for rank in self.ranks:
            row = {}
            for full_name, gradient in values.items():
                layer = int(full_name.split(".")[1])
                short = full_name.split(".attn.", 1)[1]
                if not isinstance(gradient, _RankGradient) or gradient.layer != layer:
                    raise TypeError("DP reduction received a foreign gradient descriptor")
                value = gradient.values.get(rank)
                if value is None:
                    value = self._workspace(
                        rank, self._adam[rank, layer, short].master.shape, torch.float32,
                    )
                row[full_name] = {"value": value}
            packets.append(row)

        transport = getattr(self.native, "transport", None)
        if transport is None:
            output, _ = self.resident.exchange(
                f"online-dp-step-{self._last_step + 1}", packets,
                "all_reduce", "dp",
            )
        else:
            output, _ = transport.exchange(
                f"online-dp-step-{self._last_step + 1}", packets,
                "all_reduce", "dp",
            )
        result = {}
        for full_name in values:
            layer = int(full_name.split(".")[1])
            result[full_name] = _RankGradient(
                layer,
                {rank: output[index][full_name]["value"]
                 for index, rank in enumerate(self.ranks)},
            )
        return result

    def gradient_norm_and_clip(self, values, maximum):
        self._require_optimizer_state()
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("positive finite clipping maximum required")
        if set(values) != set(self.plan.trainable_names()):
            raise ValueError("gradient norm requires the complete selected suffix")
        # DP all-reduce makes every replica equal. Use one complete local TP
        # group and count replicated operands once; this avoids multiplying
        # the mathematical norm by the number of DP replicas.
        first_dp = min(self.ranks) // 4
        group = tuple(range(first_dp * 4, first_dp * 4 + 4))
        squared, finite = 0., True
        for full_name, gradient in values.items():
            short = full_name.split(".attn.", 1)[1]
            selected = group[:1] if short in self._replicated_tp_parameters() else group
            for rank in selected:
                part, good = self._tensor_statistics(rank, gradient.values[rank], full_name)
                squared += part
                finite = finite and good
        norm = math.sqrt(squared)
        if not finite or not math.isfinite(norm):
            raise FloatingPointError("non-finite DSV4.1 online gradient")
        clip = min(1., maximum / (norm + 1e-6))
        self._last_norm = norm
        return _HostScalar(norm), _HostScalar(clip)

    def _tensor_statistics(self, rank, value, name):
        from .pypto_dsv41_optimizer_programs import make_reduce, make_stats

        shape = tuple(value.shape)
        stats, count = make_stats(shape)
        sums = self._workspace(rank, (count, 8), torch.float32)
        flags = self._workspace(rank, (count, 8), torch.float32)
        self._execute(
            ("gradient-stats", shape), rank, lambda: make_stats(shape)[0],
            value=value, sums=sums, flags=flags,
        )
        level = 0
        while count > 1:
            reduce, next_count = make_reduce(count)
            next_sums = self._workspace(rank, (next_count, 8), torch.float32)
            next_flags = self._workspace(rank, (next_count, 8), torch.float32)
            self._execute(
                ("gradient-reduce", count), rank, lambda count=count: make_reduce(count)[0],
                sums=sums, flags=flags, out_sums=next_sums, out_flags=next_flags,
            )
            sums, flags, count, level = next_sums, next_flags, next_count, level + 1
        local = self._local(rank)
        host_sums = self.resident.read_output(sums, local)
        host_flags = self.resident.read_output(flags, local)
        return float(host_sums.sum()), bool((host_flags == 1).all())

    def adamw(self, values, **options):
        self._require_optimizer_state()
        from .pypto_dsv41_online_programs import make_adam
        from .pypto_dsv41_optimizer_programs import make_publish

        rate, step, clip = options["rate"], options["step"], options["clip"]
        clip = clip.value if isinstance(clip, _HostScalar) else float(clip)
        if step != self._last_step + 1 or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Adam step/rate is out of order or invalid")
        for full_name, gradient in values.items():
            layer = int(full_name.split(".")[1])
            short = full_name.split(".attn.", 1)[1]
            for rank in self.ranks:
                state = self._adam[rank, layer, short]
                grad = gradient.values[rank]
                self._execute(
                    ("adam", tuple(state.master.shape)), rank,
                    lambda shape=tuple(state.master.shape): make_adam(shape),
                    master=state.master, gradient=grad, first=state.first, second=state.second,
                    next_master=state.candidate_master,
                    next_first=state.candidate_first,
                    next_second=state.candidate_second,
                    rate=rate, correction1=1. - .9 ** step,
                    correction2=1. - .999 ** step, clip=clip,
                )
                if state.execution_dtype == torch.bfloat16:
                    self._execute(
                        ("publish", tuple(state.master.shape)), rank,
                        lambda shape=tuple(state.master.shape): make_publish(shape),
                        master=state.candidate_master, execution=state.execution,
                    )
                elif state.execution_dtype == torch.float32:
                    self.resident.copy_tensor(
                        state.execution, state.candidate_master, self._local(rank),
                    )
                else:
                    raise TypeError("unsupported draft execution dtype")
                state.master, state.candidate_master = state.candidate_master, state.master
                state.first, state.candidate_first = state.candidate_first, state.first
                state.second, state.candidate_second = state.candidate_second, state.second
        self._last_step = step
        digest = json.dumps(
            {"step": step, "last_layers": self.plan.last_layers,
             "rate": rate, "norm": self._last_norm},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        self._publication_digest = hashlib.sha256(digest).hexdigest()

    def zero_gradients(self, layers, names):
        self._require_optimizer_state()
        if tuple(layers) != self.plan.layers or tuple(names) != self.plan.trainable_names():
            raise ValueError("zero-gradient participant changed the suffix ABI")
        result = {}
        for full_name in names:
            layer = int(full_name.split(".")[1])
            short = full_name.split(".attn.", 1)[1]
            result[full_name] = _RankGradient(layer, {
                rank: self._workspace(rank, self._adam[rank, layer, short].master.shape, torch.float32)
                for rank in self.ranks
            })
        return result

    def publication_state(self, layers):
        self._require_optimizer_state()
        from .pypto_dsv41_runtime import ResidentHeadState

        if tuple(layers) != self.plan.layers or self._publication_digest is None:
            raise RuntimeError("publication requires one completed optimizer step")
        operands = {}
        for rank in self.ranks:
            for layer in self.plan.layers:
                operands[rank, layer] = {
                    state.execution_name: state.execution
                    for (owner, selected, _), state in self._adam.items()
                    if owner == rank and selected == layer
                }
        return ResidentHeadState(
            operands, self.plan.trainable_names(), self._publication_digest,
        )

    def read_scalar(self, value):
        if isinstance(value, _DeferredLoss):
            value = value.value
        if isinstance(value, _HostScalar):
            return value.value
        if not isinstance(value, _ResidentScalar):
            raise TypeError("only resident scalar summaries may cross the host boundary")
        host = self.resident.read_output(value.value, self._local(value.rank))
        self._scalar_reads += 1
        return float(host[:, 0].sum())

    @property
    def capabilities(self):
        return MappingProxyType(production_readiness())


def create_terminal_primitives(*, draft, native, cluster=None, config):
    """Construct the validated terminal slice for focused device tests."""
    return DSV41ResidentPrimitives(draft=draft, native=native, cluster=cluster, config=config)


def create_primitives(*, draft, native, cluster=None, config):
    """Production factory.

    Refuse before online serving starts until every exact reverse/optimizer
    stage is wired.  This is preferable to spending minutes loading a dual-node
    model and then failing at the first update—or, worse, publishing zeros.
    """
    readiness = production_readiness()
    if not readiness["production_ready"]:
        raise ProductionFactoryIncomplete(
            "DSV4.1 resident online factory is incomplete: "
            + "; ".join(readiness["blockers"])
        )
    return create_terminal_primitives(draft=draft, native=native, cluster=cluster, config=config)
