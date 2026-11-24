"""Resident PyPTO primitives for Orthrus-DSV4.1 online continuation.

This module is deliberately strict: serving never falls back to Torch autograd
or to host-side model math.  The implemented resident reverse path owns

    full-vocabulary TV/KL -> TP4 LM head -> final RMSNorm -> final mHC collapse.

the complete TP4 C1A draft-Attention VJP, and the synchronized EP16 frozen
MoE tail reverse. All intermediate gradients are resident tensors. CPU
tensors are used only
for the verifier/student logits which the current serving patch already makes
an explicit host boundary.
"""

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import math

import torch

from .dsv41_device_update import SuffixGradient
from .dsv41_online_profile import active_profile, profile_online, profiled_stage, timed_stage
from .dsv41_serving_online import DSV41SequenceSpan
from .dsv41_suffix import DSV41SuffixPlan
from .pypto_dsv41_ep_vjp import EPVJPGeometry, ROUTES
from .pypto_dsv41_resident_abi import with_resident_entry


ROWS = 32
HIDDEN = 5120
HC_COPIES = 4
SUPPORTED_LAST_N = (1, 2, 4, 8)


def _make_group_layout(rows, groups, width, *, split):
    """Transpose between token-major and group-major matrices on device."""
    import pypto.language as pl

    if rows != ROWS or groups < 1 or width % 64:
        raise ValueError("grouped projection layout is not aligned")

    if split:
        @pl.jit
        def layout(source: pl.Tensor, output: pl.Out[pl.Tensor]):
            with pl.spmd(rows * groups, name_hint="orthrus_dsv41_group_split"):
                task = pl.tile.get_block_idx()
                token = task // groups
                group = task % groups
                for part in pl.range(width // 64):
                    value = pl.slice(
                        source, [1, 64],
                        [token, group * width + part * 64],
                    )
                    output[
                        group * rows + token:group * rows + token + 1,
                        part * 64:(part + 1) * 64,
                    ] = value
            return output
    else:
        @pl.jit
        def layout(source: pl.Tensor, output: pl.Out[pl.Tensor]):
            with pl.spmd(rows * groups, name_hint="orthrus_dsv41_group_merge"):
                task = pl.tile.get_block_idx()
                token = task // groups
                group = task % groups
                for part in pl.range(width // 64):
                    value = pl.slice(
                        source, [1, 64],
                        [group * rows + token, part * 64],
                    )
                    output[
                        token:token + 1,
                        group * width + part * 64:group * width + (part + 1) * 64,
                    ] = value
            return output

    @pl.jit.host
    def entry(source: pl.Tensor, output: pl.Out[pl.Tensor]):
        layout(source, output, device=0)

    return with_resident_entry(entry, layout)


def _make_rope_vjp(native_module, heads, *, inverse):
    """Wrap the native adjacent-pair RoPE as a resident backward leaf."""
    import pypto.language as pl

    rotate = native_module("qkv_proj_rope").make_rope(
        heads, inverse=inverse, head_dim=512, rope_dim=64,
        name_hint="orthrus_dsv41_rope_backward",
    )

    @pl.jit
    def kernel(x: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
               output: pl.Out[pl.Tensor], active: pl.Scalar[pl.INT32]):
        rotate(x, cos, sin, output, active)
        return output

    @pl.jit.host
    def entry(x: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
              output: pl.Out[pl.Tensor], active: pl.Scalar[pl.INT32]):
        kernel(x, cos, sin, output, active, device=0)

    return with_resident_entry(entry, kernel)


def _make_sink_compact(heads=16):
    """Compact cache-line-padded sink rows into the public parameter shape."""
    import pypto.language as pl

    @pl.jit
    def compact(source: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.spmd(1, name_hint="orthrus_dsv41_sink_compact"):
            _block = pl.tile.get_block_idx()
            row = pl.tile.full([1, heads], dtype=pl.FP32, value=0.)
            for head in pl.range(heads):
                pl.tile.write(row, [0, head], pl.read(source, [head, 0]))
            pl.store(row, [0, 0], output)
        return output

    @pl.jit.host
    def entry(source: pl.Tensor, output: pl.Out[pl.Tensor]):
        compact(source, output, device=0)

    return with_resident_entry(entry, compact)


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
        "production_ready": True,
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
            "TP4 C1A draft-attention parameter/input VJP orchestration",
            "EP16 frozen routed-expert input VJP dispatch and return",
        ),
        "blockers": (),
    }


class DSV41ResidentPrimitives:
    """Exact resident terminal VJP bound to one initialized generation.

    The public learner uses the synchronized TP4/EP16 batch methods. The
    single-rank ``reverse_frozen_tail`` entry remains a deliberate fail-fast
    guard because running only one EP rank would silently produce a partial
    gradient.
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
        self._dp_zero_storage = None
        self._dp_zero_ranks = set()
        self._dp_zero_views = {}
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

    def _workspace(self, rank, shape, dtype=torch.bfloat16, *, fill=0):
        return self.resident.workspace(tuple(shape), dtype, self._local(rank), fill=fill)

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

    def _pinned(self, rank, name, shape=None):
        """Resolve one immutable target operand by its native pin name."""
        local = self._local(rank)
        rows = [owner for (device, _), owner in self.resident.pinned.items()
                if device == local and owner.get("name") == name]
        if len(rows) != 1:
            raise RuntimeError(f"expected one resident {name!r} on rank {rank}, found {len(rows)}")
        source = rows[0]["source"]
        if shape is not None:
            source = source.view(shape)
        return self.resident.handle(source, local, readonly=True)

    def _draft_operand(self, rank, layer, name, shape):
        bank = self.draft.online_bank
        source = bank.host[bank.active_slot, rank, layer, name].view(shape)
        return self.resident.handle(source, self._local(rank), readonly=True)

    def _expert_partition(self, rank, value, partition):
        count = value.shape[0] // 4
        local = self._local(rank)
        shape = (count, *value.shape[1:])
        offset = partition * math.prod(shape) * value.dtype.itemsize
        view = self.resident.workers[local].view_tensor(value, shape, byte_offset=offset)
        self.resident.alias_leases[id(view)] = view
        return view

    @staticmethod
    def _span(row):
        begin, end = row.get("token_begin"), row.get("token_end")
        if type(begin) is not int or type(end) is not int or end - begin != ROWS:
            raise ValueError("terminal VJP requires the exact captured K32 token span")
        return begin, end

    @classmethod
    def _local_span(cls, row):
        if "sp_span" in row:
            return row["sp_span"]
        # Standalone VJP callers can supply a complete, unsharded K32 slab.
        begin, _ = cls._span(row)
        return DSV41SequenceSpan(begin, 0, ROWS)

    def _copy_rows(self, rank, output, value, *, destination=0):
        if value.shape[0]:
            self.resident.copy_tensor(
                output[destination:destination + value.shape[0]], value, self._local(rank),
            )

    def _padded_rows(self, rank, value, start, count, copies=1):
        matrix = value.view(-1, value.shape[-1]) if torch.is_tensor(value) else value
        rows = matrix[start * copies:(start + count) * copies]
        if count == ROWS:
            return rows
        padded = self._workspace(rank, (ROWS * copies, matrix.shape[-1]), value.dtype)
        self._copy_rows(rank, padded, rows)
        return padded

    def _local_rows(self, rank, row, value, copies=1):
        span = self._local_span(row)
        return self._padded_rows(rank, value, span.local_start, span.rows, copies)

    def _compact_mhc_columns(self, rank, value):
        """Materialize the four logical coefficients of each padded mHC row."""
        from .pypto_dsv41_tail_vjp import make_mhc_gradient_pack

        rows = value.shape[0]
        output = self._workspace(rank, (rows // 4, 16), torch.float32, fill=None)
        self._execute(
            ("mhc-gradient-pack", rows), rank,
            lambda: make_mhc_gradient_pack(rows), source=value, output=output,
        )
        return output.view(rows, HC_COPIES)

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

    def _loss_on_rank(self, rank, token, student, teacher):
        from .pypto_dsv41_online_programs import make_kl, make_tv

        vocabulary = token.student.shape[1]
        gradient = self._workspace(rank, (ROWS, vocabulary), fill=None)
        losses = self._workspace(rank, (ROWS, 8), torch.float32, fill=None)
        factory = make_tv if token.kind == "tv" else make_kl
        self._execute(
            ("loss", token.kind, vocabulary), rank,
            lambda: factory(ROWS, vocabulary),
            student=student, teacher=teacher, position_weights=student,
            gradient=gradient, losses=losses,
            valid=token.valid, normalizer=token.valid,
        )
        return gradient, losses

    def _tp_collective(self, values, collective, *, stage):
        """Run one complete local TP4 collective without a host boundary.

        In the two-host topology every TP group is node-local.  Calling the
        cluster-wide scheduling wrapper here would incorrectly require the
        non-owning host to enter a collective that belongs only to the owner
        replay.  We therefore use the already-created local TP communicators
        directly; DP/EP collectives still use the global transport later.
        """
        ranks = tuple(sorted(values))
        if len(ranks) != 4 or ranks != tuple(range(ranks[0] // 4 * 4, ranks[0] // 4 * 4 + 4)):
            raise ValueError("TP all-reduce requires one contiguous TP4 group")
        if collective not in ("all_gather", "reduce_scatter", "all_reduce"):
            raise ValueError("unsupported TP reverse collective")
        shape, dtype = values[ranks[0]].shape, values[ranks[0]].dtype
        if any(value.shape != shape or value.dtype != dtype for value in values.values()):
            raise ValueError("TP collective operands changed shape or dtype")
        output_shape = list(shape)
        if collective == "all_gather":
            output_shape[0] *= len(ranks)
        elif collective == "reduce_scatter":
            if output_shape[0] % len(ranks):
                raise ValueError("TP reduce-scatter rows do not divide by four")
            output_shape[0] //= len(ranks)
        output_shape = tuple(output_shape)

        transport = getattr(self.native, "transport", None)
        if transport is None:
            packets = [
                {stage: {"value": values[rank]}}
                for rank in ranks
            ]
            result, _ = self.resident.exchange(stage, packets, collective, "tp")
            return {rank: result[index][stage]["value"] for index, rank in enumerate(ranks)}

        channels = transport._channels("tp")
        local_ranks = tuple(transport.plan.local_ranks)
        if not set(ranks) <= set(local_ranks):
            raise ValueError("a TP group may not cross hosts")
        def prepare(rank):
            local = self._local(rank)
            channel = channels[local]
            output = self._workspace(rank, output_shape, dtype, fill=None)
            source = self.resident.tensor(values[rank], local)
            destination = self.resident.tensor(output, local)
            channel._validate_device_transfer(source, destination, collective)
            return output, (source, destination, f"online/{stage}/rank{rank}")

        prepared = {rank: prepare(rank) for rank in ranks}

        def transfer(rank):
            local = self._local(rank)
            output, packet = prepared[rank]
            channels[local].exchange_device_batch([packet], collective)
            self.resident.mark_workspace_written(output, local)
            return output

        return self._map_owner_ranks(ranks, transfer)

    def _tp_all_reduce(self, values, *, stage):
        return self._tp_collective(values, "all_reduce", stage=stage)

    @profiled_stage("lm_head_loss_and_vjp")
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

        from .pypto_dsv41_online_programs import make_linear, make_rms, make_vocab_gradient_shard
        from .pypto_dsv41_tail_vjp import make_hc_collapse_forward, make_hc_collapse_vjp

        local_vocabulary = vocabulary // 4
        norms = {rank: self._pinned(rank, "norm.weight", (1, HIDDEN)) for rank in ranks}

        def collapse(rank):
            row = rows[rank]
            span = self._local_span(row)
            streams = self._local_rows(rank, row, row["hidden"], HC_COPIES)
            premix = self._local_rows(rank, row, row["pre_mix"])
            if streams.shape != (ROWS * HC_COPIES, HIDDEN) or premix.shape != (ROWS, HC_COPIES):
                raise ValueError("captured final mHC state has the wrong K32 geometry")

            collapsed = self._workspace(rank, (ROWS, HIDDEN))
            self._execute(
                ("final-collapse-forward",), rank,
                lambda: make_hc_collapse_forward(ROWS, HIDDEN),
                streams=streams, premix=premix, collapsed=collapsed,
            )
            packet = self._workspace(rank, (ROWS, HIDDEN))
            self._copy_rows(rank, packet, collapsed[:span.rows], destination=span.block_start)
            return streams, premix, packet

        # The forward concatenates local SP collapses before applying the
        # final norm and each vocabulary shard. The request rows are disjoint.
        terminal_states = self._map_owner_ranks(ranks, collapse)
        collapsed_packets = {rank: state[2] for rank, state in terminal_states.items()}
        collapsed_rows = self._tp_all_reduce(collapsed_packets, stage="lm-head-sp-collapse")

        # Rank threads share read-only host inputs. Padding them per rank
        # duplicates the full vocabulary allocation and contends for CPU workers.
        loss = gradient.loss
        student = self._padded_logits(loss.student, loss.valid, vocabulary)
        teacher = self._padded_logits(loss.teacher, loss.valid, vocabulary)

        def reverse_head(rank):
            dlogits, losses = self._loss_on_rank(rank, loss, student, teacher)
            dlocal = self._workspace(rank, (ROWS, local_vocabulary), fill=None)
            self._execute(
                ("vocab-gradient-shard", vocabulary), rank,
                lambda: make_vocab_gradient_shard(ROWS, vocabulary),
                gradient=dlogits, output=dlocal, offset=rank % 4 * local_vocabulary,
            )
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
            return losses, dnormalized

        terminal_reverse = self._map_owner_ranks(ranks, reverse_head)
        if loss.value is None:
            loss.value = _ResidentScalar(terminal_reverse[ranks[0]][0], ranks[0])
        normalized_gradients = self._tp_all_reduce(
            {rank: state[1] for rank, state in terminal_reverse.items()}, stage="lm-head-input-vjp",
        )

        def reverse_collapse(rank):
            streams, premix, _ = terminal_states[rank]
            norm = norms[rank]
            dcollapsed = self._workspace(rank, (ROWS, HIDDEN))
            self._execute(
                ("final-rms-frozen-input-vjp",), rank,
                lambda: make_rms(
                    ROWS, HIDDEN, mode="frozen_input", epsilon=1e-20,
                    gamma_bf16=norm.dtype == torch.bfloat16,
                ),
                x=collapsed_rows[rank], gamma=norm,
                dy=normalized_gradients[rank], dx=dcollapsed,
            )
            dstreams = self._workspace(rank, (ROWS * HC_COPIES, HIDDEN))
            dpremix = self._workspace(rank, (ROWS, 16), torch.float32)
            self._execute(
                ("final-collapse-vjp",), rank,
                lambda: make_hc_collapse_vjp(ROWS, HIDDEN),
                streams=streams, premix=premix,
                output_gradient=self._padded_rows(rank, dcollapsed,
                    self._local_span(rows[rank]).block_start, self._local_span(rows[rank]).rows),
                stream_gradient=dstreams, premix_gradient=dpremix,
            )
            return dstreams, self._compact_mhc_columns(rank, dpremix)

        output = self._map_owner_ranks(ranks, reverse_collapse)
        hidden_gradients = {rank: value[0] for rank, value in output.items()}
        premix_gradients = {rank: value[1] for rank, value in output.items()}
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

    def _reverse_sync_stage(self, layer, synchronization, phase):
        """Validate the cluster contract and derive a peer-stable stage tag."""
        if not isinstance(synchronization, dict):
            raise ValueError("synchronized frozen-tail reverse requires a contract")
        reverse = synchronization.get("reverse")
        if (
            not isinstance(reverse, dict)
            or reverse.get("protocol") != "dsv41-synchronized-reverse-v1"
            or tuple(reverse.get("layers", ())) != self.plan.layers
            or layer not in self.plan.layers
        ):
            raise ValueError("synchronized frozen-tail reverse contract is inconsistent")
        step = synchronization.get("update_number", 0)
        if type(step) is not int or step < 0:
            raise ValueError("synchronized frozen-tail reverse has an invalid update number")
        return f"online-frozen-tail-step{step}-layer{layer}-{phase}-v1"

    def _reverse_bucket_rows(self, layer, synchronization):
        widths = synchronization["reverse"].get("ep_bucket_rows")
        return widths[self.plan.layers.index(layer)] if widths else ROUTES

    def _ep_exchange(self, stage, packets):
        """Run one fixed-packet EP16 all-to-all on every node-local rank."""
        transport = getattr(self.native, "transport", None)
        phase = stage.rsplit("-", 2)[-2]
        profile = active_profile()
        if profile is not None and profile.current is not None:
            profile.current["counts"][f"ep_{phase}_logical_send_bytes"] += sum(
                value.numel() * value.element_size()
                for packet in packets for fields in packet.values() for value in fields.values()
            )
        with timed_stage(f"ep_{phase}_exchange"):
            if transport is None:
                output, _ = self.resident.exchange(stage, packets, "all_to_all", "ep")
            else:
                # All-to-all writes every byte of each fixed-width bucket.
                output, _ = transport.exchange(
                    stage, packets, "all_to_all", "ep", write_only_receive=True,
                )
        if len(output) != len(self.ranks):
            raise RuntimeError("EP16 exchange did not return every node-local rank")
        return output

    @profiled_stage("ep_pack")
    def _ep_dispatch_packets(self, sources, bucket_rows=ROUTES):
        """Create the same fixed dispatch ABI on owner and participant nodes."""
        from .pypto_dsv41_ep_vjp import (
            EP_WORLD, MAP_PAD, PACKET_FIELDS,
            make_ep_vjp_dispatch_pack, make_ep_vjp_empty_payload,
        )
        packet_rows = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows

        # The expert slab holds one K32 block, split across the owner ranks.
        spans = [self._local_span(captured) for captured, _ in sources.values()]
        rows = sorted(
            row
            for span in spans
            for row in range(span.block_start, span.block_start + span.rows)
        )
        if sources and rows != list(range(ROWS)):
            raise ValueError("EP reverse requires disjoint owner spans covering one K32 block")

        def pack_rank(index):
            rank = self.ranks[index]
            source = sources.get(rank)
            fill = 0 if source is None else None
            sent_metadata = self._workspace(rank, (packet_rows, PACKET_FIELDS), torch.int64, fill=fill)
            sent_x = self._workspace(rank, (packet_rows, HIDDEN), fill=None)
            sent_dy = self._workspace(rank, (packet_rows, HIDDEN), fill=None)
            sent_counts = self._workspace(rank, (EP_WORLD, MAP_PAD), torch.int32, fill=fill)
            packet = {"frozen-tail": {"metadata": sent_metadata, "x": sent_x,
                                      "dy": sent_dy, "counts": sent_counts}}
            if source is None:
                self._execute(
                    ("ep-vjp-empty-payload", bucket_rows), rank,
                    lambda: make_ep_vjp_empty_payload(bucket_rows), sent_x=sent_x, sent_dy=sent_dy,
                )
                return packet
            captured, envelope = source
            normalized = self._local_rows(rank, captured, captured["normed"])
            route_ids = self._local_rows(rank, captured, captured["route_ids"])
            expert_gradient = envelope.expert
            if (
                normalized.shape != (ROWS, HIDDEN)
                or normalized.dtype != torch.bfloat16
                or route_ids.shape != (ROWS, 6)
                or route_ids.dtype != torch.int32
                or expert_gradient.shape != (ROUTES, HIDDEN)
                or expert_gradient.dtype != torch.bfloat16
            ):
                raise ValueError("captured EP dispatch boundary changed its K32 ABI")
            active = self._local_span(captured).rows
            self._execute(
                ("ep-vjp-dispatch-pack", bucket_rows), rank,
                lambda: make_ep_vjp_dispatch_pack(bucket_rows),
                normalized_x=normalized, route_output_gradient=expert_gradient,
                route_ids=route_ids, sent_metadata=sent_metadata,
                sent_x=sent_x, sent_dy=sent_dy, sent_counts=sent_counts,
                active=active, source_rank=rank,
            )
            return packet

        packets = self._map_online_ranks(pack_rank)
        originals = {
            rank: (packet["frozen-tail"]["metadata"], packet["frozen-tail"]["counts"])
            for rank, packet in zip(self.ranks, packets, strict=True)
        }
        return packets, originals

    @profiled_stage("ep_expert_owner")
    def _ep_owner_expert_packets(self, layer, received, bucket_rows):
        """Run frozen expert input VJPs for the populated slabs on each local rank."""
        from .pypto_dsv41_ep_vjp import (
            EXPERT_CAPACITY, EXPERT_SMALL_ROWS, LOCAL_EXPERTS, MAP_PAD,
            SLAB_ROWS, make_ep_vjp_dispatch_unpack, make_ep_vjp_return_pack,
        )
        from .pypto_dsv41_moe_vjp import make_frozen_w8a8_expert_input_vjp
        packet_rows = EPVJPGeometry(bucket_rows=bucket_rows).packet_rows

        layer_weights = self.native.weights.for_layer(layer)["expert"]
        profile = active_profile()
        measured = profile is not None and profile.current is not None

        def run_rank(index):
            rank = self.ranks[index]
            packet = received[index]["frozen-tail"]
            expert_x = self._workspace(rank, (SLAB_ROWS, HIDDEN), fill=None)
            expert_dy = self._workspace(rank, (SLAB_ROWS, HIDDEN), fill=None)
            expert_counts = self._workspace(rank, (LOCAL_EXPERTS,), torch.int32, fill=None)
            packet_map = self._workspace(rank, (packet_rows, MAP_PAD), torch.int32, fill=None)
            received_total = self._workspace(rank, (MAP_PAD,), torch.int32, fill=None)
            self._execute(
                ("ep-vjp-dispatch-unpack", bucket_rows), rank,
                lambda: make_ep_vjp_dispatch_unpack(bucket_rows),
                packet_metadata=packet["metadata"], packet_x=packet["x"],
                packet_dy=packet["dy"], packet_counts=packet["counts"],
                expert_x=expert_x, expert_dy=expert_dy,
                expert_counts=expert_counts, packet_map=packet_map,
                received_total=received_total, owner_rank=rank,
            )

            # Return packing reads only routed rows written by the active experts.
            expert_dx = self._workspace(rank, (SLAB_ROWS, HIDDEN), fill=None)
            with timed_stage("expert_route_control_d2h"):
                counts = self.resident.read_control(expert_counts, self._local(rank)).tolist()
            active_experts = [
                (local, EXPERT_SMALL_ROWS if count <= EXPERT_SMALL_ROWS else EXPERT_CAPACITY)
                for local, count in enumerate(counts) if count
            ]
            if measured:
                with profile.lock:
                    experts = profile.current["counts"]
                    received_rows = sum(counts)
                    experts["expert_rows_received"] += received_rows
                    experts["peak_routed_rows_per_ep_rank"] = max(
                        experts["peak_routed_rows_per_ep_rank"], received_rows,
                    )
                    experts["active_routed_expert_calls"] += len(active_experts)
                    experts["expert_padded_rows"] += sum(rows for _, rows in active_experts)
                    experts[f"expert_rows_rank_{rank}"] += received_rows
            for local_expert, rows in active_experts:
                begin = local_expert * EXPERT_CAPACITY
                end = begin + rows
                global_expert = rank * LOCAL_EXPERTS + local_expert
                weights = layer_weights(rank, global_expert)
                if not isinstance(weights, (tuple, list)) or len(weights) != 4:
                    raise RuntimeError("resident routed expert changed its W8A8 weight ABI")
                self._execute(
                    ("routed-expert-input-vjp", rows), rank,
                    lambda rows=rows: make_frozen_w8a8_expert_input_vjp(
                        rows, HIDDEN, 2304,
                    ),
                    value=expert_x[begin:end],
                    output_gradient=expert_dy[begin:end],
                    gate_up_packed=weights[0], gate_up_scale=weights[1],
                    down_packed=weights[2], down_scale=weights[3],
                    input_gradient=expert_dx[begin:end],
                )

            sent_dx = self._workspace(rank, (packet_rows, HIDDEN), fill=None)
            self._execute(
                ("ep-vjp-return-pack", bucket_rows), rank,
                lambda: make_ep_vjp_return_pack(bucket_rows),
                expert_dx=expert_dx, packet_counts=packet["counts"],
                packet_map=packet_map, sent_dx=sent_dx,
            )
            return {"frozen-tail": {"dx": sent_dx}}

        return self._map_online_ranks(run_rank)

    def _map_online_ranks(self, run_rank):
        """Dispatch independent per-card work while retaining one update profile."""
        profile = active_profile()
        if profile is None or profile.current is None:
            return self.resident.map_ranks(run_rank)

        def profile_rank(index):
            with profile_online(profile):
                return run_rank(index)

        before = self.resident.stats.get("program_evictions", 0)
        profile.parallel_ranks = True
        try:
            return self.resident.map_ranks(profile_rank)
        finally:
            profile.parallel_ranks = False
            profile.current["counts"]["device_callable_evictions"] += (
                self.resident.stats.get("program_evictions", 0) - before
            )

    def _map_owner_ranks(self, ranks, function):
        if len(ranks) == 1:
            return {ranks[0]: function(ranks[0])}
        selected = set(ranks)

        def run_rank(index):
            rank = self.ranks[index]
            if rank in selected:
                return function(rank)

        result = self._map_online_ranks(run_rank)
        return {rank: value for rank, value in zip(self.ranks, result, strict=True) if rank in selected}

    @profiled_stage("ep_restore_routes")
    def _ep_routed_input_gradients(self, returned, originals, sources, bucket_rows):
        """Restore route order only for the TP ranks that originated K32 rows."""
        from .pypto_dsv41_ep_vjp import make_ep_vjp_source_restore

        if not set(sources).issubset(self.ranks):
            raise ValueError("owner TP rank is not resident on this host")

        def restore_rank(index):
            rank = self.ranks[index]
            if rank not in sources:
                return None
            active = self._local_span(sources[rank][0]).rows
            if not active:
                return self._workspace(rank, (ROWS, HIDDEN))
            metadata, counts = originals[rank]
            route_dx = self._workspace(rank, (ROUTES, HIDDEN), fill=None)
            token_dx = self._workspace(rank, (ROWS, HIDDEN), fill=None)
            self._execute(
                ("ep-vjp-source-restore", bucket_rows), rank,
                lambda: make_ep_vjp_source_restore(bucket_rows),
                returned_dx=returned[index]["frozen-tail"]["dx"],
                sent_metadata=metadata, sent_counts=counts,
                route_dx=route_dx, token_dx=token_dx,
                active=active, source_rank=rank,
            )
            return token_dx

        gradients = self._map_online_ranks(restore_rank)
        return {rank: tensor for rank, tensor in zip(self.ranks, gradients, strict=True) if tensor is not None}

    def reverse_frozen_tail_owner_batch(
        self, *, layer, ranks, captured, states, output_gradients,
        synchronization,
    ):
        """Reverse one owner TP4 batch while every EP16 rank participates."""
        dispatch_stage = self._reverse_sync_stage(
            layer, synchronization, "dispatch",
        )
        return_stage = self._reverse_sync_stage(layer, synchronization, "return")
        bucket_rows = self._reverse_bucket_rows(layer, synchronization)
        ranks = tuple(ranks)
        if (
            len(ranks) != 4
            or ranks != tuple(range(ranks[0] // 4 * 4, ranks[0] // 4 * 4 + 4))
            or not set(ranks) <= set(self.ranks)
            or set(captured) != set(ranks)
            or set(states) != set(ranks)
            or set(output_gradients) != set(ranks)
        ):
            raise ValueError("owner frozen-tail reverse requires one complete local TP4 group")

        live_ranks = tuple(rank for rank in ranks if self._local_span(captured[rank]).rows)
        if not live_ranks:
            raise ValueError("EP reverse requires disjoint owner spans covering one K32 block")
        envelopes = self._map_owner_ranks(
            live_ranks,
            lambda rank: self._reverse_frozen_moe_envelope(
                layer=layer, rank=rank, captured=captured[rank],
                state=states[rank], output_gradient=output_gradients[rank],
            ),
        )
        sources = {rank: (captured[rank], envelope) for rank, envelope in envelopes.items()}
        packets, originals = self._ep_dispatch_packets(sources, bucket_rows)
        received = self._ep_exchange(dispatch_stage, packets)
        return_packets = self._ep_owner_expert_packets(layer, received, bucket_rows)
        returned = self._ep_exchange(return_stage, return_packets)
        routed = self._ep_routed_input_gradients(returned, originals, sources, bucket_rows)
        result = self._map_owner_ranks(
            live_ranks,
            lambda rank: self._reverse_frozen_tail_finish(
                layer=layer, rank=rank, captured=captured[rank],
                envelope=envelopes[rank], routed_input_gradient=routed[rank],
            ),
        )
        # The frozen FFN is token-local. Empty owners still participated as
        # expert destinations above, and need an empty boundary for TP replay.
        for rank in ranks:
            if rank not in result:
                result[rank] = SuffixGradient(
                    self._workspace(rank, (ROWS * HC_COPIES, HIDDEN)),
                    self._workspace(rank, (ROWS, HC_COPIES), torch.float32),
                )
        return {rank: result[rank] for rank in ranks}

    def participate_frozen_tail_peer(self, *, layer, synchronization):
        """Enter the same two EP16 collectives without inventing source rows."""
        dispatch_stage = self._reverse_sync_stage(
            layer, synchronization, "dispatch",
        )
        return_stage = self._reverse_sync_stage(layer, synchronization, "return")
        bucket_rows = self._reverse_bucket_rows(layer, synchronization)
        packets, _ = self._ep_dispatch_packets({}, bucket_rows)
        received = self._ep_exchange(dispatch_stage, packets)
        return_packets = self._ep_owner_expert_packets(layer, received, bucket_rows)
        self._ep_exchange(return_stage, return_packets)

    @profiled_stage("moe_route_vjp")
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
        if not isinstance(output_gradient, SuffixGradient):
            raise TypeError("frozen tail requires a suffix gradient")
        hidden_gradient, pre_gradient = output_gradient.hidden, output_gradient.pre_mix
        if hidden_gradient.shape != (ROWS * HC_COPIES, HIDDEN):
            raise ValueError("FFN output gradient changed the K32 four-stream geometry")
        if pre_gradient.shape != (ROWS, HC_COPIES):
            raise ValueError("FFN pre-mix gradient changed the exact four-column ABI")

        changed = self._local_rows(rank, captured, captured["changed"], HC_COPIES)
        post = self._local_rows(rank, captured, captured["post"])
        combination = self._local_rows(rank, captured, captured["comb"], HC_COPIES)
        combined = self._local_rows(rank, captured, captured["combined"])
        experts = self._local_rows(rank, captured, captured["expert_outputs"], 6)
        route_weights = self._local_rows(rank, captured, captured["route_weights"])
        route_ids = self._local_rows(rank, captured, captured["route_ids"])
        router_logits = self._local_rows(rank, captured, captured["router_logits"])
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
            route_scale=self.draft.config["routed_scaling_factor"],
        )
        return _FrozenMoEEnvelope(
            shared_gradient, expert_gradient, router_gradient,
            residual_gradient, self._compact_mhc_columns(rank, post_gradient),
            self._compact_mhc_columns(rank, combination_gradient), pre_gradient,
        )

    @profiled_stage("moe_shared_and_mhc_vjp")
    def _reverse_frozen_tail_finish(
        self, *, layer, rank, captured, envelope, routed_input_gradient,
    ):
        """Finish one rank after EP16 returns the six routed-expert dX rows."""
        if not isinstance(envelope, _FrozenMoEEnvelope):
            raise TypeError("frozen MoE finish requires its resident envelope")
        normed = self._local_rows(rank, captured, captured["normed"])
        collapsed = self._local_rows(rank, captured, captured["collapsed"])
        changed = self._local_rows(rank, captured, captured["changed"], HC_COPIES)
        incoming = self._local_rows(rank, captured, captured["pre"])
        if (
            normed.shape != (ROWS, HIDDEN) or normed.dtype != torch.bfloat16
            or collapsed.shape != (ROWS, HIDDEN) or collapsed.dtype != torch.bfloat16
            or changed.shape != (ROWS * HC_COPIES, HIDDEN)
            or incoming.shape != (ROWS, HC_COPIES)
            or routed_input_gradient.shape != (ROWS, HIDDEN)
        ):
            raise ValueError("captured FFN reverse boundary changed its K32 ABI")

        from .pypto_dsv41_moe_vjp import (
            make_frozen_w8a8_expert_input_vjp, make_moe_input_gradient_sum, make_router_input_vjp,
        )
        from .pypto_dsv41_online_programs import make_rms
        from .pypto_dsv41_tail_vjp import (
            make_hc_pre_data_vjp,
            make_hc_pre_gate_vjp,
            make_hc_pre_projection_forward,
        )

        # Native SP owns complete token rows. All four padded intermediate
        # partitions of the shared expert are packed onto every rank.
        shared_weights = self.native.weights.for_layer(layer)["expert"](rank, None)
        intermediate = shared_weights[1].shape[0] // (4 * 2)
        shared_parts = []
        for partition in range(4):
            gate, scale, down = (self._expert_partition(rank, operand, partition)
                                 for operand in shared_weights[:3])
            gradient = self._workspace(rank, (ROWS, HIDDEN))
            self._execute(
                ("shared-expert-input-vjp", intermediate), rank,
                lambda: make_frozen_w8a8_expert_input_vjp(ROWS, HIDDEN, intermediate),
                value=normed, output_gradient=envelope.shared,
                gate_up_packed=gate, gate_up_scale=scale,
                down_packed=down, down_scale=shared_weights[3],
                input_gradient=gradient,
            )
            shared_parts.append(gradient)

        # Fixed Top-K router membership is not differentiated.  The selected
        # score VJP already lives in ``envelope.router``; only the
        # frozen full-precision gate projection remains here.
        gate = self._pinned(rank, f"layers.{layer}.ffn.gate.weight")
        router_input_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("router-input-vjp", layer), rank,
            lambda: make_router_input_vjp(ROWS, HIDDEN),
            gradient=envelope.router, gate=gate, output=router_input_gradient,
            ids=self._local_rows(rank, captured, captured["route_ids"]),
        )
        moe_input_gradient = self._workspace(rank, (ROWS, HIDDEN), fill=None)
        self._execute(
            ("moe-input-gradient-sum",), rank, make_moe_input_gradient_sum,
            shared_0=shared_parts[0], shared_1=shared_parts[1],
            shared_2=shared_parts[2], shared_3=shared_parts[3],
            routed=routed_input_gradient, router=router_input_gradient, output=moe_input_gradient,
        )

        # Reverse the frozen FFN RMSNorm with the model's actual 1e-20 eps.
        norm = self._pinned(rank, f"layers.{layer}.ffn_norm.weight", (1, HIDDEN))
        collapsed_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("ffn-rms-frozen-input-vjp", layer), rank,
            lambda: make_rms(
                ROWS, HIDDEN, mode="frozen_input", epsilon=1e-20,
                gamma_bf16=norm.dtype == torch.bfloat16,
            ),
            x=collapsed, gamma=norm,
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
        return SuffixGradient(hidden_gradient, self._compact_mhc_columns(rank, incoming_gradient))

    @staticmethod
    def _attention_record(captured):
        record = captured.get("record")
        if not isinstance(record, dict):
            raise ValueError("C1A reverse requires the native SP record")
        physical = record.get("SP_physical_rows")
        positions = record.get("positions")
        if type(physical) is not int or physical < 1 or positions is None:
            raise ValueError("C1A reverse has invalid SP physical-row metadata")
        return physical, len(positions)

    def _reverse_attention_post(self, layer, rank, captured, output_gradient):
        """Reverse the frozen mHC-post shell around the draft contribution."""
        from .pypto_dsv41_tail_vjp import make_hc_post_data_vjp, make_hc_post_gate_vjp

        if not isinstance(output_gradient, SuffixGradient):
            raise TypeError("draft Attention reverse requires a suffix gradient")
        hidden_gradient, pre_gradient = output_gradient.hidden, output_gradient.pre_mix
        scattered = self._local_rows(rank, captured, captured["attention_scattered"])
        residual = self._local_rows(rank, captured, captured["hidden"], HC_COPIES)
        post = self._local_rows(rank, captured, captured["post"])
        combination = self._local_rows(rank, captured, captured["comb"], HC_COPIES)
        if (
            hidden_gradient.shape != (ROWS * HC_COPIES, HIDDEN)
            or pre_gradient.shape != (ROWS, HC_COPIES)
            or scattered.shape != (ROWS, HIDDEN)
            or residual.shape != (ROWS * HC_COPIES, HIDDEN)
            or post.shape != (ROWS, HC_COPIES)
            or combination.shape != (ROWS * HC_COPIES, HC_COPIES)
        ):
            raise ValueError("captured Attention mHC-post changed its K32 ABI")

        value_gradient = self._workspace(rank, (ROWS, HIDDEN))
        residual_gradient = self._workspace(rank, (ROWS * HC_COPIES, HIDDEN))
        post_gradient = self._workspace(rank, (ROWS, 16), torch.float32)
        combination_gradient = self._workspace(
            rank, (ROWS * HC_COPIES, 16), torch.float32,
        )
        self._execute(
            ("attn-hc-post-data-vjp", layer), rank,
            lambda: make_hc_post_data_vjp(ROWS, HIDDEN),
            output_gradient=hidden_gradient, post=post, combination=combination,
            value_gradient=value_gradient, residual_gradient=residual_gradient,
        )
        self._execute(
            ("attn-hc-post-gate-vjp", layer), rank,
            lambda: make_hc_post_gate_vjp(ROWS, HIDDEN),
            value=scattered, residual=residual, output_gradient=hidden_gradient,
            post_gradient=post_gradient,
            combination_gradient=combination_gradient,
        )
        return {
            "value": value_gradient,
            "residual": residual_gradient,
            "post": self._compact_mhc_columns(rank, post_gradient),
            "combination": self._compact_mhc_columns(rank, combination_gradient),
            "pre": pre_gradient,
        }

    def _reverse_c1a_rank(self, layer, rank, captured, contribution_gradient):
        """Run the rank-local projection and exact selected-history C1A VJP."""
        from .pypto_dsv41_attention_vjp import (
            C1AAttentionVJPGeometry, c1a_vjp_factories,
        )
        from .pypto_dsv41_online_programs import make_element, make_linear, make_rms

        geometry = C1AAttentionVJPGeometry()
        qkv = captured["qkv"]
        selection = captured.get("selection")
        history = captured["history"]
        if selection is None or history["role"].get("ratio") != 1:
            raise ValueError("resident draft reverse only supports selected C1A history")
        operands = self.draft.online_bank.operands(rank, layer)
        required = {"wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "sink", "wo_a", "wo_b"}
        if set(operands) & required != required:
            raise ValueError("resident draft bank changed the C1A operand ABI")
        if contribution_gradient.shape != (ROWS, HIDDEN):
            raise ValueError("local contribution gradient changed shape")

        # C1A suffix layers have the same kernel geometry; weights and replay
        # rows are passed per call. Reuse compiled entries across layers.
        # Wo-B and grouped Wo-A VJPs. Group-major staging lets the generic
        # batched GEMM preserve the native two-group ownership exactly.
        grouped = qkv["grouped"]
        unrotated = qkv["unrotated"]
        # These dW programs overwrite every tile and never accumulate into scratch.
        dwo_b = self._workspace(rank, operands["wo_b"].shape, torch.float32, fill=None)
        self._execute(
            ("attn-wo-b-weight-vjp",), rank,
            lambda: make_linear(
                (ROWS, 2048), (ROWS, HIDDEN), transpose_a=True,
                transpose_b=False, gradient_weight=True,
            ),
            a=grouped, b=contribution_gradient, output=dwo_b,
        )
        dgrouped = self._workspace(rank, (ROWS, 2048))
        self._execute(
            ("attn-wo-b-input-vjp",), rank,
            lambda: make_linear((ROWS, HIDDEN), (2048, HIDDEN)),
            a=contribution_gradient, b=operands["wo_b"], output=dgrouped,
        )
        grouped_gradient = self._workspace(rank, (2 * ROWS, 1024))
        grouped_input = self._workspace(rank, (2 * ROWS, 4096))
        self._execute(
            ("attn-grouped-gradient-split",), rank,
            lambda: _make_group_layout(ROWS, 2, 1024, split=True),
            source=dgrouped, output=grouped_gradient,
        )
        self._execute(
            ("attn-grouped-input-split",), rank,
            lambda: _make_group_layout(ROWS, 2, 4096, split=True),
            source=unrotated, output=grouped_input,
        )
        dwo_a_flat = self._workspace(rank, (2 * 1024, 4096), torch.float32, fill=None)
        self._execute(
            ("attn-wo-a-weight-vjp",), rank,
            lambda: make_linear(
                (ROWS, 1024), (ROWS, 4096), transpose_a=True,
                transpose_b=False, gradient_weight=True, batches=2,
            ),
            a=grouped_gradient, b=grouped_input, output=dwo_a_flat,
        )
        dunrotated_grouped = self._workspace(rank, (2 * ROWS, 4096))
        self._execute(
            ("attn-wo-a-input-vjp",), rank,
            lambda: make_linear(
                (ROWS, 1024), (1024, 4096), transpose_b=False, batches=2,
            ),
            a=grouped_gradient, b=self._draft_operand(rank, layer, "wo_a", (2 * 1024, 4096)),
            output=dunrotated_grouped,
        )
        dunrotated = self._workspace(rank, (ROWS, 8192))
        self._execute(
            ("attn-grouped-input-merge",), rank,
            lambda: _make_group_layout(ROWS, 2, 4096, split=False),
            source=dunrotated_grouped, output=dunrotated,
        )
        context_gradient = self._workspace(rank, (ROWS, 8192))
        self._execute(
            ("attn-output-rope-vjp",), rank,
            lambda: _make_rope_vjp(self.draft.native, 16, inverse=False),
            x=dunrotated, cos=captured["cos"], sin=captured["sin"],
            output=context_gradient, active=ROWS,
        )

        # Exact selected-history probability replay and temporary-block VJP.
        history_bias = self.draft.swa_history_bias[min(history["committed_tokens"], geometry.swa)]
        factories = c1a_vjp_factories(geometry)
        joined = captured["joined_history"]
        query = qkv["query"].view(geometry.rows, 512)
        probability = self._workspace(
            rank, (geometry.rows, geometry.padded_columns), torch.float32,
        )
        self._execute(
            ("attn-c1a-probability-replay",), rank,
            factories["probability_replay"],
            query=query, joined_key=joined, history_bias=history_bias,
            compressed_bias=selection["bias"], sink=operands["sink"],
            final_lse=captured["lse"], probability=probability,
        )
        probability_gradient = self._workspace(
            rank, (geometry.rows, geometry.padded_columns),
        )
        self._execute(
            ("attn-c1a-probability-gradient",), rank,
            factories["d_probability"],
            a=context_gradient.view(geometry.rows, 512), b=joined,
            output=probability_gradient,
        )
        score_gradient = self._workspace(
            rank, (geometry.rows, geometry.padded_columns),
        )
        sink_rows = self._workspace(rank, (geometry.rows, 16), torch.float32)
        block_score = self._workspace(rank, (geometry.rows, ROWS))
        block_probability = self._workspace(rank, (geometry.rows, ROWS))
        self._execute(
            ("attn-c1a-softmax-block-vjp",), rank,
            factories["softmax_block_vjp"],
            probability=probability, probability_gradient=probability_gradient,
            score_gradient=score_gradient, sink_rows=sink_rows,
            block_score=block_score, block_probability=block_probability,
        )
        query_gradient = self._workspace(rank, (geometry.rows, 512))
        self._execute(
            ("attn-c1a-query-vjp",), rank, factories["d_query"],
            a=score_gradient, b=joined, output=query_gradient,
        )
        key_gradient = self._workspace(rank, (ROWS, 512), torch.float32)
        value_gradient = self._workspace(rank, (ROWS, 512), torch.float32)
        self._execute(
            ("attn-c1a-key-block-vjp",), rank, factories["d_key_block"],
            a=block_score, b=query, output=key_gradient,
        )
        self._execute(
            ("attn-c1a-value-block-vjp",), rank, factories["d_value_block"],
            a=block_probability, b=context_gradient.view(geometry.rows, 512),
            output=value_gradient,
        )
        temporary_gradient = self._workspace(rank, (ROWS, 512))
        self._execute(
            ("attn-c1a-temporary-add",), rank,
            lambda: make_element(
                (ROWS, 512), "add", input_fp32=(True, True),
            ),
            a=key_gradient, b=value_gradient, output=temporary_gradient,
        )
        sink_padded = self._workspace(rank, (geometry.local_heads, 16), torch.float32)
        sink_gradient = self._workspace(rank, (1, 16), torch.float32)
        self._execute(
            ("attn-c1a-sink-reduce",), rank, factories["sink_reduce"],
            sink_rows=sink_rows, sink_gradient=sink_padded,
        )
        self._execute(
            ("attn-c1a-sink-compact",), rank, _make_sink_compact,
            source=sink_padded, output=sink_gradient,
        )

        # Reverse query and temporary RoPE, then both low-rank Q/KV branches.
        query_projection_gradient = self._workspace(rank, (ROWS, 8192))
        temporary_norm_gradient = self._workspace(rank, (ROWS, 512))
        self._execute(
            ("attn-query-rope-vjp",), rank,
            lambda: _make_rope_vjp(self.draft.native, 16, inverse=True),
            x=query_gradient.view(ROWS, 8192), cos=captured["cos"],
            sin=captured["sin"], output=query_projection_gradient, active=ROWS,
        )
        self._execute(
            ("attn-kv-rope-vjp",), rank,
            lambda: _make_rope_vjp(self.draft.native, 1, inverse=True),
            x=temporary_gradient, cos=captured["cos"], sin=captured["sin"],
            output=temporary_norm_gradient, active=ROWS,
        )

        dlatent = self._workspace(rank, (ROWS, 1280))
        dwq_b = self._workspace(rank, operands["wq_b"].shape, torch.float32, fill=None)
        self._execute(
            ("attn-wq-b-input-vjp",), rank,
            lambda: make_linear((ROWS, 8192), (1280, 8192)),
            a=query_projection_gradient, b=operands["wq_b"], output=dlatent,
        )
        self._execute(
            ("attn-wq-b-weight-vjp",), rank,
            lambda: make_linear(
                (ROWS, 1280), (ROWS, 8192), transpose_a=True,
                transpose_b=False, gradient_weight=True,
            ),
            a=qkv["latent"], b=query_projection_gradient, output=dwq_b,
        )
        q_gamma = self._draft_operand(rank, layer, "q_norm", (1, 1280))
        dqa = self._workspace(rank, (ROWS, 1280), fill=None)
        dq_norm = self._workspace(rank, (1, 1280), torch.float32, fill=None)
        self._execute(
            ("attn-q-norm-vjp", str(q_gamma.dtype)), rank,
            lambda: make_rms(
                ROWS, 1280, mode="trainable", epsilon=1e-20,
                gamma_bf16=q_gamma.dtype == torch.bfloat16,
            ),
            x=qkv["qa"], gamma=q_gamma, dy=dlatent, dx=dqa, dg=dq_norm,
        )
        dwq_a = self._workspace(rank, operands["wq_a"].shape, torch.float32, fill=None)
        query_input_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("attn-wq-a-weight-vjp",), rank,
            lambda: make_linear(
                (ROWS, HIDDEN), (ROWS, 1280), transpose_a=True,
                transpose_b=False, gradient_weight=True,
            ),
            a=qkv["hidden"], b=dqa, output=dwq_a,
        )
        self._execute(
            ("attn-wq-a-input-vjp",), rank,
            lambda: make_linear((ROWS, 1280), (HIDDEN, 1280)),
            a=dqa, b=operands["wq_a"], output=query_input_gradient,
        )

        kv_gamma = self._draft_operand(rank, layer, "kv_norm", (1, 512))
        dkv = self._workspace(rank, (ROWS, 512), fill=None)
        dkv_norm = self._workspace(rank, (1, 512), torch.float32, fill=None)
        self._execute(
            ("attn-kv-norm-vjp", str(kv_gamma.dtype)), rank,
            lambda: make_rms(
                ROWS, 512, mode="trainable", epsilon=1e-20,
                gamma_bf16=kv_gamma.dtype == torch.bfloat16,
            ),
            x=qkv["kv"], gamma=kv_gamma, dy=temporary_norm_gradient, dx=dkv, dg=dkv_norm,
        )
        dwkv = self._workspace(rank, operands["wkv"].shape, torch.float32, fill=None)
        kv_input_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("attn-wkv-weight-vjp",), rank,
            lambda: make_linear(
                (ROWS, HIDDEN), (ROWS, 512), transpose_a=True,
                transpose_b=False, gradient_weight=True,
            ),
            a=qkv["hidden"], b=dkv, output=dwkv,
        )
        self._execute(
            ("attn-wkv-input-vjp",), rank,
            lambda: make_linear((ROWS, 512), (HIDDEN, 512)),
            a=dkv, b=operands["wkv"], output=kv_input_gradient,
        )
        input_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("attn-q-kv-input-add",), rank,
            lambda: make_element((ROWS, HIDDEN), "add"),
            a=query_input_gradient, b=kv_input_gradient, output=input_gradient,
        )
        parameters = {
            "wq_a.weight": dwq_a,
            "q_norm.weight": dq_norm.view(1280),
            "wq_b.weight": dwq_b,
            "wkv.weight": dwkv,
            "kv_norm.weight": dkv_norm.view(512),
            "attn_sink": sink_gradient.view(16),
            "wo_a.weight": dwo_a_flat.view(2, 1024, 4096),
            "wo_b.weight": dwo_b,
        }
        return input_gradient, parameters

    def _reverse_attention_pre(
        self, layer, rank, captured, shell, normalized_gradient,
    ):
        """Reverse frozen Attention norm and exact mHC-pre replay."""
        from .pypto_dsv41_online_programs import make_rms
        from .pypto_dsv41_tail_vjp import (
            make_hc_pre_data_vjp, make_hc_pre_gate_vjp,
            make_hc_pre_projection_forward,
        )

        collapsed = self._local_rows(rank, captured, captured["collapsed"])
        residual = self._local_rows(rank, captured, captured["hidden"], HC_COPIES)
        incoming = self._local_rows(rank, captured, captured["incoming"])
        norm = self._pinned(rank, f"layers.{layer}.attn_norm", (1, HIDDEN))
        collapsed_gradient = self._workspace(rank, (ROWS, HIDDEN))
        self._execute(
            ("attn-rms-frozen-input-vjp", layer), rank,
            lambda: make_rms(
                ROWS, HIDDEN, mode="frozen_input", epsilon=1e-20,
                gamma_bf16=norm.dtype == torch.bfloat16,
            ),
            x=collapsed, gamma=norm,
            dy=normalized_gradient, dx=collapsed_gradient,
        )

        function = self._pinned(rank, f"layers.{layer}.hc_function")
        scale = self._pinned(rank, f"layers.{layer}.hc_scale")
        base = self._pinned(rank, f"layers.{layer}.hc_base")
        projection = self._workspace(rank, (ROWS, 32), torch.float32)
        hc_inverse = self._workspace(rank, (ROWS, 16), torch.float32)
        self._execute(
            ("attn-hc-pre-projection", layer), rank,
            lambda: make_hc_pre_projection_forward(ROWS, HIDDEN),
            residual=residual, function=function,
            projection=projection, inverse=hc_inverse,
        )
        projection_gradient = self._workspace(rank, (ROWS, 32), torch.float32)
        self._execute(
            ("attn-hc-pre-gate-vjp", layer), rank,
            lambda: make_hc_pre_gate_vjp(ROWS),
            projection=projection, scale=scale, base=base,
            pre_gradient=shell["pre"], post_gradient=shell["post"],
            combination_gradient=shell["combination"],
            projection_gradient=projection_gradient,
        )
        hidden_gradient = self._workspace(rank, (ROWS * HC_COPIES, HIDDEN))
        incoming_gradient = self._workspace(rank, (ROWS, 16), torch.float32)
        self._execute(
            ("attn-hc-pre-data-vjp", layer), rank,
            lambda: make_hc_pre_data_vjp(ROWS, HIDDEN),
            residual=residual, incoming=incoming, function=function,
            inverse=hc_inverse, projection_gradient=projection_gradient,
            collapsed_gradient=collapsed_gradient,
            residual_passthrough=shell["residual"],
            residual_gradient=hidden_gradient,
            incoming_gradient=incoming_gradient,
        )
        return SuffixGradient(hidden_gradient, self._compact_mhc_columns(rank, incoming_gradient))

    @profiled_stage("attention_vjp")
    def reverse_draft_attention_owner_batch(
        self, *, layer, ranks, captured, output_gradients,
    ):
        """Reverse one complete node-local TP4 C1A draft-Attention batch."""
        from .pypto_dsv41_attention_vjp import (
            make_attention_input_gradient_pack, make_attention_output_gradient_unpack,
        )

        ranks = tuple(ranks)
        if (
            len(ranks) != 4
            or ranks != tuple(range(ranks[0] // 4 * 4, ranks[0] // 4 * 4 + 4))
            or not set(ranks) <= set(self.ranks)
            or set(captured) != set(ranks)
            or set(output_gradients) != set(ranks)
        ):
            raise ValueError("draft Attention reverse requires one complete local TP4 group")
        shells = {
            rank: self._reverse_attention_post(
                layer, rank, captured[rank], output_gradients[rank],
            )
            for rank in ranks
        }

        # Reverse the native FP32 reduce-scatter: place each request gradient
        # back into its physical owner slab, all-gather slabs, then apply the
        # same SP unpack permutation used by the forward input gather.
        scattered = {}
        geometry = {}
        for rank in ranks:
            row = captured[rank]
            begin, end = self._span(row)
            span = self._local_span(row)
            capacity = row["attention_scattered"].shape[0]
            physical, total = self._attention_record(row)
            if not 0 <= begin < end <= total <= min(capacity, 4 * physical) or not 0 < physical <= ROWS:
                raise ValueError("captured Attention span is outside the SP slab")
            slab = self._workspace(rank, (ROWS, HIDDEN), fill=0)
            self._copy_rows(rank, slab, shells[rank]["value"][:span.rows],
                            destination=span.local_start)
            scattered[rank] = slab
            geometry[rank] = begin, physical, total
        gathered = self._tp_collective(
            scattered, "all_gather", stage=f"layer{layer}-attn-output-vjp",
        )
        contribution_gradients = {}
        for rank in ranks:
            begin, physical, _ = geometry[rank]
            output = self._workspace(rank, (ROWS, HIDDEN), fill=None)
            self._execute(
                ("attn-output-gradient-unpack",), rank,
                lambda: make_attention_output_gradient_unpack(ROWS, HIDDEN, 4),
                packet=gathered[rank], output=output, begin=begin, physical=physical,
            )
            contribution_gradients[rank] = output

        # Each lane owns a different weight shard, even if its SP slab is empty.
        local = self._map_owner_ranks(
            ranks,
            lambda rank: self._reverse_c1a_rank(
                layer, rank, captured[rank], contribution_gradients[rank],
            ),
        )

        # Reverse the native normalized-input all-gather: pack each rank's
        # request gradient directly into the rank-local SP rows. The native
        # wide buffer has at most 32 live rows per TP rank.
        widened = {}
        for rank in ranks:
            begin, physical, total = geometry[rank]
            wide = self._workspace(rank, (4 * ROWS, HIDDEN), torch.float32, fill=None)
            self._execute(
                ("attn-input-gradient-pack",), rank,
                lambda: make_attention_input_gradient_pack(ROWS, HIDDEN, 4),
                gradient=local[rank][0], packed=wide, begin=begin, physical=physical, active=total,
            )
            widened[rank] = wide
        reduced = self._tp_collective(
            widened, "reduce_scatter", stage=f"layer{layer}-attn-input-vjp",
        )
        result = {}
        for rank in ranks:
            narrowed = self._workspace(rank, (ROWS, HIDDEN), fill=None)
            self._execute(
                ("attn-input-gradient-narrow",), rank,
                lambda: self.draft.native("input_pack").make_sp_reduce_narrow(ROWS, HIDDEN),
                value=reduced[rank], output=narrowed,
            )
            incoming = self._reverse_attention_pre(
                layer, rank, captured[rank], shells[rank],
                self._local_rows(rank, captured[rank], narrowed),
            )
            result[rank] = incoming, local[rank][1]
        return result

    @profiled_stage("tp_gradient_assembly")
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

    def _dp_zero(self, rank, shape):
        """Give every missing DP gradient a read-only view of one zero buffer."""
        if self._dp_zero_storage is None:
            largest = max(state.master.numel() for state in self._adam.values())
            self._dp_zero_storage = torch.zeros(largest, dtype=torch.float32)
        if rank not in self._dp_zero_ranks:
            self.resident.pin(
                self._dp_zero_storage, (self._local(rank),), readonly=True,
                name=f"online-dp/zero/rank{rank}", auxiliary=True,
            )
            self._dp_zero_ranks.add(rank)
        if shape not in self._dp_zero_views:
            self._dp_zero_views[shape] = self._dp_zero_storage[:math.prod(shape)].view(shape)
        return self._dp_zero_views[shape]

    @profiled_stage("dp_gradient_reduce")
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
                    value = self._dp_zero(rank, self._adam[rank, layer, short].master.shape)
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
                "all_reduce", "dp", write_only_receive=True,
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

    @profiled_stage("gradient_norm_clip")
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
        replicated = self._replicated_tp_parameters()
        selected = {
            name: group[:1] if name.split(".attn.", 1)[1] in replicated else group
            for name in values
        }

        def rank_statistics(index):
            rank = self.ranks[index]
            return {
                name: self._tensor_statistics(rank, gradient.values[rank], name)
                for name, gradient in values.items() if rank in selected[name]
            }

        statistics = self._map_online_ranks(rank_statistics)
        squared, finite = 0., True
        # Preserve the parameter-major accumulation order used for clipping.
        for name, ranks in selected.items():
            for rank in ranks:
                part, good = statistics[self._local(rank)][name]
                squared += part
                finite = finite and good
        norm = math.sqrt(squared)
        if not finite or not math.isfinite(norm):
            raise FloatingPointError("non-finite DSV4.1 online gradient")
        clip = min(1., maximum / (norm + 1e-6))
        self._last_norm = norm
        return _HostScalar(norm), _HostScalar(clip)

    def _tensor_statistics(self, rank, value, name):
        from .pypto_dsv41_optimizer_programs import make_gradient_statistics

        shape = tuple(value.shape)
        sums = self._workspace(rank, (1, 8), torch.float32, fill=None)
        flags = self._workspace(rank, (1, 8), torch.float32, fill=None)
        self._execute(
            ("gradient-stats", shape), rank, lambda: make_gradient_statistics(shape),
            value=value, sums=sums, flags=flags,
        )
        local = self._local(rank)
        host_sums = self.resident.read_output(sums, local)
        host_flags = self.resident.read_output(flags, local)
        return float(host_sums.sum()), bool((host_flags == 1).all())

    @profiled_stage("adamw")
    def adamw(self, values, **options):
        self._require_optimizer_state()
        from .pypto_dsv41_online_programs import make_adam, make_adam_publish

        rate, step, clip = options["rate"], options["step"], options["clip"]
        clip = clip.value if isinstance(clip, _HostScalar) else float(clip)
        if step != self._last_step + 1 or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Adam step/rate is out of order or invalid")
        parameters = [
            (int(name.split(".")[1]), name.split(".attn.", 1)[1], gradient)
            for name, gradient in values.items()
        ]
        correction1, correction2 = 1. - .9 ** step, 1. - .999 ** step

        def update_rank(index):
            rank = self.ranks[index]
            for layer, short, gradient in parameters:
                state = self._adam[rank, layer, short]
                shape = tuple(state.master.shape)
                operands = dict(
                    master=state.master, gradient=gradient.values[rank],
                    first=state.first, second=state.second, next_master=state.candidate_master,
                    next_first=state.candidate_first, next_second=state.candidate_second,
                    rate=rate, correction1=correction1, correction2=correction2, clip=clip,
                )
                if state.execution_dtype == torch.bfloat16:
                    self._execute(
                        ("adam-publish", shape), rank, lambda: make_adam_publish(shape),
                        **operands, execution=state.execution,
                    )
                elif state.execution_dtype == torch.float32:
                    self._execute(("adam", shape), rank, lambda: make_adam(shape), **operands)
                    self.resident.copy_tensor(state.execution, state.candidate_master, self._local(rank))
                else:
                    raise TypeError("unsupported draft execution dtype")

        self._map_online_ranks(update_rank)
        # No rank advances its optimizer state before all candidate writes finish.
        for rank in self.ranks:
            for layer, short, _ in parameters:
                state = self._adam[rank, layer, short]
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
        # The reduction resolves absent contributions on each rank. A
        # participant can join without allocating a tensor per parameter.
        return {name: _RankGradient(int(name.split(".")[1]), {}) for name in names}

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
    """Construct the production resident online-update backend."""
    readiness = production_readiness()
    if not readiness["production_ready"]:
        raise ProductionFactoryIncomplete(
            "DSV4.1 resident online factory is incomplete: "
            + "; ".join(readiness["blockers"])
        )
    return create_terminal_primitives(draft=draft, native=native, cluster=cluster, config=config)
