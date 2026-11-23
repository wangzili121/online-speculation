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
        ),
        "blockers": (
            "resident C1A draft-attention parameter/input VJP orchestration",
            "fixed-route mHC/MoE reverse with EP16 expert exchange",
            "TP4 parameter-gradient assembly and DP4 all-reduce",
            "persistent FP32 Adam state and D2D ResidentHeadState publication",
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
        if kind not in ("tv", "kl") or not 1 <= valid < ROWS:
            raise ValueError("resident loss requires TV/KL and 1..31 authoritative rows")
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
        factory = (make_tv if token.kind == "tv" else make_kl)
        self._execute(
            ("loss", token.kind, vocabulary), rank,
            lambda: factory(ROWS, vocabulary),
            student=student, teacher=teacher, gradient=gradient, losses=losses,
            valid=token.valid, normalizer=token.valid,
        )
        return gradient, losses

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

        hidden_gradients, premix_gradients = {}, {}
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
                lambda: make_rms(ROWS, HIDDEN),
                x=collapsed, gamma=norm, y=normalized, inv=inverse,
            )

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
            dcollapsed = self._workspace(rank, (ROWS, HIDDEN))
            self._execute(
                ("final-rms-input-vjp",), rank,
                lambda: make_rms(ROWS, HIDDEN, mode="input"),
                x=collapsed, gamma=norm, inv=inverse, dy=dnormalized, dx=dcollapsed,
            )
            dstreams = self._workspace(rank, (ROWS * HC_COPIES, HIDDEN))
            dpremix = self._workspace(rank, (ROWS, 16), torch.float32)
            self._execute(
                ("final-collapse-vjp",), rank,
                lambda: make_hc_collapse_vjp(ROWS, HIDDEN),
                streams=streams, premix=premix, output_gradient=dcollapsed,
                stream_gradient=dstreams, premix_gradient=dpremix,
            )
            hidden_gradients[rank], premix_gradients[rank] = dstreams, dpremix
        return SuffixGradient(hidden_gradients, premix_gradients)

    @staticmethod
    def _unfinished(stage):
        blockers = production_readiness()["blockers"]
        raise ProductionFactoryIncomplete(f"{stage} is not production-wired: {'; '.join(blockers)}")

    def reverse_frozen_tail(self, **options):
        self._unfinished("fixed-route mHC/MoE reverse")

    def reverse_draft_attention(self, **options):
        self._unfinished("C1A draft Attention reverse")

    def tp_assemble_gradients(self, layer, values):
        self._unfinished("TP4 parameter-gradient assembly")

    def dp_all_reduce(self, values):
        self._unfinished("DP4 parameter-gradient all-reduce")

    def gradient_norm_and_clip(self, values, maximum):
        self._unfinished("resident global gradient norm")

    def adamw(self, values, **options):
        self._unfinished("persistent FP32 Adam")

    def zero_gradients(self, layers, names):
        self._unfinished("resident zero-gradient participant")

    def publication_state(self, layers):
        self._unfinished("D2D ResidentHeadState publication")

    def read_scalar(self, value):
        if isinstance(value, _DeferredLoss):
            value = value.value
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
