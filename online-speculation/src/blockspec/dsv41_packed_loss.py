"""Full-vocabulary loss on the packed draft request's resident L3 worker."""

from dataclasses import dataclass, field
from time import perf_counter

from .pypto_dsv41_online_programs import make_kl, make_tv


def make_packed_loss_graph(rows, vocabulary, dp, tp, kind):
    import pypto.language as pl

    factory = {"tv": make_tv, "forward_kl": make_kl}[kind]
    loss_vjp = factory(rows, vocabulary, input_fp32=True, inline=True)

    @pl.jit
    def rank_loss(
        student: pl.Tensor[[rows, vocabulary], pl.FP32],
        teacher: pl.Tensor[[rows, vocabulary], pl.FP32],
        gradient: pl.Out[pl.Tensor[[rows, vocabulary], pl.BF16]],
        losses: pl.Out[pl.Tensor[[rows, 8], pl.FP32]],
        valid: pl.Scalar[pl.INT32],
    ):
        return loss_vjp(student, teacher, gradient, losses, valid, valid)

    @pl.jit.host
    def dispatch(
        student: pl.Tensor[[dp, rows, vocabulary], pl.FP32],
        teacher: pl.Tensor[[dp, rows, vocabulary], pl.FP32],
        gradient: pl.Out[pl.Tensor[[dp, rows, vocabulary], pl.BF16]],
        losses: pl.Out[pl.Tensor[[dp, rows, 8], pl.FP32]],
        owner: pl.Scalar[pl.INT32],
        valid: pl.Scalar[pl.INT32],
    ):
        replica = owner // tp
        rank_loss(student[replica], teacher[replica], gradient[replica], losses[replica],
                  valid, device=owner)

    return dispatch


@dataclass(frozen=True)
class PackedLossResult:
    """Gradient remains valid until the next loss on its owning rank."""

    owner: int
    rows: int
    value: float
    gradient: object
    teacher_upload_bytes: int
    seconds: float


@dataclass
class PackedTeacherRows:
    """Contiguous teacher rows on the committed branch after one draft capture."""

    student: object
    chunks: list = field(default_factory=list)

    @property
    def remaining(self):
        return self.student.rows - sum(len(rows) for rows in self.chunks)

    def append(self, position, values):
        remaining = self.remaining
        if position != self.student.position + self.student.rows - remaining or not 1 <= len(values) <= remaining:
            raise ValueError("Packed teacher rows must extend the captured draft's committed prefix")
        self.chunks.append(values)

    def logits(self, *, finished=False):
        import torch

        if not self.chunks or self.remaining and not finished:
            return None
        if len(self.chunks) == 1:
            return self.chunks[0]
        return torch.cat(self.chunks, dim=0)


class PackedLoss:
    """Route a retained FP32 student and the authoritative teacher to one DP owner."""

    def __init__(self, proposer, graph):
        import torch
        from pypto.runtime import StackedDeviceTensor

        self.runtime, self.config, self.graph = proposer.runtime, proposer.config, graph
        self.owners = tuple(range(0, proposer.world, proposer.tp))
        self.student = tuple(proposer.ranks[rank]["logits"] for rank in self.owners)
        self.rows, self.vocabulary = self.student[0].shape
        self.stacked = StackedDeviceTensor

        def allocate(shape, dtype):
            values = [self.runtime.alloc_tensor(shape, dtype, worker_id=rank) for rank in self.owners]
            return self._stack(values, shape)

        shape = self.rows, self.vocabulary
        self.teacher = allocate(shape, torch.float32)
        self.gradient = allocate(shape, torch.bfloat16)
        self.losses = allocate((self.rows, 8), torch.float32)
        self.host_losses = torch.empty((self.rows, 8), dtype=torch.float32)

    def _stack(self, shards, shape):
        return self.stacked(shards, (len(self.owners), *shape), self.owners)

    def __call__(self, capture, teacher):
        import torch

        shape = self.rows, self.vocabulary
        if (teacher.ndim != 2 or teacher.shape[1] != self.vocabulary
                or not 1 <= teacher.shape[0] <= capture.rows <= self.rows
                or capture.logits.shape != shape or capture.logits.dtype != torch.float32
                or teacher.device.type != "cpu" or not teacher.is_floating_point()):
            raise ValueError("Packed loss requires aligned, full-vocabulary teacher and student rows")
        teacher = teacher.to(dtype=torch.float32).contiguous()
        if not torch.isfinite(teacher).all():
            raise ValueError("Packed teacher logits must be finite")
        owner = capture.owner
        replica = self.owners.index(owner)
        shards = list(self.student)
        shards[replica] = capture.logits
        start = perf_counter()
        self.runtime.copy_to(self.teacher.shards[replica].data_ptr, teacher.data_ptr(),
                             teacher.nbytes, worker_id=owner)
        self.runtime.run(self.graph, self._stack(shards, shape), self.teacher, self.gradient,
                         self.losses, owner, teacher.shape[0], config=self.config)
        self.runtime.copy_from(self.host_losses.data_ptr(), self.losses.shards[replica].data_ptr,
                               self.host_losses.nbytes, worker_id=owner)
        return PackedLossResult(owner, teacher.shape[0], float(self.host_losses[:, 0].sum()),
                                self.gradient.shards[replica], teacher.nbytes, perf_counter() - start)
