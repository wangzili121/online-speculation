"""Full-vocabulary loss on the packed draft request's resident L3 worker."""

from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter

from .dsv41_online_profile import active_profile, timed_stage
from .pypto_dsv41_online_programs import loss_vocabulary_tile, make_kl, make_tv

TEACHER_COPY_SHARDS = 16


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


def make_packed_teacher_copy(rows, vocabulary, dp, tp):
    """Snapshot selected Verify rows and reduce their finite flags on device."""
    import pypto.language as pl

    tile = loss_vocabulary_tile(vocabulary)
    shards = TEACHER_COPY_SHARDS
    flags_width = shards * 8
    tiles_per_shard = (vocabulary // tile + shards - 1) // shards

    @pl.jit
    def snapshot(
        source: pl.Tensor[[rows, vocabulary], pl.FP32],
        target: pl.InOut[pl.Tensor[[rows, vocabulary], pl.FP32]],
        flags: pl.InOut[pl.Tensor[[rows, flags_width], pl.FP32]],
        first: pl.Scalar[pl.INT32], cursor: pl.Scalar[pl.INT32], count: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(rows * shards, name_hint="orthrus_committed_teacher_rows"):
            block = pl.tile.get_block_idx()
            row = block // shards
            shard = block % shards
            if row < count:
                valid = pl.full([8, tile // 8], dtype=pl.FP32, value=1.)
                for offset in pl.range(tiles_per_shard):
                    col = (shard * tiles_per_shard + offset) * tile
                    if col < vocabulary:
                        value = pl.slice(source, [1, tile], [first + row, col])
                        target[cursor + row:cursor + row + 1, col:col + tile] = value
                        finite = pl.cmp(pl.abs(pl.reshape(value, [8, tile // 8])), 3.4028234663852886e38, cmp_type=3)
                        valid = pl.minimum(valid, finite)
                flags[cursor + row:cursor + row + 1, shard * 8:(shard + 1) * 8] = pl.reshape(pl.row_min(valid), [1, 8])

    @pl.jit.host
    def dispatch(
        source: pl.Tensor[[dp, rows, vocabulary], pl.FP32],
        target: pl.InOut[pl.Tensor[[dp, rows, vocabulary], pl.FP32]],
        flags: pl.InOut[pl.Tensor[[dp, rows, flags_width], pl.FP32]],
        owner: pl.Scalar[pl.INT32], first: pl.Scalar[pl.INT32], cursor: pl.Scalar[pl.INT32],
        count: pl.Scalar[pl.INT32],
    ):
        replica = owner // tp
        snapshot(source[replica], target[replica], flags[replica], first, cursor, count, device=owner)

    return dispatch


@dataclass(frozen=True)
class ResidentTeacherLogits:
    rows: int
    buffer: object
    flags: object

    def __len__(self):
        return self.rows


class ResidentPackedTeacherRows:
    """Retain the committed prefix across Verify calls on the request's worker."""

    def __init__(self, student, collector, buffer, flags):
        self.student, self.collector = student, collector
        self.buffer, self.flags = buffer, flags
        self.rows = 0

    @property
    def remaining(self):
        return self.student.rows - self.rows

    def append_from(self, position, first, count):
        if (position != self.student.position + self.rows or not 1 <= count <= self.remaining
                or not 0 <= first <= self.collector.loss.rows - count):
            raise ValueError("Packed teacher rows must extend the captured draft's committed prefix")
        self.collector.snapshot(self, first, count)
        self.rows += count

    def logits(self, *, finished=False):
        if not self.rows or (self.remaining and not finished):
            return None
        return ResidentTeacherLogits(self.rows, self.buffer, self.flags)

    def close(self):
        for value in (self.buffer, self.flags):
            self.collector.loss.runtime.free_tensor(value, worker_id=self.student.owner)


class PackedTeacherCopy:
    def __init__(self, loss, graph, sources):
        import torch

        self.loss, self.graph = loss, graph
        self.source = loss._stack(sources, (loss.rows, loss.vocabulary))
        self.flags_shape = (loss.rows, TEACHER_COPY_SHARDS * 8)
        self.padding = tuple(loss.runtime.alloc_tensor(self.flags_shape, torch.float32, worker_id=owner)
                             for owner in loss.owners)

    def capture(self, student):
        import torch

        loss = self.loss
        flags = loss.runtime.alloc_tensor(self.flags_shape, torch.float32, worker_id=student.owner)
        try:
            buffer = loss.runtime.alloc_tensor((loss.rows, loss.vocabulary), torch.float32, worker_id=student.owner)
        except BaseException:
            loss.runtime.free_tensor(flags, worker_id=student.owner)
            raise
        return ResidentPackedTeacherRows(student, self, buffer, flags)

    def snapshot(self, rows, first, count):
        loss = self.loss
        owner = rows.student.owner
        replica = loss.owners.index(owner)
        target, flags = list(loss.teacher.shards), list(self.padding)
        target[replica], flags[replica] = rows.buffer, rows.flags
        with timed_stage("teacher_device_snapshot"):
            loss.runtime.run(self.graph, self.source, loss._stack(target, rows.buffer.shape),
                             loss._stack(flags, rows.flags.shape), owner, first, rows.rows, count, config=loss.config)
        profile = active_profile()
        if profile is not None and profile.current is not None:
            profile.current["counts"]["teacher_device_bytes"] += count * loss.vocabulary * 4


def _require_finite_host(teacher):
    import torch

    extrema = torch.aminmax(teacher)
    if not isfinite(extrema.min) or not isfinite(extrema.max):
        raise ValueError("Packed teacher logits must be finite")


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
        self.host_flags = torch.empty((self.rows, TEACHER_COPY_SHARDS * 8), dtype=torch.float32)

    def _stack(self, shards, shape):
        return self.stacked(shards, (len(self.owners), *shape), self.owners)

    def __call__(self, capture, teacher):
        import torch

        shape = self.rows, self.vocabulary
        resident = isinstance(teacher, ResidentTeacherLogits)
        if not resident and (teacher.ndim != 2 or teacher.shape[1] != self.vocabulary
                             or teacher.device.type != "cpu" or not teacher.is_floating_point()):
            raise ValueError("Packed loss requires aligned, full-vocabulary teacher and student rows")
        count = teacher.rows if resident else teacher.shape[0]
        if (not 1 <= count <= capture.rows <= self.rows
                or capture.logits.shape != shape or capture.logits.dtype != torch.float32):
            raise ValueError("Packed loss requires aligned, full-vocabulary teacher and student rows")
        owner = capture.owner
        replica = self.owners.index(owner)
        shards = list(self.student)
        shards[replica] = capture.logits
        start = perf_counter()
        upload = 0
        if resident:
            flags = self.host_flags[:count]
            self.runtime.copy_from(flags.data_ptr(), teacher.flags.data_ptr, flags.nbytes, worker_id=owner)
            if not bool((flags == 1).all()):
                raise ValueError("Packed teacher logits must be finite")
            source = list(self.teacher.shards)
            source[replica] = teacher.buffer
            teacher_stack = self._stack(source, shape)
        else:
            teacher = teacher.to(dtype=torch.float32).contiguous()
            with timed_stage("teacher_check_host"):
                _require_finite_host(teacher)
            self.runtime.copy_to(self.teacher.shards[replica].data_ptr, teacher.data_ptr(),
                                 teacher.nbytes, worker_id=owner)
            teacher_stack, upload = self.teacher, teacher.nbytes
        self.runtime.run(self.graph, self._stack(shards, shape), teacher_stack, self.gradient,
                         self.losses, owner, count, config=self.config)
        self.runtime.copy_from(self.host_losses.data_ptr(), self.losses.shards[replica].data_ptr,
                               self.host_losses.nbytes, worker_id=owner)
        return PackedLossResult(owner, count, float(self.host_losses[:, 0].sum()),
                                self.gradient.shards[replica], upload, perf_counter() - start)
