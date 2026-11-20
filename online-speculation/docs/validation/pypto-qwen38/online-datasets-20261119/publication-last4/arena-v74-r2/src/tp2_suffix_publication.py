"""在设备上排布 last-4 大矩阵；很小的 decay/bias 仍按传输缓冲整理。"""

from dataclasses import dataclass

from .tp2_publication import FULL_BANKS


GDN_BANKS = {name: i for i, name in enumerate((
    "in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight", "in_proj_a.weight",
    "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight"))}


@dataclass(frozen=True)
class Copy:
    bank: int
    name: str
    source: tuple
    target: tuple
    dtype: str
    mode: str
    slot: int
    local_rows: int
    rank_rows: int


@dataclass(frozen=True)
class Destination:
    tensor: object
    base: int
    offset: int


def suffix_publication_plan(layout, selected):
    import torch
    from .head_loading import bank_specs
    from .oneserving_tp2 import SHAPES

    modes = ("qkv_rows", "rows", "columns", "padded_rows", "padded_rows",
             "qkv_columns", "scalar", "scalar", "replicate")
    plan = []
    for entry in layout.entries:
        p = entry.parameter
        if not 60 <= p.layer <= 63:
            raise ValueError("suffix publication requires layers 60..63")
        if p.layer == 63:
            bank = FULL_BANKS[bank_specs(63)[p.bank][0]]
            mode = "replicate" if p.offset_norm else "columns" if bank == 14 else "direct"
            slot = 0
        else:
            if not 0 <= p.bank < len(modes):
                raise ValueError("invalid GDN publication bank")
            # 训练参数顺序与 serving bank 不同，按规范名称建立映射。
            bank = GDN_BANKS[bank_specs(p.layer)[p.bank][0]]
            mode, slot = modes[bank], p.layer - 60
        if bank not in selected:
            raise ValueError("parameter has no selected publication bank")
        (rank_rows, columns), dtype = SHAPES[bank]
        local_rows = rank_rows // (3 if p.layer < 63 else 1)
        expected = {"direct": (2 * rank_rows, columns), "rows": (2 * local_rows, columns),
                    "columns": (local_rows, 2 * columns), "qkv_rows": (10240, columns),
                    "qkv_columns": (local_rows, 10240), "padded_rows": (128, columns),
                    "scalar": (1, 64), "replicate": (local_rows, columns)}[mode]
        if p.shape != expected or p.execution_dtype != ("fp32" if dtype == torch.float32 else "bf16"):
            raise ValueError("source shape or dtype does not match the serving bank")
        plan.append(Copy(bank, p.name, p.shape, (2 * rank_rows, columns),
                         p.execution_dtype, mode, slot, local_rows, rank_rows))
    if {r.bank for r in plan} != set(selected):
        raise ValueError("publication bank set does not match the trainable layout")
    for bank in selected:
        rows = [r for r in plan if r.bank == bank]
        if len(rows) != (3 if bank < 9 else 1) or {r.slot for r in rows} != (
                {0, 1, 2} if bank < 9 else {0}):
            raise ValueError("every selected bank must contain its complete layer stack")
    return tuple(plan)


def make_suffix_copy(spec):
    import pypto.language as pl

    _, n = spec.target
    m = spec.local_rows
    r = 8 if m % 8 == 0 else 1
    t = next(v for v in (1024, 512, 256, 128, 64) if n % v == 0)
    tasks, blocks = 2 * m // r * (n // t), min(64, 2 * m // r * (n // t))
    slot, rank_rows = spec.slot, spec.rank_rows
    padded = spec.mode == "padded_rows"
    split_rows, split_columns = spec.mode == "rows", spec.mode == "columns"
    qkv_rows, qkv_columns = spec.mode == "qkv_rows", spec.mode == "qkv_columns"

    @pl.jit.inline
    def copy(source: pl.Tensor, target: pl.InOut[pl.Tensor]):
        with pl.spmd(blocks, name_hint="orthrus_tp2_suffix_pack"):
            for tile in pl.range(pl.tile.get_block_idx(), tasks, blocks):
                logical = tile // (n // t) * r
                rank = logical // m
                row = logical % m
                col = tile % (n // t) * t
                dst = rank * rank_rows + slot * m + row
                if padded:
                    if row < 24:
                        target[dst:dst + r, col:col + t] = pl.slice(source, [r, t], [rank * 24 + row, col])
                    else:
                        target[dst:dst + r, col:col + t] = pl.full([r, t], dtype=pl.BF16, value=0.)
                else:
                    sr = row
                    sc = col
                    if split_rows:
                        sr = rank * m + row
                    elif split_columns:
                        sc = rank * n + col
                    elif qkv_rows:
                        if row < 1024:
                            sr = row + rank * 1024
                        elif row < 2048:
                            sr = row + 1024 + rank * 1024
                        else:
                            sr = row + 2048 + rank * 3072
                    elif qkv_columns:
                        if col < 1024:
                            sc = col + rank * 1024
                        elif col < 2048:
                            sc = col + 1024 + rank * 1024
                        else:
                            sc = col + 2048 + rank * 3072
                    target[dst:dst + r, col:col + t] = pl.slice(source, [r, t], [sr, sc])
        return target

    return copy


class SuffixPublicationPrograms:
    """只改变参数排布，不改变舍入、padding 或两张卡的发布边界。"""

    def __init__(self, layout, selected, config):
        import torch
        import pypto.language as pl
        from .training_pipeline import bind

        self.plan, self.config = suffix_publication_plan(layout, selected), config
        self.materialized = tuple(r for r in self.plan if r.mode not in {"direct", "scalar", "rows"})
        self.outputs = {r.bank: r for r in self.materialized}
        functions, arguments, examples, body = {"pl": pl}, [], [], []
        for i, spec in enumerate(self.materialized):
            functions[f"copy_{i}"] = make_suffix_copy(spec)
            arguments.append(f"s{i}: pl.Tensor")
            dtype = torch.float32 if spec.dtype == "fp32" else torch.bfloat16
            examples.append(torch.empty(spec.source, dtype=dtype, device="meta"))
            body.append(f"    t{spec.bank} = copy_{i}(s{i}, t{spec.bank})")
        for bank, spec in self.outputs.items():
            arguments.append(f"t{bank}: pl.InOut[pl.Tensor]")
            dtype = torch.float32 if spec.dtype == "fp32" else torch.bfloat16
            examples.append(torch.empty(spec.target, dtype=dtype, device="meta"))
        args = ", ".join(arguments)
        calls = ", ".join(a.split(":")[0] for a in arguments)
        text = (f"def publish_chip({args}):\n" + "\n".join(body)
                + "\n    return " + ", ".join(f"t{b}" for b in self.outputs) + "\n"
                + f"def publish_host({args}):\n    publish_chip({calls}, device=0)\n")
        self.program = bind(text, functions, "publish").compile(*examples, config=config)

    def allocate(self, worker, *, scratch=None):
        import torch
        from math import prod
        from .loading import tensor_layout

        if scratch is not None:
            from pypto.runtime.task_interface import DataType
            shape, dtype = tensor_layout(scratch)
            if dtype != torch.float32 or prod(shape) * 4 < self.allocation_bytes:
                raise ValueError("publication scratch must have sufficient FP32 storage")
        destinations, offset = {}, 0
        for bank, spec in self.outputs.items():
            dtype = torch.float32 if spec.dtype == "fp32" else torch.bfloat16
            if scratch is None:
                value = worker.alloc_tensor(spec.target, dtype)
                destinations[bank] = Destination(value, value.data_ptr, 0)
            else:
                kind = DataType.FLOAT32 if spec.dtype == "fp32" else DataType.BFLOAT16
                value = scratch.buffer.tensor(spec.target, kind, byte_offset=offset)
                destinations[bank] = Destination(value, scratch.data_ptr, offset)
            offset += (prod(spec.target) * dtype.itemsize + 63) // 64 * 64
        return destinations

    @property
    def allocation_bytes(self):
        from math import prod
        # 同一 bank 的三层共享一个输出，不能按参数条目重复计显存。
        return sum((prod(r.target) * (4 if r.dtype == "fp32" else 2) + 63) // 64 * 64
                   for r in self.outputs.values())

    def run(self, worker, layout, ordinary, special, destinations):
        views = layout.views(ordinary, special)
        values = [views[r.name] for r in self.materialized]
        values += [destinations[b].tensor for b in self.outputs]
        worker.run(self.program, *values, config=self.config)

    def sources(self, layout, ordinary, destinations):
        entries = {r.parameter.name: r for r in layout.entries}
        return {r.bank: ((ordinary.data_ptr, entries[r.name].offset * 2) if r.mode == "direct"
                         else (destinations[r.bank].base, destinations[r.bank].offset))
                for r in self.plan if r.mode not in {"scalar", "rows"}}

    def readback(self, worker, layout, buffers, destinations, packed, host_buffers):
        for bank, (base, offset) in self.sources(layout, buffers[0], destinations).items():
            host = packed[bank]
            worker.copy_from(host.data_ptr(), base, host.nbytes, src_offset=offset)
        # 只搬运每层 64 个 decay/bias 值；不再把整块大矩阵交给 CPU 排布。
        entries = {r.parameter.name: r for r in layout.entries}
        for spec in self.plan:
            if spec.mode == "rows":
                # Z 的每个 rank 切片已连续，直接按层 DMA 到最终主机 bank，免去 180 MiB 设备副本。
                entry = entries[spec.name]
                columns = spec.source[1]
                for rank in range(2):
                    host = packed[spec.bank][rank, spec.slot * spec.local_rows:(spec.slot + 1) * spec.local_rows]
                    offset = (entry.offset + rank * spec.local_rows * columns) * 2
                    worker.copy_from(host.data_ptr(), buffers[0].data_ptr, host.nbytes, src_offset=offset)
                continue
            if spec.mode != "scalar":
                continue
            entry = entries[spec.name]
            special = entry.special is not None
            index = int(special)
            offset = entry.special if special else entry.offset
            source = host_buffers[index][0, offset:offset + 64]
            worker.copy_from(source.data_ptr(), buffers[index].data_ptr, source.nbytes,
                             src_offset=offset * source.element_size())
            for rank in range(2):
                target = packed[spec.bank][rank, spec.slot]
                target[:24].copy_(source[rank * 24:(rank + 1) * 24])
                target[24:].zero_()
