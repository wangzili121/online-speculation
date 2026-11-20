"""将 last-1 的渲染参数在设备上整理成 TP2 推理 bank。"""

from .head_loading import bank_specs


FULL_BANKS = {name: index + 9 for index, name in enumerate((
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "q_norm.weight", "k_norm.weight", "o_proj.weight"))}


def publication_plan(layout, selected):
    """参数仍使用原渲染精度；这里只按行/列分片或复制 norm。"""
    plan = []
    for entry in layout.entries:
        p = entry.parameter
        if p.layer != 63:
            raise ValueError("device publication currently requires last-1")
        name = bank_specs(p.layer)[p.bank][0]
        bank = FULL_BANKS[name]
        if bank not in selected:
            raise ValueError("rendered parameter has no publication bank")
        mode = "replicate" if p.offset_norm else "columns" if name == "o_proj.weight" else "rows"
        m, n = p.shape
        if mode == "rows":
            if m % 2:
                raise ValueError("TP2 row partition requires an even number of rows")
            target = (m, n)
        elif mode == "columns":
            if n % 2:
                raise ValueError("TP2 column partition requires an even number of columns")
            target = (2 * m, n // 2)
        else:
            target = (2 * m, n)
        plan.append((bank, p.name, p.shape, target, p.execution_dtype, mode))
    if {row[0] for row in plan} != set(selected):
        raise ValueError("publication bank set does not match the trainable layout")
    return tuple(sorted(plan))


def make_copy(source_shape, target_shape, mode):
    import pypto.language as pl
    M, N = target_shape
    R = 8 if M % 8 == 0 else 1
    T = next(t for t in (1024, 512, 256, 128, 64) if N % t == 0)
    tasks = M // R * (N // T)
    blocks = min(64, tasks)
    columns, replicate = mode == "columns", mode == "replicate"

    @pl.jit.inline
    def copy(source: pl.Tensor, target: pl.Out[pl.Tensor]):
        with pl.spmd(blocks, name_hint="orthrus_tp2_pack_parameter"):
            for tile in pl.range(pl.tile.get_block_idx(), tasks, blocks):
                row = tile // (N // T) * R
                column = tile % (N // T) * T
                if columns:
                    sr = row % source_shape[0]
                    sc = column + row // source_shape[0] * N
                elif replicate:
                    sr = row % source_shape[0]
                    sc = column
                else:
                    sr = row
                    sc = column
                target[row:row + R, column:column + T] = pl.slice(source, [R, T], [sr, sc])
        return target
    return copy


class PublicationPrograms:
    def __init__(self, layout, selected, config):
        import torch
        import pypto.language as pl
        self.plan, self.config = publication_plan(layout, selected), config
        # Q/K/V 的两份行分片在原渲染缓冲里已经连续排列，可直接 DMA，免去再次复制。
        self.materialized = tuple(row for row in self.plan if row[-1] != "rows")
        namespace = {"pl": pl}
        arguments, statements, outputs, examples = [], [], [], []
        for bank, name, shape, target, dtype, mode in self.materialized:
            kernel = f"copy_{bank}"
            namespace[kernel] = make_copy(shape, target, mode)
            arguments.extend([f"s{bank}: pl.Tensor", f"t{bank}: pl.Out[pl.Tensor]"])
            statements.append(f"    t{bank} = {kernel}(s{bank}, t{bank})")
            outputs.append(f"t{bank}")
            scalar = torch.float32 if dtype == "fp32" else torch.bfloat16
            examples.extend([torch.empty(shape, dtype=scalar, device="meta"),
                             torch.empty(target, dtype=scalar, device="meta")])
        args = ", ".join(arguments)
        calls = ", ".join(a.split(":")[0] for a in arguments)
        # 和训练图一样生成一个静态 HOST 入口，避免逐参数跨主机提交。
        source = (f"@pl.jit\ndef pack({args}):\n" + "\n".join(statements)
                  + "\n    return " + ", ".join(outputs) + "\n"
                  + f"@pl.jit.host\ndef entry({args}):\n    pack({calls}, device=0)\n")
        filename = "<orthrus_tp2_publication_" + "_".join(str(row[0]) for row in self.plan) + ">"
        import linecache
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        exec(compile(source, filename, "exec"), namespace)
        self.program = namespace["entry"].compile(*examples, config=config)

    def allocate(self, worker):
        import torch
        return {bank: worker.alloc_tensor(target, torch.float32 if dtype == "fp32" else torch.bfloat16)
                for bank, _, _, target, dtype, _ in self.materialized}

    def run(self, worker, layout, ordinary, special, destinations):
        views = layout.views(ordinary, special)
        args = [value for bank, name, *_ in self.materialized for value in (views[name], destinations[bank])]
        worker.run(self.program, *args, config=self.config)

    def sources(self, layout, ordinary, destinations):
        entries = {entry.parameter.name: entry for entry in layout.entries}
        # copy_from 要求 allocation base + 显式 src_offset，不能传内部裸地址。
        return {bank: ((ordinary.data_ptr, entries[name].offset * 2) if mode == "rows"
                       else (destinations[bank].data_ptr, 0))
                for bank, name, _, _, _, mode in self.plan}

    @property
    def allocation_bytes(self):
        from math import prod
        return sum(prod(target) * (4 if dtype == "fp32" else 2)
                   for _, _, _, target, dtype, _ in self.materialized)

    def readback(self, worker, layout, buffers, destinations, packed, host_buffers):
        for bank, (base, offset) in self.sources(layout, buffers[0], destinations).items():
            host = packed[bank]
            worker.copy_from(host.data_ptr(), base, host.nbytes, src_offset=offset)
