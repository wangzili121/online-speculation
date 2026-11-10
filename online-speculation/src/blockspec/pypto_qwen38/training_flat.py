"""多层参数的扁平布局；保留每层推理 bank 的形状和精度。"""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class Entry:
    parameter: object
    offset: int
    special: int | None


class FlatLayout:
    def __init__(self, parameters):
        self.entries = []
        cursor = special = 0
        for parameter in parameters:
            count = prod(parameter.shape)
            if count % 64:
                raise ValueError("64-aligned parameter required")
            floating = parameter.execution_dtype == "fp32"
            self.entries.append(Entry(parameter,cursor,special if floating else None))
            cursor += count
            special += count if floating else 0
        self.elements = (cursor+4095)//4096*4096
        self.special_elements = max(64,(special+63)//64*64)
        self.shape = (1,self.elements)
        self.special_shape = (1,self.special_elements)
        self.table = []
        for entry in self.entries:
            if entry.special is not None:
                for fragment in range(prod(entry.parameter.shape)//64):
                    self.table.append((entry.offset+fragment*64,entry.special+fragment*64,
                                       int(entry.parameter.offset_norm),0,0,0,0,0))

    def views(self, ordinary, special):
        from pypto.runtime.task_interface import DataType
        return {entry.parameter.name:(special if entry.special is not None else ordinary).buffer.tensor(
            entry.parameter.shape, DataType.FLOAT32 if entry.special is not None else DataType.BFLOAT16,
            byte_offset=entry.special*4 if entry.special is not None else entry.offset*2)
            for entry in self.entries}

    def gradients(self, handle):
        from pypto.runtime.task_interface import DataType
        return {entry.parameter.name:handle.buffer.tensor(entry.parameter.shape,DataType.FLOAT32,
                                                         byte_offset=entry.offset*4) for entry in self.entries}

    def banks(self, ordinary, special):
        tensors = self.views(ordinary,special)
        result = {}
        for entry in self.entries:
            result.setdefault(entry.parameter.layer,[]).append(tensors[entry.parameter.name])
        return {layer:tuple(values) for layer,values in result.items()}

    def export_banks(self, ordinary, special):
        import torch
        from .prepared_head import DeviceBank
        result = {}
        for entry in self.entries:
            p = entry.parameter
            bank = DeviceBank(special if entry.special is not None else ordinary,p.shape,
                torch.float32 if entry.special is not None else torch.bfloat16,
                entry.special*4 if entry.special is not None else entry.offset*2)
            result.setdefault(p.layer,[]).append(bank)
        return {layer:tuple(values) for layer,values in result.items()}

    def schema(self):
        return [dict(name=e.parameter.name,layer=e.parameter.layer,bank=e.parameter.bank,
                     shape=e.parameter.shape,raw_dtype=e.parameter.raw_dtype,
                     offset_norm=e.parameter.offset_norm,offset=e.offset,special=e.special) for e in self.entries]


def make_initialize(layout):
    import pypto.language as pl
    N,T,R = layout.elements,2048,len(layout.table)

    @pl.jit
    def initialize(raw: pl.Tensor, specials: pl.Tensor, table: pl.Tensor, master: pl.Out[pl.Tensor]):
        source = pl.reshape(raw,[N//T,T])
        output = pl.reshape(master,[N//T,T])
        with pl.spmd((N//T+15)//16,name_hint="orthrus_suffix_master_cast"):
            first = pl.tile.get_block_idx()*16
            for part in pl.range(16):
                row = first+part
                if row < N//T:
                    output[row:row+1,0:T] = pl.cast(pl.slice(source,[1,T],[row,0]),pl.FP32)
        with pl.spmd(R,name_hint="orthrus_suffix_master_fp32_parameters"):
            row = pl.tile.get_block_idx()
            kind = pl.read(table,[row,2])
            if kind == 0:
                offset = pl.read(table,[row,0])
                special = pl.read(table,[row,1])
                value = pl.slice(specials,[1,64],[0,special])
                master[0:1,offset:offset+64] = value
        return master

    @pl.jit.host
    def entry(raw: pl.Tensor, specials: pl.Tensor, table: pl.Tensor, master: pl.Out[pl.Tensor]):
        initialize(raw,specials,table,master,device=0)
    return entry


def make_render(layout):
    import pypto.language as pl
    from .training_optimizer_programs import make_publish
    ordinary = make_publish(layout.shape,inline=True)
    R = len(layout.table)

    @pl.jit
    def render(master: pl.Tensor, table: pl.Tensor, execution: pl.Out[pl.Tensor], specials: pl.Out[pl.Tensor]):
        execution = ordinary(master,execution)
        with pl.spmd(R,name_hint="orthrus_suffix_execution_fp32_parameters"):
            row = pl.tile.get_block_idx()
            offset = pl.read(table,[row,0])
            target = pl.read(table,[row,1])
            kind = pl.read(table,[row,2])
            value = pl.slice(master,[1,64],[0,offset])
            if kind == 1:
                value = pl.add(pl.cast(pl.cast(value,pl.BF16,mode="rint"),pl.FP32),1.)
            specials[0:1,target:target+64] = value
        return execution,specials

    @pl.jit.host
    def entry(master: pl.Tensor, table: pl.Tensor, execution: pl.Out[pl.Tensor], specials: pl.Out[pl.Tensor]):
        render(master,table,execution,specials,device=0)
    return entry


def make_summary(max_norm):
    import pypto.language as pl

    @pl.jit
    def calculate(sums: pl.Tensor, flags: pl.Tensor, output: pl.Out[pl.Tensor]):
        with pl.spmd(1,name_hint="orthrus_suffix_global_norm"):
            row = pl.tile.get_block_idx()
            repeated = pl.col_expand(pl.full([8,8],dtype=pl.FP32,value=1.),sums)
            norm = pl.row_expand(pl.full([8,8],dtype=pl.FP32,value=1.),pl.sqrt(pl.row_sum(repeated)))
            scale = pl.minimum(pl.mul(pl.recip(pl.add(norm,1e-6)),max_norm),1.)
            good = pl.col_expand(pl.full([8,8],dtype=pl.FP32,value=1.),flags)
            good = pl.row_expand(pl.full([8,8],dtype=pl.FP32,value=1.),pl.row_min(good))
            finite = pl.mul(good,pl.cmp(norm,3.4028234663852886e38,cmp_type=3))
            output[row:row+1,0:8] = pl.slice(norm,[1,8],[0,0])
            output[row+1:row+2,0:8] = pl.slice(scale,[1,8],[0,0])
            output[row+2:row+3,0:8] = pl.slice(finite,[1,8],[0,0])
        return output

    @pl.jit.host
    def entry(sums: pl.Tensor, flags: pl.Tensor, output: pl.Out[pl.Tensor]):
        calculate(sums,flags,output,device=0)
    return entry


def make_flat_zero(elements, dtype="fp32"):
    """大 flat 使用有界任务网格；不能按 256 元素直接展开成上百万个任务。"""
    if (type(elements) is not int or elements < 64 or elements%64
            or elements > 4096 and elements%4096 or dtype not in {"bf16","fp32"}):
        raise ValueError("4096-aligned large or 64-aligned small flat buffer required")
    import pypto.language as pl
    N,T = elements,min(4096,elements)
    D = pl.BF16 if dtype == "bf16" else pl.FP32

    @pl.jit
    def zero(result: pl.Out[pl.Tensor]):
        out = pl.reshape(result,[N//T,T])
        with pl.spmd((N//T+15)//16,name_hint="orthrus_suffix_flat_zero"):
            first = pl.tile.get_block_idx()*16
            for part in pl.range(16):
                row = first+part
                if row < N//T:
                    out[row:row+1,0:T] = pl.full([1,T],dtype=D,value=0.)
        return result

    @pl.jit.host
    def entry(result: pl.Out[pl.Tensor]):
        zero(result,device=0)
    return entry
