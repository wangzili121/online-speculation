"""last-N 共用一组 FP32 Adam 状态，避免按参数注册大量运行时入口。"""

import math
from math import prod

from .training_flat import FlatLayout, make_initialize, make_render, make_summary, make_flat_zero
from .training_optimizer_programs import make_accumulate, make_reduce, make_stats, make_norm_equal
from .training_programs import make_adam


class SuffixOptimizerPrograms:
    def __init__(self, graph, config, *, clip_norm):
        import torch
        self.graph,self.config = graph,config
        self.layout = FlatLayout(graph.parameters)
        shape = self.layout.shape
        meta = lambda shape,dtype=torch.float32:torch.empty(shape,dtype=dtype,device="meta")
        f,b,s = meta(shape),meta(shape,torch.bfloat16),meta(self.layout.special_shape)
        table = meta((len(self.layout.table),8),torch.int32)
        self.initialize = make_initialize(self.layout).compile(b,s,table,f,config=config)
        self.render = make_render(self.layout).compile(f,table,b,s,config=config)
        self.adam = make_adam(shape).compile(*[f]*7,1e-5,.1,.001,1.,config=config)
        self.accumulate = make_accumulate(shape).compile(f,f,f,config=config)
        self.zero = make_flat_zero(self.layout.elements,"fp32").compile(f,config=config)
        self.zero_raw = make_flat_zero(self.layout.elements,"bf16").compile(b,config=config)
        self.zero_special = make_flat_zero(self.layout.special_elements,"fp32").compile(s,config=config)
        entry,rows = make_stats(shape)
        self.stat = entry.compile(f,meta((rows,8)),meta((rows,8)),config=config)
        self.reductions,self.counts = [],[rows]
        while rows > 1:
            entry,count = make_reduce(rows)
            self.reductions.append(entry.compile(meta((rows,8)),meta((rows,8)),meta((count,8)),meta((count,8)),config=config))
            self.counts.append(count)
            rows = count
        self.summary = make_summary(clip_norm).compile(meta((1,8)),meta((1,8)),meta((3,8)),config=config)
        d = graph.shape.dimension
        self.equal = make_norm_equal(d).compile(meta((1,d)),meta((1,d)),meta((1,8)),config=config)

    def programs(self):
        return [self.initialize,self.render,self.adam,self.accumulate,self.zero,self.zero_raw,self.zero_special,
                self.stat,*self.reductions,self.summary,self.equal]


class SuffixOptimizer:
    def __init__(self, programs, worker):
        import torch
        self.programs,self.worker,self.handles = programs,worker,[]
        self.layout = programs.layout
        self.steps = self.index = 0
        try:
            self.state = [self.alloc(self.layout.shape,torch.float32) for _ in range(3)]
            self.candidate = [self.alloc(self.layout.shape,torch.float32) for _ in range(3)]
            self.scratch = self.alloc(self.layout.shape,torch.float32)
            self.gradients = self.layout.gradients(self.scratch)
            self.execution = [(self.alloc(self.layout.shape,torch.bfloat16),
                               self.alloc(self.layout.special_shape,torch.float32)) for _ in range(2)]
            self.stats = [[self.alloc((count,8),torch.float32) for _ in range(2)] for count in programs.counts]
            self.summary = self.alloc((3,8),torch.float32)
            self.summary_host = torch.empty((3,8),dtype=torch.float32)
            table = torch.tensor(self.layout.table,dtype=torch.int32)
            self.table = self.alloc(tuple(table.shape),table.dtype,init=table)
            # padding 参与全局 norm，初始化一次，图只覆盖真实参数区间。
            self.run(programs.zero,self.scratch)
        except BaseException:
            self.close()
            raise

    def alloc(self, shape, dtype, **kwargs):
        tensor = self.worker.alloc_tensor(shape,dtype,**kwargs)
        self.handles.append(tensor)
        return tensor

    def run(self, program, *args):
        self.worker.run(program,*args,config=self.programs.config)

    def initialize(self, head, raw_norms):
        import torch
        raw = special = None
        try:
            raw = self.worker.alloc_tensor(self.layout.shape,torch.bfloat16)
            special = self.worker.alloc_tensor(self.layout.special_shape,torch.float32)
            self.run(self.programs.zero_raw,raw)
            self.run(self.programs.zero_special,special)
            for entry in self.layout.entries:
                p = entry.parameter
                source = raw_norms[p.name] if p.offset_norm else head.banks(p.layer)[p.bank]
                expected = torch.float32 if p.raw_dtype == "fp32" else torch.bfloat16
                if tuple(source.shape) != p.shape or source.dtype != expected or not source.is_contiguous():
                    raise ValueError("raw suffix parameter layout differs: "+p.name)
                target,offset = (special,entry.special*4) if expected == torch.float32 else (raw,entry.offset*2)
                self.worker.copy_to(target.data_ptr,source.data_ptr(),source.numel()*source.element_size(),dst_offset=offset)
            self.run(self.programs.initialize,raw,special,self.table,self.state[0])
            for tensor in (*self.state[1:],*self.candidate):
                self.run(self.programs.zero,tensor)
            self.statistics(self.state[0])
            self.render(0,self.state[0])
            tensors = self.tensors()
            for entry in self.layout.entries:
                p = entry.parameter
                if p.offset_norm:
                    expected = flags = None
                    try:
                        original = head.banks(p.layer)[p.bank]
                        expected = self.worker.alloc_tensor(p.shape,torch.float32,init=original)
                        flags = self.worker.alloc_tensor((1,8),torch.float32)
                        self.run(self.programs.equal,expected,tensors[p.name],flags)
                        value = torch.empty((1,8),dtype=torch.float32)
                        self.worker.copy_from(value.data_ptr(),flags.data_ptr,32)
                        if not bool((value == 1).all()):
                            raise ValueError("raw training norm does not match inference head: "+p.name)
                    finally:
                        for tensor in (flags,expected):
                            if tensor is not None:
                                self.worker.free_tensor(tensor)
        finally:
            for tensor in (special,raw):
                if tensor is not None:
                    self.worker.free_tensor(tensor)

    def render(self, index, master):
        self.run(self.programs.render,master,self.table,*self.execution[index])

    def tensors(self):
        return self.layout.views(*self.execution[self.index])

    def zero_grad(self):
        self.run(self.programs.zero,self.candidate[0])

    def accumulate(self):
        self.run(self.programs.accumulate,self.candidate[0],self.scratch,self.candidate[0])

    def statistics(self, value):
        self.run(self.programs.stat,value,*self.stats[0])
        for i,entry in enumerate(self.programs.reductions):
            self.run(entry,*self.stats[i],*self.stats[i+1])
        self.run(self.programs.summary,*self.stats[-1],self.summary)
        self.worker.copy_from(self.summary_host.data_ptr(),self.summary.data_ptr,96)
        norm,scale,valid = self.summary_host[:,0].tolist()
        if valid != 1. or not math.isfinite(norm) or not math.isfinite(scale):
            raise FloatingPointError("nonfinite suffix update; serving head remains unchanged")
        return norm,scale

    def step(self, publisher, learning_rate):
        norm,scale = self.statistics(self.candidate[0])
        next_step,index = self.steps+1,1-self.index
        self.run(self.programs.adam,self.state[0],self.candidate[0],*self.state[1:],*self.candidate,
                 learning_rate,1-.9**next_step,1-.999**next_step,scale)
        self.statistics(self.candidate[0])
        self.render(index,self.candidate[0])
        publisher.publish_flat(self.layout,self.execution[index],expected_version=self.steps)
        self.state,self.candidate = self.candidate,self.state
        self.steps,self.index = next_step,index
        return dict(version=self.steps,gradient_norm=norm,clip_scale=scale)

    def memory_bytes(self):
        return sum(prod(value.shape)*value.dtype.itemsize for value in self.handles)

    def close(self):
        while self.handles:
            self.worker.free_tensor(self.handles[-1])
            self.handles.pop()
