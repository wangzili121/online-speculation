"""用 PyPTO 进行权重数值转换；主机仅读取文件和传递布局元数据。"""

import ctypes
from dataclasses import dataclass
from math import prod

import torch

from .loading_programs import make_convert, make_equal, make_reduce, make_zero


def dtype_name(dtype):
    if dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("BF16/FP32 weight formats required")
    return "bf16" if dtype == torch.bfloat16 else "fp32"


def tensor_layout(value):
    """句柄使用 Torch 元数据；Runtime Tensor 视图使用 shapes 和 dtype wire value。"""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape), value.dtype
    from pypto.runtime.task_interface import DataType
    dtype = {DataType.BFLOAT16.value: torch.bfloat16, DataType.FLOAT32.value: torch.float32}.get(value.dtype)
    return tuple(value.shapes), dtype


class LoadingPrograms:
    """先编译/注册所有规格，再启动 worker；不依赖运行时动态注册。"""

    def __init__(self, config):
        self.config, self.programs, self.copies = config, {}, {}

    @staticmethod
    def meta(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    def copy(self, shape, target_shape, source_dtype, target_dtype, *, conv=False, gamma=False, size=None):
        size = tuple(size or ((4, shape[0]) if conv else shape))
        key = (tuple(shape), tuple(target_shape), source_dtype, target_dtype, conv, gamma, size)
        if key not in self.copies:
            entry, rows = make_convert(shape, target_shape, source_dtype=dtype_name(source_dtype),
                                      target_dtype=dtype_name(target_dtype), transpose_conv=conv,
                                      add_one=gamma, copy_shape=size)
            flags = self.meta((rows, 8), torch.float32)
            print("compiling weight layout", key, flush=True)
            source_meta = self.meta((1, prod(shape)), torch.int16) if conv else self.meta(shape, source_dtype)
            program = entry.compile(source_meta, self.meta((1, 2048), torch.int32),
                                    self.meta(target_shape, target_dtype), flags, 0, 0, 0, 0, *size, config=self.config)
            self.programs[("copy", key)] = program
            chain, n = [], rows
            while n > 1:
                entry, next_rows = make_reduce(n)
                rkey = ("reduce", n)
                if rkey not in self.programs:
                    self.programs[rkey] = entry.compile(self.meta((n, 8), torch.float32),
                                                        self.meta((next_rows, 8), torch.float32), config=self.config)
                chain.append((self.programs[rkey], next_rows))
                n = next_rows
            self.copies[key] = program, rows, chain
        return key

    def zero(self, shape, dtype):
        key = ("zero", tuple(shape), dtype)
        if key not in self.programs:
            self.programs[key] = make_zero(shape, dtype_name(dtype)).compile(self.meta(shape, dtype), config=self.config)
        return key

    def equal(self, elements):
        key = ("equal", elements)
        if key not in self.programs:
            value = self.meta((1, elements), torch.bfloat16)
            self.programs[key] = make_equal(elements).compile(value, value, self.meta((1, 8), torch.float32), config=self.config)
        return key


class DeviceLoader:
    """所有数据转换在设备执行；最终只读取八个验证标志。"""

    def __init__(self, registry, worker):
        self.registry, self.worker, self.handles = registry, worker, []
        columns = torch.arange(2048, dtype=torch.int32).reshape(1, 2048)
        self.columns = self.allocate(tuple(columns.shape), columns.dtype, init=columns)
        self.flags_host = torch.empty((1, 8), dtype=torch.float32).share_memory_()
        self.flag_values = (ctypes.c_float * 8).from_address(self.flags_host.data_ptr())

    def allocate(self, shape, dtype, **options):
        handle = self.worker.alloc_tensor(tuple(shape), dtype, **options)
        self.handles.append(handle)
        return handle

    def free(self, handle):
        self.worker.free_tensor(handle)
        self.handles.remove(handle)

    def detach(self, handle):
        self.handles.remove(handle)
        return handle

    def close(self):
        while self.handles:
            self.worker.free_tensor(self.handles[-1])
            self.handles.pop()

    def run(self, program, *args):
        self.worker.run(program, *args, config=self.registry.config)

    def zero(self, target):
        shape, dtype = tensor_layout(target)
        self.run(self.registry.programs[("zero", shape, dtype)], target)

    def check(self, flags, message):
        self.worker.copy_from(self.flags_host.data_ptr(), flags.data_ptr, 32)
        if any(value not in (0., 1.) for value in self.flag_values):
            raise RuntimeError(f"invalid device weight-check flags: {list(self.flag_values)}")
        if any(value != 1. for value in self.flag_values):
            raise ValueError(message)

    def copy(self, key, source, target, *, source_offset=(0, 0), target_offset=(0, 0)):
        sm, sn = key[0]
        tm, tn = key[1]
        rows, cols = key[-1]
        sr, sc = source_offset
        tr, tc = target_offset
        if (any(type(n) is not int or n < 0 for n in (*source_offset, *target_offset))
                or tensor_layout(source) != (key[0], key[2])
                or tensor_layout(target) != (key[1], key[3])
                or tr + rows > tm or tc + cols > tn
                or (key[4] and (sc != 0 or sr + cols > sm))
                or (not key[4] and (sr + rows > sm or sc + cols > sn))):
            raise ValueError("weight copy exceeds its registered tensor layout")
        program, rows, chain = self.registry.copies[key]
        if key[4]:
            from pypto.runtime.task_interface import DataType
            source = source.buffer.tensor((1, prod(key[0])), DataType.INT16,
                                          byte_offset=getattr(source, "byte_offset", 0))
        temporaries = []
        try:
            flags = self.allocate((rows, 8), torch.float32)
            temporaries.append(flags)
            self.run(program, source, self.columns, target, flags, *source_offset, *target_offset, *key[-1])
            for reducer, count in chain:
                result = self.allocate((count, 8), torch.float32)
                temporaries.append(result)
                self.run(reducer, flags, result)
                flags = result
            self.check(flags, "nonfinite checkpoint weight")
        finally:
            for handle in reversed(temporaries):
                self.free(handle)

    def equal(self, a, b):
        flags = self.allocate((1, 8), torch.float32)
        try:
            self.run(self.registry.programs[("equal", prod(a.shape))], a, b, flags)
            self.check(flags, "replicated draft parameter differs across ranks")
        finally:
            self.free(flags)


@dataclass(frozen=True)
class Piece:
    rank: int
    source: int
    destination: int
    width: int


def parameter_pieces(layer, name, source_tp, layout):
    """仅计算文件区间和目标偏移；不读取或计算权重值。"""
    from .head import KEY_WIDTH, VALUE_WIDTH, tensor_schema
    from .tp import TPLayout
    TPLayout(source_tp)
    if type(layer) is not int or not 0 <= layer < 64:
        raise ValueError("layer must be in 0..63")
    shape = tensor_schema(layer)[name]
    if name in {"q_norm.weight", "k_norm.weight", "norm.weight"}:
        return 0, [Piece(rank, 0, 0, shape[0]) for rank in range(source_tp)]
    axis = 1 if name in {"o_proj.weight", "out_proj.weight"} else 0
    sections = ((KEY_WIDTH, KEY_WIDTH, VALUE_WIDTH) if name in {"in_proj_qkv.weight", "conv1d.weight"} else (shape[axis],))
    src, dst, pieces = 0, 0, []
    for width in sections:
        begin, end = layout.interval(width)
        shard = width // source_tp
        for rank in range(begin // shard, (end - 1) // shard + 1):
            start, stop = max(begin, rank * shard), min(end, (rank + 1) * shard)
            pieces.append(Piece(rank, src + start - rank * shard, dst + start - begin, stop - start))
        src += shard
        dst += end - begin
    return axis, pieces
