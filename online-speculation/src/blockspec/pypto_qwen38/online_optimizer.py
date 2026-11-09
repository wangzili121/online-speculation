"""FP32 master AdamW：候选状态、裁剪、发布全部使用 PyPTO；主干仅作只读输入。"""

import math

from .loading import DeviceLoader, LoadingPrograms
from .training_programs import make_adam
from .training_optimizer_programs import (make_accumulate, make_global_stats, make_norm_equal,
                                         make_publish, make_reduce, make_stats)


class OptimizerPrograms:
    def __init__(self, graph, config, *, clip_norm):
        import torch
        self.graph, self.config = graph, config
        self.loader = LoadingPrograms(config)
        self.names = tuple(graph.gradients)
        self.shapes = {name: value.shape for name, value in graph.gradients.items()}
        self.entries, self.casts = {}, {}
        self.stats = {}
        meta = lambda shape, dtype=torch.float32: torch.empty(shape, dtype=dtype, device="meta")
        cache = {}
        for name, shape in self.shapes.items():
            self.casts[name] = self.loader.copy(shape, shape, torch.bfloat16, torch.float32)
            self.loader.zero(shape, torch.float32)
            if shape not in cache:
                args = [meta(shape) for _ in range(7)]
                adam = make_adam(shape).compile(*args, 1e-5, .1, .001, 1., config=config)
                accumulate = make_accumulate(shape).compile(meta(shape), meta(shape), meta(shape), config=config)
                stat, rows = make_stats(shape)
                stat = stat.compile(meta(shape), meta((rows, 8)), meta((rows, 8)), config=config)
                chain, n = [], rows
                while n > 1:
                    reducer, count = make_reduce(n)
                    compiled = reducer.compile(meta((n, 8)), meta((n, 8)), meta((count, 8)), meta((count, 8)), config=config)
                    chain.append((compiled, count))
                    n = count
                cache[shape] = adam, accumulate, stat, rows, chain
            adam, accumulate, stat, rows, chain = cache[shape]
            dtype = torch.float32 if "gamma" in name else torch.bfloat16
            publish = make_publish(shape, gamma="gamma" in name).compile(meta(shape), meta(shape, dtype), config=config)
            self.entries[name] = adam, accumulate, publish
            self.stats[name] = stat, rows, chain
        self.global_stats = make_global_stats(clip_norm).compile(*[meta((1, 8)) for _ in range(12)], meta((3, 8)), config=config)
        self.equal = make_norm_equal(graph.shape.dimension).compile(meta((1, graph.shape.dimension)),
            meta((1, graph.shape.dimension)), meta((1, 8)), config=config)

    def programs(self):
        result = list(self.loader.programs.values()) + [self.global_stats, self.equal]
        for name in self.names:
            result += list(self.entries[name])
            result += [self.stats[name][0], *(entry for entry, _ in self.stats[name][2])]
        return list({id(p): p for p in result}.values())


class DeviceOptimizer:
    def __init__(self, programs, worker):
        import torch
        self.programs, self.worker, self.handles = programs, worker, []
        self.state, self.candidate, self.execution, self.accum, self.stat_buffers = {}, {}, {}, {}, {}
        self.steps = 0
        self.index = 0
        try:
            for name, shape in programs.shapes.items():
                self.state[name] = [self.alloc(shape, torch.float32) for _ in range(3)]
                self.candidate[name] = [self.alloc(shape, torch.float32) for _ in range(3)]
                dtype = torch.float32 if "gamma" in name else torch.bfloat16
                self.execution[name] = [self.alloc(shape, dtype) for _ in range(2)]
                self.accum[name] = self.alloc(shape, torch.float32)
                _, rows, chain = programs.stats[name]
                self.stat_buffers[name] = [[self.alloc((count, 8), torch.float32) for _ in range(2)]
                                            for count in (rows, *(count for _, count in chain))]
            self.summary = self.alloc((3, 8), torch.float32)
            self.summary_host = torch.empty((3, 8), dtype=torch.float32).share_memory_()
        except BaseException:
            self.close()
            raise

    def alloc(self, shape, dtype, **kwargs):
        value = self.worker.alloc_tensor(shape, dtype, **kwargs)
        self.handles.append(value)
        return value

    def run(self, program, *args):
        self.worker.run(program, *args, config=self.programs.config)

    def initialize(self, head, raw_norms):
        """投影从已验证 bank 初始化，norm 必须提供原始 BF16，避免 1+gamma 逆变换丢信息。"""
        order = ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight")
        host = dict(zip(order, head.banks(63), strict=True))
        import torch
        loader = DeviceLoader(self.programs.loader, self.worker)
        try:
            for name, shape in self.programs.shapes.items():
                raw = raw_norms[name] if "gamma" in name else host[name]
                source = loader.allocate(shape, torch.bfloat16, init=raw.reshape(shape))
                loader.copy(self.programs.casts[name], source, self.state[name][0])
                loader.free(source)
                for value in (*self.state[name][1:], *self.candidate[name], self.accum[name]):
                    loader.zero(value)
                self.run(self.programs.entries[name][2], self.state[name][0], self.execution[name][0])
                if "gamma" in name:
                    expected = loader.allocate(shape, torch.float32, init=host[name])
                    flags = loader.allocate((1, 8), torch.float32)
                    self.run(self.programs.equal, expected, self.execution[name][0], flags)
                    loader.check(flags, "raw training norm does not match the loaded inference head")
                    loader.free(flags)
                    loader.free(expected)
        finally:
            loader.close()

    def zero_grad(self):
        loader = DeviceLoader(self.programs.loader, self.worker)
        try:
            for value in self.accum.values():
                loader.zero(value)
        finally:
            loader.close()

    def accumulate(self, gradients):
        if set(gradients) != set(self.programs.names):
            raise ValueError("all six last-layer gradients required")
        for name in self.programs.names:
            self.run(self.programs.entries[name][1], self.accum[name], gradients[name], self.accum[name])

    def statistics(self, values):
        sums, flags = [], []
        for name in self.programs.names:
            first, _, chain = self.programs.stats[name]
            buffers = self.stat_buffers[name]
            self.run(first, values[name], *buffers[0])
            for i, (entry, _) in enumerate(chain):
                self.run(entry, *buffers[i], *buffers[i+1])
            sums.append(buffers[-1][0])
            flags.append(buffers[-1][1])
        self.run(self.programs.global_stats, *sums, *flags, self.summary)
        self.worker.copy_from(self.summary_host.data_ptr(), self.summary.data_ptr, 96)
        norm, scale, valid = self.summary_host[:, 0].tolist()
        if valid != 1. or not math.isfinite(norm) or not math.isfinite(scale):
            raise FloatingPointError("nonfinite optimizer input; serving weights remain unchanged")
        return norm, scale

    def step(self, weights, learning_rate):
        norm, scale = self.statistics(self.accum)
        next_step, next_index = self.steps + 1, 1 - self.index
        for name in self.programs.names:
            master, first, second = self.state[name]
            self.run(self.programs.entries[name][0], master, self.accum[name], first, second,
                     *self.candidate[name], learning_rate, 1-.9**next_step, 1-.999**next_step, scale)
        self.statistics({name: state[0] for name, state in self.candidate.items()})
        for name in self.programs.names:
            self.run(self.programs.entries[name][2], self.candidate[name][0], self.execution[name][next_index])
        banks = tuple(self.execution[name][next_index] for name in
                      ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight"))
        weights.publish(63, banks, expected_version=self.steps)
        self.state, self.candidate = self.candidate, self.state
        self.steps, self.index = next_step, next_index
        return dict(version=self.steps, gradient_norm=norm, clip_scale=scale)

    def tensors(self):
        return {name: values[self.index] for name, values in self.execution.items()}

    def close(self):
        while self.handles:
            self.worker.free_tensor(self.handles[-1])
            self.handles.pop()
